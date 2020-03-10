// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/fp16/depthwise_convolution_direct.h"

EE depthwise_convolution_direct_A76(TensorDesc inputDesc, F16* inArray,
    TensorDesc filterDesc, const F16* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode depthwiseActivationMode)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = convDesc.stride_h;
    U32 strideW = convDesc.stride_w;
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;
    U32 dilateH = convDesc.dilatedRate_h;
    U32 dilateW = convDesc.dilatedRate_w;

    if (fdf != DF_NCHWC8)
        CHECK_STATUS(NOT_MATCH);

    oc /= 8;
    ic /= 8;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ihiw = ih*iw;
    I32 ohow = oh*ow;

    for (U32 n = 0; n < in; n++) {
        F16 *inArray_pad = (F16*)tmp;
        F16 *inArray_pad_mov = inArray_pad;
        F16 *inArray_mov = inArray + n*ic*ihiw*8;
        for (U32 c = 0; c < ic; c++) {
            // copy input into a input with padding
            for (U32 h = 0; h < paddingT; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*sizeof(F16));
                inArray_pad_mov += iw_pad*8;
            }
            for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                memset(inArray_pad_mov, 0, paddingL*8*sizeof(F16));
                inArray_pad_mov += paddingL*8;
                memcpy(inArray_pad_mov, inArray_mov, iw*8*sizeof(F16));
                inArray_pad_mov += iw*8;
                inArray_mov += iw*8;
                memset(inArray_pad_mov, 0, paddingR*8*sizeof(F16));
                inArray_pad_mov += paddingR*8;
            }
            for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*sizeof(F16));
                inArray_pad_mov += iw_pad*8;
            }

            const F16 *b = biasArray + c*8;
            F16 *in_pad = inArray_pad + c*ih_pad*iw_pad*8;
            const F16 *f = filterArray + c*fh*fw*8;
            // ohow / 8
            for (I32 hw = 0; hw < ohow-7; hw+=8) {
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                U32 in_h_1 = (hw+1)/ow*strideH;
                U32 in_w_1 = (hw+1)%ow*strideW;
                U32 in_h_2 = (hw+2)/ow*strideH;
                U32 in_w_2 = (hw+2)%ow*strideW;
                U32 in_h_3 = (hw+3)/ow*strideH;
                U32 in_w_3 = (hw+3)%ow*strideW;
                U32 in_h_4 = (hw+4)/ow*strideH;
                U32 in_w_4 = (hw+4)%ow*strideW;
                U32 in_h_5 = (hw+5)/ow*strideH;
                U32 in_w_5 = (hw+5)%ow*strideW;
                U32 in_h_6 = (hw+6)/ow*strideH;
                U32 in_w_6 = (hw+6)%ow*strideW;
                U32 in_h_7 = (hw+7)/ow*strideH;
                U32 in_w_7 = (hw+7)%ow*strideW;
                F16 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;
                //TODO handle asm combined with c. No guarantee that compile will not use vec reg in c.
                __asm__ __volatile__(
                    "ldr q8, [%[b]]\n"
                    "mov v0.16b, v8.16b\n"
                    "mov v1.16b, v8.16b\n"
                    "mov v2.16b, v8.16b\n"
                    "mov v3.16b, v8.16b\n"
                    "mov v4.16b, v8.16b\n"
                    "mov v5.16b, v8.16b\n"
                    "mov v6.16b, v8.16b\n"
                    "mov v7.16b, v8.16b\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F16 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        F16 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        F16 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;
                        F16 *in_1 = in_idx + in_h_1*iw_pad*8 + in_w_1*8;
                        F16 *in_2 = in_idx + in_h_2*iw_pad*8 + in_w_2*8;
                        F16 *in_3 = in_idx + in_h_3*iw_pad*8 + in_w_3*8;
                        F16 *in_4 = in_idx + in_h_4*iw_pad*8 + in_w_4*8;
                        F16 *in_5 = in_idx + in_h_5*iw_pad*8 + in_w_5*8;
                        F16 *in_6 = in_idx + in_h_6*iw_pad*8 + in_w_6*8;
                        F16 *in_7 = in_idx + in_h_7*iw_pad*8 + in_w_7*8;
                        __asm__ __volatile__(
                            "ldr q17, [%[f0]]\n"
                            "ldr q9, [%[in0]]\n"
                            "ldr q10, [%[in1]]\n"
                            "ldr q11, [%[in2]]\n"
                            "ldr q12, [%[in3]]\n"
                            "ldr q13, [%[in4]]\n"
                            "ldr q14, [%[in5]]\n"
                            "ldr q15, [%[in6]]\n"
                            "ldr q16, [%[in7]]\n"
                            "fmla v0.8h,  v9.8h, v17.8h\n"
                            "fmla v1.8h, v10.8h, v17.8h\n"
                            "fmla v2.8h, v11.8h, v17.8h\n"
                            "fmla v3.8h, v12.8h, v17.8h\n"
                            "fmla v4.8h, v13.8h, v17.8h\n"
                            "fmla v5.8h, v14.8h, v17.8h\n"
                            "fmla v6.8h, v15.8h, v17.8h\n"
                            "fmla v7.8h, v16.8h, v17.8h\n"
                            :
                            :[in0]"r"(in_0),
                             [in1]"r"(in_1),
                             [in2]"r"(in_2),
                             [in3]"r"(in_3),
                             [in4]"r"(in_4),
                             [in5]"r"(in_5),
                             [in6]"r"(in_6),
                             [in7]"r"(in_7),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationMode){
                    case ACTIVATION_NULL:
                         break;
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fmax v0.8h, v0.8h, v31.8h\n"
                            "fmax v1.8h, v1.8h, v31.8h\n"
                            "fmax v2.8h, v2.8h, v31.8h\n"
                            "fmax v3.8h, v3.8h, v31.8h\n"
                            "fmax v4.8h, v4.8h, v31.8h\n"
                            "fmax v5.8h, v5.8h, v31.8h\n"
                            "fmax v6.8h, v6.8h, v31.8h\n"
                            "fmax v7.8h, v7.8h, v31.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "movi v30.8h, #0x46, lsl #8\n"  // six
                            "fmax v0.8h, v0.8h, v31.8h\n"
                            "fmax v1.8h, v1.8h, v31.8h\n"
                            "fmax v2.8h, v2.8h, v31.8h\n"
                            "fmax v3.8h, v3.8h, v31.8h\n"
                            "fmax v4.8h, v4.8h, v31.8h\n"
                            "fmax v5.8h, v5.8h, v31.8h\n"
                            "fmax v6.8h, v6.8h, v31.8h\n"
                            "fmax v7.8h, v7.8h, v31.8h\n"
                            "fmin v0.8h, v0.8h, v30.8h\n"
                            "fmin v1.8h, v1.8h, v30.8h\n"
                            "fmin v2.8h, v2.8h, v30.8h\n"
                            "fmin v3.8h, v3.8h, v30.8h\n"
                            "fmin v4.8h, v4.8h, v30.8h\n"
                            "fmin v5.8h, v5.8h, v30.8h\n"
                            "fmin v6.8h, v6.8h, v30.8h\n"
                            "fmin v7.8h, v7.8h, v30.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v30", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        __asm__ __volatile__(
                            "movi v29.8h, #0x42, lsl #8\n"  // three
                            "movi v30.8h, #0x46, lsl #8\n"  // six
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fadd v21.8h,  v0.8h, v29.8h\n"
                            "fadd v22.8h,  v1.8h, v29.8h\n"
                            "fadd v23.8h,  v2.8h, v29.8h\n"
                            "fadd v24.8h,  v3.8h, v29.8h\n"
                            "fadd v25.8h,  v4.8h, v29.8h\n"
                            "fadd v26.8h,  v5.8h, v29.8h\n"
                            "fadd v27.8h,  v6.8h, v29.8h\n"
                            "fadd v28.8h,  v7.8h, v29.8h\n"
                            "fmax v21.8h, v21.8h, v31.8h\n"
                            "fmax v22.8h, v22.8h, v31.8h\n"
                            "fmax v23.8h, v23.8h, v31.8h\n"
                            "fmax v24.8h, v24.8h, v31.8h\n"
                            "fmax v25.8h, v25.8h, v31.8h\n"
                            "fmax v26.8h, v26.8h, v31.8h\n"
                            "fmax v27.8h, v27.8h, v31.8h\n"
                            "fmax v28.8h, v28.8h, v31.8h\n"
                            "fmin v21.8h, v21.8h, v30.8h\n"
                            "fmin v22.8h, v22.8h, v30.8h\n"
                            "fmin v23.8h, v23.8h, v30.8h\n"
                            "fmin v24.8h, v24.8h, v30.8h\n"
                            "fmin v25.8h, v25.8h, v30.8h\n"
                            "fmin v26.8h, v26.8h, v30.8h\n"
                            "fmin v27.8h, v27.8h, v30.8h\n"
                            "fmin v28.8h, v28.8h, v30.8h\n"
                            "fdiv v21.8h, v21.8h, v30.8h\n"
                            "fdiv v22.8h, v22.8h, v30.8h\n"
                            "fdiv v23.8h, v23.8h, v30.8h\n"
                            "fdiv v24.8h, v24.8h, v30.8h\n"
                            "fdiv v25.8h, v25.8h, v30.8h\n"
                            "fdiv v26.8h, v26.8h, v30.8h\n"
                            "fdiv v27.8h, v27.8h, v30.8h\n"
                            "fdiv v28.8h, v28.8h, v30.8h\n"
                            "fmul  v0.8h,  v0.8h, v21.8h\n"
                            "fmul  v1.8h,  v1.8h, v22.8h\n"
                            "fmul  v2.8h,  v2.8h, v23.8h\n"
                            "fmul  v3.8h,  v3.8h, v24.8h\n"
                            "fmul  v4.8h,  v4.8h, v25.8h\n"
                            "fmul  v5.8h,  v5.8h, v26.8h\n"
                            "fmul  v6.8h,  v6.8h, v27.8h\n"
                            "fmul  v7.8h,  v7.8h, v28.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                              "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__(
                    "str q0, [%[out]]\n"
                    "str q1, [%[out], #16]\n"
                    "str q2, [%[out], #32]\n"
                    "str q3, [%[out], #48]\n"
                    "str q4, [%[out], #64]\n"
                    "str q5, [%[out], #80]\n"
                    "str q6, [%[out], #96]\n"
                    "str q7, [%[out], #112]\n"
                    :[out]"+r"(out_ptr)
                    :
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                      "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
                );
            }

            // ohow_reminder % 8 / 4
            U32 ohow_s = (ohow / 8) * 8;
            for (I32 hw = ohow_s; hw < ohow-3; hw+=4) {
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                U32 in_h_1 = (hw+1)/ow*strideH;
                U32 in_w_1 = (hw+1)%ow*strideW;
                U32 in_h_2 = (hw+2)/ow*strideH;
                U32 in_w_2 = (hw+2)%ow*strideW;
                U32 in_h_3 = (hw+3)/ow*strideH;
                U32 in_w_3 = (hw+3)%ow*strideW;
                F16 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;
                //TODO handle asm combined with c. No guarantee that compile will not use vec reg in c.
                __asm__ __volatile__(
                    "ldr q8, [%[b]]\n"
                    "mov v0.16b, v8.16b\n"
                    "mov v1.16b, v8.16b\n"
                    "mov v2.16b, v8.16b\n"
                    "mov v3.16b, v8.16b\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v8"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F16 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        F16 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        F16 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;
                        F16 *in_1 = in_idx + in_h_1*iw_pad*8 + in_w_1*8;
                        F16 *in_2 = in_idx + in_h_2*iw_pad*8 + in_w_2*8;
                        F16 *in_3 = in_idx + in_h_3*iw_pad*8 + in_w_3*8;
                        __asm__ __volatile__(
                            "ldr q17, [%[f0]]\n"
                            "ldr q9, [%[in0]]\n"
                            "ldr q10, [%[in1]]\n"
                            "ldr q11, [%[in2]]\n"
                            "ldr q12, [%[in3]]\n"
                            "fmla v0.8h, v9.8h, v17.8h\n"
                            "fmla v1.8h, v10.8h, v17.8h\n"
                            "fmla v2.8h, v11.8h, v17.8h\n"
                            "fmla v3.8h, v12.8h, v17.8h\n"
                            :
                            :[in0]"r"(in_0),
                             [in1]"r"(in_1),
                             [in2]"r"(in_2),
                             [in3]"r"(in_3),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v9", "v10", "v11", "v12", "v17"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationMode){
                    case ACTIVATION_NULL:
                         break;
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fmax v0.8h, v0.8h, v31.8h\n"
                            "fmax v1.8h, v1.8h, v31.8h\n"
                            "fmax v2.8h, v2.8h, v31.8h\n"
                            "fmax v3.8h, v3.8h, v31.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "movi v30.8h, #0x46, lsl #8\n"  // six
                            "fmax v0.8h, v0.8h, v31.8h\n"
                            "fmax v1.8h, v1.8h, v31.8h\n"
                            "fmax v2.8h, v2.8h, v31.8h\n"
                            "fmax v3.8h, v3.8h, v31.8h\n"
                            "fmin v0.8h, v0.8h, v30.8h\n"
                            "fmin v1.8h, v1.8h, v30.8h\n"
                            "fmin v2.8h, v2.8h, v30.8h\n"
                            "fmin v3.8h, v3.8h, v30.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v30", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        __asm__ __volatile__(
                            "movi v29.8h, #0x42, lsl #8\n"  // three
                            "movi v30.8h, #0x46, lsl #8\n"  // six
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fadd v25.8h,  v0.8h, v29.8h\n"
                            "fadd v26.8h,  v1.8h, v29.8h\n"
                            "fadd v27.8h,  v2.8h, v29.8h\n"
                            "fadd v28.8h,  v3.8h, v29.8h\n"
                            "fmax v25.8h, v25.8h, v31.8h\n"
                            "fmax v26.8h, v26.8h, v31.8h\n"
                            "fmax v27.8h, v27.8h, v31.8h\n"
                            "fmax v28.8h, v28.8h, v31.8h\n"
                            "fmin v25.8h, v25.8h, v30.8h\n"
                            "fmin v26.8h, v26.8h, v30.8h\n"
                            "fmin v27.8h, v27.8h, v30.8h\n"
                            "fmin v28.8h, v28.8h, v30.8h\n"
                            "fdiv v25.8h, v25.8h, v30.8h\n"
                            "fdiv v26.8h, v26.8h, v30.8h\n"
                            "fdiv v27.8h, v27.8h, v30.8h\n"
                            "fdiv v28.8h, v28.8h, v30.8h\n"
                            "fmul v0.8h,   v0.8h, v25.8h\n"
                            "fmul v1.8h,   v1.8h, v26.8h\n"
                            "fmul v2.8h,   v2.8h, v27.8h\n"
                            "fmul v3.8h,   v3.8h, v28.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__(
                    "st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [%[out]]\n"
                    :[out]"+r"(out_ptr)
                    :
                    :"memory", "cc", "v0", "v1", "v2", "v3"
                );
            }

            // ohow_reminder % 4
            ohow_s = (ohow / 4) * 4;
            for (I32 hw = ohow_s; hw < ohow; hw++) {
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                F16 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;
                //TODO handle asm combined with c. No guarantee that compile will not use vec reg in c.
                __asm__ __volatile__(
                    "ldr q8, [%[b]]\n"
                    "mov v0.16b, v8.16b\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v0"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F16 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        F16 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        F16 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;
                        __asm__ __volatile__(
                            "ldr q17, [%[f0]]\n"
                            "ldr q9, [%[in0]]\n"
                            "fmla v0.8h, v9.8h, v17.8h\n"
                            :
                            :[in0]"r"(in_0),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v9", "v17"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationMode){
                    case ACTIVATION_NULL:
                         break;
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fmax v0.8h, v0.8h, v31.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "movi v30.8h, #0x46, lsl #8\n"  // six
                            "fmax v0.8h, v0.8h, v31.8h\n"
                            "fmin v0.8h, v0.8h, v30.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v30", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        __asm__ __volatile__(
                            "movi v29.8h, #0x42, lsl #8\n"  // three
                            "movi v30.8h, #0x46, lsl #8\n"  // six
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fadd v28.8h,  v0.8h, v29.8h\n"
                            "fmax v28.8h, v28.8h, v31.8h\n"
                            "fmin v28.8h, v28.8h, v30.8h\n"
                            "fdiv v28.8h, v28.8h, v30.8h\n"
                            "fmul v0.8h,   v0.8h, v28.8h\n"
                            :
                            :
                            :"memory", "cc", "v0", "v28", "v29", "v30", "v31"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__(
                    "str q0, [%[out]]\n"
                    :[out]"+r"(out_ptr)
                    :
                    :"memory", "cc", "v0"
                );
            }
        }
    }
    return SUCCESS;
}
