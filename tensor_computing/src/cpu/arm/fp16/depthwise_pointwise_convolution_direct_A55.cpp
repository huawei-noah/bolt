// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/fp16/depthwise_pointwise_convolution_direct.h"

EE depthwise_pointwise_convolution_direct_A55(TensorDesc inputDesc, F16* inArray,
    TensorDesc filterDesc, const F16* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode)
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

    if (fdf != DF_CHWC8_NCN16)
        CHECK_STATUS(NOT_MATCH);

    oc /= 8;
    ic /= 8;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ihiw = ih*iw;
    I32 ohow = oh*ow;
    F16 *pwArray = (F16*)tmp + ic*ih_pad*iw_pad*8;

    for (U32 n = 0; n < in; n++) {
        // copy input into a input with padding
        F16 *inArray_pad = (F16*)tmp;
        F16 *inArray_pad_mov = inArray_pad;
        F16 *inArray_mov = inArray + n*ic*ihiw*8;
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < paddingT; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*bytesOf(fdt));
                inArray_pad_mov += iw_pad*8;
            }
            for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                memset(inArray_pad_mov, 0, paddingL*8*bytesOf(fdt));
                inArray_pad_mov += paddingL*8;
                memcpy(inArray_pad_mov, inArray_mov, iw*8*bytesOf(fdt));
                inArray_pad_mov += iw*8;
                inArray_mov += iw*8;
                memset(inArray_pad_mov, 0, paddingR*8*bytesOf(fdt));
                inArray_pad_mov += paddingR*8;
            }
            for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*bytesOf(fdt));
                inArray_pad_mov += iw_pad*8;
            }
        }

        // dw_conv
        for (U32 c = 0; c < ic ; c++) {
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
                F16 *pw_pack_0 = pwArray + hw*ic*8 + c*8*8;
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
                    "zip1  v8.8h, v0.8h, v4.8h\n"
                    "zip1  v9.8h, v2.8h, v6.8h\n"
                    "zip1 v10.8h, v1.8h, v5.8h\n"
                    "zip1 v11.8h, v3.8h, v7.8h\n"
                    "zip2  v0.8h, v0.8h, v4.8h\n"
                    "zip2  v2.8h, v2.8h, v6.8h\n"
                    "zip2  v1.8h, v1.8h, v5.8h\n"
                    "zip2  v3.8h, v3.8h, v7.8h\n"
                    "zip1 v12.8h,  v8.8h,  v9.8h\n"
                    "zip1 v13.8h, v10.8h, v11.8h\n"
                    "zip2  v8.8h,  v8.8h,  v9.8h\n"
                    "zip2 v10.8h, v10.8h, v11.8h\n"
                    "zip1 v14.8h,  v0.8h,  v2.8h\n"
                    "zip1 v15.8h,  v1.8h,  v3.8h\n"
                    "zip2  v0.8h,  v0.8h,  v2.8h\n"
                    "zip2  v1.8h,  v1.8h,  v3.8h\n"
                    "zip1 v16.8h, v12.8h, v13.8h\n"
                    "zip2 v12.8h, v12.8h, v13.8h\n"
                    "zip1 v17.8h,  v8.8h, v10.8h\n"
                    "zip2  v8.8h,  v8.8h, v10.8h\n"
                    "zip1 v18.8h, v14.8h, v15.8h\n"
                    "zip2 v14.8h, v14.8h, v15.8h\n"
                    "zip1 v19.8h,  v0.8h,  v1.8h\n"
                    "zip2  v0.8h,  v0.8h,  v1.8h\n"
                    "str q16, [%[pw0]]\n"
                    "str q12, [%[pw0], #16]\n"
                    "str q17, [%[pw0], #32]\n"
                    "str q8, [%[pw0], #48]\n"
                    "str q18, [%[pw0], #64]\n"
                    "str q14, [%[pw0], #80]\n"
                    "str q19, [%[pw0], #96]\n"
                    "str q0, [%[pw0], #112]\n"
                    :[pw0]"+r"(pw_pack_0)
                    :
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
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
                F16 *pw_pack_0 = pwArray + hw*ic*8 + c*8*4;
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
                    "st4 {v0.8h, v1.8h, v2.8h, v3.8h}, [%[pw0]]\n"
                    :[pw0]"+r"(pw_pack_0)
                    :
                    :"memory", "cc", "v0", "v1", "v2", "v3"
                );
            }

            // ohow_reminder % 4
            ohow_s = (ohow / 4) * 4;
            for (I32 hw = ohow_s; hw < ohow; hw++) {
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                F16 *pw_pack_0 = pwArray + hw*ic*8 + c*8;
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
                    "str q0, [%[pw0]]\n"
                    :[pw0]"+r"(pw_pack_0)
                    :
                    :"memory", "cc", "v0"
                );
            }
        }

        // pw_conv
        // ohow / 8
        for (I32 hw = 0; hw < ohow-7; hw+=8) {
            const F16 *b0 = biasArray + ic*8;
            const F16 *b1 = b0 + 8;
            F16 *in_pack = pwArray + hw*ic*8;
            const F16 *f_o0c0 = filterArray + ic*fh*fw*8;
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // bias
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr d22, [%[b_0]]\n"       //b_o0
                    "ldr  x1, [%[b_0], #8]\n"
                    "ins v22.d[1], x1\n"
                    "ldr d23, [%[b_1]]\n"       //b_o1
                    "ldr  x2, [%[b_1], #8]\n"
                    "ins v23.d[1], x2\n"
                    "mov  x0, %[ic]\n"             //ic_blk
                    "mov  v2.16b, v22.16b\n"      //out_o0hw0
                    "ldr  d0, [%[in_0]]\n"           //in_hw0
                    "mov  v3.16b, v22.16b\n"      //out_o0hw1
                    "ldr  x1, [%[in_0], #8]\n"
                    "mov  v4.16b, v22.16b\n"      //out_o0hw2
                    "ins  v0.d[1], x1\n"
                    "mov  v5.16b, v22.16b\n"      //out_o0hw3
                    "ldr d18, [%[f_0]]\n"            //f_o0c0
                    "mov  v6.16b, v22.16b\n"      //out_o0hw4
                    "ldr  x2, [%[f_0], #8]\n"
                    "mov  v7.16b, v22.16b\n"      //out_o0hw5
                    "ins v18.d[1], x2\n"
                    "mov  v8.16b, v22.16b\n"      //out_o0hw6
                    "ldr d19, [%[f_0], #16]\n"            //f_o1c0
                    "mov  v9.16b, v22.16b\n"      //out_o0hw7
                    "ldr  x3, [%[f_0], #24]\n"
                    "mov v10.16b, v23.16b\n"      //out_o1hw0
                    "ins v19.d[1], x3\n"
                    "mov v11.16b, v23.16b\n"      //out_o1hw1
                    "mov v12.16b, v23.16b\n"      //out_o1hw2
                    "mov v13.16b, v23.16b\n"      //out_o1hw3
                    "mov v14.16b, v23.16b\n"      //out_o1hw4
                    "mov v15.16b, v23.16b\n"      //out_o1hw5
                    "mov v16.16b, v23.16b\n"      //out_o1hw6
                    "mov v17.16b, v23.16b\n"      //out_o1hw7
                    "0:\n"
                    "ldr  d1, [%[in_0], #16]\n"           //in_hw0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "ldr  x1, [%[in_0], #24]\n"
                    "fmla  v3.8h, v18.8h, v0.h[1]\n"
                    "ins  v1.d[1], x1\n"
                    "fmla  v4.8h, v18.8h, v0.h[2]\n"
                    "ldr d20, [%[f_0], #32]\n"            //f_o0c0
                    "fmla  v5.8h, v18.8h, v0.h[3]\n"
                    "ldr  x2, [%[f_0], #40]\n"
                    "fmla  v6.8h, v18.8h, v0.h[4]\n"
                    "ins v20.d[1], x2\n"
                    "fmla  v7.8h, v18.8h, v0.h[5]\n"
                    "ldr d21, [%[f_0], #48]\n"            //f_o1c0
                    "fmla  v8.8h, v18.8h, v0.h[6]\n"
                    "ldr  x3, [%[f_0], #56]\n"
                    "fmla  v9.8h, v18.8h, v0.h[7]\n"
                    "ins v21.d[1], x3\n"
                    "fmla v10.8h, v19.8h, v0.h[0]\n"
                    "fmla v11.8h, v19.8h, v0.h[1]\n"
                    "fmla v12.8h, v19.8h, v0.h[2]\n"
                    "fmla v13.8h, v19.8h, v0.h[3]\n"
                    "fmla v14.8h, v19.8h, v0.h[4]\n"
                    "fmla v15.8h, v19.8h, v0.h[5]\n"
                    "fmla v16.8h, v19.8h, v0.h[6]\n"
                    "fmla v17.8h, v19.8h, v0.h[7]\n"

                    "ldr  d0, [%[in_0], #32]\n"           //in_hw0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "ldr  x1, [%[in_0], #40]\n"
                    "fmla  v3.8h, v20.8h, v1.h[1]\n"
                    "ins  v0.d[1], x1\n"
                    "fmla  v4.8h, v20.8h, v1.h[2]\n"
                    "ldr d18, [%[f_0], #64]\n"            //f_o0c0
                    "fmla  v5.8h, v20.8h, v1.h[3]\n"
                    "ldr  x2, [%[f_0], #72]\n"
                    "fmla  v6.8h, v20.8h, v1.h[4]\n"
                    "ins v18.d[1], x2\n"
                    "fmla  v7.8h, v20.8h, v1.h[5]\n"
                    "ldr d19, [%[f_0], #80]\n"            //f_o1c0
                    "fmla  v8.8h, v20.8h, v1.h[6]\n"
                    "ldr  x3, [%[f_0], #88]\n"
                    "fmla  v9.8h, v20.8h, v1.h[7]\n"
                    "ins v19.d[1], x3\n"
                    "fmla v10.8h, v21.8h, v1.h[0]\n"
                    "add %[in_0], %[in_0], #32\n"
                    "fmla v11.8h, v21.8h, v1.h[1]\n"
                    "add %[f_0], %[f_0], #64\n"
                    "fmla v12.8h, v21.8h, v1.h[2]\n"
                    "subs x0, x0, #2\n"
                    "fmla v13.8h, v21.8h, v1.h[3]\n"
                    "fmla v14.8h, v21.8h, v1.h[4]\n"
                    "fmla v15.8h, v21.8h, v1.h[5]\n"
                    "fmla v16.8h, v21.8h, v1.h[6]\n"
                    "fmla v17.8h, v21.8h, v1.h[7]\n"
                    "bne 0b\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 11f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax  v6.8h,  v6.8h, v0.8h\n"
                    "fmax  v7.8h,  v7.8h, v0.8h\n"
                    "fmax  v8.8h,  v8.8h, v0.8h\n"
                    "fmax  v9.8h,  v9.8h, v0.8h\n"
                    "fmax v10.8h, v10.8h, v0.8h\n"
                    "fmax v11.8h, v11.8h, v0.8h\n"
                    "fmax v12.8h, v12.8h, v0.8h\n"
                    "fmax v13.8h, v13.8h, v0.8h\n"
                    "fmax v14.8h, v14.8h, v0.8h\n"
                    "fmax v15.8h, v15.8h, v0.8h\n"
                    "fmax v16.8h, v16.8h, v0.8h\n"
                    "fmax v17.8h, v17.8h, v0.8h\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 12f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"     //six
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax  v6.8h,  v6.8h, v0.8h\n"
                    "fmax  v7.8h,  v7.8h, v0.8h\n"
                    "fmax  v8.8h,  v8.8h, v0.8h\n"
                    "fmax  v9.8h,  v9.8h, v0.8h\n"
                    "fmax v10.8h, v10.8h, v0.8h\n"
                    "fmax v11.8h, v11.8h, v0.8h\n"
                    "fmax v12.8h, v12.8h, v0.8h\n"
                    "fmax v13.8h, v13.8h, v0.8h\n"
                    "fmax v14.8h, v14.8h, v0.8h\n"
                    "fmax v15.8h, v15.8h, v0.8h\n"
                    "fmax v16.8h, v16.8h, v0.8h\n"
                    "fmax v17.8h, v17.8h, v0.8h\n"
                    "fmin  v2.8h,  v2.8h, v1.8h\n"
                    "fmin  v3.8h,  v3.8h, v1.8h\n"
                    "fmin  v4.8h,  v4.8h, v1.8h\n"
                    "fmin  v5.8h,  v5.8h, v1.8h\n"
                    "fmin  v6.8h,  v6.8h, v1.8h\n"
                    "fmin  v7.8h,  v7.8h, v1.8h\n"
                    "fmin  v8.8h,  v8.8h, v1.8h\n"
                    "fmin  v9.8h,  v9.8h, v1.8h\n"
                    "fmin v10.8h, v10.8h, v1.8h\n"
                    "fmin v11.8h, v11.8h, v1.8h\n"
                    "fmin v12.8h, v12.8h, v1.8h\n"
                    "fmin v13.8h, v13.8h, v1.8h\n"
                    "fmin v14.8h, v14.8h, v1.8h\n"
                    "fmin v15.8h, v15.8h, v1.8h\n"
                    "fmin v16.8h, v16.8h, v1.8h\n"
                    "fmin v17.8h, v17.8h, v1.8h\n"

                    "12:\n"
                    "cmp %[pointwiseActivationMode], %[am_h_swish]\n"
                    "bne 13f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "movi v18.8h, #0x42, lsl #8\n"     //three
                    "fadd v19.8h,  v2.8h, v18.8h\n"
                    "fadd v20.8h,  v3.8h, v18.8h\n"
                    "fadd v21.8h,  v4.8h, v18.8h\n"
                    "fadd v22.8h,  v5.8h, v18.8h\n"
                    "fadd v23.8h,  v6.8h, v18.8h\n"
                    "fadd v24.8h,  v7.8h, v18.8h\n"
                    "fadd v25.8h,  v8.8h, v18.8h\n"
                    "fadd v26.8h,  v9.8h, v18.8h\n"
                    "fmax v19.8h, v19.8h,  v0.8h\n"
                    "fmax v20.8h, v20.8h,  v0.8h\n"
                    "fmax v21.8h, v21.8h,  v0.8h\n"
                    "fmax v22.8h, v22.8h,  v0.8h\n"
                    "fmax v23.8h, v23.8h,  v0.8h\n"
                    "fmax v24.8h, v24.8h,  v0.8h\n"
                    "fmax v25.8h, v25.8h,  v0.8h\n"
                    "fmax v26.8h, v26.8h,  v0.8h\n"
                    "fmin v19.8h, v19.8h,  v1.8h\n"
                    "fmin v20.8h, v20.8h,  v1.8h\n"
                    "fmin v21.8h, v21.8h,  v1.8h\n"
                    "fmin v22.8h, v22.8h,  v1.8h\n"
                    "fmin v23.8h, v23.8h,  v1.8h\n"
                    "fmin v24.8h, v24.8h,  v1.8h\n"
                    "fmin v25.8h, v25.8h,  v1.8h\n"
                    "fmin v26.8h, v26.8h,  v1.8h\n"
                    "fdiv v19.8h, v19.8h,  v1.8h\n"
                    "fdiv v20.8h, v20.8h,  v1.8h\n"
                    "fdiv v21.8h, v21.8h,  v1.8h\n"
                    "fdiv v22.8h, v22.8h,  v1.8h\n"
                    "fdiv v23.8h, v23.8h,  v1.8h\n"
                    "fdiv v24.8h, v24.8h,  v1.8h\n"
                    "fdiv v25.8h, v25.8h,  v1.8h\n"
                    "fdiv v26.8h, v26.8h,  v1.8h\n"
                    "fmul  v2.8h, v19.8h,  v2.8h\n"
                    "fmul  v3.8h, v20.8h,  v3.8h\n"
                    "fmul  v4.8h, v21.8h,  v4.8h\n"
                    "fmul  v5.8h, v22.8h,  v5.8h\n"
                    "fmul  v6.8h, v23.8h,  v6.8h\n"
                    "fmul  v7.8h, v24.8h,  v7.8h\n"
                    "fmul  v8.8h, v25.8h,  v8.8h\n"
                    "fmul  v9.8h, v26.8h,  v9.8h\n"

                    "fadd v19.8h, v10.8h, v18.8h\n"
                    "fadd v20.8h, v11.8h, v18.8h\n"
                    "fadd v21.8h, v12.8h, v18.8h\n"
                    "fadd v22.8h, v13.8h, v18.8h\n"
                    "fadd v23.8h, v14.8h, v18.8h\n"
                    "fadd v24.8h, v15.8h, v18.8h\n"
                    "fadd v25.8h, v16.8h, v18.8h\n"
                    "fadd v26.8h, v17.8h, v18.8h\n"
                    "fmax v19.8h, v19.8h,  v0.8h\n"
                    "fmax v20.8h, v20.8h,  v0.8h\n"
                    "fmax v21.8h, v21.8h,  v0.8h\n"
                    "fmax v22.8h, v22.8h,  v0.8h\n"
                    "fmax v23.8h, v23.8h,  v0.8h\n"
                    "fmax v24.8h, v24.8h,  v0.8h\n"
                    "fmax v25.8h, v25.8h,  v0.8h\n"
                    "fmax v26.8h, v26.8h,  v0.8h\n"
                    "fmin v19.8h, v19.8h,  v1.8h\n"
                    "fmin v20.8h, v20.8h,  v1.8h\n"
                    "fmin v21.8h, v21.8h,  v1.8h\n"
                    "fmin v22.8h, v22.8h,  v1.8h\n"
                    "fmin v23.8h, v23.8h,  v1.8h\n"
                    "fmin v24.8h, v24.8h,  v1.8h\n"
                    "fmin v25.8h, v25.8h,  v1.8h\n"
                    "fmin v26.8h, v26.8h,  v1.8h\n"
                    "fdiv v19.8h, v19.8h,  v1.8h\n"
                    "fdiv v20.8h, v20.8h,  v1.8h\n"
                    "fdiv v21.8h, v21.8h,  v1.8h\n"
                    "fdiv v22.8h, v22.8h,  v1.8h\n"
                    "fdiv v23.8h, v23.8h,  v1.8h\n"
                    "fdiv v24.8h, v24.8h,  v1.8h\n"
                    "fdiv v25.8h, v25.8h,  v1.8h\n"
                    "fdiv v26.8h, v26.8h,  v1.8h\n"
                    "fmul v10.8h, v19.8h, v10.8h\n"
                    "fmul v11.8h, v20.8h, v11.8h\n"
                    "fmul v12.8h, v21.8h, v12.8h\n"
                    "fmul v13.8h, v22.8h, v13.8h\n"
                    "fmul v14.8h, v23.8h, v14.8h\n"
                    "fmul v15.8h, v24.8h, v15.8h\n"
                    "fmul v16.8h, v25.8h, v16.8h\n"
                    "fmul v17.8h, v26.8h, v17.8h\n"

                    "13:\n"
                    "st1 {v2.8h, v3.8h, v4.8h, v5.8h}, [%[out_0]], #64\n"
                    "st1 {v6.8h, v7.8h, v8.8h, v9.8h}, [%[out_0]], #64\n"
                    "st1 {v10.8h, v11.8h, v12.8h, v13.8h}, [%[out_1]], #64\n"
                    "st1 {v14.8h, v15.8h, v16.8h, v17.8h}, [%[out_1]], #64\n"
                    :[out_0]"+r"(out_o0hw0),
                     [out_1]"+r"(out_o1hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                                     "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                                     "v21", "v22", "v23", "v24", "v25", "v26", "x0", "x1", "x2", "x3"
                );
                b0 += 16;
                b1 += 16;
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + ic*fh*fw*8 + (oc-1)*8*ic*8;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + (oc-1)*ohow*8 + hw*8;
                // bias
                const F16 *b_o0 = biasArray + ic*8 + (oc-1)*8;
                __asm__ __volatile__(
                    "ldr q12, [%[b_0]]\n"       //b_o0
                    "mov x0, %[ic]\n" // ic_blk
                    "ldr d0, [%[in_0]]\n"   //in_hw0
                    "mov v2.16b, v12.16b\n" //out_o0hw0
                    "ldr x1, [%[in_0], #8]\n"
                    "mov v3.16b, v12.16b\n" //out_o0hw1
                    "ins v0.d[1], x1\n"
                    "mov v4.16b, v12.16b\n" //out_o0hw2
                    "ldr d10, [%[f_0]]\n"   //f_o0c0
                    "mov v5.16b, v12.16b\n" //out_o0hw3
                    "ldr x2, [%[f_0], #8]\n"
                    "mov v6.16b, v12.16b\n" //out_o0hw4
                    "ins v10.d[1], x2\n"
                    "mov v7.16b, v12.16b\n" //out_o0hw5
                    "mov v8.16b, v12.16b\n" //out_o0hw6
                    "mov v9.16b, v12.16b\n" //out_o0hw7
                    "0:\n"
                    "ldr d1, [%[in_0], #16]\n" //in_hw0
                    "fmla v2.8h, v10.8h, v0.h[0]\n"
                    "ldr x1, [%[in_0], #24]\n"
                    "fmla v3.8h, v10.8h, v0.h[1]\n"
                    "ins v1.d[1], x1\n"
                    "fmla v4.8h, v10.8h, v0.h[2]\n"
                    "ldr d11, [%[f_0], #16]\n" //f_o0c0
                    "fmla v5.8h, v10.8h, v0.h[3]\n"
                    "ldr x2, [%[f_0], #24]\n"
                    "fmla v6.8h, v10.8h, v0.h[4]\n"
                    "ins v11.d[1], x2\n"
                    "fmla v7.8h, v10.8h, v0.h[5]\n"
                    "subs x0, x0, #2\n"
                    "fmla v8.8h, v10.8h, v0.h[6]\n"
                    "fmla v9.8h, v10.8h, v0.h[7]\n"

                    "ldr d0, [%[in_0], #32]\n" //in_hw0
                    "fmla v2.8h, v11.8h, v1.h[0]\n"
                    "ldr x1, [%[in_0], #40]\n"
                    "fmla v3.8h, v11.8h, v1.h[1]\n"
                    "ins v0.d[1], x1\n"
                    "fmla v4.8h, v11.8h, v1.h[2]\n"
                    "ldr d10, [%[f_0], #32]\n" //f_o0c0
                    "fmla v5.8h, v11.8h, v1.h[3]\n"
                    "ldr x2, [%[f_0], #40]\n"
                    "fmla v6.8h, v11.8h, v1.h[4]\n"
                    "ins v10.d[1], x2\n"
                    "fmla v7.8h, v11.8h, v1.h[5]\n"
                    "add %[in_0], %[in_0], #32\n"
                    "fmla v8.8h, v11.8h, v1.h[6]\n"
                    "add %[f_0], %[f_0], #32\n"
                    "fmla v9.8h, v11.8h, v1.h[7]\n"
                    "bne 0b\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 11f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax  v6.8h,  v6.8h, v0.8h\n"
                    "fmax  v7.8h,  v7.8h, v0.8h\n"
                    "fmax  v8.8h,  v8.8h, v0.8h\n"
                    "fmax  v9.8h,  v9.8h, v0.8h\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 12f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"     //six
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax  v6.8h,  v6.8h, v0.8h\n"
                    "fmax  v7.8h,  v7.8h, v0.8h\n"
                    "fmax  v8.8h,  v8.8h, v0.8h\n"
                    "fmax  v9.8h,  v9.8h, v0.8h\n"
                    "fmin  v2.8h,  v2.8h, v1.8h\n"
                    "fmin  v3.8h,  v3.8h, v1.8h\n"
                    "fmin  v4.8h,  v4.8h, v1.8h\n"
                    "fmin  v5.8h,  v5.8h, v1.8h\n"
                    "fmin  v6.8h,  v6.8h, v1.8h\n"
                    "fmin  v7.8h,  v7.8h, v1.8h\n"
                    "fmin  v8.8h,  v8.8h, v1.8h\n"
                    "fmin  v9.8h,  v9.8h, v1.8h\n"

                    "12:\n"
                    "cmp %[pointwiseActivationMode], %[am_h_swish]\n"
                    "bne 13f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "movi v10.8h, #0x42, lsl #8\n"     //three
                    "fadd v11.8h,  v2.8h, v10.8h\n"
                    "fadd v12.8h,  v3.8h, v10.8h\n"
                    "fadd v13.8h,  v4.8h, v10.8h\n"
                    "fadd v14.8h,  v5.8h, v10.8h\n"
                    "fadd v15.8h,  v6.8h, v10.8h\n"
                    "fadd v16.8h,  v7.8h, v10.8h\n"
                    "fadd v17.8h,  v8.8h, v10.8h\n"
                    "fadd v18.8h,  v9.8h, v10.8h\n"
                    "fmax v11.8h, v11.8h,  v0.8h\n"
                    "fmax v12.8h, v12.8h,  v0.8h\n"
                    "fmax v13.8h, v13.8h,  v0.8h\n"
                    "fmax v14.8h, v14.8h,  v0.8h\n"
                    "fmax v15.8h, v15.8h,  v0.8h\n"
                    "fmax v16.8h, v16.8h,  v0.8h\n"
                    "fmax v17.8h, v17.8h,  v0.8h\n"
                    "fmax v18.8h, v18.8h,  v0.8h\n"
                    "fmin v11.8h, v11.8h,  v1.8h\n"
                    "fmin v12.8h, v12.8h,  v1.8h\n"
                    "fmin v13.8h, v13.8h,  v1.8h\n"
                    "fmin v14.8h, v14.8h,  v1.8h\n"
                    "fmin v15.8h, v15.8h,  v1.8h\n"
                    "fmin v16.8h, v16.8h,  v1.8h\n"
                    "fmin v17.8h, v17.8h,  v1.8h\n"
                    "fmin v18.8h, v18.8h,  v1.8h\n"
                    "fdiv v11.8h, v11.8h,  v1.8h\n"
                    "fdiv v12.8h, v12.8h,  v1.8h\n"
                    "fdiv v13.8h, v13.8h,  v1.8h\n"
                    "fdiv v14.8h, v14.8h,  v1.8h\n"
                    "fdiv v15.8h, v15.8h,  v1.8h\n"
                    "fdiv v16.8h, v16.8h,  v1.8h\n"
                    "fdiv v17.8h, v17.8h,  v1.8h\n"
                    "fdiv v18.8h, v18.8h,  v1.8h\n"
                    "fmul  v2.8h, v11.8h,  v2.8h\n"
                    "fmul  v3.8h, v12.8h,  v3.8h\n"
                    "fmul  v4.8h, v13.8h,  v4.8h\n"
                    "fmul  v5.8h, v14.8h,  v5.8h\n"
                    "fmul  v6.8h, v15.8h,  v6.8h\n"
                    "fmul  v7.8h, v16.8h,  v7.8h\n"
                    "fmul  v8.8h, v17.8h,  v8.8h\n"
                    "fmul  v9.8h, v18.8h,  v9.8h\n"

                    "13:\n"
                    "str q2, [%[out_0]]\n" //out_o0hw0
                    "str q3, [%[out_0], #16]\n" //out_o0hw0
                    "str q4, [%[out_0], #32]\n" //out_o0hw0
                    "str q5, [%[out_0], #48]\n" //out_o0hw0
                    "str q6, [%[out_0], #64]\n" //out_o0hw0
                    "str q7, [%[out_0], #80]\n" //out_o0hw0
                    "str q8, [%[out_0], #96]\n" //out_o0hw0
                    "str q9, [%[out_0], #112]\n" //out_o0hw0
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_r)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_o0),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                                     "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "x0", "x1", "x2"
                );
            }
        }

        // ohow_remainder % 8 / 4
        U32 ohow_s = (ohow / 8) * 8;
        for (I32 hw = ohow_s; hw < ohow-3; hw+=4) {
            const F16 *b0 = biasArray + ic*8;
            const F16 *b1 = b0 + 8;
            const F16 *f_o0c0 = filterArray + ic*fh*fw*8;
            F16 *in_pack = pwArray + hw*ic*8;
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // bias
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr d22, [%[b_0]]\n"       //b_o0
                    "ldr  x1, [%[b_0], #8]\n"
                    "ins v22.d[1], x1\n"
                    "ldr d23, [%[b_1]]\n"       //b_o1
                    "ldr  x2, [%[b_1], #8]\n"
                    "ins v23.d[1], x2\n"
                    "mov  x0, %[ic]\n"             //ic_blk
                    "mov  v2.16b, v22.16b\n"      //out_o0hw0
                    "ldr  d0, [%[in_0]]\n"           //in_hw0
                    "mov  v3.16b, v22.16b\n"      //out_o0hw1
                    "ldr d18, [%[f_0]]\n"            //f_o0c0
                    "mov  v4.16b, v22.16b\n"      //out_o0hw2
                    "ldr  x2, [%[f_0], #8]\n"
                    "mov  v5.16b, v22.16b\n"      //out_o0hw3
                    "ins v18.d[1], x2\n"
                    "mov v10.16b, v23.16b\n"      //out_o1hw0
                    "ldr d19, [%[f_0], #16]\n"            //f_o1c0
                    "mov v11.16b, v23.16b\n"      //out_o1hw1
                    "ldr  x3, [%[f_0], #24]\n"
                    "mov v12.16b, v23.16b\n"      //out_o1hw2
                    "ins v19.d[1], x3\n"
                    "mov v13.16b, v23.16b\n"      //out_o1hw3
                    "0:\n"
                    "ldr  d1, [%[in_0], #8]\n"           //in_hw0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "ldr d20, [%[f_0], #32]\n"            //f_o0c0
                    "fmla  v3.8h, v18.8h, v0.h[1]\n"
                    "ldr  x2, [%[f_0], #40]\n"
                    "fmla  v4.8h, v18.8h, v0.h[2]\n"
                    "ins v20.d[1], x2\n"
                    "fmla  v5.8h, v18.8h, v0.h[3]\n"
                    "ldr d21, [%[f_0], #48]\n"            //f_o1c0
                    "fmla v10.8h, v19.8h, v0.h[0]\n"
                    "ldr  x3, [%[f_0], #56]\n"
                    "fmla v11.8h, v19.8h, v0.h[1]\n"
                    "ins v21.d[1], x3\n"
                    "fmla v12.8h, v19.8h, v0.h[2]\n"
                    "subs x0, x0, #2\n"
                    "fmla v13.8h, v19.8h, v0.h[3]\n"

                    "ldr  d0, [%[in_0], #16]\n"           //in_hw0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "ldr d18, [%[f_0], #64]\n"            //f_o0c0
                    "fmla  v3.8h, v20.8h, v1.h[1]\n"
                    "ldr  x2, [%[f_0], #72]\n"
                    "fmla  v4.8h, v20.8h, v1.h[2]\n"
                    "ldr d19, [%[f_0], #80]\n"            //f_o1c0
                    "fmla  v5.8h, v20.8h, v1.h[3]\n"
                    "ins v18.d[1], x2\n"
                    "fmla v10.8h, v21.8h, v1.h[0]\n"
                    "ldr  x3, [%[f_0], #88]\n"
                    "fmla v11.8h, v21.8h, v1.h[1]\n"
                    "ins v19.d[1], x3\n"
                    "fmla v12.8h, v21.8h, v1.h[2]\n"
                    "add %[in_0], %[in_0], #16\n"
                    "fmla v13.8h, v21.8h, v1.h[3]\n"
                    "add %[f_0], %[f_0], #64\n"
                    "bne 0b\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 11f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax v10.8h, v10.8h, v0.8h\n"
                    "fmax v11.8h, v11.8h, v0.8h\n"
                    "fmax v12.8h, v12.8h, v0.8h\n"
                    "fmax v13.8h, v13.8h, v0.8h\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 12f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"     //six
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmax v10.8h, v10.8h, v0.8h\n"
                    "fmax v11.8h, v11.8h, v0.8h\n"
                    "fmax v12.8h, v12.8h, v0.8h\n"
                    "fmax v13.8h, v13.8h, v0.8h\n"
                    "fmin  v2.8h,  v2.8h, v1.8h\n"
                    "fmin  v3.8h,  v3.8h, v1.8h\n"
                    "fmin  v4.8h,  v4.8h, v1.8h\n"
                    "fmin  v5.8h,  v5.8h, v1.8h\n"
                    "fmin v10.8h, v10.8h, v1.8h\n"
                    "fmin v11.8h, v11.8h, v1.8h\n"
                    "fmin v12.8h, v12.8h, v1.8h\n"
                    "fmin v13.8h, v13.8h, v1.8h\n"

                    "12:\n"
                    "cmp %[pointwiseActivationMode], %[am_h_swish]\n"
                    "bne 13f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "movi v18.8h, #0x42, lsl #8\n"     //three
                    "fadd  v6.8h,  v2.8h, v18.8h\n"
                    "fadd  v7.8h,  v3.8h, v18.8h\n"
                    "fadd  v8.8h,  v4.8h, v18.8h\n"
                    "fadd  v9.8h,  v5.8h, v18.8h\n"
                    "fadd v19.8h, v10.8h, v18.8h\n"
                    "fadd v20.8h, v11.8h, v18.8h\n"
                    "fadd v21.8h, v12.8h, v18.8h\n"
                    "fadd v22.8h, v13.8h, v18.8h\n"
                    "fmax  v6.8h,  v6.8h,  v0.8h\n"
                    "fmax  v7.8h,  v7.8h,  v0.8h\n"
                    "fmax  v8.8h,  v8.8h,  v0.8h\n"
                    "fmax  v9.8h,  v9.8h,  v0.8h\n"
                    "fmax v19.8h, v19.8h,  v0.8h\n"
                    "fmax v20.8h, v20.8h,  v0.8h\n"
                    "fmax v21.8h, v21.8h,  v0.8h\n"
                    "fmax v22.8h, v22.8h,  v0.8h\n"
                    "fmin  v6.8h,  v6.8h,  v1.8h\n"
                    "fmin  v7.8h,  v7.8h,  v1.8h\n"
                    "fmin  v8.8h,  v8.8h,  v1.8h\n"
                    "fmin  v9.8h,  v9.8h,  v1.8h\n"
                    "fmin v19.8h, v19.8h,  v1.8h\n"
                    "fmin v20.8h, v20.8h,  v1.8h\n"
                    "fmin v21.8h, v21.8h,  v1.8h\n"
                    "fmin v22.8h, v22.8h,  v1.8h\n"
                    "fdiv  v6.8h,  v6.8h,  v1.8h\n"
                    "fdiv  v7.8h,  v7.8h,  v1.8h\n"
                    "fdiv  v8.8h,  v8.8h,  v1.8h\n"
                    "fdiv  v9.8h,  v9.8h,  v1.8h\n"
                    "fdiv v19.8h, v19.8h,  v1.8h\n"
                    "fdiv v20.8h, v20.8h,  v1.8h\n"
                    "fdiv v21.8h, v21.8h,  v1.8h\n"
                    "fdiv v22.8h, v22.8h,  v1.8h\n"
                    "fmul  v2.8h,  v6.8h,  v2.8h\n"
                    "fmul  v3.8h,  v7.8h,  v3.8h\n"
                    "fmul  v4.8h,  v8.8h,  v4.8h\n"
                    "fmul  v5.8h,  v9.8h,  v5.8h\n"
                    "fmul v10.8h, v19.8h, v10.8h\n"
                    "fmul v11.8h, v20.8h, v11.8h\n"
                    "fmul v12.8h, v21.8h, v12.8h\n"
                    "fmul v13.8h, v22.8h, v13.8h\n"

                    "13:\n"
                    "str   q2, [%[out_0]]\n"           //out_o0hw0
                    "str   q3, [%[out_0], #16]\n"      //out_o0hw1
                    "str   q4, [%[out_0], #32]\n"      //out_o0hw2
                    "str   q5, [%[out_0], #48]\n"      //out_o0hw3
                    "str  q10, [%[out_1]]\n"           //out_o1hw0
                    "str  q11, [%[out_1], #16]\n"      //out_o1hw1
                    "str  q12, [%[out_1], #32]\n"      //out_o1hw2
                    "str  q13, [%[out_1], #48]\n"      //out_o1hw3
                    :[out_0]"+r"(out_o0hw0),
                     [out_1]"+r"(out_o1hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                                     "v11", "v12", "v13", "v18", "v19", "v20", "v21", "v22", "v23", "x0", "x1", "x2", "x3"
                );
                b0 += 16;
                b1 += 16;
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + ic*fh*fw*8 + (oc-1)*8*ic*8;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + (oc-1)*ohow*8 + hw*8;
                // bias
                const F16 *b_o0 = biasArray + ic*8 + (oc-1)*8;
                __asm__ __volatile__(
                    "ldr d22, [%[b_0]]\n"       //b_o0
                    "ldr  x1, [%[b_0], #8]\n"
                    "ins v22.d[1], x1\n"
                    "mov  x0, %[ic]\n"             //ic_blk
                    "mov  v2.16b, v22.16b\n"      //out_o0hw0
                    "ldr  d0, [%[in_0]]\n"           //in_hw0
                    "mov  v3.16b, v22.16b\n"      //out_o0hw1
                    "ldr d18, [%[f_0]]\n"            //f_o0c0
                    "mov  v4.16b, v22.16b\n"      //out_o0hw2
                    "ldr  x2, [%[f_0], #8]\n"
                    "mov  v5.16b, v22.16b\n"      //out_o0hw3
                    "ins v18.d[1], x2\n"
                    "0:\n"
                    "ldr  d1, [%[in_0], #8]\n"           //in_hw0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "ldr d20, [%[f_0], #16]\n"            //f_o0c0
                    "fmla  v3.8h, v18.8h, v0.h[1]\n"
                    "ldr  x2, [%[f_0], #24]\n"
                    "fmla  v4.8h, v18.8h, v0.h[2]\n"
                    "ins v20.d[1], x2\n"
                    "fmla  v5.8h, v18.8h, v0.h[3]\n"
                    "subs x0, x0, #2\n"

                    "ldr  d0, [%[in_0], #16]\n"           //in_hw0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "ldr d18, [%[f_0], #32]\n"            //f_o0c0
                    "fmla  v3.8h, v20.8h, v1.h[1]\n"
                    "ldr  x2, [%[f_0], #40]\n"
                    "fmla  v4.8h, v20.8h, v1.h[2]\n"
                    "ins v18.d[1], x2\n"
                    "fmla  v5.8h, v20.8h, v1.h[3]\n"
                    "add %[in_0], %[in_0], #16\n"
                    "add %[f_0], %[f_0], #32\n"
                    "bne 0b\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 11f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 12f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax  v3.8h,  v3.8h, v0.8h\n"
                    "fmax  v4.8h,  v4.8h, v0.8h\n"
                    "fmax  v5.8h,  v5.8h, v0.8h\n"
                    "fmin  v2.8h,  v2.8h, v1.8h\n"
                    "fmin  v3.8h,  v3.8h, v1.8h\n"
                    "fmin  v4.8h,  v4.8h, v1.8h\n"
                    "fmin  v5.8h,  v5.8h, v1.8h\n"

                    "12:\n"
                    "cmp %[pointwiseActivationMode], %[am_h_swish]\n"
                    "bne 13f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "movi v18.8h, #0x42, lsl #8\n"     //three
                    "fadd  v6.8h,  v2.8h, v18.8h\n"
                    "fadd  v7.8h,  v3.8h, v18.8h\n"
                    "fadd  v8.8h,  v4.8h, v18.8h\n"
                    "fadd  v9.8h,  v5.8h, v18.8h\n"
                    "fmax  v6.8h,  v6.8h,  v0.8h\n"
                    "fmax  v7.8h,  v7.8h,  v0.8h\n"
                    "fmax  v8.8h,  v8.8h,  v0.8h\n"
                    "fmax  v9.8h,  v9.8h,  v0.8h\n"
                    "fmin  v6.8h,  v6.8h,  v1.8h\n"
                    "fmin  v7.8h,  v7.8h,  v1.8h\n"
                    "fmin  v8.8h,  v8.8h,  v1.8h\n"
                    "fmin  v9.8h,  v9.8h,  v1.8h\n"
                    "fdiv  v6.8h,  v6.8h,  v1.8h\n"
                    "fdiv  v7.8h,  v7.8h,  v1.8h\n"
                    "fdiv  v8.8h,  v8.8h,  v1.8h\n"
                    "fdiv  v9.8h,  v9.8h,  v1.8h\n"
                    "fmul  v2.8h,  v6.8h,  v2.8h\n"
                    "fmul  v3.8h,  v7.8h,  v3.8h\n"
                    "fmul  v4.8h,  v8.8h,  v4.8h\n"
                    "fmul  v5.8h,  v9.8h,  v5.8h\n"

                    "13:\n"
                    "str   q2, [%[out_0]]\n"           //out_o0hw0
                    "str   q3, [%[out_0], #16]\n"      //out_o0hw1
                    "str   q4, [%[out_0], #32]\n"      //out_o0hw2
                    "str   q5, [%[out_0], #48]\n"      //out_o0hw3
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_r)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_o0),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", 
                                     "v18", "v20", "v22", "x0", "x1", "x2"
                );
            }
        }

        // ohow_reminder % 4
        ohow_s = (ohow / 4) * 4;
        for (I32 hw = ohow_s; hw < ohow; hw++) {
            const F16 *b0 = biasArray + ic*8;
            const F16 *b1 = b0 + 8;
            const F16 *f_o0c0 = filterArray + ic*fh*fw*8;
            F16 *in_pack = pwArray + hw*ic*8;
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // bias
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr d22, [%[b_0]]\n"       //b_o0
                    "ldr  x1, [%[b_0], #8]\n"
                    "ins v22.d[1], x1\n"
                    "ldr d23, [%[b_1]]\n"       //b_o1
                    "mov  x0, %[ic]\n"             //ic_blk
                    "ldr  x2, [%[b_1], #8]\n"
                    "ins v23.d[1], x2\n"
                    "ldr  h0, [%[in_0]]\n"           //in_hw0
                    "mov  v2.16b, v22.16b\n"      //out_o0hw0
                    "ldr d18, [%[f_0]]\n"            //f_o0c0
                    "mov v10.16b, v23.16b\n"      //out_o1hw0
                    "ldr  x2, [%[f_0], #8]\n"
                    "ins v18.d[1], x2\n"
                    "ldr d19, [%[f_0], #16]\n"            //f_o1c0
                    "ldr  x3, [%[f_0], #24]\n"
                    "ins v19.d[1], x3\n"
                    "0:\n"
                    "ldr  h1, [%[in_0], #2]\n"           //in_hw0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "ldr d20, [%[f_0], #32]\n"            //f_o0c0
                    "fmla v10.8h, v19.8h, v0.h[0]\n"
                    "ldr  x2, [%[f_0], #40]\n"
                    "ins v20.d[1], x2\n"
                    "ldr d21, [%[f_0], #48]\n"            //f_o1c0
                    "subs x0, x0, #2\n"
                    "ldr  x3, [%[f_0], #56]\n"
                    "ins v21.d[1], x3\n"

                    "ldr  h0, [%[in_0], #4]\n"           //in_hw0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "ldr d18, [%[f_0], #64]\n"            //f_o0c0
                    "fmla v10.8h, v21.8h, v1.h[0]\n"
                    "ldr  x2, [%[f_0], #72]\n"
                    "ins v18.d[1], x2\n"
                    "ldr d19, [%[f_0], #80]\n"            //f_o1c0
                    "add %[in_0], %[in_0], #4\n"
                    "ldr  x3, [%[f_0], #88]\n"
                    "ins v19.d[1], x3\n"
                    "add %[f_0], %[f_0], #64\n"
                    "bne 0b\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 11f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax v10.8h, v10.8h, v0.8h\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 12f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmax v10.8h, v10.8h, v0.8h\n"
                    "fmin  v2.8h,  v2.8h, v1.8h\n"
                    "fmin v10.8h, v10.8h, v1.8h\n"

                    "12:\n"
                    "cmp %[pointwiseActivationMode], %[am_h_swish]\n"
                    "bne 13f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "movi v18.8h, #0x42, lsl #8\n"     //three
                    "fadd v19.8h,  v2.8h, v18.8h\n"
                    "fadd v20.8h, v10.8h, v18.8h\n"
                    "fmax v19.8h, v19.8h,  v0.8h\n"
                    "fmax v20.8h, v20.8h,  v0.8h\n"
                    "fmin v19.8h, v19.8h,  v1.8h\n"
                    "fmin v20.8h, v20.8h,  v1.8h\n"
                    "fdiv v19.8h, v19.8h,  v1.8h\n"
                    "fdiv v20.8h, v20.8h,  v1.8h\n"
                    "fmul  v2.8h, v19.8h,  v2.8h\n"
                    "fmul v10.8h, v20.8h, v10.8h\n"

                    "13:\n"
                    "str   q2, [%[out_0]]\n"           //out_o0hw0
                    "str  q10, [%[out_1]]\n"           //out_o1hw0
                    :[out_0]"+r"(out_o0hw0),
                     [out_1]"+r"(out_o1hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v10", "v18", "v19", "v20", "v21", "v22", "v23", "x0", "x1", "x2", "x3"
                );
                b0 += 16;
                b1 += 16;
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + ic*fh*fw*8 + (oc-1)*8*ic*8;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + (oc-1)*ohow*8 + hw*8;
                // bias
                const F16 *b_o0 = biasArray + ic*8 + (oc-1)*8;
                __asm__ __volatile__(
                    "ldr d22, [%[b_0]]\n"       //b_o0
                    "mov  x0, %[ic]\n"             //ic_blk
                    "ldr  x1, [%[b_0], #8]\n"
                    "ins v22.d[1], x1\n"
                    "ldr  h0, [%[in_0]]\n"           //in_hw0
                    "mov  v2.16b, v22.16b\n"      //out_o0hw0
                    "ldr d18, [%[f_0]]\n"            //f_o0c0
                    "ldr  x2, [%[f_0], #8]\n"
                    "ins v18.d[1], x2\n"
                    "0:\n"
                    "ldr  h1, [%[in_0], #2]\n"           //in_hw0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "ldr d20, [%[f_0], #16]\n"            //f_o0c0
                    "subs x0, x0, #2\n"
                    "ldr  x2, [%[f_0], #24]\n"
                    "ins v20.d[1], x2\n"

                    "ldr  h0, [%[in_0], #4]\n"           //in_hw0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "ldr d18, [%[f_0], #32]\n"            //f_o0c0
                    "ldr  x2, [%[f_0], #40]\n"
                    "ins v18.d[1], x2\n"
                    "add %[in_0], %[in_0], #4\n"
                    "add %[f_0], %[f_0], #32\n"
                    "bne 0b\n"

                    "cmp %[pointwiseActivationMode], %[am_relu]\n"
                    "bne 11f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v0.8h\n"

                    "11:\n"
                    "cmp %[pointwiseActivationMode], %[am_relu6]\n"
                    "bne 12f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "fmax  v2.8h,  v2.8h, v0.8h\n"
                    "fmin  v2.8h,  v2.8h, v1.8h\n"

                    "12:\n"
                    "cmp %[pointwiseActivationMode], %[am_h_swish]\n"
                    "bne 13f\n"
                    "eor v0.16b, v0.16b, v0.16b\n"     //zero
                    "movi v1.8h, #0x46, lsl #8\n"      //six
                    "movi v18.8h, #0x42, lsl #8\n"     //three
                    "fadd v20.8h,  v2.8h, v18.8h\n"
                    "fmax v20.8h, v20.8h,  v0.8h\n"
                    "fmin v20.8h, v20.8h,  v1.8h\n"
                    "fdiv v20.8h, v20.8h,  v1.8h\n"
                    "fmul  v2.8h, v20.8h,  v2.8h\n"

                    "13:\n"
                    "str   q2, [%[out_0]]\n"           //out_o0hw0
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_r)
                    :[ic]"r"((I64)ic*8),
                     [b_0]"r"(b_o0),
                     [pointwiseActivationMode]"r"((I64)pointwiseActivationMode),
                     [am_relu]"r"((I64)ACTIVATION_RELU),
                     [am_relu6]"r"((I64)ACTIVATION_RELU6),
                     [am_h_swish]"r"((I64)ACTIVATION_H_SWISH)
                    :"memory", "cc", "v0", "v1", "v2", "v10", "v18", "v20", "v22", "x0", "x1", "x2"
                );
            }
        }
    }
    return SUCCESS;
}
