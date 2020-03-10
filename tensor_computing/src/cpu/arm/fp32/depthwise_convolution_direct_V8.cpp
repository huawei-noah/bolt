// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/fp32/depthwise_convolution.h"

EE depthwise_convolution_direct_V8(TensorDesc inputDesc, F32* inArray,
    TensorDesc filterDesc, const F32* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* outArray,
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

    if (fdf != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }

    oc /= 8;
    ic /= 8;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ihiw = ih*iw;
    I32 ohow = oh*ow;

    for (U32 n = 0; n < in; n++) {
        F32 *inArray_pad = (F32*)tmp;
        F32 *inArray_pad_mov = inArray_pad;
        F32 *inArray_mov = inArray + n*ic*ihiw*8;
        for (U32 c = 0; c < ic; c++) {
            // copy input into a input with padding
            for (U32 h = 0; h < paddingT; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*sizeof(F32));
                inArray_pad_mov += iw_pad*8;
            }
            for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                memset(inArray_pad_mov, 0, paddingL*8*sizeof(F32));
                inArray_pad_mov += paddingL*8;
                memcpy(inArray_pad_mov, inArray_mov, iw*8*sizeof(F32));
                inArray_pad_mov += iw*8;
                inArray_mov += iw*8;
                memset(inArray_pad_mov, 0, paddingR*8*sizeof(F32));
                inArray_pad_mov += paddingR*8;
            }
            for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                memset(inArray_pad_mov, 0, iw_pad*8*sizeof(F32));
                inArray_pad_mov += iw_pad*8;
            }

            const F32 *b = biasArray + c*8;
            F32 *in_pad = inArray_pad + c*ih_pad*iw_pad*8;
            const F32 *f = filterArray + c*fh*fw*8;
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

                F32 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;
                __asm__ __volatile__(
                    "ldr q14, [%[b]]\n"
                    "ldr q15, [%[b], #16]\n"
                    "mov v0.16b, v14.16b\n"
                    "mov v1.16b, v15.16b\n"
                    "mov v2.16b, v14.16b\n"
                    "mov v3.16b, v15.16b\n"
                    "mov v4.16b, v14.16b\n"
                    "mov v5.16b, v15.16b\n"
                    "mov v6.16b, v14.16b\n"
                    "mov v7.16b, v15.16b\n"
                    "mov v8.16b, v14.16b\n"
                    "mov v9.16b, v15.16b\n"
                    "mov v10.16b, v14.16b\n"
                    "mov v11.16b, v15.16b\n"
                    "mov v12.16b, v14.16b\n"
                    "mov v13.16b, v15.16b\n"
                    :
                    : [b] "r"(b)
                    : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F32 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        F32 *in_idx = in_pad + fh_idx * dilateH * iw_pad * 8 + fw_idx * dilateW * 8;
                        F32 *in_0 = in_idx + in_h_0 * iw_pad * 8 + in_w_0 * 8;
                        F32 *in_1 = in_idx + in_h_1 * iw_pad * 8 + in_w_1 * 8;
                        F32 *in_2 = in_idx + in_h_2 * iw_pad * 8 + in_w_2 * 8;
                        F32 *in_3 = in_idx + in_h_3 * iw_pad * 8 + in_w_3 * 8;
                        F32 *in_4 = in_idx + in_h_4 * iw_pad * 8 + in_w_4 * 8;
                        F32 *in_5 = in_idx + in_h_5 * iw_pad * 8 + in_w_5 * 8;
                        F32 *in_6 = in_idx + in_h_6 * iw_pad * 8 + in_w_6 * 8;
                        F32 *in_7 = in_idx + in_h_7 * iw_pad * 8 + in_w_7 * 8;

                        __asm__ __volatile__(
                            "ldp q16, q17, [%[f0]]\n"
                            "ldp q30, q31, [%[in0]]\n"
                            "ldp q18, q19, [%[in1]]\n"
                            "ldp q20, q21, [%[in2]]\n"
                            "ldp q22, q23, [%[in3]]\n"
                            "ldp q24, q25, [%[in4]]\n"
                            "ldp q26, q27, [%[in5]]\n"
                            "ldp q28, q29, [%[in6]]\n"

                            "fmla v0.4s, v30.4s, v16.4s\n"
                            "fmla v1.4s, v31.4s, v17.4s\n"
                            "fmla v2.4s, v18.4s, v16.4s\n"
                            "ldp q30, q31, [%[in7]]\n"
                            "fmla v3.4s, v19.4s, v17.4s\n"
                            "fmla v4.4s, v20.4s, v16.4s\n"
                            "fmla v5.4s, v21.4s, v17.4s\n"
                            "fmla v6.4s, v22.4s, v16.4s\n"
                            "fmla v7.4s, v23.4s, v17.4s\n"
                            "fmla v8.4s, v24.4s, v16.4s\n"
                            "fmla v9.4s, v25.4s, v17.4s\n"
                            "fmla v10.4s, v26.4s, v16.4s\n"
                            "fmla v11.4s, v27.4s, v17.4s\n"
                            "fmla v12.4s, v28.4s, v16.4s\n"
                            "fmla v13.4s, v29.4s, v17.4s\n"
                            "fmla v14.4s, v30.4s, v16.4s\n"
                            "fmla v15.4s, v31.4s, v17.4s\n"
                            :
                            : [in0] "r"(in_0),
                              [in1] "r"(in_1),
                              [in2] "r"(in_2),
                              [in3] "r"(in_3),
                              [in4] "r"(in_4),
                              [in5] "r"(in_5),
                              [in6] "r"(in_6),
                              [in7] "r"(in_7),
                              [f0] "r"(f_0)
                            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                              "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
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
                            "fmax v0.4s, v0.4s, v31.4s\n"
                            "fmax v1.4s, v1.4s, v31.4s\n"
                            "fmax v2.4s, v2.4s, v31.4s\n"
                            "fmax v3.4s, v3.4s, v31.4s\n"
                            "fmax v4.4s, v4.4s, v31.4s\n"
                            "fmax v5.4s, v5.4s, v31.4s\n"
                            "fmax v6.4s, v6.4s, v31.4s\n"
                            "fmax v7.4s, v7.4s, v31.4s\n"
                            "fmax v8.4s, v8.4s, v31.4s\n"
                            "fmax v9.4s, v9.4s, v31.4s\n"
                            "fmax v10.4s, v10.4s, v31.4s\n"
                            "fmax v11.4s, v11.4s, v31.4s\n"
                            "fmax v12.4s, v12.4s, v31.4s\n"
                            "fmax v13.4s, v13.4s, v31.4s\n"
                            "fmax v14.4s, v14.4s, v31.4s\n"
                            "fmax v15.4s, v15.4s, v31.4s\n"
                            :
                            :
                            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fmov v30.4s, 6.0\n"              // six
                            "fmax v0.4s, v0.4s, v31.4s\n"
                            "fmax v1.4s, v1.4s, v31.4s\n"
                            "fmax v2.4s, v2.4s, v31.4s\n"
                            "fmax v3.4s, v3.4s, v31.4s\n"
                            "fmax v4.4s, v4.4s, v31.4s\n"
                            "fmax v5.4s, v5.4s, v31.4s\n"
                            "fmax v6.4s, v6.4s, v31.4s\n"
                            "fmax v7.4s, v7.4s, v31.4s\n"
                            "fmax v8.4s, v8.4s, v31.4s\n"
                            "fmax v9.4s, v9.4s, v31.4s\n"
                            "fmax v10.4s, v10.4s, v31.4s\n"
                            "fmax v11.4s, v11.4s, v31.4s\n"
                            "fmax v12.4s, v12.4s, v31.4s\n"
                            "fmax v13.4s, v13.4s, v31.4s\n"
                            "fmax v14.4s, v14.4s, v31.4s\n"
                            "fmax v15.4s, v15.4s, v31.4s\n"

                            "fmin v0.4s, v0.4s, v30.4s\n"
                            "fmin v1.4s, v1.4s, v30.4s\n"
                            "fmin v2.4s, v2.4s, v30.4s\n"
                            "fmin v3.4s, v3.4s, v30.4s\n"
                            "fmin v4.4s, v4.4s, v30.4s\n"
                            "fmin v5.4s, v5.4s, v30.4s\n"
                            "fmin v6.4s, v6.4s, v30.4s\n"
                            "fmin v7.4s, v7.4s, v30.4s\n"
                            "fmin v8.4s, v8.4s, v30.4s\n"
                            "fmin v9.4s, v9.4s, v30.4s\n"
                            "fmin v10.4s, v10.4s, v30.4s\n"
                            "fmin v11.4s, v11.4s, v30.4s\n"
                            "fmin v12.4s, v12.4s, v30.4s\n"
                            "fmin v13.4s, v13.4s, v30.4s\n"
                            "fmin v14.4s, v14.4s, v30.4s\n"
                            "fmin v15.4s, v15.4s, v30.4s\n"
                            :
                            :
                            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v30", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        __asm__ __volatile__(
                            "fmov v29.4s, 3.0\n"              // three
                            "fmov v30.4s, 6.0\n"              // six
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fadd v22.4s,  v0.4s, v29.4s\n"
                            "fadd v23.4s,  v1.4s, v29.4s\n"
                            "fadd v16.4s,  v2.4s, v29.4s\n"
                            "fadd v17.4s,  v3.4s, v29.4s\n"
                            "fadd v18.4s,  v4.4s, v29.4s\n"
                            "fadd v19.4s,  v5.4s, v29.4s\n"
                            "fadd v20.4s,  v6.4s, v29.4s\n"
                            "fadd v21.4s,  v7.4s, v29.4s\n"

                            "fmax v22.4s, v22.4s, v31.4s\n"
                            "fmax v23.4s, v23.4s, v31.4s\n"
                            "fmax v16.4s, v16.4s, v31.4s\n"
                            "fmax v17.4s, v17.4s, v31.4s\n"
                            "fmax v18.4s, v18.4s, v31.4s\n"
                            "fmax v19.4s, v19.4s, v31.4s\n"
                            "fmax v20.4s, v20.4s, v31.4s\n"
                            "fmax v21.4s, v21.4s, v31.4s\n"

                            "fmin v22.4s, v22.4s, v30.4s\n"
                            "fmin v23.4s, v23.4s, v30.4s\n"
                            "fmin v16.4s, v16.4s, v30.4s\n"
                            "fmin v17.4s, v17.4s, v30.4s\n"
                            "fmin v18.4s, v18.4s, v30.4s\n"
                            "fmin v19.4s, v19.4s, v30.4s\n"
                            "fmin v20.4s, v20.4s, v30.4s\n"
                            "fmin v21.4s, v21.4s, v30.4s\n"

                            "fdiv v22.4s, v22.4s, v30.4s\n"
                            "fdiv v23.4s, v23.4s, v30.4s\n"
                            "fdiv v16.4s, v16.4s, v30.4s\n"
                            "fdiv v17.4s, v17.4s, v30.4s\n"
                            "fdiv v18.4s, v18.4s, v30.4s\n"
                            "fdiv v19.4s, v19.4s, v30.4s\n"
                            "fdiv v20.4s, v20.4s, v30.4s\n"
                            "fdiv v21.4s, v21.4s, v30.4s\n"

                            "fmul  v0.4s,  v0.4s, v22.4s\n"
                            "fmul  v1.4s,  v1.4s, v23.4s\n"
                            "fmul  v2.4s,  v2.4s, v16.4s\n"
                            "fmul  v3.4s,  v3.4s, v17.4s\n"
                            "fmul  v4.4s,  v4.4s, v18.4s\n"
                            "fmul  v5.4s,  v5.4s, v19.4s\n"
                            "fmul  v6.4s,  v6.4s, v20.4s\n"
                            "fmul  v7.4s,  v7.4s, v21.4s\n"


                            "fadd v22.4s,  v8.4s, v29.4s\n"
                            "fadd v23.4s,  v9.4s, v29.4s\n"
                            "fadd v16.4s,  v10.4s, v29.4s\n"
                            "fadd v17.4s,  v11.4s, v29.4s\n"
                            "fadd v18.4s,  v12.4s, v29.4s\n"
                            "fadd v19.4s,  v13.4s, v29.4s\n"
                            "fadd v20.4s,  v14.4s, v29.4s\n"
                            "fadd v21.4s,  v15.4s, v29.4s\n"

                            "fmax v22.4s, v22.4s, v31.4s\n"
                            "fmax v23.4s, v23.4s, v31.4s\n"
                            "fmax v16.4s, v16.4s, v31.4s\n"
                            "fmax v17.4s, v17.4s, v31.4s\n"
                            "fmax v18.4s, v18.4s, v31.4s\n"
                            "fmax v19.4s, v19.4s, v31.4s\n"
                            "fmax v20.4s, v20.4s, v31.4s\n"
                            "fmax v21.4s, v21.4s, v31.4s\n"

                            "fmin v22.4s, v22.4s, v30.4s\n"
                            "fmin v23.4s, v23.4s, v30.4s\n"
                            "fmin v16.4s, v16.4s, v30.4s\n"
                            "fmin v17.4s, v17.4s, v30.4s\n"
                            "fmin v18.4s, v18.4s, v30.4s\n"
                            "fmin v19.4s, v19.4s, v30.4s\n"
                            "fmin v20.4s, v20.4s, v30.4s\n"
                            "fmin v21.4s, v21.4s, v30.4s\n"

                            "fdiv v22.4s, v22.4s, v30.4s\n"
                            "fdiv v23.4s, v23.4s, v30.4s\n"
                            "fdiv v16.4s, v16.4s, v30.4s\n"
                            "fdiv v17.4s, v17.4s, v30.4s\n"
                            "fdiv v18.4s, v18.4s, v30.4s\n"
                            "fdiv v19.4s, v19.4s, v30.4s\n"
                            "fdiv v20.4s, v20.4s, v30.4s\n"
                            "fdiv v21.4s, v21.4s, v30.4s\n"

                            "fmul  v8.4s,  v8.4s, v22.4s\n"
                            "fmul  v9.4s,  v9.4s, v23.4s\n"
                            "fmul  v10.4s,  v10.4s, v16.4s\n"
                            "fmul  v11.4s,  v11.4s, v17.4s\n"
                            "fmul  v12.4s,  v12.4s, v18.4s\n"
                            "fmul  v13.4s,  v13.4s, v19.4s\n"
                            "fmul  v14.4s,  v14.4s, v20.4s\n"
                            "fmul  v15.4s,  v15.4s, v21.4s\n"
                            :
                            :
                            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                              "v18", "v19", "v20", "v21", "v22", "v23", "v29", "v30", "v31"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__(
                    "stp q0, q1, [%[out]]\n"
                    "stp q2, q3, [%[out], #32]\n"
                    "stp q4, q5, [%[out], #64]\n"
                    "stp q6, q7, [%[out], #96]\n"
                    "stp q8, q9, [%[out], #128]\n"
                    "stp q10, q11, [%[out], #160]\n"
                    "stp q12, q13, [%[out], #192]\n"
                    "stp q14, q15, [%[out], #224]\n"
                    : [out] "+r"(out_ptr)
                    :
                    : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                      "v11", "v12", "v13", "v14", "v15"
                );
            }

            U32 ohow_s = (ohow / 8) * 8;
            U32 ohow_tail = ohow - ohow_s;
            if (ohow_tail >= 4) {
                I32 hw = ohow_s;
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                U32 in_h_1 = (hw+1)/ow*strideH;
                U32 in_w_1 = (hw+1)%ow*strideW;
                U32 in_h_2 = (hw+2)/ow*strideH;
                U32 in_w_2 = (hw+2)%ow*strideW;
                U32 in_h_3 = (hw+3)/ow*strideH;
                U32 in_w_3 = (hw+3)%ow*strideW;
                F32 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;

                __asm__ __volatile__(
                    "ldr q14, [%[b]]\n"
                    "ldr q15, [%[b], #16]\n"
                    "mov v0.16b, v14.16b\n"
                    "mov v1.16b, v15.16b\n"
                    "mov v2.16b, v14.16b\n"
                    "mov v3.16b, v15.16b\n"
                    "mov v4.16b, v14.16b\n"
                    "mov v5.16b, v15.16b\n"
                    "mov v6.16b, v14.16b\n"
                    "mov v7.16b, v15.16b\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v14", "v15"
                );

                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F32 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        F32 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        F32 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;
                        F32 *in_1 = in_idx + in_h_1*iw_pad*8 + in_w_1*8;
                        F32 *in_2 = in_idx + in_h_2*iw_pad*8 + in_w_2*8;
                        F32 *in_3 = in_idx + in_h_3*iw_pad*8 + in_w_3*8;

                        __asm__ __volatile__(
                            "ldp q14, q15, [%[f0]]\n"
                            "ldp q16, q17, [%[in0]]\n"
                            "ldp q18, q19, [%[in1]]\n"
                            "ldp q20, q21, [%[in2]]\n"
                            "ldp q22, q23, [%[in3]]\n"

                            "fmla v0.4s,  v16.4s, v14.4s\n"
                            "fmla v1.4s, v17.4s, v15.4s\n"
                            "fmla v2.4s, v18.4s, v14.4s\n"
                            "fmla v3.4s, v19.4s, v15.4s\n"
                            "fmla v4.4s, v20.4s, v14.4s\n"
                            "fmla v5.4s, v21.4s, v15.4s\n"
                            "fmla v6.4s, v22.4s, v14.4s\n"
                            "fmla v7.4s, v23.4s, v15.4s\n"
                            :
                            :[in0]"r"(in_0),
                             [in1]"r"(in_1),
                             [in2]"r"(in_2),
                             [in3]"r"(in_3),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v14", "v15", "v16", "v17",
                             "v18", "v19", "v20", "v21", "v22", "v23"
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
                            "fmax v0.4s, v0.4s, v31.4s\n"
                            "fmax v1.4s, v1.4s, v31.4s\n"
                            "fmax v2.4s, v2.4s, v31.4s\n"
                            "fmax v3.4s, v3.4s, v31.4s\n"
                            "fmax v4.4s, v4.4s, v31.4s\n"
                            "fmax v5.4s, v5.4s, v31.4s\n"
                            "fmax v6.4s, v6.4s, v31.4s\n"
                            "fmax v7.4s, v7.4s, v31.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fmov v30.4s, 6.0\n"  // six
                            "fmax v0.4s, v0.4s, v31.4s\n"
                            "fmax v1.4s, v1.4s, v31.4s\n"
                            "fmax v2.4s, v2.4s, v31.4s\n"
                            "fmax v3.4s, v3.4s, v31.4s\n"
                            "fmax v4.4s, v4.4s, v31.4s\n"
                            "fmax v5.4s, v5.4s, v31.4s\n"
                            "fmax v6.4s, v6.4s, v31.4s\n"
                            "fmax v7.4s, v7.4s, v31.4s\n"

                            "fmin v0.4s, v0.4s, v30.4s\n"
                            "fmin v1.4s, v1.4s, v30.4s\n"
                            "fmin v2.4s, v2.4s, v30.4s\n"
                            "fmin v3.4s, v3.4s, v30.4s\n"
                            "fmin v4.4s, v4.4s, v30.4s\n"
                            "fmin v5.4s, v5.4s, v30.4s\n"
                            "fmin v6.4s, v6.4s, v30.4s\n"
                            "fmin v7.4s, v7.4s, v30.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v30", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        __asm__ __volatile__(
                            "fmov v29.4s, 3.0\n"  // three
                            "fmov v30.4s, 6.0\n"  // six
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fadd v14.4s,  v0.4s, v29.4s\n"
                            "fadd v15.4s,  v1.4s, v29.4s\n"
                            "fadd v16.4s,  v2.4s, v29.4s\n"
                            "fadd v17.4s,  v3.4s, v29.4s\n"
                            "fadd v18.4s,  v4.4s, v29.4s\n"
                            "fadd v19.4s,  v5.4s, v29.4s\n"
                            "fadd v20.4s,  v6.4s, v29.4s\n"
                            "fadd v21.4s,  v7.4s, v29.4s\n"

                            "fmax v14.4s, v14.4s, v31.4s\n"
                            "fmax v15.4s, v15.4s, v31.4s\n"
                            "fmax v16.4s, v16.4s, v31.4s\n"
                            "fmax v17.4s, v17.4s, v31.4s\n"
                            "fmax v18.4s, v18.4s, v31.4s\n"
                            "fmax v19.4s, v19.4s, v31.4s\n"
                            "fmax v20.4s, v20.4s, v31.4s\n"
                            "fmax v21.4s, v21.4s, v31.4s\n"

                            "fmin v14.4s, v14.4s, v30.4s\n"
                            "fmin v15.4s, v15.4s, v30.4s\n"
                            "fmin v16.4s, v16.4s, v30.4s\n"
                            "fmin v17.4s, v17.4s, v30.4s\n"
                            "fmin v18.4s, v18.4s, v30.4s\n"
                            "fmin v19.4s, v19.4s, v30.4s\n"
                            "fmin v20.4s, v20.4s, v30.4s\n"
                            "fmin v21.4s, v21.4s, v30.4s\n"

                            "fdiv v14.4s, v14.4s, v30.4s\n"
                            "fdiv v15.4s, v15.4s, v30.4s\n"
                            "fdiv v16.4s, v16.4s, v30.4s\n"
                            "fdiv v17.4s, v17.4s, v30.4s\n"
                            "fdiv v18.4s, v18.4s, v30.4s\n"
                            "fdiv v19.4s, v19.4s, v30.4s\n"
                            "fdiv v20.4s, v20.4s, v30.4s\n"
                            "fdiv v21.4s, v21.4s, v30.4s\n"

                            "fmul  v0.4s,  v0.4s, v14.4s\n"
                            "fmul  v1.4s,  v1.4s, v15.4s\n"
                            "fmul  v2.4s,  v2.4s, v16.4s\n"
                            "fmul  v3.4s,  v3.4s, v17.4s\n"
                            "fmul  v4.4s,  v4.4s, v18.4s\n"
                            "fmul  v5.4s,  v5.4s, v19.4s\n"
                            "fmul  v6.4s,  v6.4s, v20.4s\n"
                            "fmul  v7.4s,  v7.4s, v21.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v14", "v15", "v16", "v17",
                             "v18", "v19", "v20", "v21", "v29", "v30", "v31"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__(
                    "stp q0, q1, [%[out]]\n"
                    "stp q2, q3, [%[out], #32]\n"
                    "stp q4, q5, [%[out], #64]\n"
                    "stp q6, q7, [%[out], #96]\n"
                    :[out]"+r"(out_ptr)
                    :
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                );

                ohow_s += 4;
            }

            for (I32 hw = ohow_s; hw < ohow; hw++) {
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                F32 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;

                __asm__ __volatile__(
                    "ldr q0, [%[b]]\n"
                    "ldr q1, [%[b], #16]\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "v0", "v1"
                );

                 for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F32 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        F32 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        F32 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;

                        __asm__ __volatile__(
                            "ldp q14, q15, [%[f0]]\n"
                            "ldp q16, q17, [%[in0]]\n"

                            "fmla v0.4s, v16.4s, v14.4s\n"
                            "fmla v1.4s, v17.4s, v15.4s\n"
                            :
                            :[in0]"r"(in_0),
                             [f0]"r"(f_0)
                            :"memory", "cc", "v0", "v1", "v14", "v15"
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
                            "fmax v0.4s, v0.4s, v31.4s\n"
                            "fmax v1.4s, v1.4s, v31.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fmov v30.4s, 6.0\n"  // six
                            "fmax v0.4s, v0.4s, v31.4s\n"
                            "fmax v1.4s, v1.4s, v31.4s\n"

                            "fmin v0.4s, v0.4s, v30.4s\n"
                            "fmin v1.4s, v1.4s, v30.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v30", "v31"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        __asm__ __volatile__(
                            "fmov v29.4s, 3.0\n"  // three
                            "fmov v30.4s, 6.0\n"  // six
                            "eor v31.16b, v31.16b, v31.16b\n" // zero
                            "fadd v14.4s,  v0.4s, v29.4s\n"
                            "fadd v15.4s,  v1.4s, v29.4s\n"

                            "fmax v14.4s, v14.4s, v31.4s\n"
                            "fmax v15.4s, v15.4s, v31.4s\n"

                            "fmin v14.4s, v14.4s, v30.4s\n"
                            "fmin v15.4s, v15.4s, v30.4s\n"

                            "fdiv v14.4s, v14.4s, v30.4s\n"
                            "fdiv v15.4s, v15.4s, v30.4s\n"

                            "fmul  v0.4s,  v0.4s, v14.4s\n"
                            "fmul  v1.4s,  v1.4s, v15.4s\n"
                            :
                            :
                            :"memory", "cc", "v0", "v1", "v14", "v15", "v29", "v30", "v31"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__(
                    "stp q0, q1, [%[out]]\n"
                    :[out]"+r"(out_ptr)
                    :
                    :"memory", "cc", "v0", "v1"
                );
            }
        }
    }
    return SUCCESS;
}
