// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>
#include "cpu/arm/fp16/convolution_gemm.h"

EE convolution_gemm_A76(TensorDesc inputDesc, F16* inArray,
    TensorDesc filterDesc, const F16* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F16* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* outArray,
    ActivationMode activationMode)
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

    if (fdf != DF_NHWCN16)
        CHECK_STATUS(NOT_MATCH);

    I64 activation = 0;
    switch (activationMode) {
        case ACTIVATION_NULL:
            activation = 0;
            break;
        case ACTIVATION_RELU:
            activation = 1;
            break;
        default:
            return NOT_SUPPORTED;
    }
    oc /= 8;
    ic /= 8;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    I32 ohow = oh*ow;
    U32 ihiw = ih_pad*iw_pad;
    F16 *inArray_pad;
    EE ret = SUCCESS;
    for (U32 n = 0; n < in; n++) {
        if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            inArray_pad = inArray + n*ic*ih*iw*8;
        } else {
            // copy input into a input with padding
            inArray_pad = (F16*)tmp;
            F16 *inArray_pad_mov = inArray_pad;
            F16 *inArray_mov = inArray + n*ic*ih*iw*8;
            for (U32 c = 0; c < ic; c++) {
                for (U32 h = 0; h < paddingT; h++) {
                    memset(inArray_pad_mov, 0, iw_pad*8*bytesOf(idt));
                    inArray_pad_mov += iw_pad*8;
                }
                for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                    memset(inArray_pad_mov, 0, paddingL*8*bytesOf(idt));
                    inArray_pad_mov += paddingL*8;
                    memcpy(inArray_pad_mov, inArray_mov, iw*8*bytesOf(idt));
                    inArray_pad_mov += iw*8;
                    inArray_mov += iw*8;
                    memset(inArray_pad_mov, 0, paddingR*8*bytesOf(idt));
                    inArray_pad_mov += paddingR*8;
                }
                for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                    memset(inArray_pad_mov, 0, iw_pad*8*bytesOf(idt));
                    inArray_pad_mov += iw_pad*8;
                }
            }
        }
        // ohow / 8
        for (I32 hw = 0; hw < ohow-7; hw+=8) {
            const F16 *b0 = biasArray;
            const F16 *b1 = biasArray + 8;
            const F16 *f_o0c0 = filterArray;
            F16 *in_pack = ((F16*)tmp) + ic*ih_pad*iw_pad*8;
            // pack input
            // NCHWc8 => NHWChw8 + im2col
            U32 in_h_0 = (hw/ow)*strideH;
            U32 in_w_0 = (hw%ow)*strideW;
            U32 in_h_1 = ((hw+1)/ow)*strideH;
            U32 in_w_1 = ((hw+1)%ow)*strideW;
            U32 in_h_2 = ((hw+2)/ow)*strideH;
            U32 in_w_2 = ((hw+2)%ow)*strideW;
            U32 in_h_3 = ((hw+3)/ow)*strideH;
            U32 in_w_3 = ((hw+3)%ow)*strideW;
            U32 in_h_4 = ((hw+4)/ow)*strideH;
            U32 in_w_4 = ((hw+4)%ow)*strideW;
            U32 in_h_5 = ((hw+5)/ow)*strideH;
            U32 in_w_5 = ((hw+5)%ow)*strideW;
            U32 in_h_6 = ((hw+6)/ow)*strideH;
            U32 in_w_6 = ((hw+6)%ow)*strideW;
            U32 in_h_7 = ((hw+7)/ow)*strideH;
            U32 in_w_7 = ((hw+7)%ow)*strideW;
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        F16 *in_hw8c8 = inArray_pad + c*ihiw*8 + fh_idx*dilateH*iw_pad*8+ fw_idx*dilateW*8;
                        F16 *in_0 = in_hw8c8 + in_h_0*iw_pad*8 + in_w_0*8;
                        F16 *in_1 = in_hw8c8 + in_h_1*iw_pad*8 + in_w_1*8;
                        F16 *in_2 = in_hw8c8 + in_h_2*iw_pad*8 + in_w_2*8;
                        F16 *in_3 = in_hw8c8 + in_h_3*iw_pad*8 + in_w_3*8;
                        F16 *in_4 = in_hw8c8 + in_h_4*iw_pad*8 + in_w_4*8;
                        F16 *in_5 = in_hw8c8 + in_h_5*iw_pad*8 + in_w_5*8;
                        F16 *in_6 = in_hw8c8 + in_h_6*iw_pad*8 + in_w_6*8;
                        F16 *in_7 = in_hw8c8 + in_h_7*iw_pad*8 + in_w_7*8;

                        // NHWChw8
                        F16 *in_pack_c8hw8 = in_pack + fh_idx*fw*ic*8*8 + fw_idx*ic*8*8 + c*8*8;
                        /*
                         * for (U32 c8 = 0; c8 < 8; c8++) {
                         *     for (U32 hw8 = 0; hw8 < 8; hw8++) {
                         *         in_pack_c8hw8[c8*8 + hw8] = in_hw8c8[hw8*8 + c8];
                         *     }
                         * }
                         */
                        float16x8_t v0 = vld1q_f16(in_0);
                        float16x8_t v1 = vld1q_f16(in_1);
                        float16x8_t v2 = vld1q_f16(in_2);
                        float16x8_t v3 = vld1q_f16(in_3);
                        float16x8_t v4 = vld1q_f16(in_4);
                        float16x8_t v5 = vld1q_f16(in_5);
                        float16x8_t v6 = vld1q_f16(in_6);
                        float16x8_t v7 = vld1q_f16(in_7);
                        vst1q_f16(in_pack_c8hw8,
                            vzip1q_f16(
                                vzip1q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                                vzip1q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                        vst1q_f16(in_pack_c8hw8 + 8,
                            vzip2q_f16(
                                vzip1q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                                vzip1q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                        vst1q_f16(in_pack_c8hw8 + 8*2,
                            vzip1q_f16(
                                vzip2q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                                vzip2q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                        vst1q_f16(in_pack_c8hw8 + 8*3,
                            vzip2q_f16(
                                vzip2q_f16(vzip1q_f16(v0, v4), vzip1q_f16(v2, v6)),
                                vzip2q_f16(vzip1q_f16(v1, v5), vzip1q_f16(v3, v7))));
                        vst1q_f16(in_pack_c8hw8 + 8*4,
                            vzip1q_f16(
                                vzip1q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                                vzip1q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                        vst1q_f16(in_pack_c8hw8 + 8*5,
                            vzip2q_f16(
                                vzip1q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                                vzip1q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                        vst1q_f16(in_pack_c8hw8 + 8*6,
                            vzip1q_f16(
                                vzip2q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                                vzip2q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                        vst1q_f16(in_pack_c8hw8 + 8*7,
                            vzip2q_f16(
                                vzip2q_f16(vzip2q_f16(v0, v4), vzip2q_f16(v2, v6)),
                                vzip2q_f16(vzip2q_f16(v1, v5), vzip2q_f16(v3, v7))));
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc)-1; o+=2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // bias
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q22, [%[b_0]]\n"       //b_o0
                    "ldr q23, [%[b_1]]\n"       //b_o1
                    "mov  x0, %[ic]\n"             //ic_blk
                    "mov  v2.16b, v22.16b\n"      //out_o0hw0
                    "ldr  q0, [%[in_0]]\n"           //in_hw0
                    "mov  v3.16b, v22.16b\n"      //out_o0hw1
                    "mov  v4.16b, v22.16b\n"      //out_o0hw2
                    "mov  v5.16b, v22.16b\n"      //out_o0hw3
                    "ldr q18, [%[f_0]]\n"            //f_o0c0
                    "mov  v6.16b, v22.16b\n"      //out_o0hw4
                    "mov  v7.16b, v22.16b\n"      //out_o0hw5
                    "mov  v8.16b, v22.16b\n"      //out_o0hw6
                    "ldr q19, [%[f_0], #16]\n"            //f_o1c0
                    "mov  v9.16b, v22.16b\n"      //out_o0hw7
                    "mov v10.16b, v23.16b\n"      //out_o1hw0
                    "mov v11.16b, v23.16b\n"      //out_o1hw1
                    "mov v12.16b, v23.16b\n"      //out_o1hw2
                    "mov v13.16b, v23.16b\n"      //out_o1hw3
                    "mov v14.16b, v23.16b\n"      //out_o1hw4
                    "mov v15.16b, v23.16b\n"      //out_o1hw5
                    "mov v16.16b, v23.16b\n"      //out_o1hw6
                    "mov v17.16b, v23.16b\n"      //out_o1hw7
                    "0:\n"
                    "ldr  q1, [%[in_0], #16]\n"           //in_hw0
                    "ldr q20, [%[f_0], #32]\n"            //f_o0c0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "fmla  v3.8h, v18.8h, v0.h[1]\n"
                    "ldr q21, [%[f_0], #48]\n"            //f_o1c0
                    "fmla  v4.8h, v18.8h, v0.h[2]\n"
                    "fmla  v5.8h, v18.8h, v0.h[3]\n"
                    "fmla  v6.8h, v18.8h, v0.h[4]\n"
                    "fmla  v7.8h, v18.8h, v0.h[5]\n"
                    "fmla  v8.8h, v18.8h, v0.h[6]\n"
                    "fmla  v9.8h, v18.8h, v0.h[7]\n"
                    "fmla v10.8h, v19.8h, v0.h[0]\n"
                    "fmla v11.8h, v19.8h, v0.h[1]\n"
                    "fmla v12.8h, v19.8h, v0.h[2]\n"
                    "fmla v13.8h, v19.8h, v0.h[3]\n"
                    "fmla v14.8h, v19.8h, v0.h[4]\n"
                    "fmla v15.8h, v19.8h, v0.h[5]\n"
                    "fmla v16.8h, v19.8h, v0.h[6]\n"
                    "fmla v17.8h, v19.8h, v0.h[7]\n"
                    "subs x0, x0, #2\n"

                    "ldr  q0, [%[in_0], #32]\n"           //in_hw0
                    "ldr q18, [%[f_0], #64]\n"            //f_o0c0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "fmla  v3.8h, v20.8h, v1.h[1]\n"
                    "ldr q19, [%[f_0], #80]\n"            //f_o1c0
                    "fmla  v4.8h, v20.8h, v1.h[2]\n"
                    "fmla  v5.8h, v20.8h, v1.h[3]\n"
                    "fmla  v6.8h, v20.8h, v1.h[4]\n"
                    "fmla  v7.8h, v20.8h, v1.h[5]\n"
                    "fmla  v8.8h, v20.8h, v1.h[6]\n"
                    "fmla  v9.8h, v20.8h, v1.h[7]\n"
                    "fmla v10.8h, v21.8h, v1.h[0]\n"
                    "fmla v11.8h, v21.8h, v1.h[1]\n"
                    "fmla v12.8h, v21.8h, v1.h[2]\n"
                    "fmla v13.8h, v21.8h, v1.h[3]\n"
                    "add %[in_0], %[in_0], #32\n"
                    "add %[f_0], %[f_0], #64\n"
                    "fmla v14.8h, v21.8h, v1.h[4]\n"
                    "fmla v15.8h, v21.8h, v1.h[5]\n"
                    "fmla v16.8h, v21.8h, v1.h[6]\n"
                    "fmla v17.8h, v21.8h, v1.h[7]\n"
                    "bne 0b\n"
                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v1.8h\n"     //max(v2, 0)
                    "fmax  v3.8h,  v3.8h, v1.8h\n"
                    "fmax  v4.8h,  v4.8h, v1.8h\n"
                    "fmax  v5.8h,  v5.8h, v1.8h\n"
                    "fmax  v6.8h,  v6.8h, v1.8h\n"
                    "fmax  v7.8h,  v7.8h, v1.8h\n"
                    "fmax  v8.8h,  v8.8h, v1.8h\n"
                    "fmax  v9.8h,  v9.8h, v1.8h\n"
                    "fmax v10.8h, v10.8h, v1.8h\n"
                    "fmax v11.8h, v11.8h, v1.8h\n"
                    "fmax v12.8h, v12.8h, v1.8h\n"
                    "fmax v13.8h, v13.8h, v1.8h\n"
                    "fmax v14.8h, v14.8h, v1.8h\n"
                    "fmax v15.8h, v15.8h, v1.8h\n"
                    "fmax v16.8h, v16.8h, v1.8h\n"
                    "fmax v17.8h, v17.8h, v1.8h\n"
                    "1:\n"
                    "str   q2, [%[out_0]]\n"           //out_o0hw0
                    "str   q3, [%[out_0], #16]\n"      //out_o0hw1
                    "str   q4, [%[out_0], #32]\n"      //out_o0hw2
                    "str   q5, [%[out_0], #48]\n"      //out_o0hw3
                    "str   q6, [%[out_0], #64]\n"      //out_o0hw4
                    "str   q7, [%[out_0], #80]\n"      //out_o0hw5
                    "str   q8, [%[out_0], #96]\n"      //out_o0hw6
                    "str   q9, [%[out_0], #112]\n"     //out_o0hw7
                    "str  q10, [%[out_1]]\n"           //out_o1hw0
                    "str  q11, [%[out_1], #16]\n"      //out_o1hw1
                    "str  q12, [%[out_1], #32]\n"      //out_o1hw2
                    "str  q13, [%[out_1], #48]\n"      //out_o1hw3
                    "str  q14, [%[out_1], #64]\n"      //out_o1hw4
                    "str  q15, [%[out_1], #80]\n"      //out_o1hw5
                    "str  q16, [%[out_1], #96]\n"      //out_o1hw6
                    "str  q17, [%[out_1], #112]\n"     //out_o1hw7
                    :[out_0]"+r"(out_o0hw0),
                     [out_1]"+r"(out_o1hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                        "v21", "v22", "v23", "x0"
                );
                b0 += 16;
                b1 += 16;
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + (oc-1)*8*fh*fw*ic*8;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + (oc-1)*ohow*8 + hw*8;
                // bias
                const F16 *b_o0 = biasArray + (oc-1)*8;
                __asm__ __volatile__(
                    "ldr q12, [%[b_0]]\n"       //b_o0
                    "mov x0, %[ic]\n" // ic_blk
                    "ldr q0, [%[in_0]]\n"   //in_hw0
                    "ldr q10, [%[f_0]]\n"   //f_o0c0
                    "mov v2.16b, v12.16b\n" //out_o0hw0
                    "mov v3.16b, v12.16b\n" //out_o0hw1
                    "mov v4.16b, v12.16b\n" //out_o0hw2
                    "mov v5.16b, v12.16b\n" //out_o0hw3
                    "mov v6.16b, v12.16b\n" //out_o0hw4
                    "mov v7.16b, v12.16b\n" //out_o0hw5
                    "mov v8.16b, v12.16b\n" //out_o0hw6
                    "mov v9.16b, v12.16b\n" //out_o0hw7
                    "0:\n"
                    "ldr q1, [%[in_0], #16]\n" //in_hw0
                    "ldr q11, [%[f_0], #16]\n" //f_o0c0
                    "fmla v2.8h, v10.8h, v0.h[0]\n"
                    "fmla v3.8h, v10.8h, v0.h[1]\n"
                    "fmla v4.8h, v10.8h, v0.h[2]\n"
                    "fmla v5.8h, v10.8h, v0.h[3]\n"
                    "fmla v6.8h, v10.8h, v0.h[4]\n"
                    "fmla v7.8h, v10.8h, v0.h[5]\n"
                    "fmla v8.8h, v10.8h, v0.h[6]\n"
                    "fmla v9.8h, v10.8h, v0.h[7]\n"
                    "subs x0, x0, #2\n"

                    "ldr q0, [%[in_0], #32]\n" //in_hw0
                    "ldr q10, [%[f_0], #32]\n" //f_o0c0
                    "fmla v2.8h, v11.8h, v1.h[0]\n"
                    "fmla v3.8h, v11.8h, v1.h[1]\n"
                    "fmla v4.8h, v11.8h, v1.h[2]\n"
                    "fmla v5.8h, v11.8h, v1.h[3]\n"
                    "add %[in_0], %[in_0], #32\n"
                    "add %[f_0], %[f_0], #32\n"
                    "fmla v6.8h, v11.8h, v1.h[4]\n"
                    "fmla v7.8h, v11.8h, v1.h[5]\n"
                    "fmla v8.8h, v11.8h, v1.h[6]\n"
                    "fmla v9.8h, v11.8h, v1.h[7]\n"
                    "bne 0b\n"
                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n" // zero
                    "fmax v2.8h, v2.8h, v1.8h\n"   //max(v2, 0)
                    "fmax v3.8h, v3.8h, v1.8h\n"
                    "fmax v4.8h, v4.8h, v1.8h\n"
                    "fmax v5.8h, v5.8h, v1.8h\n"
                    "fmax v6.8h, v6.8h, v1.8h\n"
                    "fmax v7.8h, v7.8h, v1.8h\n"
                    "fmax v8.8h, v8.8h, v1.8h\n"
                    "fmax v9.8h, v9.8h, v1.8h\n"
                    "1:\n"
                    "str q2, [%[out_0]]\n"       //out_o0hw0
                    "str q3, [%[out_0], #16]\n"  //out_o0hw0
                    "str q4, [%[out_0], #32]\n"  //out_o0hw0
                    "str q5, [%[out_0], #48]\n"  //out_o0hw0
                    "str q6, [%[out_0], #64]\n"  //out_o0hw0
                    "str q7, [%[out_0], #80]\n"  //out_o0hw0
                    "str q8, [%[out_0], #96]\n"  //out_o0hw0
                    "str q9, [%[out_0], #112]\n" //out_o0hw0
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_r)
                    :[ic]"r"((I64)ic*8*fh*fw),
                     [b_0]"r"(b_o0),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "x0"
                );
            }
        }

        // ohow_reminder % 8 / 4
        U32 ohow_s = (ohow / 8) * 8;

        for (I32 hw = ohow_s; hw < ohow-3; hw+=4) {
            const F16 *b0 = biasArray;
            const F16 *b1 = biasArray + 8;
            const F16 *f_o0c0 = filterArray;
            F16 *in_pack = ((F16*)tmp) + ic*ih_pad*iw_pad*8;
            // pack input
            // NCHWc8 => NHWChw4 + im2col
            U32 in_h_0 = (hw/ow)*strideH;
            U32 in_w_0 = (hw%ow)*strideW;
            U32 in_h_1 = ((hw+1)/ow)*strideH;
            U32 in_w_1 = ((hw+1)%ow)*strideW;
            U32 in_h_2 = ((hw+2)/ow)*strideH;
            U32 in_w_2 = ((hw+2)%ow)*strideW;
            U32 in_h_3 = ((hw+3)/ow)*strideH;
            U32 in_w_3 = ((hw+3)%ow)*strideW;
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        F16 *in_hw4c8 = inArray_pad + c*ihiw*8 + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        F16 *in_0 = in_hw4c8 + in_h_0*iw_pad*8 + in_w_0*8;
                        F16 *in_1 = in_hw4c8 + in_h_1*iw_pad*8 + in_w_1*8;
                        F16 *in_2 = in_hw4c8 + in_h_2*iw_pad*8 + in_w_2*8;
                        F16 *in_3 = in_hw4c8 + in_h_3*iw_pad*8 + in_w_3*8;
                        F16 *in_pack_c8hw4 = in_pack + fh_idx*fw*ic*8*4 + fw_idx*ic*8*4 + c*8*4;

                        /*
                         * for (U32 c8 = 0; c8 < 8; c8++) {
                         *     for (U32 hw4 = 0; hw4 < 4; hw4++) {
                         *         in_pack_c8hw4[c8*4 + hw4] = in_hw4c8[hw4*8 + c8];
                         *     }
                         * }
                         */

                        __asm__ __volatile__(
                            "ldr q0, [%[in_0]]\n"
                            "ldr q1, [%[in_1]]\n"
                            "ldr q2, [%[in_2]]\n"
                            "ldr q3, [%[in_3]]\n"
                            "st4 {v0.8h, v1.8h, v2.8h, v3.8h}, [%[in_pack_0]]\n"
                            :[in_pack_0]"+r"(in_pack_c8hw4)
                            :[in_0]"r"(in_0),
                             [in_1]"r"(in_1),
                             [in_2]"r"(in_2),
                             [in_3]"r"(in_3)
                            :"memory", "cc", "v0", "v1", "v2", "v3"
                        );
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // bias
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q22, [%[b_0]]\n"         //b_o0
                    "ldr q23, [%[b_1]]\n"         //b_o1
                    "mov  x0, %[ic]\n"            //ic_blk
                    "ldr  d0, [%[in_0]]\n"        //in_hw0
                    "ldr q18, [%[f_0]]\n"         //f_o0c0
                    "ldr q19, [%[f_0], #16]\n"    //f_o1c0
                    "mov  v2.16b, v22.16b\n"      //out_o0hw0
                    "mov  v3.16b, v22.16b\n"      //out_o0hw1
                    "mov  v4.16b, v22.16b\n"      //out_o0hw2
                    "mov  v5.16b, v22.16b\n"      //out_o0hw3
                    "mov v10.16b, v23.16b\n"      //out_o1hw0
                    "mov v11.16b, v23.16b\n"      //out_o1hw1
                    "mov v12.16b, v23.16b\n"      //out_o1hw2
                    "mov v13.16b, v23.16b\n"      //out_o1hw3
                    "0:\n"
                    "ldr  d1, [%[in_0], #8]\n"           //in_hw0
                    "ldr q20, [%[f_0], #32]\n"            //f_o0c0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "fmla  v3.8h, v18.8h, v0.h[1]\n"
                    "ldr q21, [%[f_0], #48]\n"            //f_o1c0
                    "fmla  v4.8h, v18.8h, v0.h[2]\n"
                    "fmla  v5.8h, v18.8h, v0.h[3]\n"
                    "fmla v10.8h, v19.8h, v0.h[0]\n"
                    "fmla v11.8h, v19.8h, v0.h[1]\n"
                    "fmla v12.8h, v19.8h, v0.h[2]\n"
                    "fmla v13.8h, v19.8h, v0.h[3]\n"
                    "subs x0, x0, #2\n"

                    "ldr  d0, [%[in_0], #16]\n"           //in_hw0
                    "ldr q18, [%[f_0], #64]\n"            //f_o0c0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "fmla  v3.8h, v20.8h, v1.h[1]\n"
                    "ldr q19, [%[f_0], #80]\n"            //f_o1c0
                    "fmla  v4.8h, v20.8h, v1.h[2]\n"
                    "fmla  v5.8h, v20.8h, v1.h[3]\n"
                    "add %[in_0], %[in_0], #16\n"
                    "add %[f_0], %[f_0], #64\n"
                    "fmla v10.8h, v21.8h, v1.h[0]\n"
                    "fmla v11.8h, v21.8h, v1.h[1]\n"
                    "fmla v12.8h, v21.8h, v1.h[2]\n"
                    "fmla v13.8h, v21.8h, v1.h[3]\n"
                    "bne 0b\n"
                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "fmax  v2.8h,  v2.8h, v1.8h\n"     //max(v2, 0)
                    "fmax  v3.8h,  v3.8h, v1.8h\n"
                    "fmax  v4.8h,  v4.8h, v1.8h\n"
                    "fmax  v5.8h,  v5.8h, v1.8h\n"
                    "fmax v10.8h, v10.8h, v1.8h\n"
                    "fmax v11.8h, v11.8h, v1.8h\n"
                    "fmax v12.8h, v12.8h, v1.8h\n"
                    "fmax v13.8h, v13.8h, v1.8h\n"
                    "1:\n"
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
                    :[ic]"r"((I64)ic*8*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v10", "v11", "v12", "v13", "v18", "v19", "v20", "v21", "v22", "v23", "x0"
                );
                b0 += 16;
                b1 += 16;
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + (oc-1)*8*fh*fw*ic*8;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + (oc-1)*ohow*8 + hw*8;
                // bias
                const F16 *b_o0 = biasArray + (oc-1)*8;
                __asm__ __volatile__(
                    "ldr q22, [%[b_0]]\n"             //b_o0
                    "ldr  d0, [%[in_0]]\n"            //in_hw0
                    "ldr q18, [%[f_0]]\n"             //f_o0c0
                    "mov  x0, %[ic]\n"                //ic_blk
                    "mov  v2.16b, v22.16b\n"          //out_o0hw0
                    "mov  v3.16b, v22.16b\n"          //out_o0hw1
                    "mov  v4.16b, v22.16b\n"          //out_o0hw2
                    "mov  v5.16b, v22.16b\n"          //out_o0hw3
                    "0:\n"
                    "ldr  d1, [%[in_0], #8]\n"         //in_hw0
                    "ldr q20, [%[f_0], #16]\n"         //f_o0c0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "fmla  v3.8h, v18.8h, v0.h[1]\n"
                    "fmla  v4.8h, v18.8h, v0.h[2]\n"
                    "fmla  v5.8h, v18.8h, v0.h[3]\n"
                    "subs x0, x0, #2\n"

                    "ldr  d0, [%[in_0], #16]\n"        //in_hw0
                    "ldr q18, [%[f_0], #32]\n"         //f_o0c0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "fmla  v3.8h, v20.8h, v1.h[1]\n"
                    "fmla  v4.8h, v20.8h, v1.h[2]\n"
                    "fmla  v5.8h, v20.8h, v1.h[3]\n"
                    "add %[in_0], %[in_0], #16\n"
                    "add %[f_0], %[f_0], #32\n"
                    "bne 0b\n"
                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "fmax v2.8h, v2.8h, v1.8h\n"       //max(v2, 0)
                    "fmax v3.8h, v3.8h, v1.8h\n"
                    "fmax v4.8h, v4.8h, v1.8h\n"
                    "fmax v5.8h, v5.8h, v1.8h\n"
                    "1:\n"
                    "str   q2, [%[out_0]]\n"           //out_o0hw0
                    "str   q3, [%[out_0], #16]\n"      //out_o0hw1
                    "str   q4, [%[out_0], #32]\n"      //out_o0hw2
                    "str   q5, [%[out_0], #48]\n"      //out_o0hw3
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_r)
                    :[ic]"r"((I64)ic*8*fh*fw),
                     [b_0]"r"(b_o0),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v18", "v20", "v22", "x0"
                );
            }
        }

        // ohow_reminder % 4
        ohow_s = (ohow / 4) * 4;
        for (I32 hw = ohow_s; hw < ohow; hw++) {
            const F16 *b0 = biasArray;
            const F16 *b1 = biasArray + 8;
            const F16 *f_o0c0 = filterArray;
            F16 *in_pack = ((F16*)tmp) + ic*ih_pad*iw_pad*8;
            // pack input
            // NCHWc8 => NHWChw4 + im2col
            U32 in_h_0 = (hw/ow)*strideH;
            U32 in_w_0 = (hw%ow)*strideW;
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        F16 *in_hw1c8 = inArray_pad + c*ihiw*8 + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        F16 *in_0 = in_hw1c8 + in_h_0*iw_pad*8 + in_w_0*8;
                        F16 *in_pack_c8hw1 = in_pack + fh_idx*fw*ic*8 + fw_idx*ic*8 + c*8;
                        /*
                         * for (U32 c8 = 0; c8 < 8; c8++) {
                         *         in_pack_c8hw1[c8] = in_0[c8];
                         * }
                         */
                        memcpy(in_pack_c8hw1, in_0, 8*bytesOf(idt));
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc-1); o+=2) {
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                F16 *out_o1hw0 = out_o0hw0 + ohow*8;
                // bias
                const F16 *b_o0 = b0;
                const F16 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q22, [%[b_0]]\n"            //b_o0
                    "ldr q23, [%[b_1]]\n"            //b_o1
                    "ldr  h0, [%[in_0]]\n"           //in_hw0
                    "ldr q18, [%[f_0]]\n"            //f_o0c0
                    "ldr q19, [%[f_0], #16]\n"       //f_o1c0
                    "mov  x0, %[ic]\n"               //ic_blk
                    "mov  v2.16b, v22.16b\n"         //out_o0hw0
                    "mov v10.16b, v23.16b\n"         //out_o1hw0
                    "0:\n"
                    "ldr  h1, [%[in_0], #2]\n"        //in_hw0
                    "ldr q20, [%[f_0], #32]\n"        //f_o0c0
                    "ldr q21, [%[f_0], #48]\n"        //f_o1c0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "fmla v10.8h, v19.8h, v0.h[0]\n"
                    "subs x0, x0, #2\n"

                    "ldr  h0, [%[in_0], #4]\n"        //in_hw0
                    "ldr q18, [%[f_0], #64]\n"        //f_o0c0
                    "ldr q19, [%[f_0], #80]\n"        //f_o1c0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "fmla v10.8h, v21.8h, v1.h[0]\n"
                    "add %[in_0], %[in_0], #4\n"
                    "add %[f_0], %[f_0], #64\n"
                    "bne 0b\n"
                    "cbz %[activation], 1f\n"
                    "eor  v1.16b, v1.16b, v1.16b\n"    // zero
                    "fmax  v2.8h,  v2.8h, v1.8h\n"     //max(v2, 0)
                    "fmax v10.8h, v10.8h, v1.8h\n"
                    "1:\n"
                    "str   q2, [%[out_0]]\n"           //out_o0hw0
                    "str  q10, [%[out_1]]\n"           //out_o1hw0
                    :[out_0]"+r"(out_o0hw0),
                     [out_1]"+r"(out_o1hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*8*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v2", "v10", "v18", "v19", "v20", "v21", "v22", "v23", "x0"
                );
                b0 += 16;
                b1 += 16;
            }
            if (oc & 1) {
                // oc%2 != 0
                const F16 *f_r = filterArray + (oc-1)*8*fh*fw*ic*8;
                F16 *in_hw0 = in_pack;
                F16 *out_o0hw0 = outArray + n*oc*ohow*8 + (oc-1)*ohow*8 + hw*8;
                // bias
                const F16 *b_o0 = biasArray + (oc-1)*8;
                __asm__ __volatile__(
                    "ldr q22, [%[b_0]]\n"            //b_o0
                    "ldr  h0, [%[in_0]]\n"           //in_hw0
                    "ldr q18, [%[f_0]]\n"            //f_o0c0
                    "mov  x0, %[ic]\n"               //ic_blk
                    "mov  v2.16b, v22.16b\n"         //out_o0hw0
                    "0:\n"
                    "ldr  h1, [%[in_0], #2]\n"       //in_hw0
                    "ldr q20, [%[f_0], #16]\n"       //f_o0c0
                    "fmla  v2.8h, v18.8h, v0.h[0]\n"
                    "subs x0, x0, #2\n"

                    "ldr  h0, [%[in_0], #4]\n"       //in_hw0
                    "ldr q18, [%[f_0], #32]\n"       //f_o0c0
                    "fmla  v2.8h, v20.8h, v1.h[0]\n"
                    "add %[in_0], %[in_0], #4\n"
                    "add %[f_0], %[f_0], #32\n"
                    "bne 0b\n"
                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"   // zero
                    "fmax v2.8h, v2.8h, v1.8h\n"     //max(v2, 0)
                    "1:\n"
                    "str   q2, [%[out_0]]\n"         //out_o0hw0
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_r)
                    :[ic]"r"((I64)ic*8*fh*fw),
                     [b_0]"r"(b_o0),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v2", "v10", "v18", "v20", "v22", "x0"
                );
            }
        }
    }
    return ret;
}
