// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <cstring>
#include "cpu/arm/fp32/tensor_computing_fp32.h"

EE convolution_gemm_icnchw_V8(TensorDesc inputDesc, F32* inArray,
    TensorDesc filterDesc, const F32* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* outArray,
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

    if (fdf != DF_NHWCN8) {
        CHECK_STATUS(NOT_MATCH);
    }

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
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    I32 ohow = oh*ow;
    U32 ihiw = ih_pad*iw_pad;
    F32 *inArray_pad;
    EE ret = SUCCESS;
    for (U32 n = 0; n < in; n++) {
        if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            inArray_pad = inArray + n*ic*ih*iw;
        } else {
            // copy input into a input with padding
            inArray_pad = (F32*)tmp;
            F32 *inArray_pad_mov = inArray_pad;
            F32 *inArray_mov = inArray + n*ic*ih*iw;
            for (U32 c = 0; c < ic; c++) {
                for (U32 h = 0; h < paddingT; h++) {
                    memset(inArray_pad_mov, 0, iw_pad*bytesOf(idt));
                    inArray_pad_mov += iw_pad;
                }
                for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                    memset(inArray_pad_mov, 0, paddingL*bytesOf(idt));
                    inArray_pad_mov += paddingL;
                    memcpy(inArray_pad_mov, inArray_mov, iw*bytesOf(idt));
                    inArray_pad_mov += iw;
                    inArray_mov += iw;
                    memset(inArray_pad_mov, 0, paddingR*bytesOf(idt));
                    inArray_pad_mov += paddingR;
                }
                for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                    memset(inArray_pad_mov, 0, iw_pad*bytesOf(idt));
                    inArray_pad_mov += iw_pad;
                }
            }
        }
        // ohow / 12
        for (I32 hw = 0; hw < ohow - 11; hw += 12) {
            const F32 *b0 = biasArray;
            const F32 *b1 = biasArray + 4;
            F32 *in_pack = ((F32*)tmp) + ic*ih_pad*iw_pad;
            // pack input
            // NCHW => NHWChw12 + im2col
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
            U32 in_h_8 = ((hw+8)/ow)*strideH;
            U32 in_w_8 = ((hw+8)%ow)*strideW;
            U32 in_h_9 = ((hw+9)/ow)*strideH;
            U32 in_w_9 = ((hw+9)%ow)*strideW;
            U32 in_h_10 = ((hw+10)/ow)*strideH;
            U32 in_w_10 = ((hw+10)%ow)*strideW;
            U32 in_h_11 = ((hw+11)/ow)*strideH;
            U32 in_w_11 = ((hw+11)%ow)*strideW;
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        F32 *in_hw = inArray_pad + c*ihiw + fh_idx*dilateH*iw_pad + dilateW*fw_idx;
                        F32 *in_0 = in_hw + in_h_0*iw_pad + in_w_0;
                        F32 *in_1 = in_hw + in_h_1*iw_pad + in_w_1;
                        F32 *in_2 = in_hw + in_h_2*iw_pad + in_w_2;
                        F32 *in_3 = in_hw + in_h_3*iw_pad + in_w_3;
                        F32 *in_4 = in_hw + in_h_4*iw_pad + in_w_4;
                        F32 *in_5 = in_hw + in_h_5*iw_pad + in_w_5;
                        F32 *in_6 = in_hw + in_h_6*iw_pad + in_w_6;
                        F32 *in_7 = in_hw + in_h_7*iw_pad + in_w_7;
                        F32 *in_8 = in_hw + in_h_8*iw_pad + in_w_8;
                        F32 *in_9 = in_hw + in_h_9*iw_pad + in_w_9;
                        F32 *in_10 = in_hw + in_h_10*iw_pad + in_w_10;
                        F32 *in_11 = in_hw + in_h_11*iw_pad + in_w_11;
                        F32 *in_pack_hw12 = in_pack + fh_idx*fw*ic*12 + fw_idx*ic*12 + c*12;
                        *in_pack_hw12 = *in_0;
                        *(in_pack_hw12+1) = *in_1;
                        *(in_pack_hw12+2) = *in_2;
                        *(in_pack_hw12+3) = *in_3;
                        *(in_pack_hw12+4) = *in_4;
                        *(in_pack_hw12+5) = *in_5;
                        *(in_pack_hw12+6) = *in_6;
                        *(in_pack_hw12+7) = *in_7;
                        *(in_pack_hw12+8) = *in_8;
                        *(in_pack_hw12+9) = *in_9;
                        *(in_pack_hw12+10) = *in_10;
                        *(in_pack_hw12+11) = *in_11;
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o*8*fh*fw*ic;
                F32 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;

                // bias
                const F32 *b_o0 = b0;
                const F32 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q27, [%[b_0]]\n"
                    "ldr q28, [%[b_1]]\n"
                    //give in address to x3
                    "mov x3, %[in_0]\n"

                    //give f address to x0
                    "mov x0, %[f_0]\n"

                    "mov  x2, %[ic]\n"
                    
                    "mov  v5.16b, v27.16b\n"
                    "ldr  q1, [%[in_0]]\n"           //in_hw0
                    "mov  v7.16b, v27.16b\n"
                    "mov  v9.16b, v27.16b\n"
                    "mov  v11.16b, v27.16b\n"
                    "ldr q0, [%[f_0]]\n"            //f_o0c0
                    "mov  v13.16b, v27.16b\n"
                    "mov  v15.16b, v27.16b\n"
                    "mov  v17.16b, v27.16b\n"
                    "ldr q3, [%[in_0], #16]\n"
                    "mov  v19.16b, v27.16b\n"
                    "mov v21.16b, v27.16b\n"
                    "mov v23.16b, v27.16b\n"
                    "mov v25.16b, v27.16b\n"

                    "mov v6.16b, v28.16b\n"
                    "mov v8.16b, v28.16b\n"
                    "mov v10.16b, v28.16b\n"
                    "mov v12.16b, v28.16b\n"
                    "mov v14.16b, v28.16b\n"
                    "mov v16.16b, v28.16b\n"
                    "mov v18.16b, v28.16b\n"
                    "mov v20.16b, v28.16b\n"
                    "mov v22.16b, v28.16b\n"
                    "mov v24.16b, v28.16b\n"
                    "mov v26.16b, v28.16b\n"
                    "0:\n"
                    "fmla  v5.4s, v0.4s, v1.s[0]\n"
                    "fmla  v7.4s, v0.4s, v1.s[1]\n"
                    "ldr q2, [x3, 32]\n"
                    "ldr q29, [x0, 16]\n"
                    "fmla  v9.4s, v0.4s, v1.s[2]\n"
                    "fmla v11.4s, v0.4s, v1.s[3]\n"

                    "fmla v13.4s, v0.4s, v3.s[0]\n"
                    "fmla v15.4s, v0.4s, v3.s[1]\n"
                    "fmla v17.4s, v0.4s, v3.s[2]\n"
                    "fmla v19.4s, v0.4s, v3.s[3]\n"

                    "fmla v21.4s, v0.4s, v2.s[0]\n"
                    "fmla v23.4s, v0.4s, v2.s[1]\n"
                    "fmla v25.4s, v0.4s, v2.s[2]\n"
                    "fmla v27.4s, v0.4s, v2.s[3]\n"

                    "fmla  v6.4s, v29.4s, v1.s[0]\n"
                    "fmla  v8.4s, v29.4s, v1.s[1]\n"
                    "fmla v10.4s, v29.4s, v1.s[2]\n"
                    "fmla v12.4s, v29.4s, v1.s[3]\n"

                    "fmla v14.4s, v29.4s, v3.s[0]\n"
                    "fmla v16.4s, v29.4s, v3.s[1]\n"
                    "ldr q1, [x3, 48]!\n"
                    "ldr q0, [x0, 32]!\n"
                    "fmla v18.4s, v29.4s, v3.s[2]\n"
                    "fmla v20.4s, v29.4s, v3.s[3]\n"

                    "fmla v22.4s, v29.4s, v2.s[0]\n"
                    "fmla v24.4s, v29.4s, v2.s[1]\n"
                    "ldr q3, [x3, 16]\n"
                    "subs x2, x2, #1\n"
                    "fmla v26.4s, v29.4s, v2.s[2]\n"
                    "fmla v28.4s, v29.4s, v2.s[3]\n"
                    "bne 0b\n"

                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "fmax  v5.4s,  v5.4s, v1.4s\n"
                    "fmax  v6.4s,  v6.4s, v1.4s\n"
                    "fmax  v7.4s,  v7.4s, v1.4s\n"
                    "fmax  v8.4s,  v8.4s, v1.4s\n"
                    "fmax  v9.4s,  v9.4s, v1.4s\n"
                    "fmax v10.4s, v10.4s, v1.4s\n"
                    "fmax v11.4s, v11.4s, v1.4s\n"
                    "fmax v12.4s, v12.4s, v1.4s\n"
                    "fmax v13.4s, v13.4s, v1.4s\n"
                    "fmax v14.4s, v14.4s, v1.4s\n"
                    "fmax v15.4s, v15.4s, v1.4s\n"
                    "fmax v16.4s, v16.4s, v1.4s\n"
                    "fmax v17.4s, v17.4s, v1.4s\n"
                    "fmax v18.4s, v18.4s, v1.4s\n"
                    "fmax v19.4s, v19.4s, v1.4s\n"
                    "fmax v20.4s, v20.4s, v1.4s\n"
                    "fmax v21.4s, v21.4s, v1.4s\n"
                    "fmax v22.4s, v22.4s, v1.4s\n"
                    "fmax v23.4s, v23.4s, v1.4s\n"
                    "fmax v24.4s, v24.4s, v1.4s\n"
                    "fmax v25.4s, v25.4s, v1.4s\n"
                    "fmax v26.4s, v26.4s, v1.4s\n"
                    "fmax v27.4s, v27.4s, v1.4s\n"
                    "fmax v28.4s, v28.4s, v1.4s\n"
                    "1:\n"
                    "str   q5, [%[out_0]]\n"
                    "str   q6, [%[out_0], #16]\n"
                    "str   q7, [%[out_0], #32]\n"
                    "str   q8, [%[out_0], #48]\n"
                    "str   q9, [%[out_0], #64]\n"
                    "str   q10, [%[out_0], #80]\n"
                    "str   q11, [%[out_0], #96]\n"
                    "str   q12, [%[out_0], #112]\n"
                    "str   q13, [%[out_0], #128]\n"
                    "str   q14, [%[out_0], #144]\n"
                    "str   q15, [%[out_0], #160]\n"
                    "str   q16, [%[out_0], #176]\n"
                    "str   q17, [%[out_0], #192]\n"
                    "str   q18, [%[out_0], #208]\n"
                    "str   q19, [%[out_0], #224]\n"
                    "str   q20, [%[out_0], #240]\n"
                    "str   q21, [%[out_0], #256]\n"
                    "str   q22, [%[out_0], #272]\n"
                    "str   q23, [%[out_0], #288]\n"
                    "str   q24, [%[out_0], #304]\n"
                    "str   q25, [%[out_0], #320]\n"
                    "str   q26, [%[out_0], #336]\n"
                    "str   q27, [%[out_0], #352]\n"
                    "str   q28, [%[out_0], #368]\n"
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9", "v10",
                        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x0", "x1", "x2", "x3"
                );
                b0 += 8;
                b1 += 8;
            }
        }

        U32 ohow_s = (ohow / 12) * 12;
        U32 ohow_tail = ohow - ohow_s;

        if (ohow_tail >= 8) {
            I32 hw = ohow_s;
            const F32 *b0 = biasArray;
            const F32 *b1 = biasArray + 4;
            F32 *in_pack = ((F32*)tmp) + ic*ih_pad*iw_pad;
            // pack input
            // NCHW => NHWChw8 + im2col
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
                        F32 *in_hw = inArray_pad + c*ihiw + fh_idx*dilateH*iw_pad + dilateW*fw_idx;
                        F32 *in_0 = in_hw + in_h_0*iw_pad + in_w_0;
                        F32 *in_1 = in_hw + in_h_1*iw_pad + in_w_1;
                        F32 *in_2 = in_hw + in_h_2*iw_pad + in_w_2;
                        F32 *in_3 = in_hw + in_h_3*iw_pad + in_w_3;
                        F32 *in_4 = in_hw + in_h_4*iw_pad + in_w_4;
                        F32 *in_5 = in_hw + in_h_5*iw_pad + in_w_5;
                        F32 *in_6 = in_hw + in_h_6*iw_pad + in_w_6;
                        F32 *in_7 = in_hw + in_h_7*iw_pad + in_w_7;
                        F32 *in_pack_hw8 = in_pack + fh_idx*fw*ic*8 + fw_idx*ic*8 + c*8;
                        *in_pack_hw8 = *in_0;
                        *(in_pack_hw8+1) = *in_1;
                        *(in_pack_hw8+2) = *in_2;
                        *(in_pack_hw8+3) = *in_3;
                        *(in_pack_hw8+4) = *in_4;
                        *(in_pack_hw8+5) = *in_5;
                        *(in_pack_hw8+6) = *in_6;
                        *(in_pack_hw8+7) = *in_7;
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o*8*fh*fw*ic;
                F32 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                // bias
                const F32 *b_o0 = b0;
                const F32 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q27, [%[b_0]]\n"
                    "ldr q28, [%[b_1]]\n"
                    //give in address to x3
                    "mov x3, %[in_0]\n"

                    //give f address to x0
                    "mov x0, %[f_0]\n"

                    "mov  x2, %[ic]\n"
                    
                    "mov  v5.16b, v27.16b\n"
                    "ldr  q1, [%[in_0]]\n"           //in_hw0
                    "mov  v7.16b, v27.16b\n"
                    "mov  v9.16b, v27.16b\n"
                    "mov  v11.16b, v27.16b\n"
                    "ldr q0, [%[f_0]]\n"            //f_o0c0
                    "mov  v13.16b, v27.16b\n"
                    "mov  v15.16b, v27.16b\n"
                    "mov  v17.16b, v27.16b\n"
                    "mov  v19.16b, v27.16b\n"

                    "mov v6.16b, v28.16b\n"
                    "mov v8.16b, v28.16b\n"
                    "mov v10.16b, v28.16b\n"
                    "mov v12.16b, v28.16b\n"
                    "mov v14.16b, v28.16b\n"
                    "mov v16.16b, v28.16b\n"
                    "mov v18.16b, v28.16b\n"
                    "mov v20.16b, v28.16b\n"
                    "0:\n"
                    "ldr q3, [x3, 16]!\n"
                    "ldr q29, [x0, 16]\n"
                    "fmla  v5.4s, v0.4s, v1.s[0]\n"
                    "fmla  v7.4s, v0.4s, v1.s[1]\n"
                    "fmla  v9.4s, v0.4s, v1.s[2]\n"
                    "fmla v11.4s, v0.4s, v1.s[3]\n"

                    "fmla v13.4s, v0.4s, v3.s[0]\n"
                    "fmla v15.4s, v0.4s, v3.s[1]\n"
                    "fmla v17.4s, v0.4s, v3.s[2]\n"
                    "fmla v19.4s, v0.4s, v3.s[3]\n"

                    "fmla  v6.4s, v29.4s, v1.s[0]\n"
                    "fmla  v8.4s, v29.4s, v1.s[1]\n"
                    "fmla v10.4s, v29.4s, v1.s[2]\n"
                    "fmla v12.4s, v29.4s, v1.s[3]\n"

                    "fmla v14.4s, v29.4s, v3.s[0]\n"
                    "fmla v16.4s, v29.4s, v3.s[1]\n"
                    "ldr q1, [x3, 16]!\n"
                    "ldr q0, [x0, 32]!\n"
                    "subs x2, x2, #1\n"
                    "fmla v18.4s, v29.4s, v3.s[2]\n"
                    "fmla v20.4s, v29.4s, v3.s[3]\n"
                    "bne 0b\n"

                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "fmax  v5.4s,  v5.4s, v1.4s\n"
                    "fmax  v6.4s,  v6.4s, v1.4s\n"
                    "fmax  v7.4s,  v7.4s, v1.4s\n"
                    "fmax  v8.4s,  v8.4s, v1.4s\n"
                    "fmax  v9.4s,  v9.4s, v1.4s\n"
                    "fmax v10.4s, v10.4s, v1.4s\n"
                    "fmax v11.4s, v11.4s, v1.4s\n"
                    "fmax v12.4s, v12.4s, v1.4s\n"
                    "fmax v13.4s, v13.4s, v1.4s\n"
                    "fmax v14.4s, v14.4s, v1.4s\n"
                    "fmax v15.4s, v15.4s, v1.4s\n"
                    "fmax v16.4s, v16.4s, v1.4s\n"
                    "fmax v17.4s, v17.4s, v1.4s\n"
                    "fmax v18.4s, v18.4s, v1.4s\n"
                    "fmax v19.4s, v19.4s, v1.4s\n"
                    "fmax v20.4s, v20.4s, v1.4s\n"
                    "1:\n"
                    "str   q5, [%[out_0]]\n"
                    "str   q6, [%[out_0], #16]\n"
                    "str   q7, [%[out_0], #32]\n"
                    "str   q8, [%[out_0], #48]\n"
                    "str   q9, [%[out_0], #64]\n"
                    "str   q10, [%[out_0], #80]\n"
                    "str   q11, [%[out_0], #96]\n"
                    "str   q12, [%[out_0], #112]\n"
                    "str   q13, [%[out_0], #128]\n"
                    "str   q14, [%[out_0], #144]\n"
                    "str   q15, [%[out_0], #160]\n"
                    "str   q16, [%[out_0], #176]\n"
                    "str   q17, [%[out_0], #192]\n"
                    "str   q18, [%[out_0], #208]\n"
                    "str   q19, [%[out_0], #224]\n"
                    "str   q20, [%[out_0], #240]\n"
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                     "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                     "v27", "v28", "v29", "x0", "x1", "x2", "x3"
                );
                b0 += 8;
                b1 += 8;
            }
            ohow_s += 8;
            ohow_tail -= 8;
        }

        if (ohow_tail >= 4) {
            I32 hw = ohow_s;
            const F32 *b0 = biasArray;
            const F32 *b1 = biasArray + 4;
            F32 *in_pack = ((F32*)tmp) + ic*ih_pad*iw_pad;
            // pack input
            // NCHW => NHWChw4 + im2col
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
                        F32 *in_hw = inArray_pad + c*ihiw + fh_idx*dilateH*iw_pad + dilateW*fw_idx;
                        F32 *in_0 = in_hw + in_h_0*iw_pad + in_w_0;
                        F32 *in_1 = in_hw + in_h_1*iw_pad + in_w_1;
                        F32 *in_2 = in_hw + in_h_2*iw_pad + in_w_2;
                        F32 *in_3 = in_hw + in_h_3*iw_pad + in_w_3;
                        F32 *in_pack_hw4 = in_pack + fh_idx*fw*ic*4 + fw_idx*ic*4 + c*4;
                        *in_pack_hw4 = *in_0;
                        *(in_pack_hw4+1) = *in_1;
                        *(in_pack_hw4+2) = *in_2;
                        *(in_pack_hw4+3) = *in_3;
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o*8*fh*fw*ic;
                F32 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                // bias
                const F32 *b_o0 = b0;
                const F32 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q27, [%[b_0]]\n"
                    "ldr q28, [%[b_1]]\n"
                    //give in address to x3
                    "mov x3, %[in_0]\n"

                    //give f address to x0
                    "mov x0, %[f_0]\n"

                    "mov  x2, %[ic]\n"
                    
                    "mov  v5.16b, v27.16b\n"
                    "ldr  q1, [%[in_0]]\n"           //in_hw0
                    "mov  v7.16b, v27.16b\n"
                    "mov  v9.16b, v27.16b\n"
                    "mov  v11.16b, v27.16b\n"
                    "ldr q0, [%[f_0]]\n"            //f_o0c0

                    "mov v6.16b, v28.16b\n"
                    "mov v8.16b, v28.16b\n"
                    "mov v10.16b, v28.16b\n"
                    "mov v12.16b, v28.16b\n"
                    "0:\n"
                    "ldr q3, [x3, 16]!\n"
                    "ldr q29, [x0, 16]\n"
                    "fmla  v5.4s, v0.4s, v1.s[0]\n"
                    "fmla  v7.4s, v0.4s, v1.s[1]\n"
                    "fmla  v9.4s, v0.4s, v1.s[2]\n"
                    "fmla v11.4s, v0.4s, v1.s[3]\n"

                    "fmla  v6.4s, v29.4s, v1.s[0]\n"
                    "fmla  v8.4s, v29.4s, v1.s[1]\n"
                    "ldr q0, [x0, 32]!\n"
                    "subs x2, x2, #1\n"
                    "fmla v10.4s, v29.4s, v1.s[2]\n"
                    "fmla v12.4s, v29.4s, v1.s[3]\n"

                    "mov	v1.16b, v3.16b\n"
                    "bne 0b\n"

                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "fmax  v5.4s,  v5.4s, v1.4s\n"
                    "fmax  v6.4s,  v6.4s, v1.4s\n"
                    "fmax  v7.4s,  v7.4s, v1.4s\n"
                    "fmax  v8.4s,  v8.4s, v1.4s\n"
                    "fmax  v9.4s,  v9.4s, v1.4s\n"
                    "fmax v10.4s, v10.4s, v1.4s\n"
                    "fmax v11.4s, v11.4s, v1.4s\n"
                    "fmax v12.4s, v12.4s, v1.4s\n"
                    "1:\n"
                    "str   q5, [%[out_0]]\n"
                    "str   q6, [%[out_0], #16]\n"
                    "str   q7, [%[out_0], #32]\n"
                    "str   q8, [%[out_0], #48]\n"
                    "str   q9, [%[out_0], #64]\n"
                    "str   q10, [%[out_0], #80]\n"
                    "str   q11, [%[out_0], #96]\n"
                    "str   q12, [%[out_0], #112]\n"
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v3", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                     "v12", "v27", "v28", "v29", "x0", "x1", "x2", "x3"
                );
                b0 += 8;
                b1 += 8;
            }
            ohow_s += 4;
            ohow_tail -= 4;
        }

        for (I32 hw = ohow_s; hw < ohow; hw++) {
            const F32 *b0 = biasArray;
            const F32 *b1 = biasArray + 4;
            F32 *in_pack = ((F32*)tmp) + ic*ih_pad*iw_pad;
            // pack input
            // NCHW => NCHWc8hw1 + im2col
            U32 in_h_0 = (hw/ow)*strideH;
            U32 in_w_0 = (hw%ow)*strideW;
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        F32 *in_hw = inArray_pad + c*ihiw + fh_idx*dilateH*iw_pad + dilateW*fw_idx;
                        F32 *in_0 = in_hw + in_h_0*iw_pad + in_w_0;
                        F32 *in_pack_hw1 = in_pack + fh_idx*fw*ic + fw_idx*ic + c;
                        *in_pack_hw1 = *in_0;
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o*8*fh*fw*ic;
                F32 *out_o0hw0 = outArray + n*oc*ohow*8 + o*ohow*8 + hw*8;
                // bias
                const F32 *b_o0 = b0;
                const F32 *b_o1 = b1;
                __asm__ __volatile__(
                    "ldr q5, [%[b_0]]\n"
                    "ldr q6, [%[b_1]]\n"
                    //give in address to x3
                    "mov x3, %[in_0]\n"

                    //give f address to x0
                    "mov x0, %[f_0]\n"

                    "mov  x2, %[ic]\n"

                    "0:\n"
                    "ldr q0, [x0], #16\n"
                    "subs x2, x2, #1\n"
                    "ldr q29, [x0], #16\n"
                    "ldr s1, [x3], #4\n"
                    "fmla  v5.4s, v0.4s, v1.s[0]\n"
                    "fmla  v6.4s, v29.4s, v1.s[0]\n"

                    "bne 0b\n"

                    "cbz %[activation], 1f\n"
                    "eor v1.16b, v1.16b, v1.16b\n"     //zero
                    "fmax  v5.4s,  v5.4s, v1.4s\n"
                    "fmax  v6.4s,  v6.4s, v1.4s\n"
                    "1:\n"
                    "str   q5, [%[out_0]]\n"
                    "str   q6, [%[out_0], #16]\n"
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "v0", "v1", "v5", "v6", "v29", "x0", "x1", "x2", "x3"
                );
                b0 += 8;
                b1 += 8;
            }
        }
    }
    return ret;
}
