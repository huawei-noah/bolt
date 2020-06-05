// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef __aarch64__
#include <cstring>
#include "cpu/arm/fp32/tensor_computing_fp32.h"

EE convolution_gemm_icnchw_V7(TensorDesc inputDesc, F32* inArray,
    TensorDesc filterDesc, const F32* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* outArray,
    ActivationDesc activationDesc)
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
    switch (activationDesc.mode) {
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
        // ohow / 6
        for (I32 hw = 0; hw < ohow - 5; hw += 6) {
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
                        F32 *in_pack_hw6 = in_pack + (fh_idx*fw*ic + fw_idx*ic + c)*6;
                        *in_pack_hw6 = *in_0;
                        *(in_pack_hw6+1) = *in_1;
                        *(in_pack_hw6+2) = *in_2;
                        *(in_pack_hw6+3) = *in_3;
                        *(in_pack_hw6+4) = *in_4;
                        *(in_pack_hw6+5) = *in_5;
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
                    "vld1.f32 {d10-d11}, [%[b_0]]\n"
                    "vld1.f32 {d12-d13}, [%[b_1]]\n"
                    "mov r2, %[ic]\n"
                    
                    "vld1.f32 {d2-d3}, [%[in_0]]!\n"           //in_hw0
                    "vmov.f32  q7, q5\n"
                    "vmov.f32  q9, q5\n"
                    "vmov.f32 q11, q5\n"
                    "vld1.f32 {d0-d1}, [%[f_0]]!\n"            //f_o0c0
                    "vmov.f32 q13, q5\n"
                    "vmov.f32 q15, q5\n"

                    "vmov.f32  q8, q6\n"
                    "vmov.f32 q10, q6\n"
                    "vmov.f32 q12, q6\n"
                    "vmov.f32 q14, q6\n"
                    "vmov.f32  q3, q6\n"
                    "0:\n"
                    "vld1.f32 {d4}, [%[in_0]]!\n"
                    "vld1.f32 {d8-d9}, [%[f_0]]!\n"
                    "vmla.f32  q5, q0, d2[0]\n"
                    "vmla.f32  q7, q0, d2[1]\n"
                    "vmla.f32  q9, q0, d3[0]\n"
                    "vmla.f32 q11, q0, d3[1]\n"
                    "vmla.f32 q13, q0, d4[0]\n"
                    "vmla.f32 q15, q0, d4[1]\n"
                    "vld1.f32 {d0-d1}, [%[f_0]]!\n"

                    "vmla.f32  q6, q4, d2[0]\n"
                    "vmla.f32  q8, q4, d2[1]\n"
                    "vmla.f32 q10, q4, d3[0]\n"
                    "vmla.f32 q12, q4, d3[1]\n"
                    "vld1.f32 {d2-d3}, [%[in_0]]!\n"
                    "vmla.f32 q14, q4, d4[0]\n"
                    "vmla.f32  q3, q4, d4[1]\n"
                    "subs r2, r2, #1\n"
                    "bne 0b\n"

                    "cmp %[activation], #0\n"
                    "beq 1f\n"
                    "veor q1, q1, q1\n"     //zero
                    "vmax.f32  q5,  q5, q1\n"
                    "vmax.f32  q6,  q6, q1\n"
                    "vmax.f32  q7,  q7, q1\n"
                    "vmax.f32  q8,  q8, q1\n"
                    "vmax.f32  q9,  q9, q1\n"
                    "vmax.f32 q10, q10, q1\n"
                    "vmax.f32 q11, q11, q1\n"
                    "vmax.f32 q12, q12, q1\n"
                    "vmax.f32 q13, q13, q1\n"
                    "vmax.f32 q14, q14, q1\n"
                    "vmax.f32 q15, q15, q1\n"
                    "vmax.f32  q3,  q3, q1\n"
                    "1:\n"
                    "vst1.f32 {d10-d11}, [%[out_0]]!\n"
                    "vst1.f32 {d12-d13}, [%[out_0]]!\n"
                    "vst1.f32 {d14-d15}, [%[out_0]]!\n"
                    "vst1.f32 {d16-d17}, [%[out_0]]!\n"
                    "vst1.f32 {d18-d19}, [%[out_0]]!\n"
                    "vst1.f32 {d20-d21}, [%[out_0]]!\n"
                    "vst1.f32 {d22-d23}, [%[out_0]]!\n"
                    "vst1.f32 {d24-d25}, [%[out_0]]!\n"
                    "vst1.f32 {d26-d27}, [%[out_0]]!\n"
                    "vst1.f32 {d28-d29}, [%[out_0]]!\n"
                    "vst1.f32 {d30-d31}, [%[out_0]]!\n"
                    "vst1.f32 {d6-d7},   [%[out_0]]\n"
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                        "q11", "q12", "q13", "q14", "q15",
                        "r2"
                );
                b0 += 8;
                b1 += 8;
            }
        }

        U32 ohow_s = (ohow / 6) * 6;
        U32 ohow_tail = ohow - ohow_s;

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
                    "vld1.f32 {d10-d11}, [%[b_0]]\n"
                    "vld1.f32 {d12-d13}, [%[b_1]]\n"
                    "mov  r2, %[ic]\n"

                    "vld1.f32  {d2-d3}, [%[in_0]]!\n"           //in_hw0
                    "vmov.f32  q7, q5\n"
                    "vmov.f32  q9, q5\n"
                    "vmov.f32 q11, q5\n"
                    "vld1.f32 {d0-d1}, [%[f_0]]!\n"            //f_o0c0

                    "vmov.f32  q8, q6\n"
                    "vmov.f32 q10, q6\n"
                    "vmov.f32 q12, q6\n"
                    "0:\n"
                    "vld1.f32 {d4-d5}, [%[in_0]]!\n"
                    "vld1.f32 {d8-d9}, [%[f_0]]!\n"
                    "vmla.f32  q5, q0, d2[0]\n"
                    "vmla.f32  q7, q0, d2[1]\n"
                    "vmla.f32  q9, q0, d3[0]\n"
                    "vmla.f32 q11, q0, d3[1]\n"

                    "vmla.f32  q6, q4, d2[0]\n"
                    "vmla.f32  q8, q4, d2[1]\n"
                    "subs r2, r2, #1\n"
                    "vmla.f32 q10, q4, d3[0]\n"
                    "vmla.f32 q12, q4, d3[1]\n"
                    "vmov.f32 q1, q2\n"
                    "bne 0b\n"

                    "cmp %[activation], #0\n"
                    "beq 1f\n"
                    "veor q1, q1, q1\n"     //zero
                    "vmax.f32  q5,  q5, q1\n"
                    "vmax.f32  q6,  q6, q1\n"
                    "vmax.f32  q7,  q7, q1\n"
                    "vmax.f32  q8,  q8, q1\n"
                    "vmax.f32  q9,  q9, q1\n"
                    "vmax.f32 q10, q10, q1\n"
                    "vmax.f32 q11, q11, q1\n"
                    "vmax.f32 q12, q12, q1\n"
                    "1:\n"
                    "vst1.f32 {d10-d11}, [%[out_0]]!\n"
                    "vst1.f32 {d12-d13}, [%[out_0]]!\n"
                    "vst1.f32 {d14-d15}, [%[out_0]]!\n"
                    "vst1.f32 {d16-d17}, [%[out_0]]!\n"
                    "vst1.f32 {d18-d19}, [%[out_0]]!\n"
                    "vst1.f32 {d20-d21}, [%[out_0]]!\n"
                    "vst1.f32 {d22-d23}, [%[out_0]]!\n"
                    "vst1.f32 {d24-d25}, [%[out_0]]\n"
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "q0", "q1", "q3", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
                     "q12", "q4", "r2"
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
                    "vld1.f32 {d10-d11}, [%[b_0]]\n"
                    "vld1.f32 {d12-d13}, [%[b_1]]\n"
                    "mov r2, %[ic]\n"

                    "0:\n"
                    "vld1.f32 {d0-d1}, [%[f_0]]!\n"
                    "vld1.f32 {d8-d9}, [%[f_0]]!\n"
                    "vld1.f32 {d2[0]}, [%[in_0]]!\n"
                    "subs r2, r2, #1\n"
                    "vmla.f32 q5, q0, d2[0]\n"
                    "vmla.f32 q6, q4, d2[0]\n"
                    "bne 0b\n"

                    "cmp %[activation], #0\n"
                    "beq 1f\n"
                    "veor q1, q1, q1\n"     //zero
                    "vmax.f32 q5, q5, q1\n"
                    "vmax.f32 q6, q6, q1\n"
                    "1:\n"
                    "vst1.f32 {d10-d11}, [%[out_0]]!\n"
                    "vst1.f32 {d12-d13}, [%[out_0]]\n"
                    :[out_0]"+r"(out_o0hw0),
                     [in_0]"+r"(in_hw0),
                     [f_0]"+r"(f_o0c0)
                    :[ic]"r"((I64)ic*fh*fw),
                     [b_0]"r"(b_o0),
                     [b_1]"r"(b_o1),
                     [activation]"r"(activation)
                    :"memory", "cc", "q0", "q1", "q5", "q6", "q4", "r2"
                );
                b0 += 8;
                b1 += 8;
            }
        }
    }
    return ret;
}
#endif
