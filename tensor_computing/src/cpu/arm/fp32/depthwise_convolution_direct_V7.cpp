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
#include "cpu/arm/fp32/depthwise_convolution.h"

EE depthwise_convolution_direct_V7(TensorDesc inputDesc, F32* inArray,
    TensorDesc filterDesc, const F32* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* outArray,
    ActivationDesc depthwiseActivationDesc)
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
            // ohow / 6
            for (I32 hw = 0; hw < ohow-5; hw+=6) {
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

                F32 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;
                __asm__ __volatile__(
                    "vld1.f32 {d0-d3}, [%[b]]\n"
                    "vmov.f32  q2, q0\n"
                    "vmov.f32  q3, q1\n"
                    "vmov.f32  q4, q0\n"
                    "vmov.f32  q5, q1\n"
                    "vmov.f32  q6, q0\n"
                    "vmov.f32  q7, q1\n"
                    "vmov.f32  q8, q0\n"
                    "vmov.f32  q9, q1\n"
                    "vmov.f32 q10, q0\n"
                    "vmov.f32 q11, q1\n"
                    : [b] "+r"(b)
                    :
                    : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
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
                        __asm__ __volatile__(
                            "vld1.f32 {d24-d27}, [%[f0]]\n"
                            "vld1.f32 {d28-d29}, [%[in0]]!\n"
                            "vld1.f32 {d30-d31}, [%[in0]]\n"

                            "vmla.f32  q0, q14, q12\n"
                            "vld1.f32 {d28-d29}, [%[in1]]!\n"
                            "vmla.f32  q1, q15, q13\n"
                            "vld1.f32 {d30-d31}, [%[in1]]\n"
                            "vmla.f32  q2, q14, q12\n"
                            "vld1.f32 {d28-d29}, [%[in2]]!\n"
                            "vmla.f32  q3, q15, q13\n"
                            "vld1.f32 {d30-d31}, [%[in2]]\n"
                            "vmla.f32  q4, q14, q12\n"
                            "vld1.f32 {d28-d29}, [%[in3]]!\n"
                            "vmla.f32  q5, q15, q13\n"
                            "vld1.f32 {d30-d31}, [%[in3]]\n"
                            "vmla.f32  q6, q14, q12\n"
                            "vld1.f32 {d28-d29}, [%[in4]]!\n"
                            "vmla.f32  q7, q15, q13\n"
                            "vld1.f32 {d30-d31}, [%[in4]]\n"
                            "vmla.f32  q8, q14, q12\n"
                            "vld1.f32 {d28-d29}, [%[in5]]!\n"
                            "vmla.f32  q9, q15, q13\n"
                            "vld1.f32 {d30-d31}, [%[in5]]\n"
                            "vmla.f32 q10, q14, q12\n"
                            "vmla.f32 q11, q15, q13\n"
                            : [in0] "+r"(in_0),
                              [in1] "+r"(in_1),
                              [in2] "+r"(in_2),
                              [in3] "+r"(in_3),
                              [in4] "+r"(in_4),
                              [in5] "+r"(in_5)
                            : [f0] "r"(f_0)
                            : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                              "q11", "q12", "q13", "q14", "q15"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationDesc.mode){
                    case ACTIVATION_NULL:
                         break;
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "veor q15, q15, q15\n" // zero
                            "vmax.f32  q0,  q0, q15\n"
                            "vmax.f32  q1,  q1, q15\n"
                            "vmax.f32  q2,  q2, q15\n"
                            "vmax.f32  q3,  q3, q15\n"
                            "vmax.f32  q4,  q4, q15\n"
                            "vmax.f32  q5,  q5, q15\n"
                            "vmax.f32  q6,  q6, q15\n"
                            "vmax.f32  q7,  q7, q15\n"
                            "vmax.f32  q8,  q8, q15\n"
                            "vmax.f32  q9,  q9, q15\n"
                            "vmax.f32 q10, q10, q15\n"
                            "vmax.f32 q11, q11, q15\n"
                            :
                            :
                            : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                              "q11", "q12", "q13", "q14", "q15"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "veor q15, q15, q15\n" // zero
                            "vmov.f32 q14, #6.0\n"              // six
                            "vmax.f32  q0,  q0, q15\n"
                            "vmax.f32  q1,  q1, q15\n"
                            "vmax.f32  q2,  q2, q15\n"
                            "vmax.f32  q3,  q3, q15\n"
                            "vmax.f32  q4,  q4, q15\n"
                            "vmax.f32  q5,  q5, q15\n"
                            "vmax.f32  q6,  q6, q15\n"
                            "vmax.f32  q7,  q7, q15\n"
                            "vmax.f32  q8,  q8, q15\n"
                            "vmax.f32  q9,  q9, q15\n"
                            "vmax.f32 q10, q10, q15\n"
                            "vmax.f32 q11, q11, q15\n"

                            "vmin.f32  q0,  q0, q14\n"
                            "vmin.f32  q1,  q1, q14\n"
                            "vmin.f32  q2,  q2, q14\n"
                            "vmin.f32  q3,  q3, q14\n"
                            "vmin.f32  q4,  q4, q14\n"
                            "vmin.f32  q5,  q5, q14\n"
                            "vmin.f32  q6,  q6, q14\n"
                            "vmin.f32  q7,  q7, q14\n"
                            "vmin.f32  q8,  q8, q14\n"
                            "vmin.f32  q9,  q9, q14\n"
                            "vmin.f32 q10, q10, q14\n"
                            "vmin.f32 q11, q11, q14\n"
                            :
                            :
                            : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                              "q11", "q12", "q13", "q14", "q15"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        F32 *out_0 = out_ptr;
                        F32 *out_24 = out_ptr + 24;
                        __asm__ __volatile__(
                            "vstm %[out], {d0-d11}\n"
                            "vmov.f32 q13, #3.0\n"      // three
                            "vmov.f32 q14, #6.0\n"      // six
                            "veor q15, q15, q15\n" // zero
                            "vrecpe.f32 q12, q14\n"
                            "vrecps.f32 q0, q14, q12\n"
                            "vmul.f32 q12, q0, q12\n"

                            "vadd.f32 q0,  q6, q13\n"
                            "vadd.f32 q1,  q7, q13\n"
                            "vadd.f32 q2,  q8, q13\n"
                            "vadd.f32 q3,  q9, q13\n"
                            "vadd.f32 q4, q10, q13\n"
                            "vadd.f32 q5, q11, q13\n"
                            "vmax.f32 q0,  q0, q15\n"
                            "vmax.f32 q1,  q1, q15\n"
                            "vmax.f32 q2,  q2, q15\n"
                            "vmax.f32 q3,  q3, q15\n"
                            "vmax.f32 q4,  q4, q15\n"
                            "vmax.f32 q5,  q5, q15\n"
                            "vmin.f32 q0,  q0, q14\n"
                            "vmin.f32 q1,  q1, q14\n"
                            "vmin.f32 q2,  q2, q14\n"
                            "vmin.f32 q3,  q3, q14\n"
                            "vmin.f32 q4,  q4, q14\n"
                            "vmin.f32 q5,  q5, q14\n"
                            "vmul.f32 q0,  q0, q12\n"
                            "vmul.f32 q1,  q1, q12\n"
                            "vmul.f32 q2,  q2, q12\n"
                            "vmul.f32 q3,  q3, q12\n"
                            "vmul.f32 q4,  q4, q12\n"
                            "vmul.f32 q5,  q5, q12\n"
                            "vmul.f32 q0,  q0, q6\n"
                            "vld1.f32 {d12-d13}, [%[out_0]]!\n"
                            "vst1.f32 {d0-d1}, [%[out_24]]!\n"
                            "vmul.f32 q1,  q1, q7\n"
                            "vld1.f32 {d14-d15}, [%[out_0]]!\n"
                            "vst1.f32 {d2-d3}, [%[out_24]]!\n"
                            "vmul.f32 q2,  q2, q8\n"
                            "vld1.f32 {d16-d17}, [%[out_0]]!\n"
                            "vst1.f32 {d4-d5}, [%[out_24]]!\n"
                            "vmul.f32 q3,  q3, q9\n"
                            "vld1.f32 {d18-d19}, [%[out_0]]!\n"
                            "vst1.f32 {d6-d7}, [%[out_24]]!\n"
                            "vmul.f32 q4,  q4, q10\n"
                            "vld1.f32 {d20-d21}, [%[out_0]]!\n"
                            "vst1.f32 {d8-d9}, [%[out_24]]!\n"
                            "vmul.f32 q5,  q5, q11\n"
                            "vld1.f32 {d22-d23}, [%[out_0]]\n"
                            "vst1.f32 {d10-d11}, [%[out_24]]\n"

                            "vadd.f32 q0,  q6, q13\n"
                            "vadd.f32 q1,  q7, q13\n"
                            "vadd.f32 q2,  q8, q13\n"
                            "vadd.f32 q3,  q9, q13\n"
                            "vadd.f32 q4, q10, q13\n"
                            "vadd.f32 q5, q11, q13\n"
                            "vmax.f32 q0,  q0, q15\n"
                            "vmax.f32 q1,  q1, q15\n"
                            "vmax.f32 q2,  q2, q15\n"
                            "vmax.f32 q3,  q3, q15\n"
                            "vmax.f32 q4,  q4, q15\n"
                            "vmax.f32 q5,  q5, q15\n"
                            "vmin.f32 q0,  q0, q14\n"
                            "vmin.f32 q1,  q1, q14\n"
                            "vmin.f32 q2,  q2, q14\n"
                            "vmin.f32 q3,  q3, q14\n"
                            "vmin.f32 q4,  q4, q14\n"
                            "vmin.f32 q5,  q5, q14\n"
                            "vmul.f32 q0,  q0, q12\n"
                            "vmul.f32 q1,  q1, q12\n"
                            "vmul.f32 q2,  q2, q12\n"
                            "vmul.f32 q3,  q3, q12\n"
                            "vmul.f32 q4,  q4, q12\n"
                            "vmul.f32 q5,  q5, q12\n"
                            "vmul.f32 q0,  q0, q6\n"
                            "vst1.f32 {d0-d1}, [%[out]]!\n"
                            "vmul.f32 q1,  q1, q7\n"
                            "vst1.f32 {d2-d3}, [%[out]]!\n"
                            "vmul.f32 q2,  q2, q8\n"
                            "vst1.f32 {d4-d5}, [%[out]]!\n"
                            "vmul.f32 q3,  q3, q9\n"
                            "vst1.f32 {d6-d7}, [%[out]]!\n"
                            "vmul.f32 q4,  q4, q10\n"
                            "vst1.f32 {d8-d9}, [%[out]]!\n"
                            "vmul.f32 q5,  q5, q11\n"
                            "vst1.f32 {d10-d11}, [%[out]]\n"
                            : [out] "+r"(out_ptr),
                              [out_0] "+r"(out_0),
                              [out_24] "+r"(out_24)
                            :
                            : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                              "q11", "q12", "q13", "q14", "q15"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }
                if (depthwiseActivationDesc.mode != ACTIVATION_H_SWISH) {
                    __asm__ __volatile__(
                        "vstm %[out]!, {d0-d15}\n"
                        "vstm %[out], {d16-d23}\n"
                        : [out] "+r"(out_ptr)
                        :
                        : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                          "q11", "q12", "q13", "q14", "q15"
                    );
                }
            }

            U32 ohow_s = (ohow / 6) * 6;
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
                    "vld1.f32 {d0-d1}, [%[b]]!\n"
                    "vld1.f32 {d2-d3}, [%[b]]\n"
                    "vmov.f32 q2, q0\n"
                    "vmov.f32 q3, q1\n"
                    "vmov.f32 q4, q0\n"
                    "vmov.f32 q5, q1\n"
                    "vmov.f32 q6, q0\n"
                    "vmov.f32 q7, q1\n"
                    :[b]"+r"(b)
                    :
                    :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
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
                            "vld1.f32 {d28-d31}, [%[f0]]\n"
                            "vld1.f32 {d16-d19}, [%[in0]]\n"
                            "vld1.f32 {d20-d23}, [%[in1]]\n"
                            "vld1.f32 {d24-d27}, [%[in2]]\n"

                            "vmla.f32 q0,  q8, q14\n"
                            "vmla.f32 q1,  q9, q15\n"
                            "vld1.f32 {d16-d19}, [%[in3]]\n"
                            "vmla.f32 q2, q10, q14\n"
                            "vmla.f32 q3, q11, q15\n"
                            "vmla.f32 q4, q12, q14\n"
                            "vmla.f32 q5, q13, q15\n"
                            "vmla.f32 q6,  q8, q14\n"
                            "vmla.f32 q7,  q9, q15\n"
                            :
                            :[in0]"r"(in_0),
                             [in1]"r"(in_1),
                             [in2]"r"(in_2),
                             [in3]"r"(in_3),
                             [f0]"r"(f_0)
                            :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                             "q11", "q12", "q13", "q14", "q15"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationDesc.mode){
                    case ACTIVATION_NULL:
                         break;
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "veor q15, q15, q15\n" // zero
                            "vmax.f32 q0, q0, q15\n"
                            "vmax.f32 q1, q1, q15\n"
                            "vmax.f32 q2, q2, q15\n"
                            "vmax.f32 q3, q3, q15\n"
                            "vmax.f32 q4, q4, q15\n"
                            "vmax.f32 q5, q5, q15\n"
                            "vmax.f32 q6, q6, q15\n"
                            "vmax.f32 q7, q7, q15\n"
                            :
                            :
                            :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q15"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "veor q15, q15, q15\n" // zero
                            "vmov.f32 q14, #6.0\n"  // six
                            "vmax.f32 q0, q0, q15\n"
                            "vmax.f32 q1, q1, q15\n"
                            "vmax.f32 q2, q2, q15\n"
                            "vmax.f32 q3, q3, q15\n"
                            "vmax.f32 q4, q4, q15\n"
                            "vmax.f32 q5, q5, q15\n"
                            "vmax.f32 q6, q6, q15\n"
                            "vmax.f32 q7, q7, q15\n"
                                       
                            "vmin.f32 q0, q0, q14\n"
                            "vmin.f32 q1, q1, q14\n"
                            "vmin.f32 q2, q2, q14\n"
                            "vmin.f32 q3, q3, q14\n"
                            "vmin.f32 q4, q4, q14\n"
                            "vmin.f32 q5, q5, q14\n"
                            "vmin.f32 q6, q6, q14\n"
                            "vmin.f32 q7, q7, q14\n"
                            :
                            :
                            :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q14", "q15"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        __asm__ __volatile__(
                            "vmov.f32 q13, #3.0\n"  // three
                            "vmov.f32 q14, #6.0\n"  // six
                            "veor q15, q15, q15\n" // zero
                            "vadd.f32  q8, q0, q13\n"
                            "vadd.f32  q9, q1, q13\n"
                            "vadd.f32 q10, q2, q13\n"
                            "vadd.f32 q11, q3, q13\n"
                            "vmax.f32  q8,  q8, q15\n"
                            "vmax.f32  q9,  q9, q15\n"
                            "vmax.f32 q10, q10, q15\n"
                            "vmax.f32 q11, q11, q15\n"
                            "vmin.f32  q8,  q8, q14\n"
                            "vmin.f32  q9,  q9, q14\n"
                            "vmin.f32 q10, q10, q14\n"
                            "vmin.f32 q11, q11, q14\n"
                            "vrecpe.f32 q12, q14\n"
                            "vrecps.f32 q14, q14, q12\n"
                            "vmul.f32 q12, q14, q12\n"
                            "vmul.f32  q8,  q8, q12\n"
                            "vmul.f32  q9,  q9, q12\n"
                            "vmul.f32 q10, q10, q12\n"
                            "vmul.f32 q11, q11, q12\n"
                            "vmov.f32 q14, #6.0\n"  // six
                            "vmul.f32 q0, q0,  q8\n"
                            "vmul.f32 q1, q1,  q9\n"
                            "vmul.f32 q2, q2, q10\n"
                            "vmul.f32 q3, q3, q11\n"

                            "vadd.f32  q8, q4, q13\n"
                            "vadd.f32  q9, q5, q13\n"
                            "vadd.f32 q10, q6, q13\n"
                            "vadd.f32 q11, q7, q13\n"
                            "vmax.f32  q8,  q8, q15\n"
                            "vmax.f32  q9,  q9, q15\n"
                            "vmax.f32 q10, q10, q15\n"
                            "vmax.f32 q11, q11, q15\n"
                            "vmin.f32  q8,  q8, q14\n"
                            "vmin.f32  q9,  q9, q14\n"
                            "vmin.f32 q10, q10, q14\n"
                            "vmin.f32 q11, q11, q14\n"
                            "vrecpe.f32 q12, q14\n"
                            "vrecps.f32 q14, q14, q12\n"
                            "vmul.f32 q12, q14, q12\n"
                            "vmul.f32  q8,  q8, q12\n"
                            "vmul.f32  q9,  q9, q12\n"
                            "vmul.f32 q10, q10, q12\n"
                            "vmul.f32 q11, q11, q12\n"
                            "vmul.f32 q4, q4,  q8\n"
                            "vmul.f32 q5, q5,  q9\n"
                            "vmul.f32 q6, q6, q10\n"
                            "vmul.f32 q7, q7, q11\n"
                            :
                            :
                            :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                             "q11", "q12", "q13", "q14", "q15"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }

                __asm__ __volatile__(
                    "vstm %[out], {d0-d15}\n"
                    :[out]"+r"(out_ptr)
                    :
                    :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                );

                ohow_s += 4;
            }

            for (I32 hw = ohow_s; hw < ohow; hw++) {
                U32 in_h_0 = hw/ow*strideH;
                U32 in_w_0 = hw%ow*strideW;
                F32 *out_ptr = outArray + ((n * ic + c) * ohow + hw) * 8;

                __asm__ __volatile__(
                    "vld1.f32 {d0-d3}, [%[b]]\n"
                    :
                    :[b]"r"(b)
                    :"memory", "cc", "q0", "q1"
                );

                 for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        const F32 *f_0 = f + fh_idx*fw*8 + fw_idx*8;
                        F32 *in_idx = in_pad + fh_idx*dilateH*iw_pad*8 + fw_idx*dilateW*8;
                        F32 *in_0 = in_idx + in_h_0*iw_pad*8 + in_w_0*8;

                        __asm__ __volatile__(
                            "vld1.f32 {d28-d31}, [%[f0]]\n"
                            "vld1.f32 {d24-d27}, [%[in0]]\n"

                            "vmla.f32 q0, q12, q14\n"
                            "vmla.f32 q1, q13, q15\n"
                            :
                            :[in0]"r"(in_0),
                             [f0]"r"(f_0)
                            :"memory", "cc", "q0", "q1", "q12", "q13", "q14", "q15"
                        );
                    }
                }

                // activation
                switch (depthwiseActivationDesc.mode){
                    case ACTIVATION_NULL:
                         break;
                    case ACTIVATION_RELU:{
                        __asm__ __volatile__(
                            "veor q15, q15, q15\n" // zero
                            "vmax.f32 q0, q0, q15\n"
                            "vmax.f32 q1, q1, q15\n"
                            :
                            :
                            :"memory", "cc", "q0", "q1", "q15"
                        );
                        break;
                    }
                    case ACTIVATION_RELU6:{
                        __asm__ __volatile__(
                            "veor q15, q15, q15\n" // zero
                            "vmov.f32 q14, #6.0\n"  // six
                            "vmax.f32 q0, q0, q15\n"
                            "vmax.f32 q1, q1, q15\n"

                            "vmin.f32 q0, q0, q14\n"
                            "vmin.f32 q1, q1, q14\n"
                            :
                            :
                            :"memory", "cc", "q0", "q1", "q14", "q15"
                        );
                        break;
                    }
                    case ACTIVATION_H_SWISH:{
                        __asm__ __volatile__(
                            "vmov.f32 q13, #3.0\n"  // three
                            "vmov.f32 q14, #6.0\n"  // six
                            "veor q15, q15, q15\n" // zero
                            "vadd.f32 q11, q0, q13\n"
                            "vadd.f32 q12, q1, q13\n"

                            "vmax.f32 q11, q11, q15\n"
                            "vmax.f32 q12, q12, q15\n"

                            "vmin.f32 q11, q11, q14\n"
                            "vmin.f32 q12, q12, q14\n"

                            "vrecpe.f32 q13, q14\n"
                            "vrecps.f32 q14, q14, q13\n"
                            "vmul.f32 q14, q14, q13\n"
                            "vmul.f32 q11, q11, q14\n"
                            "vmul.f32 q12, q12, q14\n"

                            "vmul.f32 q0, q0, q11\n"
                            "vmul.f32 q1, q1, q12\n"
                            :
                            :
                            :"memory", "cc", "q0", "q1", "q11", "q12", "q13", "q14", "q15"
                        );
                        break;
                    }
                    default:
                        return NOT_SUPPORTED;
                }
                __asm__ __volatile__(
                    "vst1.f32 {d0-d3}, [%[out]]\n"
                    :[out]"+r"(out_ptr)
                    :
                    :"memory", "cc", "q0", "q1"
                );
            }
        }
    }
    return SUCCESS;
}
#endif
