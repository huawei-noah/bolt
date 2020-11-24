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
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#include <string.h>
#ifdef _USE_OPENMP
#include <omp.h>
#endif

EE convolution_gemm_V7(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc)
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
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;

    if (fdf != DF_NHWCN8) {
        CHECK_STATUS(NOT_MATCH);
    }

    oc /= 8;
    ic /= 8;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    I32 ohow = oh * ow;
    U32 ihiw = ih_pad * iw_pad;
    F32 *inArray_pad;
    EE ret = SUCCESS;
    for (U32 n = 0; n < in; n++) {
        if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            inArray_pad = inArray + n * ic * ih * iw * 8;
        } else {
            // copy input into a input with padding
            inArray_pad = (F32 *)tmp;
            F32 *inArray_pad_mov = inArray_pad;
            F32 *inArray_mov = inArray + n * ic * ih * iw * 8;
            for (U32 c = 0; c < ic; c++) {
                for (U32 h = 0; h < paddingT; h++) {
                    memset(inArray_pad_mov, 0, iw_pad * 8 * bytesOf(idt));
                    inArray_pad_mov += iw_pad * 8;
                }
                for (U32 h = paddingT; h < ih_pad - paddingB; h++) {
                    memset(inArray_pad_mov, 0, paddingL * 8 * bytesOf(idt));
                    inArray_pad_mov += paddingL * 8;
                    memcpy(inArray_pad_mov, inArray_mov, iw * 8 * bytesOf(idt));
                    inArray_pad_mov += iw * 8;
                    inArray_mov += iw * 8;
                    memset(inArray_pad_mov, 0, paddingR * 8 * bytesOf(idt));
                    inArray_pad_mov += paddingR * 8;
                }
                for (U32 h = ih_pad - paddingB; h < ih_pad; h++) {
                    memset(inArray_pad_mov, 0, iw_pad * 8 * bytesOf(idt));
                    inArray_pad_mov += iw_pad * 8;
                }
            }
        }

        // ohow / 6
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 hw = 0; hw < ohow - 5; hw += 6) {
            const F32 *b0 = biasArray;
            const F32 *b1 = biasArray + 4;
#ifdef _USE_OPENMP
            // For NDK on ARMv7, OpenMP loop cannot reference more than 14 outside variables
            U32 paddingT = convParamSpec.padding_top;
            U32 paddingB = convParamSpec.padding_bottom;
            U32 paddingL = convParamSpec.padding_left;
            U32 paddingR = convParamSpec.padding_right;
            U32 fh = filterDesc.dims[1];
            U32 fw = filterDesc.dims[0];
            U32 thread_private_buffer_offset = 6 * fh * fw * ic * 8 * omp_get_thread_num();
#else
            U32 thread_private_buffer_offset = 0;
#endif
            F32 *in_pack = ((F32 *)tmp) + ic * ihiw * 8 + thread_private_buffer_offset;
            // pack input
            // NCHWc8 => NHWChw6 + im2col
            U32 in_h[6] = {0};
            U32 in_w[6] = {0};
            for (U32 i = 0; i < 6; i++) {
                in_h[i] = ((hw + i) / ow) * convParamSpec.stride_h;
                in_w[i] = ((hw + i) % ow) * convParamSpec.stride_w;
            }

            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        F32 *in_hw6c8 = inArray_pad + c * ihiw * 8 +
                            fh_idx * convParamSpec.dilatedRate_h * iw_pad * 8 +
                            fw_idx * convParamSpec.dilatedRate_w * 8;
                        F32 *in_0 = in_hw6c8 + in_h[0] * iw_pad * 8 + in_w[0] * 8;
                        F32 *in_1 = in_hw6c8 + in_h[1] * iw_pad * 8 + in_w[1] * 8;
                        F32 *in_2 = in_hw6c8 + in_h[2] * iw_pad * 8 + in_w[2] * 8;
                        F32 *in_3 = in_hw6c8 + in_h[3] * iw_pad * 8 + in_w[3] * 8;
                        F32 *in_4 = in_hw6c8 + in_h[4] * iw_pad * 8 + in_w[4] * 8;
                        F32 *in_5 = in_hw6c8 + in_h[5] * iw_pad * 8 + in_w[5] * 8;

                        // NHWChw6
                        F32 *in_pack_c8hw6 =
                            in_pack + fh_idx * fw * ic * 6 * 8 + fw_idx * ic * 6 * 8 + c * 6 * 8;

                        __asm__ __volatile__("vld1.f32 {d0-d3}, [%[in_0]]\n"
                                             "vld1.f32 {d4-d7}, [%[in_1]]\n"
                                             "vld1.f32 {d8-d11}, [%[in_2]]\n"
                                             "vld1.f32 {d12-d15}, [%[in_3]]\n"
                                             "vld1.f32 {d16-d19}, [%[in_4]]\n"
                                             "vld1.f32 {d20-d23}, [%[in_5]]\n"

                                             "vzip.32 q0, q2\n"
                                             "vzip.32 q4, q6\n"
                                             "vzip.32 q8, q10\n"

                                             "vst1.f32 {d0}, [%[pack]]!\n"
                                             "vst1.f32 {d8}, [%[pack]]!\n"
                                             "vst1.f32 {d16}, [%[pack]]!\n"
                                             "vst1.f32 {d1}, [%[pack]]!\n"
                                             "vst1.f32 {d9}, [%[pack]]!\n"
                                             "vst1.f32 {d17}, [%[pack]]!\n"
                                             "vst1.f32 {d4}, [%[pack]]!\n"
                                             "vst1.f32 {d12}, [%[pack]]!\n"
                                             "vst1.f32 {d20}, [%[pack]]!\n"
                                             "vst1.f32 {d5}, [%[pack]]!\n"
                                             "vst1.f32 {d13}, [%[pack]]!\n"
                                             "vst1.f32 {d21}, [%[pack]]!\n"

                                             "vzip.32 q1, q3\n"
                                             "vzip.32 q5, q7\n"
                                             "vzip.32 q9, q11\n"

                                             "vst1.f32 {d2}, [%[pack]]!\n"
                                             "vst1.f32 {d10}, [%[pack]]!\n"
                                             "vst1.f32 {d18}, [%[pack]]!\n"
                                             "vst1.f32 {d3}, [%[pack]]!\n"
                                             "vst1.f32 {d11}, [%[pack]]!\n"
                                             "vst1.f32 {d19}, [%[pack]]!\n"
                                             "vst1.f32 {d6}, [%[pack]]!\n"
                                             "vst1.f32 {d14}, [%[pack]]!\n"
                                             "vst1.f32 {d22}, [%[pack]]!\n"
                                             "vst1.f32 {d7}, [%[pack]]!\n"
                                             "vst1.f32 {d15}, [%[pack]]!\n"
                                             "vst1.f32 {d23}, [%[pack]]!\n"
                                             : [pack] "+r"(in_pack_c8hw6), [in_0] "+r"(in_0),
                                             [in_1] "+r"(in_1), [in_2] "+r"(in_2),
                                             [in_3] "+r"(in_3), [in_4] "+r"(in_4), [in_5] "+r"(in_5)
                                             :
                                             : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5",
                                             "q6", "q7", "q8", "q9", "q10", "q11");
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * fh * fw * ic * 8;
                F32 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                // bias
                const F32 *b_o0 = b0 + o * 8;
                const F32 *b_o1 = b1 + o * 8;
                __asm__ __volatile__(
                    "vld1.f32 {d8-d9}, [%[b_0]]\n"
                    "vld1.f32 {d10-d11}, [%[b_1]]\n"
                    "vld1.f32  {d0-d3}, [%[in_0]]!\n"
                    "vld1.f32  {d4-d7}, [%[f_0]]!\n"

                    "vmov.f32  q6, q4\n"
                    "vmov.f32  q8, q4\n"
                    "vmov.f32  q10, q4\n"
                    "vmov.f32  q12, q4\n"
                    "vmov.f32  q14, q4\n"

                    "mov  r2, %[ic]\n"

                    "vmov.f32  q7, q5\n"
                    "vmov.f32  q9, q5\n"
                    "vmov.f32  q11, q5\n"
                    "vmov.f32  q13, q5\n"
                    "vmov.f32  q15, q5\n"

                    "0:\n"
                    "vmla.f32  q4, q2, d0[0]\n"
                    "vmla.f32  q6, q2, d0[1]\n"
                    "vmla.f32  q8, q2, d1[0]\n"
                    "vmla.f32  q10, q2, d1[1]\n"
                    "vmla.f32  q12, q2, d2[0]\n"
                    "vmla.f32  q14, q2, d2[1]\n"

                    "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                    "vmla.f32  q5, q3, d0[0]\n"
                    "vmla.f32  q7, q3, d0[1]\n"
                    "vmla.f32  q9, q3, d1[0]\n"
                    "vmla.f32  q11, q3, d1[1]\n"
                    "vld1.f32  {d0-d1}, [%[in_0]]!\n"
                    "vmla.f32  q13, q3, d2[0]\n"
                    "vmla.f32  q15, q3, d2[1]\n"

                    "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                    "subs r2, r2, #4\n"

                    "vmla.f32  q4, q2, d3[0]\n"
                    "vmla.f32  q6, q2, d3[1]\n"
                    "vmla.f32  q8, q2, d0[0]\n"
                    "vmla.f32  q10, q2, d0[1]\n"
                    "vmla.f32  q12, q2, d1[0]\n"
                    "vmla.f32  q14, q2, d1[1]\n"

                    "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                    "vmla.f32  q5, q3, d3[0]\n"
                    "vmla.f32  q7, q3, d3[1]\n"
                    "vld1.f32  {d2-d3}, [%[in_0]]!\n"
                    "vmla.f32  q9, q3, d0[0]\n"
                    "vmla.f32  q11, q3, d0[1]\n"
                    "vmla.f32  q13, q3, d1[0]\n"
                    "vmla.f32  q15, q3, d1[1]\n"

                    "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                    "vld1.f32  {d0-d1}, [%[in_0]]!\n"

                    "vmla.f32  q4, q2, d2[0]\n"
                    "vmla.f32  q6, q2, d2[1]\n"
                    "vmla.f32  q8, q2, d3[0]\n"
                    "vmla.f32  q10, q2, d3[1]\n"
                    "vmla.f32  q12, q2, d0[0]\n"
                    "vmla.f32  q14, q2, d0[1]\n"

                    "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                    "vmla.f32  q5, q3, d2[0]\n"
                    "vmla.f32  q7, q3, d2[1]\n"
                    "vmla.f32  q9, q3, d3[0]\n"
                    "vmla.f32  q11, q3, d3[1]\n"
                    "vld1.f32  {d2-d3}, [%[in_0]]!\n"
                    "vmla.f32  q13, q3, d0[0]\n"
                    "vmla.f32  q15, q3, d0[1]\n"

                    "vld1.f32  {d6-d7}, [%[f_0]]!\n"

                    "vmla.f32  q4, q2, d1[0]\n"
                    "vmla.f32  q6, q2, d1[1]\n"
                    "vmla.f32  q8, q2, d2[0]\n"
                    "vmla.f32  q10, q2, d2[1]\n"
                    "vmla.f32  q12, q2, d3[0]\n"
                    "vmla.f32  q14, q2, d3[1]\n"

                    "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                    "vmla.f32  q5, q3, d1[0]\n"
                    "vmla.f32  q7, q3, d1[1]\n"
                    "vld1.f32  {d0-d1}, [%[in_0]]!\n"
                    "vmla.f32  q9, q3, d2[0]\n"
                    "vmla.f32  q11, q3, d2[1]\n"
                    "vmla.f32  q13, q3, d3[0]\n"
                    "vmla.f32  q15, q3, d3[1]\n"

                    "vld1.f32  {d2-d3}, [%[in_0]]!\n"
                    "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                    "bne 0b\n"
                    : [out_0] "+r"(out_o0hw0), [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                    : [ic] "r"((I64)ic * 8 * fh * fw), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                    : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "q14", "q15", "r2");

                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("veor q1, q1, q1\n"  // zero
                                             "vmax.f32  q4,  q4, q1\n"
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
                                             :
                                             :
                                             : "memory", "cc", "q1", "q4", "q5", "q6", "q7", "q8",
                                             "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("veor q1, q1, q1\n"    // zero
                                             "vmov.f32 q2, #6.0\n"  // six
                                             "vmax.f32  q4,  q4, q1\n"
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
                                             "vmin.f32  q4,  q4, q2\n"
                                             "vmin.f32  q5,  q5, q2\n"
                                             "vmin.f32  q6,  q6, q2\n"
                                             "vmin.f32  q7,  q7, q2\n"
                                             "vmin.f32  q8,  q8, q2\n"
                                             "vmin.f32  q9,  q9, q2\n"
                                             "vmin.f32 q10, q10, q2\n"
                                             "vmin.f32 q11, q11, q2\n"
                                             "vmin.f32 q12, q12, q2\n"
                                             "vmin.f32 q13, q13, q2\n"
                                             "vmin.f32 q14, q14, q2\n"
                                             "vmin.f32 q15, q15, q2\n"
                                             :
                                             :
                                             : "memory", "cc", "q1", "q2", "q4", "q5", "q6", "q7",
                                             "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                        break;
                    }
                    default: {
                        CHECK_STATUS(NOT_SUPPORTED);
                    }
                }

                __asm__ __volatile__("vst1.f32  {q4}, [%[out_0]]!\n"
                                     "vst1.f32  {q5}, [%[out_0]]!\n"
                                     "vst1.f32  {q6}, [%[out_0]]!\n"
                                     "vst1.f32  {q7}, [%[out_0]]!\n"
                                     "vst1.f32  {q8}, [%[out_0]]!\n"
                                     "vst1.f32  {q9}, [%[out_0]]!\n"
                                     "vst1.f32 {q10}, [%[out_0]]!\n"
                                     "vst1.f32 {q11}, [%[out_0]]!\n"
                                     "vst1.f32 {q12}, [%[out_0]]!\n"
                                     "vst1.f32 {q13}, [%[out_0]]!\n"
                                     "vst1.f32 {q14}, [%[out_0]]!\n"
                                     "vst1.f32 {q15}, [%[out_0]]!\n"
                                     : [out_0] "+r"(out_o0hw0)
                                     :
                                     : "memory", "cc", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                                     "q11", "q12", "q13", "q14", "q15");
            }
        }

        U32 ohow_s = (ohow / 6) * 6;
        U32 ohow_tail = ohow - ohow_s;

        if (ohow_tail >= 4) {
            I32 hw = ohow_s;
            const F32 *b0 = biasArray;
            const F32 *b1 = biasArray + 4;
            F32 *in_pack = ((F32 *)tmp) + ic * ih_pad * iw_pad * 8;
            // pack input
            // NCHWc8 => NHWChw4 + im2col
            U32 in_h[4] = {0};
            U32 in_w[4] = {0};

            for (U32 i = 0; i < 4; i++) {
                in_h[i] = ((hw + i) / ow) * convParamSpec.stride_h;
                in_w[i] = ((hw + i) % ow) * convParamSpec.stride_w;
            }
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        F32 *in_hw4c8 = inArray_pad + c * ihiw * 8 +
                            fh_idx * convParamSpec.dilatedRate_h * iw_pad * 8 +
                            fw_idx * convParamSpec.dilatedRate_w * 8;
                        F32 *in_0 = in_hw4c8 + in_h[0] * iw_pad * 8 + in_w[0] * 8;
                        F32 *in_1 = in_hw4c8 + in_h[1] * iw_pad * 8 + in_w[1] * 8;
                        F32 *in_2 = in_hw4c8 + in_h[2] * iw_pad * 8 + in_w[2] * 8;
                        F32 *in_3 = in_hw4c8 + in_h[3] * iw_pad * 8 + in_w[3] * 8;
                        F32 *in_pack_c8hw4 =
                            in_pack + fh_idx * fw * ic * 8 * 4 + fw_idx * ic * 8 * 4 + c * 8 * 4;

                        __asm__ __volatile__(
                            "vld1.f32 {d0-d3}, [%[in_0]]\n"
                            "vld1.f32 {d4-d7}, [%[in_1]]\n"
                            "vld1.f32 {d8-d11}, [%[in_2]]\n"
                            "vld1.f32 {d12-d15}, [%[in_3]]\n"

                            "vzip.32 q0, q4\n"
                            "vzip.32 q2, q6\n"

                            "vzip.32 q0, q2\n"
                            "vzip.32 q4, q6\n"

                            "vst1.f32 {q0}, [%[pack]]!\n"
                            "vst1.f32 {q2}, [%[pack]]!\n"
                            "vst1.f32 {q4}, [%[pack]]!\n"
                            "vst1.f32 {q6}, [%[pack]]!\n"

                            "vzip.32 q1, q5\n"
                            "vzip.32 q3, q7\n"

                            "vzip.32 q1, q3\n"
                            "vzip.32 q5, q7\n"

                            "vst1.f32 {q1}, [%[pack]]!\n"
                            "vst1.f32 {q3}, [%[pack]]!\n"
                            "vst1.f32 {q5}, [%[pack]]!\n"
                            "vst1.f32 {q7}, [%[pack]]!\n"
                            : [pack] "+r"(in_pack_c8hw4), [in_0] "+r"(in_0), [in_1] "+r"(in_1),
                            [in_2] "+r"(in_2), [in_3] "+r"(in_3)
                            :
                            : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * fh * fw * ic * 8;
                F32 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                // bias
                const F32 *b_o0 = b0;
                const F32 *b_o1 = b1;
                __asm__ __volatile__(
                    "vld1.f32 {d8-d9}, [%[b_0]]\n"
                    "vld1.f32 {d10-d11}, [%[b_1]]\n"
                    "vld1.f32  {d0-d1}, [%[in_0]]!\n"
                    "vld1.f32  {d4-d7}, [%[f_0]]!\n"

                    "vmov.f32  q6, q4\n"
                    "vmov.f32  q8, q4\n"
                    "vmov.f32  q10, q4\n"

                    "mov  r2, %[ic]\n"

                    "vmov.f32  q7, q5\n"
                    "vmov.f32  q9, q5\n"
                    "vmov.f32  q11, q5\n"

                    "0:\n"
                    "vmla.f32  q4, q2, d0[0]\n"
                    "vmla.f32  q6, q2, d0[1]\n"
                    "vmla.f32  q8, q2, d1[0]\n"
                    "vmla.f32  q10, q2, d1[1]\n"

                    "vld1.f32  {d2-d3}, [%[in_0]]!\n"
                    "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                    "vmla.f32  q5, q3, d0[0]\n"
                    "vmla.f32  q7, q3, d0[1]\n"
                    "vmla.f32  q9, q3, d1[0]\n"
                    "vmla.f32  q11, q3, d1[1]\n"

                    "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                    "subs r2, r2, #2\n"

                    "vmla.f32  q4, q2, d2[0]\n"
                    "vmla.f32  q6, q2, d2[1]\n"
                    "vmla.f32  q8, q2, d3[0]\n"
                    "vmla.f32  q10, q2, d3[1]\n"

                    "vld1.f32  {d0-d1}, [%[in_0]]!\n"
                    "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                    "vmla.f32  q5, q3, d2[0]\n"
                    "vmla.f32  q7, q3, d2[1]\n"
                    "vmla.f32  q9, q3, d3[0]\n"
                    "vmla.f32  q11, q3, d3[1]\n"

                    "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                    "bne 0b\n"
                    : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                    : [ic] "r"((I64)ic * 8 * fh * fw), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                    : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "r2");

                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("veor q1, q1, q1\n"  // zero
                                             "vmax.f32  q4,  q4, q1\n"
                                             "vmax.f32  q5,  q5, q1\n"
                                             "vmax.f32  q6,  q6, q1\n"
                                             "vmax.f32  q7,  q7, q1\n"
                                             "vmax.f32  q8,  q8, q1\n"
                                             "vmax.f32  q9,  q9, q1\n"
                                             "vmax.f32 q10, q10, q1\n"
                                             "vmax.f32 q11, q11, q1\n"
                                             :
                                             :
                                             : "memory", "cc", "q1", "q4", "q5", "q6", "q7", "q8",
                                             "q9", "q10", "q11");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("veor q1, q1, q1\n"    // zero
                                             "vmov.f32 q2, #6.0\n"  // six
                                             "vmax.f32  q4,  q4, q1\n"
                                             "vmax.f32  q5,  q5, q1\n"
                                             "vmax.f32  q6,  q6, q1\n"
                                             "vmax.f32  q7,  q7, q1\n"
                                             "vmax.f32  q8,  q8, q1\n"
                                             "vmax.f32  q9,  q9, q1\n"
                                             "vmax.f32 q10, q10, q1\n"
                                             "vmax.f32 q11, q11, q1\n"
                                             "vmin.f32  q4,  q4, q2\n"
                                             "vmin.f32  q5,  q5, q2\n"
                                             "vmin.f32  q6,  q6, q2\n"
                                             "vmin.f32  q7,  q7, q2\n"
                                             "vmin.f32  q8,  q8, q2\n"
                                             "vmin.f32  q9,  q9, q2\n"
                                             "vmin.f32 q10, q10, q2\n"
                                             "vmin.f32 q11, q11, q2\n"
                                             :
                                             :
                                             : "memory", "cc", "q1", "q2", "q4", "q5", "q6", "q7",
                                             "q8", "q9", "q10", "q11");
                        break;
                    }
                    default: {
                        CHECK_STATUS(NOT_SUPPORTED);
                    }
                }

                __asm__ __volatile__(
                    "vst1.f32  {q4}, [%[out_0]]!\n"
                    "vst1.f32  {q5}, [%[out_0]]!\n"
                    "vst1.f32  {q6}, [%[out_0]]!\n"
                    "vst1.f32  {q7}, [%[out_0]]!\n"
                    "vst1.f32  {q8}, [%[out_0]]!\n"
                    "vst1.f32  {q9}, [%[out_0]]!\n"
                    "vst1.f32 {q10}, [%[out_0]]!\n"
                    "vst1.f32 {q11}, [%[out_0]]!\n"
                    : [out_0] "+r"(out_o0hw0)
                    :
                    : "memory", "cc", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
                b0 += 8;
                b1 += 8;
            }
            ohow_s += 4;
            ohow_tail -= 4;
        }

        // I32 ohow_s = (ohow / 4) * 4;

        for (I32 hw = ohow_s; hw < ohow; hw++) {
            const F32 *b0 = biasArray;
            const F32 *b1 = biasArray + 4;
            F32 *in_pack = ((F32 *)tmp) + ic * ih_pad * iw_pad * 8;
            // pack input
            // NCHW => NCHWc8hw1 + im2col
            U32 in_h_0 = (hw / ow) * convParamSpec.stride_h;
            U32 in_w_0 = (hw % ow) * convParamSpec.stride_w;
            for (U32 c = 0; c < ic; c++) {
                for (U32 fh_idx = 0; fh_idx < fh; fh_idx++) {
                    for (U32 fw_idx = 0; fw_idx < fw; fw_idx++) {
                        F32 *in_hw1c8 = inArray_pad + c * ihiw * 8 +
                            fh_idx * convParamSpec.dilatedRate_h * iw_pad * 8 +
                            fw_idx * convParamSpec.dilatedRate_w * 8;
                        F32 *in_0 = in_hw1c8 + in_h_0 * iw_pad * 8 + in_w_0 * 8;
                        F32 *in_pack_c8hw1 =
                            in_pack + fh_idx * fw * ic * 8 + fw_idx * ic * 8 + c * 8;

                        memcpy(in_pack_c8hw1, in_0, 8 * bytesOf(idt));
                    }
                }
            }

            // compute
            for (I32 o = 0; o < I32(oc); o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * fh * fw * ic * 8;
                F32 *out_o0hw0 = outArray + n * oc * ohow * 8 + o * ohow * 8 + hw * 8;
                // bias
                const F32 *b_o0 = b0;
                const F32 *b_o1 = b1;
                __asm__ __volatile__(
                    "vld1.f32 {d8-d9}, [%[b_0]]\n"
                    "vld1.f32 {d10-d11}, [%[b_1]]\n"
                    "vld1.f32  {d0}, [%[in_0]]!\n"
                    "vld1.f32  {d4-d7}, [%[f_0]]!\n"
                    "mov  r2, %[ic]\n"
                    "0:\n"
                    "vmla.f32  q4, q2, d0[0]\n"

                    "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                    "vmla.f32  q5, q3, d0[0]\n"

                    "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                    "subs r2, r2, #2\n"

                    "vmla.f32  q4, q2, d0[1]\n"

                    "vld1.f32  {d4-d5}, [%[f_0]]!\n"

                    "vmla.f32  q5, q3, d0[1]\n"

                    "vld1.f32  {d0}, [%[in_0]]!\n"
                    "vld1.f32  {d6-d7}, [%[f_0]]!\n"
                    "bne 0b\n"
                    : [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                    : [ic] "r"((I64)ic * 8 * fh * fw), [b_0] "r"(b_o0), [b_1] "r"(b_o1)
                    : "memory", "cc", "q0", "q2", "q3", "q4", "q5", "r2");

                switch (activationDesc.mode) {
                    case ACTIVATION_NULL:
                        break;
                    case ACTIVATION_RELU: {
                        __asm__ __volatile__("veor q1, q1, q1\n"  // zero
                                             "vmax.f32  q4,  q4, q1\n"
                                             "vmax.f32  q5,  q5, q1\n"
                                             :
                                             :
                                             : "memory", "cc", "q1", "q4", "v5");
                        break;
                    }
                    case ACTIVATION_RELU6: {
                        __asm__ __volatile__("veor q1, q1, q1\n"    // zero
                                             "vmov.f32 q2, #6.0\n"  // six
                                             "vmax.f32  q4,  q4, q1\n"
                                             "vmax.f32  q5,  q5, q1\n"
                                             "vmin.f32  q4,  q4, q2\n"
                                             "vmin.f32  q5,  q5, q2\n"
                                             :
                                             :
                                             : "memory", "cc", "q1", "q2", "q4", "v5");
                        break;
                    }
                    default: {
                        CHECK_STATUS(NOT_SUPPORTED);
                    }
                }

                __asm__ __volatile__("vst1.f32  {q4}, [%[out_0]]!\n"
                                     "vst1.f32  {q5}, [%[out_0]]!\n"
                                     : [out_0] "+r"(out_o0hw0)
                                     :
                                     : "memory", "cc", "q4", "q5");
                b0 += 8;
                b1 += 8;
            }
        }
    }
    return ret;
}
#endif
