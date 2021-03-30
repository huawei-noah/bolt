// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp32/tensor_computing_fp32.h"
#include "cpu/arm/transform_functions.h"
#ifdef _USE_OPENMP
#include <omp.h>
#endif

EE convolution_gemm_icnchw_V7(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec p,
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
    U32 in, ic, it, ih, iw;
    U32 fn, fc, ft, fh, fw;
    U32 on, oc, ot, oh, ow;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        it = ft = ot = 1;
        p.dilatedRate_t = p.stride_t = 1;
        p.padding_before = p.padding_after = 0;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }
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
    U32 it_pad = it + p.padding_before + p.padding_after;
    U32 ih_pad = ih + p.padding_top + p.padding_bottom;
    U32 iw_pad = iw + p.padding_left + p.padding_right;
    I64 K = ic * ft * fh * fw;
    I32 ohow = ot * oh * ow;
    F32 *in_pack = ((F32 *)tmp) + ic * it_pad * ih_pad * iw_pad;
    EE ret = SUCCESS;
    U32 params[12] = {ic, it_pad, ih_pad, iw_pad, fc, ft, fh, fw, oc, ot, oh, ow};
    for (U32 n = 0; n < in; n++) {
        F32 *inArray_pad = convolution_input_padding_per_channel<F32, 1>(
            n, ic, it, ih, iw, p, inArray, (F32 *)tmp);
        // ohow / 6
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (I32 hw = 0; hw < ohow - 5; hw += 6) {
#ifdef _USE_OPENMP
            // For NDK on ARMv7, OpenMP loop cannot reference more than 14 outside variables
            F32 *thread_in_pack = in_pack + 6 * K * omp_get_thread_num();
#else
            F32 *thread_in_pack = in_pack;
#endif
            // pack input
            // NCHW => NHWChw6 + im2col
            convolution_nchw_input_pack<F32, 6>(params[0], params[1], params[2], params[3], p,
                params[5], params[6], params[7], params[9], params[10], params[11], inArray_pad, hw,
                thread_in_pack);

            // compute
            for (U32 o = 0; o < params[8]; o++) {
                F32 *in_hw0 = thread_in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * K;
                F32 *out_o0hw0 = outArray + ((n * params[8] + o) * ohow + hw) * 8;

                // bias
                const F32 *b_o0 = biasArray + o * 8;
                const F32 *b_o1 = b_o0 + 4;
                __asm__ __volatile__(
                    "vld1.f32 {d10-d11}, [%[b_0]]\n"
                    "vld1.f32 {d12-d13}, [%[b_1]]\n"
                    "mov r2, %[ic]\n"

                    "vld1.f32 {d2-d3}, [%[in_0]]!\n"  // in_hw0
                    "vmov.f32  q7, q5\n"
                    "vmov.f32  q9, q5\n"
                    "vmov.f32 q11, q5\n"
                    "vld1.f32 {d0-d1}, [%[f_0]]!\n"  // f_o0c0
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
                    "veor q1, q1, q1\n"  // zero
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
                    : [out_0] "+r"(out_o0hw0), [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                    : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1), [activation] "r"(activation)
                    : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "q14", "q15", "r2");
            }
        }

        U32 ohow_s = (ohow / 6) * 6;
        U32 ohow_tail = ohow - ohow_s;

        if (ohow_tail >= 4) {
            I32 hw = ohow_s;
            // pack input
            // NCHW => NHWChw4 + im2col
            convolution_nchw_input_pack<F32, 4>(
                ic, it_pad, ih_pad, iw_pad, p, ft, fh, fw, ot, oh, ow, inArray_pad, hw, in_pack);

            // compute
            for (U32 o = 0; o < oc; o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * K;
                F32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                // bias
                const F32 *b_o0 = biasArray + o * 8;
                const F32 *b_o1 = b_o0 + 4;
                __asm__ __volatile__(
                    "vld1.f32 {d10-d11}, [%[b_0]]\n"
                    "vld1.f32 {d12-d13}, [%[b_1]]\n"
                    "mov  r2, %[ic]\n"

                    "vld1.f32  {d2-d3}, [%[in_0]]!\n"  // in_hw0
                    "vmov.f32  q7, q5\n"
                    "vmov.f32  q9, q5\n"
                    "vmov.f32 q11, q5\n"
                    "vld1.f32 {d0-d1}, [%[f_0]]!\n"  // f_o0c0

                    "vmov.f32  q8, q6\n"
                    "vmov.f32 q10, q6\n"
                    "vmov.f32 q12, q6\n"
                    "0:\n"
                    "vld1.f32 {d6-d7}, [%[in_0]]!\n"
                    "vld1.f32 {d8-d9}, [%[f_0]]!\n"
                    "vmla.f32  q5, q0, d2[0]\n"
                    "vmla.f32  q7, q0, d2[1]\n"
                    "vmla.f32  q9, q0, d3[0]\n"
                    "vmla.f32 q11, q0, d3[1]\n"
                    "vld1.f32 {d0-d1}, [%[f_0]]!\n"

                    "vmla.f32  q6, q4, d2[0]\n"
                    "vmla.f32  q8, q4, d2[1]\n"
                    "subs r2, r2, #1\n"
                    "vmla.f32 q10, q4, d3[0]\n"
                    "vmla.f32 q12, q4, d3[1]\n"
                    "vmov.f32 q1, q3\n"
                    "bne 0b\n"

                    "cmp %[activation], #0\n"
                    "beq 1f\n"
                    "veor q1, q1, q1\n"  // zero
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
                    : [out_0] "+r"(out_o0hw0), [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                    : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1), [activation] "r"(activation)
                    : "memory", "cc", "q0", "q1", "q3", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
                    "q12", "q4", "r2");
            }
            ohow_s += 4;
            ohow_tail -= 4;
        }

        for (I32 hw = ohow_s; hw < ohow; hw++) {
            // pack input
            // NCHW => NCHWc8hw1 + im2col
            convolution_nchw_input_pack<F32, 1>(
                ic, it_pad, ih_pad, iw_pad, p, ft, fh, fw, ot, oh, ow, inArray_pad, hw, in_pack);

            // compute
            for (U32 o = 0; o < oc; o++) {
                F32 *in_hw0 = in_pack;
                const F32 *f_o0c0 = filterArray + o * 8 * K;
                F32 *out_o0hw0 = outArray + ((n * oc + o) * ohow + hw) * 8;
                // bias
                const F32 *b_o0 = biasArray + o * 8;
                const F32 *b_o1 = b_o0 + 4;
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
                    "veor q1, q1, q1\n"  // zero
                    "vmax.f32 q5, q5, q1\n"
                    "vmax.f32 q6, q6, q1\n"
                    "1:\n"
                    "vst1.f32 {d10-d11}, [%[out_0]]!\n"
                    "vst1.f32 {d12-d13}, [%[out_0]]\n"
                    : [out_0] "+r"(out_o0hw0), [in_0] "+r"(in_hw0), [f_0] "+r"(f_o0c0)
                    : [ic] "r"(K), [b_0] "r"(b_o0), [b_1] "r"(b_o1), [activation] "r"(activation)
                    : "memory", "cc", "q0", "q1", "q5", "q6", "q4", "r2");
            }
        }
    }
    return ret;
}
