// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <arm_neon.h>
#include "cpu/arm/fp16/tensor_computing_fp16.h"

EE scale_nchwc8_fp16(
    F16 *input, F16 *alpha, F16 *beta, I32 in, I32 ic, I32 elements_per_channel, F16 *output)
{
    float16x8_t one = vdupq_n_f16(1.);
    float16x8_t zero = vdupq_n_f16(0.);
    ic /= 8;
#ifdef _USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
    for (int j = 0; j < in * ic; j++) {
        int n = j / ic;
        int c = j % ic;
        int c8 = c * 8;
        int index = j * elements_per_channel * 8;
        float16x8_t alpha_vec = (alpha == nullptr) ? one : vld1q_f16(alpha + c8);
        float16x8_t beta_vec = (beta == nullptr) ? zero : vld1q_f16(beta + c8);
        for (I32 i = 0; i < elements_per_channel; i++) {
            float16x8_t in_vec = vld1q_f16(input + index);
            float16x8_t out_vec = vfmaq_f16(beta_vec, alpha_vec, in_vec);
            vst1q_f16(output + index, out_vec);
            index += 8;
        }
    }
    return SUCCESS;
}

template <bool icoc_equal>
EE scale_nchw_fp16(
    F16 *input, F16 *alpha, F16 *beta, I32 in, I32 ic, I32 elements_per_channel, F16 *output)
{
    float16x8_t one = vdupq_n_f16(1.);
    float16x8_t zero = vdupq_n_f16(0.);
#ifdef _USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
    for (int j = 0; j < in * ic; j++) {
        int n = j / ic;
        int c = j % ic;
        U32 dst = j * elements_per_channel, src = 0;
        float16x8_t alpha_vec = (alpha == nullptr) ? one : vdupq_n_f16(alpha[c]);
        float16x8_t beta_vec = (beta == nullptr) ? zero : vdupq_n_f16(beta[c]);
        I32 i = 0;
        for (; i < elements_per_channel - 7; i += 8) {
            if (icoc_equal) {
                src = dst;
            } else {
                src = n * elements_per_channel + i;
            }
            float16x8_t in_vec = vld1q_f16(input + src);
            float16x8_t out_vec = vfmaq_f16(beta_vec, alpha_vec, in_vec);
            vst1q_f16(output + dst, out_vec);
            dst += 8;
        }
        for (; i < elements_per_channel; i++) {
            if (icoc_equal) {
                src = dst;
            } else {
                src = n * elements_per_channel + i;
            }
            float alpha_s = (alpha == nullptr) ? 1 : alpha[c];
            float beta_s = (beta == nullptr) ? 0 : beta[c];
            output[dst] = alpha_s * input[src] + beta_s;
            dst++;
        }
    }
    return SUCCESS;
}

template <bool icoc_equal>
EE scale_nhwc_fp16(
    F16 *input, F16 *alpha, F16 *beta, I32 in, I32 ic, I32 elements_per_channel, F16 *output)
{
    float16x8_t one = vdupq_n_f16(1.);
    float16x8_t zero = vdupq_n_f16(0.);
    float16x8_t in_vec;
    float in_s;
    for (I32 n = 0, src = 0, dst = 0; n < in; n++) {
        for (I32 i = 0; i < elements_per_channel; i++, src++) {
            I32 c = 0;
            for (; c < ic - 7; c += 8) {
                float16x8_t alpha_vec = (alpha == nullptr) ? one : vld1q_f16(alpha + c);
                float16x8_t beta_vec = (beta == nullptr) ? zero : vld1q_f16(beta + c);
                if (icoc_equal) {
                    in_vec = vld1q_f16(input + dst);
                } else {
                    in_vec = vdupq_n_f16(input[src]);
                }
                float16x8_t out_vec = vfmaq_f16(beta_vec, alpha_vec, in_vec);
                vst1q_f16(output + dst, out_vec);
                dst += 8;
            }
            for (; c < ic; c++) {
                float alpha_s = (alpha == nullptr) ? 1 : alpha[c];
                float beta_s = (beta == nullptr) ? 0 : beta[c];
                if (icoc_equal) {
                    in_s = input[dst];
                } else {
                    in_s = input[src];
                }
                output[dst] = alpha_s * in_s + beta_s;
                dst++;
            }
        }
    }
    return SUCCESS;
}

EE scale_fp16(F16 *input,
    I32 axis,
    I32 nDims,
    F16 *alpha,
    F16 *beta,
    I32 on,
    I32 oc,
    I32 elements_per_channel,
    I32 ic,
    F16 *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;
    // If oc is 1, it means that weights/vectors have only one param, so we need use the calculation logic of nchw.
    if (axis < (nDims - 1) || oc == 1) {
        if (ic == oc) {
            ret = scale_nchw_fp16<true>(input, alpha, beta, on, oc, elements_per_channel, output);
        } else {
            ret = scale_nchw_fp16<false>(input, alpha, beta, on, oc, elements_per_channel, output);
        }
    } else if (axis == nDims - 1) {
        if (ic == oc) {
            ret = scale_nhwc_fp16<true>(input, alpha, beta, on, oc, elements_per_channel, output);
        } else {
            ret = scale_nhwc_fp16<false>(input, alpha, beta, on, oc, elements_per_channel, output);
        }
    } else if (axis == nDims) {
        ret = scale_nchwc8_fp16(input, alpha, beta, on, oc, elements_per_channel, output);
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}
