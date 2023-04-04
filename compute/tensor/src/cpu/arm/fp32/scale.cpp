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
#include "cpu/arm/fp32/tensor_computing_fp32.h"

EE scale_nchwc8_fp32(
    F32 *input, F32 *alpha, F32 *beta, I32 in, I32 ic, I32 elements_per_channel, F32 *output)
{
    float32x4_t one = vdupq_n_f32(float32_t(1.));
    float32x4_t zero = vdupq_n_f32(float32_t(0.));
    ic /= 8;
#ifdef _USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
    for (int j = 0; j < in * ic; j++) {
        int n = j / ic;
        int c = j % ic;
        int c8 = c * 8;
        int index = j * elements_per_channel * 8;
        float32x4_t alpha_vec0 = (alpha == nullptr) ? one : vld1q_f32(alpha + c8);
        float32x4_t alpha_vec1 = (alpha == nullptr) ? one : vld1q_f32(alpha + c8 + 4);
        float32x4_t beta_vec0 = (beta == nullptr) ? zero : vld1q_f32(beta + c8);
        float32x4_t beta_vec1 = (beta == nullptr) ? zero : vld1q_f32(beta + c8 + 4);
        for (I32 i = 0; i < elements_per_channel; i++) {
            float32x4_t in_vec = vld1q_f32(input + index);
            float32x4_t out_vec = vfmaq_f32(beta_vec0, alpha_vec0, in_vec);
            vst1q_f32(output + index, out_vec);

            in_vec = vld1q_f32(input + index + 4);
            out_vec = vfmaq_f32(beta_vec1, alpha_vec1, in_vec);
            vst1q_f32(output + index + 4, out_vec);
            index += 8;
        }
    }
    return SUCCESS;
}

template <bool icoc_equal>
EE scale_nchw_fp32(
    F32 *input, F32 *alpha, F32 *beta, I32 in, I32 ic, I32 elements_per_channel, F32 *output)
{
    float32x4_t one = vdupq_n_f32(1.);
    float32x4_t zero = vdupq_n_f32(0.);
#ifdef _USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
    for (int j = 0; j < in * ic; j++) {
        int n = j / ic;
        int c = j % ic;
        U32 dst = j * elements_per_channel, src = 0;
        float32x4_t alpha_vec = (alpha == nullptr) ? one : vdupq_n_f32(alpha[c]);
        float32x4_t beta_vec = (beta == nullptr) ? zero : vdupq_n_f32(beta[c]);
        I32 i = 0;
        for (; i < elements_per_channel - 3; i += 4) {
            if (icoc_equal) {
                src = dst;
            } else {
                src = n * elements_per_channel + i;
            }
            float32x4_t in_vec = vld1q_f32(input + src);
            float32x4_t out_vec = vfmaq_f32(beta_vec, alpha_vec, in_vec);
            vst1q_f32(output + dst, out_vec);
            dst += 4;
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
EE scale_nhwc_fp32(
    F32 *input, F32 *alpha, F32 *beta, I32 in, I32 ic, I32 elements_per_channel, F32 *output)
{
    float32x4_t one = vdupq_n_f32(1.);
    float32x4_t zero = vdupq_n_f32(0.);
    float32x4_t in_vec;
    float in_s;
    for (I32 n = 0, src = 0, dst = 0; n < in; n++) {
        for (I32 i = 0; i < elements_per_channel; i++, src++) {
            I32 c = 0;
            for (; c < ic - 3; c += 4) {
                float32x4_t alpha_vec = (alpha == nullptr) ? one : vld1q_f32(alpha + c);
                float32x4_t beta_vec = (beta == nullptr) ? zero : vld1q_f32(beta + c);
                if (icoc_equal) {
                    in_vec = vld1q_f32(input + dst);
                } else {
                    in_vec = vdupq_n_f32(input[src]);
                }
                float32x4_t out_vec = vfmaq_f32(beta_vec, alpha_vec, in_vec);
                vst1q_f32(output + dst, out_vec);
                dst += 4;
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

EE scale_fp32(F32 *input,
    I32 axis,
    I32 nDims,
    F32 *alpha,
    F32 *beta,
    I32 on,
    I32 oc,
    I32 elements_per_channel,
    I32 ic,
    F32 *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;
    // If oc is 1, it means that weights/vectors have only one param, so we need use the calculation logic of nchw.
    if (axis < (nDims - 1) || oc == 1) {
        if (ic == oc) {
            ret = scale_nchw_fp32<true>(input, alpha, beta, on, oc, elements_per_channel, output);
        } else {
            ret = scale_nchw_fp32<false>(input, alpha, beta, on, oc, elements_per_channel, output);
        }
    } else if (axis == nDims - 1) {
        if (ic == oc) {
            ret = scale_nhwc_fp32<true>(input, alpha, beta, on, oc, elements_per_channel, output);
        } else {
            ret = scale_nhwc_fp32<false>(input, alpha, beta, on, oc, elements_per_channel, output);
        }
    } else if (axis == nDims) {
        ret = scale_nchwc8_fp32(input, alpha, beta, on, oc, elements_per_channel, output);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return ret;
}
