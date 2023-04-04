// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/int32/tensor_computing_int32.h"

static EE scale_nchwc8_int32(
    I32 *input, I32 *alpha, I32 *beta, I32 in, I32 ic, I32 elements_per_channel, I32 *output)
{
    __m256i in_vec, out_vec;
    __m256i one = _mm256_set1_epi32(1);
    __m256i zero = _mm256_set1_epi32(0);
    U32 index = 0;
    for (I32 n = 0; n < in; n++) {
        for (I32 c = 0; c < ic; c += 8) {
            __m256i alpha_vec = (alpha == nullptr) ? one : _mm256_loadu_si256((const __m256i *)(alpha + c));
            __m256i beta_vec = (beta == nullptr) ? zero : _mm256_loadu_si256((const __m256i *)(beta + c));
            for (I32 i = 0; i < elements_per_channel; i++) {
                in_vec = _mm256_loadu_si256((const __m256i *)(input + index));
                out_vec = _mm256_add_epi32(_mm256_mul_epi32(alpha_vec, in_vec), beta_vec);
                _mm256_storeu_si256((__m256i *)(output + index), out_vec);
                index += 8;
            }
        }
    }
    return SUCCESS;
}

template <bool icoc_equal>
static EE scale_nchw_int32(
    I32 *input, I32 *alpha, I32 *beta, I32 in, I32 ic, I32 elements_per_channel, I32 *output)
{
    __m256i one = _mm256_set1_epi32(1);
    __m256i zero = _mm256_set1_epi32(0);
    U32 dst = 0, src = 0;
    for (I32 n = 0; n < in; n++) {
        for (I32 c = 0; c < ic; c++) {
            __m256i alpha_vec = (alpha == nullptr) ? one : _mm256_set1_epi32(alpha[c]);
            __m256i beta_vec = (beta == nullptr) ? zero : _mm256_set1_epi32(beta[c]);
            I32 i = 0;
            for (; i < elements_per_channel - 7; i += 8) {
                if (icoc_equal) {
                    src = (n * ic + c) * elements_per_channel + i;
                } else {
                    src = n * elements_per_channel + i;
                }
                __m256i in_vec = _mm256_loadu_si256((const __m256i *)(input + src));
                if (alpha != nullptr) {
                    in_vec = _mm256_mul_epi32(alpha_vec, in_vec);
                }
                __m256i out_vec = _mm256_add_epi32(in_vec, beta_vec);
                _mm256_storeu_si256((__m256i *)(output + dst), out_vec);
                dst += 8;
            }
            for (; i < elements_per_channel; i++) {
                if (icoc_equal) {
                    src = (n * ic + c) * elements_per_channel + i;
                } else {
                    src = n * elements_per_channel + i;
                }
                int alpha_s = (alpha == nullptr) ? 1 : alpha[c];
                int beta_s = (beta == nullptr) ? 0 : beta[c];
                output[dst] = alpha_s * input[src] + beta_s;
                dst++;
            }
        }
    }
    return SUCCESS;
}

template <bool icoc_equal>
static EE scale_nhwc_int32(
    I32 *input, I32 *alpha, I32 *beta, I32 in, I32 ic, I32 elements_per_channel, I32 *output)
{
    __m256i one = _mm256_set1_epi32(1);
    __m256i zero = _mm256_set1_epi32(0);
    __m256i in_vec;
    int in_s;
    U32 dst = 0, src = 0;
    for (I32 n = 0; n < in; n++) {
        for (I32 i = 0; i < elements_per_channel; i++, src++) {
            I32 c = 0;
            for (; c < ic - 7; c += 8) {
                __m256i alpha_vec = (alpha == nullptr) ? one : _mm256_loadu_si256((const __m256i *)(alpha + c));
                __m256i beta_vec = (beta == nullptr) ? zero : _mm256_loadu_si256((const __m256i *)(beta + c));
                if (icoc_equal) {
                    in_vec = _mm256_loadu_si256((const __m256i *)(input + dst));
                } else {
                    in_vec = _mm256_set1_epi32(input[src]);
                }
                __m256i out_vec = _mm256_add_epi32(_mm256_mul_epi32(alpha_vec, in_vec), beta_vec);
                _mm256_storeu_si256((__m256i *)(output + dst), out_vec);
                dst += 8;
            }
            for (; c < ic; c++) {
                int alpha_s = (alpha == nullptr) ? 1 : alpha[c];
                int beta_s = (beta == nullptr) ? 0 : beta[c];
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

EE scale_int32(I32 *input,
    I32 axis,
    I32 nDims,
    I32 *alpha,
    I32 *beta,
    I32 on,
    I32 oc,
    I32 elements_per_channel,
    I32 ic,
    I32 *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;
    // If oc is 1, it means that weights/vectors have only one param, so we need use the calculation logic of nchw.
    if (axis == 1 || axis == 0 || oc == 1) {
        if (ic == oc) {
            ret = scale_nchw_int32<true>(input, alpha, beta, on, oc, elements_per_channel, output);
        } else {
            ret = scale_nchw_int32<false>(input, alpha, beta, on, oc, elements_per_channel, output);
        }
    } else if (axis == nDims - 1) {
        if (ic == oc) {
            ret = scale_nhwc_int32<true>(input, alpha, beta, on, oc, elements_per_channel, output);
        } else {
            ret = scale_nhwc_int32<false>(input, alpha, beta, on, oc, elements_per_channel, output);
        }
    } else if (axis == nDims) {
        ret = scale_nchwc8_int32(input, alpha, beta, on, oc, elements_per_channel, output);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return ret;
}
