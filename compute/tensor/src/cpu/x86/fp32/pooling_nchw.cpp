// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"

#define UNROLL_W 32

typedef void (*pooling_max_func)(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride);
typedef void (*pooling_mean_func)(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize);

void pooling_max_w32(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __m256 x1, x2, x3, x4;
    if (stride == 1) {
        x1 = _mm256_loadu_ps(curI);
        x2 = _mm256_loadu_ps(curI + 8);
        x3 = _mm256_loadu_ps(curI + 16);
        x4 = _mm256_loadu_ps(curI + 24);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_max_ps(x1, _mm256_loadu_ps(curI));
                x2 = _mm256_max_ps(x2, _mm256_loadu_ps(curI + 8));
                x3 = _mm256_max_ps(x3, _mm256_loadu_ps(curI + 16));
                x4 = _mm256_max_ps(x4, _mm256_loadu_ps(curI + 24));
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        x1 = _mm256_i32gather_ps(curI, v256index, 4);
        x2 = _mm256_i32gather_ps(curI + 8 * stride, v256index, 4);
        x3 = _mm256_i32gather_ps(curI + 16 * stride, v256index, 4);
        x4 = _mm256_i32gather_ps(curI + 24 * stride, v256index, 4);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_max_ps(x1, _mm256_i32gather_ps(curI, v256index, 4));
                x2 = _mm256_max_ps(x2, _mm256_i32gather_ps(curI + 8 * stride, v256index, 4));
                x3 = _mm256_max_ps(x3, _mm256_i32gather_ps(curI + 16 * stride, v256index, 4));
                x4 = _mm256_max_ps(x4, _mm256_i32gather_ps(curI + 24 * stride, v256index, 4));
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, x1);
    _mm256_storeu_ps(curO + 8, x2);
    _mm256_storeu_ps(curO + 16, x3);
    _mm256_storeu_ps(curO + 24, x4);
}

void pooling_max_w16(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __m256 x1, x2;
    if (stride == 1) {
        x1 = _mm256_loadu_ps(curI);
        x2 = _mm256_loadu_ps(curI + 8);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_max_ps(x1, _mm256_loadu_ps(curI));
                x2 = _mm256_max_ps(x2, _mm256_loadu_ps(curI + 8));
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        x1 = _mm256_i32gather_ps(curI, v256index, 4);
        x2 = _mm256_i32gather_ps(curI + 8 * stride, v256index, 4);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_max_ps(x1, _mm256_i32gather_ps(curI, v256index, 4));
                x2 = _mm256_max_ps(x2, _mm256_i32gather_ps(curI + 8 * stride, v256index, 4));
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, x1);
    _mm256_storeu_ps(curO + 8, x2);
}

void pooling_max_w8(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __m256 x1;
    if (stride == 1) {
        x1 = _mm256_loadu_ps(curI);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_max_ps(x1, _mm256_loadu_ps(curI));
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        x1 = _mm256_i32gather_ps(curI, v256index, 4);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_max_ps(x1, _mm256_i32gather_ps(curI, v256index, 4));
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, x1);
}

void pooling_max_w0(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    *curO = *curI;
    for (U32 h = 0; h < kh; ++h) {
        for (U32 w = 0; w < kw; ++w) {
            *curO = UNI_MAX(*curO, *curI);
            curI += 1;
        }
        curI += iStep;
    }
}

void pooling_mean_w32(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __m256 x0 = _mm256_set1_ps(1.0f / poolSize);
    __m256 x1 = _mm256_setzero_ps();
    __m256 x2 = _mm256_setzero_ps();
    __m256 x3 = _mm256_setzero_ps();
    __m256 x4 = _mm256_setzero_ps();

    if (stride == 1) {
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_add_ps(x1, _mm256_loadu_ps(curI));
                x2 = _mm256_add_ps(x2, _mm256_loadu_ps(curI + 8));
                x3 = _mm256_add_ps(x3, _mm256_loadu_ps(curI + 16));
                x4 = _mm256_add_ps(x4, _mm256_loadu_ps(curI + 24));
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_add_ps(x1, _mm256_i32gather_ps(curI, v256index, 4));
                x2 = _mm256_add_ps(x2, _mm256_i32gather_ps(curI + 8 * stride, v256index, 4));
                x3 = _mm256_add_ps(x3, _mm256_i32gather_ps(curI + 16 * stride, v256index, 4));
                x4 = _mm256_add_ps(x4, _mm256_i32gather_ps(curI + 24 * stride, v256index, 4));
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, _mm256_mul_ps(x1, x0));
    _mm256_storeu_ps(curO + 8, _mm256_mul_ps(x2, x0));
    _mm256_storeu_ps(curO + 16, _mm256_mul_ps(x3, x0));
    _mm256_storeu_ps(curO + 24, _mm256_mul_ps(x4, x0));
}

void pooling_mean_w16(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __m256 x0 = _mm256_set1_ps(1.0f / poolSize);
    __m256 x1 = _mm256_setzero_ps();
    __m256 x2 = _mm256_setzero_ps();

    if (stride == 1) {
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_add_ps(x1, _mm256_loadu_ps(curI));
                x2 = _mm256_add_ps(x2, _mm256_loadu_ps(curI + 8));
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_add_ps(x1, _mm256_i32gather_ps(curI, v256index, 4));
                x2 = _mm256_add_ps(x2, _mm256_i32gather_ps(curI + 8 * stride, v256index, 4));
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, _mm256_mul_ps(x1, x0));
    _mm256_storeu_ps(curO + 8, _mm256_mul_ps(x2, x0));
}

void pooling_mean_w8(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __m256 x0 = _mm256_set1_ps(1.0f / poolSize);
    __m256 x1 = _mm256_setzero_ps();

    if (stride == 1) {
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_add_ps(x1, _mm256_loadu_ps(curI));
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        for (U32 h = 0; h < kh; ++h) {
            for (U32 w = 0; w < kw; ++w) {
                x1 = _mm256_add_ps(x1, _mm256_i32gather_ps(curI, v256index, 4));
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, _mm256_mul_ps(x1, x0));
}

void pooling_mean_w0(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    *curO = 0;
    for (U32 h = 0; h < kh; ++h) {
        for (U32 w = 0; w < kw; ++w) {
            *curO += *curI;
            curI += 1;
        }
        curI += iStep;
    }
    *curO /= poolSize;
}

void pooling_max_with_idx_w32(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __m256 x1, x2, x3, x4;
    __m256 t1, t2, t3, t4;
    __m256 b1, b2;
    __m256 i1, i2, i3, i4;
    __m256 diff = _mm256_set_ps(stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
    i1 = _mm256_add_ps(diff, _mm256_set1_ps(w));
    i2 = _mm256_add_ps(diff, _mm256_set1_ps(w + stride * 8));
    i3 = _mm256_add_ps(diff, _mm256_set1_ps(w + stride * 16));
    i4 = _mm256_add_ps(diff, _mm256_set1_ps(w + stride * 24));
    if (stride == 1) {
        x1 = _mm256_loadu_ps(curI);
        x2 = _mm256_loadu_ps(curI + 8);
        x3 = _mm256_loadu_ps(curI + 16);
        x4 = _mm256_loadu_ps(curI + 24);
        for (U32 fh = 0; fh < kh; ++fh) {
            for (U32 fw = 0; fw < kw; ++fw) {
                t1 = _mm256_loadu_ps(curI);
                t2 = _mm256_loadu_ps(curI + 8);
                t3 = _mm256_loadu_ps(curI + 16);
                t4 = _mm256_loadu_ps(curI + 24);
                b1 = _mm256_cmp_ps(x1, t1, 1);
                b2 = _mm256_cmp_ps(x2, t2, 1);
                x1 = _mm256_blendv_ps(x1, t1, b1);
                x2 = _mm256_blendv_ps(x2, t2, b2);
                i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
                i2 = _mm256_blendv_ps(i2, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + 8)), b2);
                b1 = _mm256_cmp_ps(x3, t3, 1);
                b2 = _mm256_cmp_ps(x4, t4, 1);
                x3 = _mm256_blendv_ps(x3, t3, b1);
                x4 = _mm256_blendv_ps(x4, t4, b2);
                i3 = _mm256_blendv_ps(i3, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + 16)), b1);
                i4 = _mm256_blendv_ps(i4, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + 24)), b2);
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        x1 = _mm256_i32gather_ps(curI, v256index, 4);
        x2 = _mm256_i32gather_ps(curI + 8 * stride, v256index, 4);
        x3 = _mm256_i32gather_ps(curI + 16 * stride, v256index, 4);
        x4 = _mm256_i32gather_ps(curI + 24 * stride, v256index, 4);
        for (U32 fh = 0; fh < kh; ++fh) {
            for (U32 fw = 0; fw < kw; ++fw) {
                t1 = _mm256_i32gather_ps(curI, v256index, 4);
                t2 = _mm256_i32gather_ps(curI + 8 * stride, v256index, 4);
                t3 = _mm256_i32gather_ps(curI + 16 * stride, v256index, 4);
                t4 = _mm256_i32gather_ps(curI + 24 * stride, v256index, 4);
                b1 = _mm256_cmp_ps(x1, t1, 1);
                b2 = _mm256_cmp_ps(x2, t2, 1);
                x1 = _mm256_blendv_ps(x1, t1, b1);
                x2 = _mm256_blendv_ps(x2, t2, b2);
                i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
                i2 = _mm256_blendv_ps(i2, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + stride * 8)), b2);
                b1 = _mm256_cmp_ps(x3, t3, 1);
                b2 = _mm256_cmp_ps(x4, t4, 1);
                x3 = _mm256_blendv_ps(x3, t3, b1);
                x4 = _mm256_blendv_ps(x4, t4, b2);
                i3 = _mm256_blendv_ps(i3, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + stride * 16)), b1);
                i4 = _mm256_blendv_ps(i4, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + stride * 24)), b2);
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, x1);
    _mm256_storeu_ps(curO + 8, x2);
    _mm256_storeu_ps(curO + 16, x3);
    _mm256_storeu_ps(curO + 24, x4);
    _mm256_storeu_si256((__m256i*)(idx), _mm256_cvtps_epi32(i1));
    _mm256_storeu_si256((__m256i*)(idx + 8), _mm256_cvtps_epi32(i2));
    _mm256_storeu_si256((__m256i*)(idx + 16), _mm256_cvtps_epi32(i3));
    _mm256_storeu_si256((__m256i*)(idx + 24), _mm256_cvtps_epi32(i4));
}

void pooling_max_with_idx_w16(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __m256 x1, x2;
    __m256 t1, t2;
    __m256 b1, b2;
    __m256 i1, i2;
    __m256 diff = _mm256_set_ps(stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
    i1 = _mm256_add_ps(diff, _mm256_set1_ps(w));
    i2 = _mm256_add_ps(diff, _mm256_set1_ps(w + stride * 8));
    if (stride == 1) {
        x1 = _mm256_loadu_ps(curI);
        x2 = _mm256_loadu_ps(curI + 8);
        for (U32 fh = 0; fh < kh; ++fh) {
            for (U32 fw = 0; fw < kw; ++fw) {
                t1 = _mm256_loadu_ps(curI);
                t2 = _mm256_loadu_ps(curI + 8);
                b1 = _mm256_cmp_ps(x1, t1, 1);
                b2 = _mm256_cmp_ps(x2, t2, 1);
                x1 = _mm256_blendv_ps(x1, t1, b1);
                x2 = _mm256_blendv_ps(x2, t2, b2);
                i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
                i2 = _mm256_blendv_ps(i2, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + 8)), b2);
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        x1 = _mm256_i32gather_ps(curI, v256index, 4);
        x2 = _mm256_i32gather_ps(curI + 8 * stride, v256index, 4);
        for (U32 fh = 0; fh < kh; ++fh) {
            for (U32 fw = 0; fw < kw; ++fw) {
                t1 = _mm256_i32gather_ps(curI, v256index, 4);
                t2 = _mm256_i32gather_ps(curI + 8 * stride, v256index, 4);
                b1 = _mm256_cmp_ps(x1, t1, 1);
                b2 = _mm256_cmp_ps(x2, t2, 1);
                x1 = _mm256_blendv_ps(x1, t1, b1);
                x2 = _mm256_blendv_ps(x2, t2, b2);
                i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
                i2 = _mm256_blendv_ps(i2, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + stride * 8)), b2);
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, x1);
    _mm256_storeu_ps(curO + 8, x2);
    _mm256_storeu_si256((__m256i*)(idx), _mm256_cvtps_epi32(i1));
    _mm256_storeu_si256((__m256i*)(idx + 8), _mm256_cvtps_epi32(i2));
}

void pooling_max_with_idx_w8(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    __m256 x1;
    __m256 t1;
    __m256 b1;
    __m256 i1;
    __m256 diff = _mm256_set_ps(stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
    i1 = _mm256_add_ps(diff, _mm256_set1_ps(w));
    if (stride == 1) {
        x1 = _mm256_loadu_ps(curI);
        for (U32 fh = 0; fh < kh; ++fh) {
            for (U32 fw = 0; fw < kw; ++fw) {
                t1 = _mm256_loadu_ps(curI);
                b1 = _mm256_cmp_ps(x1, t1, 1);
                x1 = _mm256_blendv_ps(x1, t1, b1);
                i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
                curI += 1;
            }
            curI += iStep;
        }
    } else {
        __m256i v256index = _mm256_set_epi32(
            stride * 7, stride * 6, stride * 5, stride * 4, stride * 3, stride * 2, stride, 0);
        x1 = _mm256_i32gather_ps(curI, v256index, 4);
        for (U32 fh = 0; fh < kh; ++fh) {
            for (U32 fw = 0; fw < kw; ++fw) {
                t1 = _mm256_i32gather_ps(curI, v256index, 4);
                b1 = _mm256_cmp_ps(x1, t1, 1);
                x1 = _mm256_blendv_ps(x1, t1, b1);
                i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
                curI += 1;
            }
            curI += iStep;
        }
    }
    _mm256_storeu_ps(curO, x1);
    _mm256_storeu_si256((__m256i*)(idx), _mm256_cvtps_epi32(i1));
}

void pooling_max_with_idx_w0(const F32 *curI, F32 *curO, I32 *idx, U32 iw, U32 w, U32 kw, U32 kh, U32 iStep, U32 stride)
{
    *curO = *curI;
    *idx = w;
    for (U32 fh = 0; fh < kh; ++fh) {
        for (U32 fw = 0; fw < kw; ++fw) {
            if (*curO < *curI) {
                *curO = *curI;
                *idx = w + fh * iw + fw;
            }
            curI += 1;
        }
        curI += iStep;
    }
}


EE pooling_nchw_fp32(
    TensorDesc inputDesc, const F32 *input, PoolingParamSpec p, TensorDesc outputDesc, F32 *output, I32 *idx)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        CHECK_STATUS(tensor3dGet(outputDesc, &odt, &odf, &on, &oc, &oh));
        iw = ow = 1;
    } else if (tensorIs2d(inputDesc)) {
        CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &ih, &iw));
        CHECK_STATUS(tensor2dGet(outputDesc, &odt, &odf, &oh, &ow));
        in = ic = 1;
        on = oc = 1;
    } else {
        return NOT_SUPPORTED;
    }

    if (idt != odt || idt != DT_F32) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (in != on || ic != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }

    PoolingMode pm = p.mode;
    U32 strideH = p.stride_h;
    U32 strideW = p.stride_w;
    U32 paddingT = p.pad_top;
    U32 paddingL = p.pad_left;
    U32 kernelSizeH = p.kernel_h;
    U32 kernelSizeW = p.kernel_w;
    if (paddingT >= kernelSizeH || paddingL >= kernelSizeW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 owInter = (iw + paddingL - kernelSizeW) / strideW + 1;
    U32 wSizes[5] = {1, 8, 16, 16, 32};
    pooling_max_func pooling_max_without_idx[5] = {
        pooling_max_w0, pooling_max_w8, pooling_max_w16, pooling_max_w16, pooling_max_w32};
    pooling_max_func pooling_max_with_idx[5] = {
        pooling_max_with_idx_w0, pooling_max_with_idx_w8, pooling_max_with_idx_w16, pooling_max_with_idx_w16, pooling_max_with_idx_w32};
    pooling_max_func *pooling_max = pooling_max_without_idx;
    if (idx != nullptr) {
        pooling_max = pooling_max_with_idx;
    }
    pooling_mean_func pooling_mean[5] = {
        pooling_mean_w0, pooling_mean_w8, pooling_mean_w16, pooling_mean_w16, pooling_mean_w32};
    U32 loop = in * ic * oh;
    EE ret = SUCCESS;

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 l = 0; l < loop; ++l) {
        U32 n = l / (ic * oh);
        U32 c = l % (ic * oh) / oh;
        U32 h = l % oh;
        const F32 *tmpI = input + n * ic * ih * iw + c * ih * iw;
        F32 *tmpO = output + n * ic * oh * ow + c * oh * ow;
        I32 *tmpIdx = idx + n * ic * oh * ow + c * oh * ow;
        int hstart = (int)h * (int)strideH - (int)paddingT;
        int hend = UNI_MIN(hstart + kernelSizeH, ih);
        hstart = UNI_MAX(hstart, 0);
        U32 kh = hend - hstart;
        F32 poolSize = kernelSizeH * kernelSizeW;
        U32 wSize = 0;
        for (U32 w = 0; w < ow; w += wSize) {
            if (w < owInter) {
                wSize = UNI_MIN(owInter - w, UNROLL_W);
            } else {
                wSize = 1;
            }
            wSize = wSizes[wSize >> 3];
            int wstart = (int)w * (int)strideW - (int)paddingL;
            int wend = UNI_MIN(wstart + kernelSizeW, iw);
            wstart = UNI_MAX(wstart, 0);

            const F32 *curI = tmpI + (hstart * iw + wstart);
            F32 *curO = tmpO + (h * ow + w);
            I32 *curIdx = tmpIdx + (h * ow + w);
            U32 kw = wend - wstart;
            U32 iStep = iw - kw;
            if (!p.count_include_pad) {
                poolSize = kh * kw;
            }
            if (kw < kernelSizeW) {
                wSize = 1;
            }
            switch (pm) {
                case POOLING_MAX: {
                    pooling_max[wSize >> 3](curI, curO, curIdx, iw, hstart * iw + wstart + c * ih * iw, kw, kh, iStep, strideW);
                    break;
                }
                case POOLING_MEAN: {
                    pooling_mean[wSize >> 3](curI, curO, kw, kh, iStep, strideW, poolSize);
                    break;
                }
                default:
                    ret = NOT_SUPPORTED;
            }
        }
    }

    return ret;
}
