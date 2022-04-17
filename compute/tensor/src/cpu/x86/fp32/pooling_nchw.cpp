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

typedef void (*pooling_max_func)(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride);
typedef void (*pooling_mean_func)(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize);

void pooling_max_w32(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
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

void pooling_max_w16(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
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

void pooling_max_w8(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
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

void pooling_max_w0(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride)
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

EE pooling_nchw_fp32(
    TensorDesc inputDesc, const F32 *input, PoolingParamSpec p, TensorDesc outputDesc, F32 *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in = 0, ic = 0, ih = 0, iw = 0, on = 0, oc = 0, oh = 0, ow = 0;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idt != odt || idt != DT_F32) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (in != on || ic != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (idf != DF_NCHW || odf != idf) {
        CHECK_STATUS(NOT_MATCH);
    }

    PoolingMode pm = p.mode;
    U32 strideH = p.stride_h;
    U32 strideW = p.stride_w;
    U32 paddingT = p.pad_top;
    U32 paddingL = p.pad_left;
    U32 kernelSizeH = p.kernel_h;
    U32 kernelSizeW = p.kernel_w;
    U32 wSize, kh, kw, iStep;
    F32 *curO;
    const F32 *curI;
    if (paddingT >= kernelSizeH || paddingL >= kernelSizeW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 owInter = (iw + paddingL - kernelSizeW) / strideW + 1;
    U32 wSizes[5] = {1, 8, 16, 16, 32};
    pooling_max_func pooling_max[5] = {
        pooling_max_w0, pooling_max_w8, pooling_max_w16, pooling_max_w16, pooling_max_w32};
    pooling_mean_func pooling_mean[5] = {
        pooling_mean_w0, pooling_mean_w8, pooling_mean_w16, pooling_mean_w16, pooling_mean_w32};
    F32 poolSize = kernelSizeH * kernelSizeW;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            for (U32 h = 0; h < oh; h++) {
                int hstart = (int)h * (int)strideH - (int)paddingT;
                int hend = UNI_MIN(hstart + kernelSizeH, ih);
                hstart = UNI_MAX(hstart, 0);
                kh = hend - hstart;
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

                    curI = input + (hstart * iw + wstart);
                    curO = output + (h * ow + w);
                    kw = wend - wstart;
                    iStep = iw - kw;
                    if (!p.count_include_pad) {
                        poolSize = kh * kw;
                    }
                    if (kw < kernelSizeW) {
                        wSize = 1;
                    }
                    switch (pm) {
                        case POOLING_MAX: {
                            pooling_max[wSize >> 3](curI, curO, kw, kh, iStep, strideW);
                            break;
                        }
                        case POOLING_MEAN: {
                            pooling_mean[wSize >> 3](curI, curO, kw, kh, iStep, strideW, poolSize);
                            break;
                        }
                        default:
                            CHECK_STATUS(NOT_SUPPORTED);
                    }
                }
            }
            input += ih * iw;
            output += oh * ow;
        }
    }
    return SUCCESS;
}
