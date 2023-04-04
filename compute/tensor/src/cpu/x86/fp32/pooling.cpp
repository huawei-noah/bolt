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
#include "cpu/x86/fp32/pooling_kernel.h"

#define UNROLL_W 4

typedef void (*pooling_max_func)(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride);
typedef void (*pooling_mean_func)(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize);

EE pooling_fp32(
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

    if (idt != odt || idt != DT_F32) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (in != on || ic != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (idf != DF_NCHWC8 || odf != idf) {
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

    ic /= 8;
    U32 owInter = (iw + paddingL - kernelSizeW) / strideW + 1;
    U32 wSizes[3] = {1, 2, 4};
    pooling_max_func pooling_max_without_idx[3] = {pooling_max_w1, pooling_max_w2, pooling_max_w4};
    pooling_max_func pooling_max_with_idx[3] = {pooling_max_with_idx_w1, pooling_max_with_idx_w2, pooling_max_with_idx_w4};
    pooling_max_func *pooling_max = pooling_max_without_idx;
    if (idx != nullptr) {
        pooling_max = pooling_max_with_idx;
    }
    pooling_mean_func pooling_mean[3] = {pooling_mean_w1, pooling_mean_w2, pooling_mean_w4};
 
    U32 loop = in * ic * oh;
    EE ret = SUCCESS;

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 l = 0; l < loop; ++l) {
        U32 n = l / (ic * oh);
        U32 c = l % (ic * oh) / oh;
        U32 h = l % oh;

        const F32 *tmpI = input + n * ic * 8 * ih * iw + c * 8 * ih * iw;
        F32 *tmpO = output + n * ic * 8 * oh * ow + c * 8 * oh * ow;
        I32 *tmpIdx = idx + n * ic * 8 * oh * ow + c * 8 * oh * ow;
        U32 wSize = 0;
        for (U32 w = 0; w < ow; w += wSize) {
            if (w < owInter) {
                wSize = UNI_MIN(owInter - w, UNROLL_W);
            } else {
                wSize = 1;
            }
            wSize = wSizes[wSize >> 1];
            int hstart = (int)h * (int)strideH - (int)paddingT;
            int wstart = (int)w * (int)strideW - (int)paddingL;
            int hend = UNI_MIN(hstart + kernelSizeH, ih);
            int wend = UNI_MIN(wstart + kernelSizeW, iw);
            hstart = UNI_MAX(hstart, 0);
            wstart = UNI_MAX(wstart, 0);

            const F32 *curI = tmpI + (hstart * iw + wstart) * 8;
            F32 *curO = tmpO + (h * ow + w) * 8;
            I32 *curIdx = tmpIdx + (h * ow + w) * 8;
            U32 kh = hend - hstart;
            U32 kw = wend - wstart;
            U32 iStep = (iw - kw) * 32;
            F32 poolSize = kernelSizeH * kernelSizeW;
            if (!p.count_include_pad) {
                poolSize = kh * kw;
            }
            if (kw < kernelSizeW) {
                wSize = 1;
            }
            switch (pm) {
                case POOLING_MAX: {
                    pooling_max[wSize >> 1](curI, curO, curIdx, kw, kh, iw, ih * iw, hstart * iw + wstart + c * ih * iw * 8, iStep, strideW * 32);
                    break;
                }
                case POOLING_MEAN: {
                    pooling_mean[wSize >> 1](
                        curI, curO, kw, kh, iStep, strideW * 32, poolSize);
                    break;
                }
                default:
                    ret = NOT_SUPPORTED;
            }
        }
    }
    return ret;
}
