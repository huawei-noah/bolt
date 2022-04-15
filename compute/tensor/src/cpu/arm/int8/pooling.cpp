// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/int8/tensor_computing_int8.h"

template <PoolingMode pm>
EE pooling_c8_int8(const I32 &tstart,
    const I32 &tend,
    const I32 &hstart,
    const I32 &hend,
    const I32 &wstart,
    const I32 &wend,
    const I32 &poolSize,
    const I32 &khkw,
    const U8 *_input,
    const I32 &it,
    const I32 &ih,
    const I32 &iw,
    U8 *_output,
    void *scale)
{
    const INT8 *input = (const INT8 *)_input;
    INT8 *output = (INT8 *)_output;
    F32 *inputScale = (F32 *)scale;
    F32 *outputScale = inputScale + 1;
    int16x8_t out_mean;
    int8x8_t out1;
    short factor = 256 / khkw;
    if (pm == POOLING_MAX) {
        *outputScale = *inputScale;
        out1 = vdup_n_s8(-128);
    } else {
        if (khkw > 256) {
            return NOT_SUPPORTED;
        }
        *outputScale = *inputScale * factor * khkw / 256;
        out_mean = vdupq_n_s16(0);
    }
    for (int kernelT = tstart; kernelT < tend; kernelT++) {
        for (int kernelH = hstart; kernelH < hend; kernelH++) {
            for (int kernelW = wstart; kernelW < wend; kernelW++) {
                U32 index = ((kernelT * ih + kernelH) * iw + kernelW) * 8;
                int8x8_t in1 = vld1_s8(input + index);
                if (pm == POOLING_MAX) {
                    out1 = vmax_s8(out1, in1);
                } else {
                    out_mean = vaddw_s8(out_mean, in1);
                }
            }
        }
    }
    if (pm == POOLING_MEAN) {
        short pool_factor = factor * khkw / poolSize;
        if (pool_factor > 1) {
            out_mean = vmulq_n_s16(out_mean, pool_factor);
        }
        out1 = vshrn_n_s16(out_mean, 8);
    }
    vst1_s8(output, out1);
    return SUCCESS;
}

template EE pooling_c8_int8<POOLING_MAX>(const I32 &tstart,
    const I32 &tend,
    const I32 &hstart,
    const I32 &hend,
    const I32 &wstart,
    const I32 &wend,
    const I32 &poolSize,
    const I32 &khkw,
    const U8 *_input,
    const I32 &it,
    const I32 &ih,
    const I32 &iw,
    U8 *_output,
    void *_scale);

template EE pooling_c8_int8<POOLING_MEAN>(const I32 &tstart,
    const I32 &tend,
    const I32 &hstart,
    const I32 &hend,
    const I32 &wstart,
    const I32 &wend,
    const I32 &poolSize,
    const I32 &khkw,
    const U8 *_input,
    const I32 &it,
    const I32 &ih,
    const I32 &iw,
    U8 *_output,
    void *_scale);
