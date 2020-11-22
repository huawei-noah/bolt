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

EE pooling_c8_int8(const INT8 *input,
    U32 stride,
    int hstart,
    int hend,
    int wstart,
    int wend,
    INT8 *output,
    PoolingParamSpec poolingParamSpec,
    void *scale)
{
    EE ret = SUCCESS;
    PoolingMode pm = poolingParamSpec.mode;
    U32 kernelSizeH = poolingParamSpec.kernel_h;
    U32 kernelSizeW = poolingParamSpec.kernel_w;
    if (kernelSizeH * kernelSizeW > 256 && pm == POOLING_MEAN) {
        ret = NOT_SUPPORTED;
    }
    short khkw = kernelSizeH * kernelSizeW;
    short factor = 256 / khkw;
    F32 *inputScale = (F32 *)scale;
    F32 *outputScale = inputScale + 1;
    switch (pm) {
        case POOLING_MAX: {
            *outputScale = *inputScale;
            break;
        }
        case POOLING_MEAN: {
            *outputScale = *inputScale * factor * khkw / 256;
            break;
        }
        default: {
            ret = NOT_SUPPORTED;
            break;
        }
    }
    int8x8_t in1, out1;
    int16x8_t out_mean = {0};
    out1 = vdup_n_s8(-128);
    short pool_size = (hend - hstart) * (wend - wstart);
    for (int kernelH = hstart; kernelH < hend; kernelH++) {
        for (int kernelW = wstart; kernelW < wend; kernelW++) {
            const U32 index = (kernelH * stride + kernelW) * 8;
            in1 = vld1_s8(input + index);
            switch (pm) {
                case POOLING_MAX:
                    out1 = vmax_s8(out1, in1);
                    break;
                case POOLING_MEAN:
                    out_mean = vaddw_s8(out_mean, in1);
                    break;
                default:
                    ret = NOT_SUPPORTED;
                    break;
            }
        }
    }
    if (pm == POOLING_MEAN) {
        short pool_factor = factor * khkw / pool_size;
        if (pool_factor > 1) {
            out_mean = vmulq_n_s16(out_mean, pool_factor);
        }
        out1 = vshrn_n_s16(out_mean, 8);
    }
    vst1_s8(output, out1);
    return ret;
}
