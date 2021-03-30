// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp16/tensor_computing_fp16.h"

EE pooling_c8_fp16(I32 tstart,
    I32 tend,
    I32 hstart,
    I32 hend,
    I32 wstart,
    I32 wend,
    I32 poolSize,
    const F16 *input,
    I32 it,
    I32 ih,
    I32 iw,
    PoolingParamSpec p,
    F16 *output)
{
    EE ret = SUCCESS;
    PoolingMode pm = p.mode;
    if (poolSize > 256 && p.mode == POOLING_MEAN) {
        float32x4_t poolSize_v = vdupq_n_f32(poolSize);
        float16x4_t temp0 = vdup_n_f16(0);
        float16x4_t temp1 = temp0;
        float32x4_t out0 = vdupq_n_f32(0);
        float32x4_t out1 = out0;
        int count = 0;
        for (int kernelT = tstart; kernelT < tend; kernelT++) {
            for (int kernelH = hstart; kernelH < hend; kernelH++) {
                for (int kernelW = wstart; kernelW < wend; kernelW++, count++) {
                    U32 index = ((kernelT * ih + kernelH) * iw + kernelW) * 8;
                    float16x4_t in0 = vld1_f16(input + index);
                    float16x4_t in1 = vld1_f16(input + index + 4);
                    temp0 = vadd_f16(temp0, in0);
                    temp1 = vadd_f16(temp1, in1);
                    if (count % 256 == 255) {
                        out0 = vaddq_f32(out0, vcvt_f32_f16(temp0));
                        out1 = vaddq_f32(out1, vcvt_f32_f16(temp1));
                        temp0 = vdup_n_f16(0);
                        temp1 = temp0;
                    }
                }
            }
        }
        out0 = vaddq_f32(out0, vcvt_f32_f16(temp0));
        out1 = vaddq_f32(out1, vcvt_f32_f16(temp1));
        out0 = (pm == POOLING_MAX) ? out0 : vdivq_f32(out0, poolSize_v);
        out1 = (pm == POOLING_MAX) ? out1 : vdivq_f32(out1, poolSize_v);
        vst1_f16(output, vcvt_f16_f32(out0));
        vst1_f16(output + 4, vcvt_f16_f32(out1));
    } else {
        float16x8_t poolSize_v = vdupq_n_f16(poolSize);
        float16x8_t out1 = vdupq_n_f16(float16_t((pm == POOLING_MAX) ? UNI_F16_MIN : 0));
        for (int kernelT = tstart; kernelT < tend; kernelT++) {
            for (int kernelH = hstart; kernelH < hend; kernelH++) {
                for (int kernelW = wstart; kernelW < wend; kernelW++) {
                    U32 index = ((kernelT * ih + kernelH) * iw + kernelW) * 8;
                    float16x8_t in1 = vld1q_f16(input + index);
                    switch (pm) {
                        case POOLING_MAX:
                            out1 = vmaxq_f16(in1, out1);
                            break;
                        case POOLING_MEAN:
                            out1 = vaddq_f16(out1, in1);
                            break;
                        default:
                            ret = NOT_SUPPORTED;
                            break;
                    }
                }
            }
        }
        vst1q_f16(output, ((pm == POOLING_MAX) ? out1 : vdivq_f16(out1, poolSize_v)));
    }
    return ret;
}
