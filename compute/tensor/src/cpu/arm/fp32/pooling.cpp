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

template <PoolingMode pm>
EE pooling_c8_fp32(const I32 &tstart,
    const I32 &tend,
    const I32 &hstart,
    const I32 &hend,
    const I32 &wstart,
    const I32 &wend,
    const I32 &poolSize,
    const I32 &kernelSize,
    const U8 *_input,
    const I32 &it,
    const I32 &ih,
    const I32 &iw,
    U8 *_output,
    void *_scale)
{
    const F32 *input = (const F32 *)_input;
    F32 *output = (F32 *)_output;
    UNUSED(_scale);
    float32x4_t out0 = vdupq_n_f32((pm == POOLING_MAX) ? -FLT_MAX : 0);
    float32x4_t out1 = out0;
    for (int kernelT = tstart; kernelT < tend; kernelT++) {
        for (int kernelH = hstart; kernelH < hend; kernelH++) {
            for (int kernelW = wstart; kernelW < wend; kernelW++) {
                U32 index = ((kernelT * ih + kernelH) * iw + kernelW) * 8;
                float32x4_t in0 = vld1q_f32(input + index);
                float32x4_t in1 = vld1q_f32(input + index + 4);
                if (pm == POOLING_MAX) {
                    out0 = vmaxq_f32(in0, out0);
                    out1 = vmaxq_f32(in1, out1);
                } else {
                    out0 = vaddq_f32(out0, in0);
                    out1 = vaddq_f32(out1, in1);
                }
            }
        }
    }
    if (pm == POOLING_MEAN) {
        out0 = vmulq_n_f32(out0, 1. / poolSize);
        out1 = vmulq_n_f32(out1, 1. / poolSize);
    }
    vst1q_f32(output, out0);
    vst1q_f32(output + 4, out1);
    return SUCCESS;
}
template EE pooling_c8_fp32<POOLING_MEAN>(const I32 &tstart,
    const I32 &tend,
    const I32 &hstart,
    const I32 &hend,
    const I32 &wstart,
    const I32 &wend,
    const I32 &poolSize,
    const I32 &kernelSize,
    const U8 *_input,
    const I32 &it,
    const I32 &ih,
    const I32 &iw,
    U8 *_output,
    void *_scale);
template EE pooling_c8_fp32<POOLING_MAX>(const I32 &tstart,
    const I32 &tend,
    const I32 &hstart,
    const I32 &hend,
    const I32 &wstart,
    const I32 &wend,
    const I32 &poolSize,
    const I32 &kernelSize,
    const U8 *_input,
    const I32 &it,
    const I32 &ih,
    const I32 &iw,
    U8 *_output,
    void *_scale);

EE pooling_bp_c8_fp32(const F32 *input,
    int hstart,
    int hend,
    int wstart,
    int wend,
    int pool,
    F32 *output,
    U32 stride,
    PoolingParamSpec poolingParamSpec)
{
    EE ret = SUCCESS;
    PoolingMode pm = poolingParamSpec.mode;
    if (pm != POOLING_MEAN) {
        ret = NOT_SUPPORTED;
    }
    float32x4_t poolSize = vdupq_n_f32(1. / pool);
    float32x4_t in0 = vmulq_f32(vld1q_f32(input), poolSize);
    float32x4_t in1 = vmulq_f32(vld1q_f32(input + 4), poolSize);
    for (int kernelH = hstart; kernelH < hend; kernelH++) {
        for (int kernelW = wstart; kernelW < wend; kernelW++) {
            U32 index = (kernelH * stride + kernelW) * 8;
            float32x4_t out0 = vaddq_f32(vld1q_f32(output + index), in0);
            float32x4_t out1 = vaddq_f32(vld1q_f32(output + index + 4), in1);
            vst1q_f32(output + index, out0);
            vst1q_f32(output + index + 4, out1);
        }
    }
    return ret;
}
