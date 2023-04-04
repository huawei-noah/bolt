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

#ifdef _USE_F16_MIX_PRECISION
#define _USE_F16_MIX_PRECISION_POOLING
#endif

template <PoolingMode pm>
EE pooling_c8_fp16(const I32 &tstart,
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
    const F16 *input = (const F16 *)_input;
    F16 *output = (F16 *)_output;
    UNUSED(_scale);
    float32x4_t out0, out1;
    float16x8_t out2;
    if (pm == POOLING_MAX) {
        out2 = vdupq_n_f16(UNI_F16_MIN);
    } else {
#ifdef _USE_F16_MIX_PRECISION_POOLING
        out0 = out1 = vdupq_n_f32(0);
#else
        out2 = vdupq_n_f16(0);
#endif
    }
    for (int kernelT = tstart; kernelT < tend; kernelT++) {
        for (int kernelH = hstart; kernelH < hend; kernelH++) {
            for (int kernelW = wstart; kernelW < wend; kernelW++) {
                U32 index = ((kernelT * ih + kernelH) * iw + kernelW) * 8;
                if (pm == POOLING_MAX) {
                    float16x8_t in1 = vld1q_f16(input + index);
                    out2 = vmaxq_f16(in1, out2);
                } else {
#ifdef _USE_F16_MIX_PRECISION_POOLING
                    float16x4_t in0 = vld1_f16(input + index);
                    float16x4_t in1 = vld1_f16(input + index + 4);
                    out0 = vaddq_f32(out0, vcvt_f32_f16(in0));
                    out1 = vaddq_f32(out1, vcvt_f32_f16(in1));
#else
                    float16x8_t in1 = vld1q_f16(input + index);
                    out2 = vaddq_f16(in1, out2);
#endif
                }
            }
        }
    }
    if (pm == POOLING_MAX) {
        vst1q_f16(output, out2);
    } else {
#ifdef _USE_F16_MIX_PRECISION_POOLING
        out0 = vmulq_n_f32(out0, 1. / poolSize);
        out1 = vmulq_n_f32(out1, 1. / poolSize);
        vst1_f16(output, vcvt_f16_f32(out0));
        vst1_f16(output + 4, vcvt_f16_f32(out1));
#else
        vst1q_f16(output, vmulq_n_f16(out2, 1. / poolSize));
#endif
    }
    return SUCCESS;
}

template EE pooling_c8_fp16<POOLING_MAX>(const I32 &tstart,
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

template EE pooling_c8_fp16<POOLING_MEAN>(const I32 &tstart,
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
