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

EE clip_fp32(F32 *input, F32 *output, I32 len, F32 minValue, F32 maxValue)
{
    if (nullptr == input
       || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    float32x4_t min_v = vdupq_n_f32(minValue);
    float32x4_t max_v  = vdupq_n_f32(maxValue);

    I32 i = 0;
    for (i = 0; i < len - 3; i += 4) {
        float32x4_t in = vld1q_f32(input + i);
        float32x4_t tmp_v = vminq_f32(max_v, vmaxq_f32(min_v, in));
        vst1q_f32(output+i, tmp_v);
    }
    for (; i < len; i++) {
        F32 value = input[i];
        value = (value > minValue) ? value : minValue;
        value = (value < maxValue) ? value : maxValue;
        output[i] = value;
    }
    return SUCCESS;
}
