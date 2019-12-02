// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <arm_neon.h>
#include "cpu/arm/tensor_computing_arm.h"

inline EE clip_fp16(F16 *input, F16 *output, I32 len, F16 minValue, F16 maxValue) {
    float16x8_t min_v = vdupq_n_f16(minValue);
    float16x8_t max_v  = vdupq_n_f16(maxValue);

    I32 i = 0;
    for (i = 0; i < len - 7; i += 8) {
        float16x8_t in = vld1q_f16(input + i);
        float16x8_t tmp_v = vminq_f16(max_v, vmaxq_f16(min_v, in));
        vst1q_f16(output+i, tmp_v);
    }
    for (; i < len; i++) {
        F16 value = input[i];
        value = (value > minValue) ? value : minValue;
        value = (value < maxValue) ? value : maxValue;
        output[i] = value;
    }
    return SUCCESS;
}

EE clip_arm(void *minValue, void *maxValue, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output)
{
    UNUSED(outputDesc);

    if (nullptr == minValue
       || nullptr == maxValue
       || nullptr == input
       || nullptr == output)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = clip_fp16((F16 *)input, (F16 *)output, tensorNumElements(inputDesc), *((F16 *)minValue), *((F16 *)maxValue));
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
