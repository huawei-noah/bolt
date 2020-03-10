// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <math.h>
#include "cpu/arm/fp16/tensor_computing_fp16.h"

EE softmax_fp16(TensorDesc inputDesc, const F16* input,
    TensorDesc outputDesc, F16* output)
{
    UNUSED(outputDesc);
    if(nullptr == input || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    U32 size = tensorNumElements(inputDesc);
    I32 len = inputDesc.dims[0];
    U32 loop_outer = size / len;
    if (len == 0)
        return NOT_MATCH;
    for(U32 loop=0; loop<loop_outer; loop++) {
        const F16 *input_current_ptr  = input + loop * len;
        F16 *output_current_ptr = output + loop * len;

        float16x8_t max_v, sub_v, sum_v, tmp_v;
        F32 max_s, tmp_s;
        max_s = array_max_f16(input_current_ptr, len);
        max_v = vdupq_n_f16(max_s);
        sum_v = vdupq_n_f16(0);

        I32 i = 0;
        F32 sum_s = 0;
        for(i = 0; i < len - 7; i+=8) {
            float16x8_t in = vld1q_f16(input_current_ptr + i);
            sub_v = vsubq_f16(in, max_v);
            tmp_v = vexpq_f16_f32(sub_v);
            sum_v = vaddq_f16(sum_v, tmp_v);
            vst1q_f16(output_current_ptr + i, tmp_v);
        }
        sum_s += vaddvq_f16(sum_v);
        for(; i < len; i++){
            tmp_s = exp(input_current_ptr[i] - max_s);
            output_current_ptr[i] = tmp_s;
            sum_s += tmp_s;
        }

        array_scale_f16(output_current_ptr, output_current_ptr, len, 1.0 / sum_s, 0);
    }

    return SUCCESS;
}
