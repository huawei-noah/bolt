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

#include "cpu/arm/arm_neon_expand.h"
#include "cpu/arm/tensor_computing_arm.h"

inline EE softmax_fp16(TensorDesc inputDesc, const F16* input,
    TensorDesc outputDesc, F16* output)
{
    UNUSED(outputDesc);

    U32 size = tensorNumElements(inputDesc);
    U32 loop_inner = inputDesc.dims[0];
    U32 loop_outer = size / loop_inner;
    F16 buffer[8];
    for(U32 loop=0; loop<loop_outer; loop++) {
        I32 len = loop_inner;
        if(len == 0)
            return NOT_MATCH;
        const F16 *input_current_ptr  = input + loop * len;
        F16 *output_current_ptr = output + loop * len;

        float16x8_t max_v, sub_v, sum_v, tmp_v;
        F16 max_s, tmp_s;
        max_s = array_max_f16(input_current_ptr, len);
        max_v = vdupq_n_f16(max_s);
        sum_v = vdupq_n_f16(0);

        I32 i = 0;
        for(i = 0; i < len - 7; i+=8) {
            float16x8_t in = vld1q_f16(input_current_ptr + i);
            sub_v = vsubq_f16(in, max_v);
            tmp_v = vexpq_f16_03_percent_error(sub_v);
            sum_v = vaddq_f16(sum_v, tmp_v);
            vst1q_f16(output_current_ptr + i, tmp_v);
        }
        vst1q_f16(buffer, sum_v);
        F16 sum_s = 0;
        for(U32 j = 0; j < 8; j++) {
            sum_s += buffer[j];
        }
        for(; i < len; i++){
            tmp_s = exp(input_current_ptr[i] - max_s);
            output_current_ptr[i] = tmp_s;
            sum_s += tmp_s;
        }

        F16 sum_inv_s = 1 / sum_s;
        float16x8_t sum_inv_v = vdupq_n_f16(sum_inv_s);
        for(i = 0; i < len - 7; i+=8) {
            tmp_v = vld1q_f16(output_current_ptr + i);
            tmp_v = vmulq_f16(tmp_v, sum_inv_v);
            vst1q_f16(output_current_ptr + i, tmp_v);
        }
        for(; i < len; i++){
            output_current_ptr[i] = output_current_ptr[i] * sum_inv_s;
        }
    }

    return SUCCESS;
}

EE softmax_arm(TensorDesc inputDesc, const void* input,
    TensorDesc outputDesc, void* output)
{
    if(nullptr == input || nullptr == output)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
        case DT_F16: {
            ret = softmax_fp16(inputDesc, (const F16*)input, outputDesc, (F16*)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
