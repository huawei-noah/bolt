// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>
#include "cpu/arm/fp32/tensor_computing_fp32.h"

EE eltwise_fp32(std::vector<void*>input, U32 num, U32 len, void *output, EltwiseMode eltwiseMode) {
    U32 len_tail = len % 4;
    U32 len_main = len - len_tail;

    F32 *output_ptr = (F32 *)output;
    for (U32 i = 0; i < len_main; i += 4){
        float32x4_t tmp_v = vld1q_f32((F32*)(input[0]) + i);

        for (U32 j = 1; j < num; j++) {
            float32x4_t value_v = vld1q_f32((F32*)(input[j]) + i);
            switch (eltwiseMode) {
                case ELTWISE_SUM:
                    tmp_v = vaddq_f32(value_v, tmp_v);
                    break;
                case ELTWISE_MAX:
                    tmp_v = vmaxq_f32(value_v, tmp_v);
                    break;
                case ELTWISE_PROD:
                    tmp_v = vmulq_f32(value_v, tmp_v);
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        vst1q_f32(output_ptr + i, tmp_v);
    }
    for (U32 i = len_main; i < len; i++){
        F32 tmp_s = *((F32*)input[0] + i);

        for (U32 j = 1; j < num; j++) {
            F32 value_s = *((F32*)input[j] + i);
            switch (eltwiseMode) {
                case ELTWISE_SUM:
                    tmp_s = value_s + tmp_s;
                    break;
                case ELTWISE_MAX:
                    tmp_s = (value_s > tmp_s) ? value_s : tmp_s;
                    break;
                case ELTWISE_PROD:
                    tmp_s *= value_s;
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        output_ptr[i] = tmp_s;
    }
    return SUCCESS;
} 
