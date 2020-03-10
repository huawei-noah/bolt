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
#include "cpu/arm/fp16/tensor_computing_fp16.h"

EE scale_nchwc8_fp16(F16* alpha, F16* beta, F16* data, U32 in, U32 ic, U32 elements_per_channel)
{
    if (nullptr == data || nullptr == alpha)
        CHECK_STATUS(NULL_POINTER);
    const U32 align_size = 8;
    float16x8_t in_vec, out_vec;
    float16x8_t zero  = vdupq_n_f16(float16_t(0.));
    ic = ic / align_size;
    for (U32 n = 0; n < in; n++) {
        for (U32 c = 0; c < ic; c++) {
            float16x8_t alpha_vec = vld1q_f16(alpha + c * align_size);
            float16x8_t beta_vec  = (beta == nullptr) ? zero : vld1q_f16(beta + c * align_size);
            for (U32 i = 0; i < elements_per_channel; i++) {
                U32 index = ((n * ic + c) * elements_per_channel + i) * align_size;
                in_vec = vld1q_f16(data + index);
                out_vec = vfmaq_f16(beta_vec, alpha_vec, in_vec);
                vst1q_f16(data+index, out_vec);
            }
        }
    }

    return SUCCESS;
}
