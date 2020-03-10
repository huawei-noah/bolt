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

inline void array_norm_scale_fp16(F16 *input, F16 *output, I32 len, F32 mean, F32 var, F16 *alpha, F16 *beta) {
    F32 eps = 1e-6;
    F32 std_value = sqrt(var + eps);
    float16x8_t mean_v = vdupq_n_f16(mean);
    float16x8_t std_v  = vdupq_n_f16(std_value);

    I32 i = 0;
    for(i = 0; i < len - 7; i += 8){
        float16x8_t in = vld1q_f16(input + i);
        float16x8_t alpha_v = vld1q_f16(alpha + i);
        float16x8_t beta_v  = vld1q_f16(beta + i);

        float16x8_t tmp_v = vsubq_f16(in, mean_v);
        tmp_v = vdivq_f16(tmp_v, std_v);
        tmp_v = vfmaq_f16(beta_v, alpha_v, tmp_v);
        vst1q_f16(output+i, tmp_v);
    }
    for(; i < len; i++){
        output[i] = alpha[i] * (input[i] - mean) / std_value + beta[i];
    }
}

EE layer_normalization_fp16(F16 *alpha, F16 *beta,
    TensorDesc inputDesc, F16* input,
    TensorDesc outputDesc, F16* output)
{
    UNUSED(outputDesc);
    if (nullptr == alpha || nullptr == beta || nullptr == input || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    U32 size = tensorNumElements(inputDesc);
    I32 size_inner = inputDesc.dims[0];
    I32 size_outer = size / size_inner;
    for(I32 i = 0; i < size_outer; i++) {
        F16 *current_input = input + i * size_inner;
        F16 *current_output = output + i * size_inner;
        F32 mean = array_mean_f16(current_input, size_inner);
        F32 var  = array_var_f16(current_input, size_inner, mean);

        array_norm_scale_fp16(current_input, current_output, size_inner, mean, var, alpha, beta);
    }
    
    return SUCCESS;
}
