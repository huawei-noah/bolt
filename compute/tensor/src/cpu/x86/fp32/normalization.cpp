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
#include "cpu/x86/fp32/tensor_computing_fp32.h"

inline void array_norm_scale_fp32(
    F32 *input, F32 *output, I32 len, F32 mean, F32 var, F32 *alpha, F32 *beta)
{
    F32 eps = 1e-6;
    F32 std_value = sqrt(var + eps);
    __m256 mean_v = _mm256_set1_ps(mean);
    __m256 std_v = _mm256_set1_ps(std_value);

    I32 i = 0;
    for (i = 0; i < len - 7; i += 8) {
        __m256 in = _mm256_loadu_ps(input + i);
        __m256 alpha_v = _mm256_loadu_ps(alpha + i);
        __m256 beta_v = _mm256_loadu_ps(beta + i);

        __m256 tmp_v = _mm256_sub_ps(in, mean_v);
        tmp_v = _mm256_div_ps(tmp_v, std_v);
        tmp_v = _mm256_fmadd_ps(alpha_v, tmp_v, beta_v);
        _mm256_storeu_ps(output + i, tmp_v);
    }
    for (; i < len; i++) {
        output[i] = alpha[i] * (input[i] - mean) / std_value + beta[i];
    }
}

EE layer_normalization_fp32(
    TensorDesc inputDesc, F32 *input, F32 *alpha, F32 *beta, TensorDesc outputDesc, F32 *output)
{
    UNUSED(outputDesc);
    if (nullptr == alpha || nullptr == beta || nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    U32 size = tensorNumElements(inputDesc);
    I32 size_inner = inputDesc.dims[0];
    I32 size_outer = size / size_inner;
    for (I32 i = 0; i < size_outer; i++) {
        F32 *current_input = input + i * size_inner;
        F32 *current_output = output + i * size_inner;
        F32 mean = array_mean_f32(current_input, size_inner);
        F32 var = array_var_f32(current_input, size_inner, mean);

        array_norm_scale_fp32(current_input, current_output, size_inner, mean, var, alpha, beta);
    }

    return SUCCESS;
}
