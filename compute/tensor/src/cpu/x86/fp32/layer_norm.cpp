// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <stdlib.h>
#include <math.h>
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "tensor_transpose.h"

static F32 eps = 1e-6;

inline static void array_norm_scale_fp32(
    F32 *input, F32 *output, I32 len, F32 mean, F32 var, F32 *alpha, F32 *beta)
{
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

static EE layer_norm_nhwc(
    TensorDesc inputDesc, F32 *input, F32 *alpha, F32 *beta, TensorDesc outputDesc, F32 *output)
{
    UNUSED(outputDesc);
    U32 size = tensorNumElements(inputDesc);
    I32 size_inner = inputDesc.dims[0];
    I32 size_outer = size / size_inner;

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
    for (I32 i = 0; i < size_outer; i++) {
        F32 *current_input = input + i * size_inner;
        F32 *current_output = output + i * size_inner;
        F32 mean = array_mean_f32(current_input, size_inner);
        F32 var = array_var_f32(current_input, size_inner, mean);

        array_norm_scale_fp32(current_input, current_output, size_inner, mean, var, alpha, beta);
    }
    return SUCCESS;
}

static EE layer_norm_nchwc8(
    TensorDesc inputDesc, F32 *input, F32 *alpha, F32 *beta, TensorDesc outputDesc, F32 *output)
{
    UNUSED(outputDesc);
    int n = inputDesc.dims[inputDesc.nDims - 1];
    int c = inputDesc.dims[inputDesc.nDims - 2];
    int hw = 1;
    for (unsigned int i = 0; i < inputDesc.nDims - 2; i++) {
        hw *= inputDesc.dims[i];
    }
    int c8 = c / 8;
    int nums = n * hw;
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS) schedule(static)
#endif
    for (int x = 0; x < nums; ++x) {
        int i = x / hw;
        int j = x % hw;
        __m256 sum_v = _mm256_set1_ps(0);
        for (int k = 0; k < c8; k++) {
            int id = ((i * c8 + k) * hw + j) * 8;
            sum_v = _mm256_add_ps(sum_v, _mm256_loadu_ps(input + id));
        }
        F32 mean = _mm256_sum_ps(sum_v) / c;
        __m256 mean_v = _mm256_set1_ps(mean);

        sum_v = _mm256_set1_ps(0);
        for (int k = 0; k < c8; k++) {
            int id = ((i * c8 + k) * hw + j) * 8;
            __m256 tmp_v = _mm256_sub_ps(_mm256_loadu_ps(input + id), mean_v);
            sum_v = _mm256_fmadd_ps(tmp_v, tmp_v, sum_v);
        }
        F32 var = _mm256_sum_ps(sum_v) / c;
        F32 std_value = sqrt(var + eps);

        __m256 std_v = _mm256_set1_ps(std_value);
        for (int k = 0, kk = 0; k < c8; k++, kk += 8) {
            int id = ((i * c8 + k) * hw + j) * 8;
            __m256 in = _mm256_loadu_ps(input + id);
            __m256 alpha_v = _mm256_loadu_ps(alpha + kk);
            __m256 beta_v = _mm256_loadu_ps(beta + kk);

            __m256 tmp_v = _mm256_sub_ps(in, mean_v);
            tmp_v = _mm256_div_ps(tmp_v, std_v);
            tmp_v = _mm256_fmadd_ps(alpha_v, tmp_v, beta_v);
            _mm256_storeu_ps(output + id, tmp_v);
        }
    }
    return SUCCESS;
}

EE layer_norm_fp32(TensorDesc inputDesc,
    F32 *input,
    LayerNormParamSpec p,
    F32 *alpha,
    F32 *beta,
    TensorDesc outputDesc,
    F32 *output)
{
    if (nullptr == alpha || nullptr == beta || nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    EE ret = NOT_SUPPORTED;
    if (inputDesc.df == DF_NCHWC8 && inputDesc.nDims == 4) {
        if (p.axis == 1) {
            ret = layer_norm_nchwc8(inputDesc, input, alpha, beta, outputDesc, output);
        } else if (p.axis == -1) {
            F32* trans_input = (F32*)malloc(tensorNumBytes(outputDesc));
            transformToNCHW(inputDesc, input, outputDesc, trans_input);
            ret = layer_norm_nhwc(outputDesc, trans_input, alpha, beta, outputDesc, output);
            free(trans_input);
        }
    } else {
        if (p.axis == -1) {
            ret = layer_norm_nhwc(inputDesc, input, alpha, beta, outputDesc, output);
        }
    }
    return ret;
}
