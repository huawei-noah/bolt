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

#include "cpu/general/general_functions.h"
#include "cpu/general/tensor_computing_general.h"

static float eps = 1e-6;

template <typename T>
inline static EE array_norm_scale_template(
    T *input, T *output, I32 len, F32 mean, F32 var, T *alpha, T *beta)
{
    F32 std_value = sqrt(var + eps);
    for (I32 i = 0; i < len; i++) {
        output[i] = alpha[i] * (input[i] - mean) / std_value + beta[i];
    }
    return SUCCESS;
}

template <typename T>
static EE layer_norm_nhwc(
    TensorDesc inputDesc, T *input, T *alpha, T *beta, TensorDesc outputDesc, T *output)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.df != outputDesc.df) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 size = tensorNumElements(inputDesc);
    I32 size_inner = inputDesc.dims[0];
    I32 size_outer = size / size_inner;
    for (I32 i = 0; i < size_outer; i++) {
        T *current_input = input + i * size_inner;
        T *current_output = output + i * size_inner;
        F32 mean = array_mean_template<T>(current_input, size_inner);
        F32 var = array_var_template<T>(current_input, size_inner, mean);

        array_norm_scale_template<T>(
            current_input, current_output, size_inner, mean, var, alpha, beta);
    }
    return SUCCESS;
}

template <typename T>
static EE layer_norm_nchwc8(
    TensorDesc inputDesc, T *input, T *alpha, T *beta, TensorDesc outputDesc, T *output)
{
    int n = inputDesc.dims[inputDesc.nDims - 1];
    int c = inputDesc.dims[inputDesc.nDims - 2];
    int hw = 1;
    for (unsigned int i = 0; i < inputDesc.nDims - 2; i++) {
        hw *= inputDesc.dims[i];
    }
    int c8 = c / 8;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < hw; j++) {
            F32 sum = 0;
            for (int k = 0; k < c8; k++) {
                int id = ((i * c8 + k) * hw + j) * 8;
                for (int a = id; a < id + 8; a++) {
                    sum += input[a];
                }
            }
            F32 mean = sum / c;

            sum = 0;
            for (int k = 0; k < c8; k++) {
                int id = ((i * c8 + k) * hw + j) * 8;
                for (int a = id; a < id + 8; a++) {
                    F32 tmp = input[a] - mean;
                    sum += tmp * tmp;
                }
            }
            F32 var = sum / c;

            F32 std_value = sqrt(var + eps);
            for (int k = 0, kk = 0; k < c8; k++) {
                int id = ((i * c8 + k) * hw + j) * 8;
                for (int a = id; a < id + 8; a++, kk++) {
                    output[a] = alpha[kk] * ((input[a] - mean) / std_value) + beta[kk];
                }
            }
        }
    }
    return SUCCESS;
}

template <typename T>
static EE layer_norm_template(TensorDesc inputDesc,
    T *input,
    LayerNormParamSpec p,
    T *alpha,
    T *beta,
    TensorDesc outputDesc,
    T *output)
{
    if (nullptr == alpha || nullptr == beta || nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    EE ret = NOT_SUPPORTED;
    if (inputDesc.df == DF_NCHWC8) {
        if (p.axis == 1) {
            ret = layer_norm_nchwc8(inputDesc, input, alpha, beta, outputDesc, output);
        }
    } else {
        if (p.axis == -1) {
            ret = layer_norm_nhwc(inputDesc, input, alpha, beta, outputDesc, output);
        }
    }
    return ret;
}

EE layer_norm_general(TensorDesc inputDesc,
    void *input,
    LayerNormParamSpec p,
    void *alpha,
    void *beta,
    TensorDesc outputDesc,
    void *output)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = layer_norm_template<F32>(
                inputDesc, (F32 *)input, p, (F32 *)alpha, (F32 *)beta, outputDesc, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = layer_norm_template<F16>(
                inputDesc, (F16 *)input, p, (F16 *)alpha, (F16 *)beta, outputDesc, (F16 *)output);
            break;
        }
#endif
        default:
            break;
    }
    return ret;
}
