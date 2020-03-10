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

#include "cpu/general/common_general.h"
#include "cpu/general/tensor_computing_general.h"

template<typename T>
inline EE array_norm_scale(T *input, T *output, I32 len, F32 mean, F32 var, T *alpha, T *beta) {
    F32 eps = 1e-6;
    F32 std_value = sqrt(var + eps);
    for(I32 i = 0; i < len; i++){
        output[i] = alpha[i] * (input[i] - mean) / std_value + beta[i];
    }
    return SUCCESS;
}

template<typename T>
inline EE layer_normalization(T *alpha, T *beta,
    TensorDesc inputDesc, T* input,
    TensorDesc outputDesc, T* output)
{
    if (nullptr == input || nullptr == output)
        CHECK_STATUS(NULL_POINTER);
    if(inputDesc.dt != outputDesc.dt || inputDesc.df != outputDesc.df)
        CHECK_STATUS(NOT_MATCH);

    U32 size = tensorNumElements(inputDesc);
    I32 size_inner = inputDesc.dims[0];
    I32 size_outer = size / size_inner;
    for(I32 i = 0; i < size_outer; i++) {
        T *current_input = input + i * size_inner;
        T *current_output = output + i * size_inner;
        F32 mean = array_mean<T>(current_input, size_inner);
        F32 var  = array_var<T>(current_input, size_inner, mean);

        array_norm_scale<T>(current_input, current_output, size_inner, mean, var, alpha, beta);
    }
    
    return SUCCESS;
}


EE layer_normalization_general(void *alpha, void *beta,
    TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output)
{
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = layer_normalization<F32>((F32*)alpha, (F32*)beta, inputDesc, (F32*)input, outputDesc, (F32*)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = layer_normalization<F16>((F16*)alpha, (F16*)beta, inputDesc, (F16*)input, outputDesc, (F16*)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
