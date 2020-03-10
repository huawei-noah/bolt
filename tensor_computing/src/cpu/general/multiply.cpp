// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/general/tensor_computing_general.h"

template<typename T>
EE multiply(T* input, T* output, U32 len, F32 alpha, F32 beta)
{
    if (nullptr == input
        || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    for (U32 i = 0; i < len; i++) {
        F32 value = input[i];
        output[i] = alpha * value + beta;
    }
    return SUCCESS;
}

EE multiply_general(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output)
{
    UNUSED(outputDesc);

    if (nullptr == alpha
        || nullptr == beta)
        CHECK_STATUS(NULL_POINTER);

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = multiply<F32>((F32 *)input, (F32 *)output, tensorNumElements(inputDesc), *((F32 *)alpha), *((F32 *)beta));
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = multiply<F16>((F16 *)input, (F16 *)output, tensorNumElements(inputDesc), *((F32 *)alpha), *((F32 *)beta));
            break;
        }
#endif
        case DT_I32: {
            ret = multiply<I32>((I32 *)input, (I32 *)output, tensorNumElements(inputDesc), *((F32 *)alpha), *((F32 *)beta));
            break;
        }
        case DT_U32: {
            ret = multiply<U32>((U32 *)input, (U32 *)output, tensorNumElements(inputDesc), *((F32 *)alpha), *((F32 *)beta));
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
