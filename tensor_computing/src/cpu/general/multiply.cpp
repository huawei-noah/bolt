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
EE multiply(T* input, T* output, U32 len, T alpha, T beta) {
    for (U32 i = 0; i < len; i++) {
        T value = input[i];
        output[i] = alpha * value + beta;
    }
    return SUCCESS;
}

EE multiply_general(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output)
{
    UNUSED(outputDesc);

    if (nullptr == alpha
        || nullptr == beta
        || nullptr == input
        || nullptr == output)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = multiply<F16>((F16 *)input, (F16 *)output, tensorNumElements(inputDesc), *((F16 *)alpha), *((F16 *)beta));
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
