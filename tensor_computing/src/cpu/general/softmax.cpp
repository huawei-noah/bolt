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
#include "cpu/general/tensor_computing_general.h"


template<typename T>
T array_max(const T* input, U32 len) {
    T tmp = input[0];
    for (U32 i = 1; i < len; i++) {
        if(input[i] > tmp)
            tmp = input[i];
    }
    return tmp;
}

template<typename T>
EE softmax(TensorDesc inputDesc, const T* input,
    TensorDesc outputDesc, T* output)
{
    UNUSED(outputDesc);

    U32 size = tensorNumElements(inputDesc);
    U32 loop_inner = inputDesc.dims[0];
    U32 loop_outer = size / loop_inner;

    if (nullptr == input || nullptr == output)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);

    for (U32 loop = 0; loop < loop_outer; loop++) {
        const T *in = input + loop * loop_inner;
        T *out = output + loop * loop_inner;

        T max_value = array_max<T>(in, loop_inner);
        T sum = 0;
        for (U32 i = 0; i < loop_inner; i++) {
            T tmp = exp(in[i] - max_value);
            sum += tmp;
            out[i] = tmp;
        }
        sum = 1 / sum;
        for (U32 i = 0; i < loop_inner; i++) {
            out[i] *= sum;
        }
    }
    return SUCCESS;
}

EE softmax_general(TensorDesc inputDesc, const void* input,
    TensorDesc outputDesc, void* output)
{
    if (nullptr == input || nullptr == output)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
        case DT_F16: {
            ret = softmax<F16>(inputDesc, (const F16*)input, outputDesc, (F16*)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
