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
F32 array_mean(const T* input, U32 len, U32 stride) {
    F32 sum = 0;
    U32 j = 0;
    for (U32 i = 0; i < len; i++, j+=stride) {
        sum += input[j];
    }
    return sum / len;
}

template<typename T>
EE axis_mean(TensorDesc inputDesc, const T* input,
    I32 axis,
    TensorDesc outputDesc, T* output)
{
    UNUSED(outputDesc);

    if (nullptr == input || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    if (axis < 0)
        axis = inputDesc.nDims + axis;
    axis = inputDesc.nDims - 1 - axis;
    U32 loopInner = 1;
    for (int i = 0; i < axis; i++) {
        loopInner *= inputDesc.dims[i];
    }
    U32 loopOuter = 1;
    for (U32 i = axis+1; i < inputDesc.nDims; i++) {
        loopOuter *= inputDesc.dims[i];
    }

    U32 len = inputDesc.dims[axis];
    for (U32 i = 0; i < loopOuter; i++) {
        for (U32 j = 0; j < loopInner; j++) {
            const T* array = input + i * (len * loopInner) + j;
            output[i*loopInner+j] = array_mean<T>(array, len, loopInner);
        }
    }
    return SUCCESS;
}

EE axis_mean_general(TensorDesc inputDesc, const void* input,
    I32 axis,
    TensorDesc outputDesc, void* output)
{
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = axis_mean<F32>(inputDesc, (const F32*)input, axis, outputDesc, (F32*)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = axis_mean<F16>(inputDesc, (const F16*)input, axis, outputDesc, (F16*)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
