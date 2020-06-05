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
F32 reductionKernel(const T* input, const float *mask, ReductionMode reductionMode, float coeff, U32 len, U32 stride) {
    F32 sum = 0;
    U32 j = 0;
    U32 count = 0;
    for (U32 i = 0; i < len; i++, j+=stride) {
        if (mask == nullptr || (mask != nullptr && mask[i] == 1)) {
            if (reductionMode == REDUCTION_SUM || reductionMode == REDUCTION_MEAN)
                sum += input[j];
            else
                CHECK_STATUS(NOT_SUPPORTED);
            count ++;
        }
    }
    F32 result = sum;
    if (reductionMode == REDUCTION_MEAN)
        result /= count;
    result *= coeff;
    return result;
}

template<typename T>
EE reduction(TensorDesc inputDesc, const T* input,
    TensorDesc maskDesc, const float* mask,
    I32 axis,
    ReductionMode reductionMode,
    float coeff,
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
    U32 maskLen = tensorNumElements(maskDesc);
    maskLen = (maskLen > 0) ? maskLen : len;
    U32 axisDim = maskLen / len;
    for (U32 i = 0; i < loopOuter; i++) {
        for (U32 j = 0; j < maskLen; j += len) {
            U32 axisIndex = j / len;
            U32 outputIndex = (i * axisDim + axisIndex) * loopInner;
            for (U32 k = 0; k < loopInner; k++) {
                const T* array = input + i * (len * loopInner) + k;
                const float *maskPtr = (mask == nullptr) ? nullptr : mask + j;
                output[outputIndex+k] = reductionKernel<T>(array, maskPtr, reductionMode, coeff, len, loopInner);
            }
        }
    }
    return SUCCESS;
}

EE reduction_general(TensorDesc inputDesc, const void* input,
    TensorDesc maskDesc, const void* mask,
    I32 axis,
    ReductionMode reductionMode,
    float coeff,
    TensorDesc outputDesc, void* output)
{
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = reduction<F32>(inputDesc, (const F32*)input, maskDesc, (const float*)mask, axis, reductionMode, coeff, outputDesc, (F32*)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = reduction<F16>(inputDesc, (const F16*)input, maskDesc, (const float *)mask, axis, reductionMode, coeff, outputDesc, (F16*)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
