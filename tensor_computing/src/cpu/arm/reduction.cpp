// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>
#include "cpu/arm/tensor_computing_arm.h"
#include "cpu/arm/arm_functions.h"

template<typename T>
EE reduction(TensorDesc inputDesc, const T* input,
    TensorDesc maskDesc, const float* mask,
    I32 axis,
    ReductionMode reductionMode,
    F32 coeff,
    TensorDesc outputDesc, T* output)
{
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
        if (loopInner == 1) {
            if (mask != nullptr) {
                return NOT_SUPPORTED;
            }
            const T* array = input + i * len;
            switch (reductionMode) {
                case REDUCTION_SUM:
                    output[i] = array_sum(inputDesc.dt, array, len);
                    break;
                case REDUCTION_MEAN:
                    output[i] = array_mean(inputDesc.dt, array, len);
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        } else {
            for (U32 j = 0; j < maskLen; j+=len) {
                U32 axisIndex = j / len;
                U32 outputIndex = (i * axisDim + axisIndex) * loopInner;
                if (reductionMode == REDUCTION_SUM || reductionMode == REDUCTION_MEAN) {
                    memset(output + outputIndex, 0, loopInner*bytesOf(inputDesc.dt));
                } else {
                    return NOT_SUPPORTED;
                }
                U32 count = 0;
                for (U32 k = 0; k < len; k++) {
                    if (mask == nullptr || (mask != nullptr && mask[j+k] == 1)) {
                        if (reductionMode == REDUCTION_SUM || reductionMode == REDUCTION_MEAN) {
                            array_add(inputDesc.dt, output+outputIndex,
                                &input[(i * len + k) * loopInner],
                                output+outputIndex, loopInner);
                            count++;
                        } else {
                            return NOT_SUPPORTED;
                        }
                    }
                }
                if (reductionMode == REDUCTION_MEAN) {
                    array_scale(inputDesc.dt, output+outputIndex, output+outputIndex, loopInner, 1.0/count, 0);
                }
            }
        }
    }
    if (coeff != 1) {
        array_scale(outputDesc.dt, output, output, tensorNumElements(outputDesc), coeff, 0);
    }
    return SUCCESS;
}

EE reduction_arm(TensorDesc inputDesc, const void* input,
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
            ret = reduction<F16>(inputDesc, (const F16*)input, maskDesc, (const float*)mask, axis, reductionMode, coeff, outputDesc, (F16*)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
