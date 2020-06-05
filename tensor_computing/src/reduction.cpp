// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "tensor_computing.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif

EE reduction(TensorDesc inputDesc, const void* input,
    TensorDesc maskDesc, const void* mask,
    I32 axis,
    ReductionMode reductionMode,
    float coeff,
    TensorDesc outputDesc, void* output, Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = reduction_general(inputDesc, input, maskDesc, mask, axis, reductionMode, coeff, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = reduction_arm(inputDesc, input, maskDesc, mask, axis, reductionMode, coeff, outputDesc, output);
#endif
    }
    return ret;
}

EE reduction_infer_output_size(TensorDesc inputDesc, TensorDesc maskDesc, int axis, bool keepDim, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc)
        CHECK_STATUS(NULL_POINTER);

    *outputDesc = inputDesc;
    if (axis < 0)
        axis += inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;
    int num = 1;
    if (tensorNumElements(maskDesc) == 0)
        (*outputDesc).dims[axis] = 1;
    else {
        num = maskDesc.dims[1];
        (*outputDesc).dims[axis] = num;
    }

    if (!keepDim && num < 2) {
        for (int i = axis; i < (I32)(inputDesc.nDims)-1; i++) {
            (*outputDesc).dims[i] = (*outputDesc).dims[i+1];
        }
        (*outputDesc).nDims = inputDesc.nDims - 1;
    }
    return SUCCESS;
}
