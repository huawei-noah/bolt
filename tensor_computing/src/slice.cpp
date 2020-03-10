// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <vector>
#include "tensor_computing.h"
#include "cpu/general/tensor_computing_general.h"
#include "cpu/arm/tensor_computing_arm.h"


EE slice_infer_output_size(TensorDesc inputDesc, std::vector<TensorDesc>* outputDesc, U32 axis, U32 *slice_point)
{
    if (nullptr == outputDesc)
        CHECK_STATUS(NULL_POINTER);

    U32 num = (*outputDesc).size();
    I32 target_axis = inputDesc.nDims - 1 - axis;
    for (U32 i = 0; i < num; i++) {
        (*outputDesc)[i] = inputDesc;

        I32 prev_point = 0;
        if (i > 0)
            prev_point = slice_point[i-1];
        I32 next_point = inputDesc.dims[target_axis];
        if (i < num - 1)
            next_point = slice_point[i];
        (*outputDesc)[i].dims[target_axis] = next_point - prev_point;
    }
    return SUCCESS;
}

EE slice(TensorDesc inputDesc, void* input,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output, Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = slice_general(inputDesc, input, outputDesc, output);
            break;
        case ARM_A55:
            ret = slice_arm(inputDesc, input, outputDesc, output);
            break;
        case ARM_A76:
            ret = slice_arm(inputDesc, input, outputDesc, output);
            break;
        case ARM_V8:
            ret = slice_arm(inputDesc, input, outputDesc, output);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
