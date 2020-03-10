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
#include "cpu/general/tensor_computing_general.h"
#include "cpu/arm/tensor_computing_arm.h"


EE reshape(TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output, Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = reshape_general(inputDesc, input, outputDesc, output);
            break;
        case ARM_A55:
            ret = reshape_arm(inputDesc, input, outputDesc, output);
            break;
        case ARM_A76:
            ret = reshape_arm(inputDesc, input, outputDesc, output);
            break;
        case ARM_V8:
            ret = reshape_arm(inputDesc, input, outputDesc, output);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE reshape_infer_output_size(TensorDesc inputDesc, TensorDesc* outputDesc, I32 *shape, I32 shape_size)
{
    if (nullptr == outputDesc || nullptr == shape) {
        return NULL_POINTER;
    }

    *outputDesc = inputDesc;
    (*outputDesc).nDims = shape_size;
    if (shape_size == 2)
        (*outputDesc).df = DF_NORMAL;
    if (shape_size == 4)
        (*outputDesc).df = DF_NCHW;

    U32 factor = 1;
    I32 count = 0;
    for(I32 i = 0; i < shape_size; i++) {
        I32 value = shape[i];
        if (value == 0) {
            value = inputDesc.dims[inputDesc.nDims-1-i];
        }
        if (value == -1) {
            value = 0;
            count ++;
        } else {
            factor *= value;
        }

        (*outputDesc).dims[shape_size-1-i] = value;
    }
    if (count > 1) {
        return NOT_SUPPORTED;
    }

    for (I32 i = 0; i < shape_size; i++) {
        if ((*outputDesc).dims[i] == 0) {
            (*outputDesc).dims[i] = tensorNumElements(inputDesc) / factor;
        }
    }

    return SUCCESS;
}
