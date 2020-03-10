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

EE split_infer_output_size(TensorDesc inputDesc, std::vector<TensorDesc>* outputDesc)
{
    if (nullptr == outputDesc)
        CHECK_STATUS(NULL_POINTER);

    for(U32 i = 0; i < (*outputDesc).size(); i++) {
        (*outputDesc)[i]  = inputDesc;
    }
    return SUCCESS;
}

EE split(TensorDesc inputDesc, void* input,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output, Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = split_general(inputDesc, input, outputDesc, output);
            break;
        case ARM_A55:
            ret = split_arm(inputDesc, input, outputDesc, output);
            break;
        case ARM_A76:
            ret = split_arm(inputDesc, input, outputDesc, output);
            break;
        case ARM_V8:
            ret = split_arm(inputDesc, input, outputDesc, output);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
