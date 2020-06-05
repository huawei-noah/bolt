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
#include <vector>
#include "tensor_computing.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif    
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif    
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

inline EE concat_infer_output_size_cpu(std::vector<TensorDesc> inputDesc, TensorDesc* outputDesc, I32 axis)
{
    if (inputDesc.size() < 1) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (inputDesc.size() == 1) {
        *outputDesc = inputDesc[0];
        return SUCCESS;
    }

    for (U32 i = 1; i < inputDesc.size(); i++) {
        if (inputDesc[i].nDims != 0) {
            *outputDesc = inputDesc[i];
            break;
        }
    }
    I32 dim = outputDesc->nDims;
    axis = (axis + dim) % dim;
    axis = dim - 1 - axis;
    outputDesc->dims[axis] = 0;

    for (U32 i = 0; i < inputDesc.size(); i++) {
        if (inputDesc[i].nDims == 0)
            continue;

        if (inputDesc[i].nDims != (U32)dim)
            return NOT_MATCH;

        for (I32 j = 0; j < dim; j++) {
            if (j == axis)
                outputDesc->dims[j] += inputDesc[i].dims[j];
            else {
                outputDesc->dims[j] = UNI_MAX(inputDesc[i].dims[j], outputDesc->dims[j]);
                if (inputDesc[i].dims[j] != 0 && outputDesc->dims[j] != 0 && outputDesc->dims[j] != inputDesc[i].dims[j]) {
                    return NOT_MATCH;
                }
            }
        }
    }
    return SUCCESS;
}

EE concat_infer_output_size(std::vector<TensorDesc> inputDesc, TensorDesc* outputDesc, I32 axis, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if(arch == MALI){
#ifdef _USE_MALI
        ret = concat_infer_output_size_mali(inputDesc, outputDesc, axis, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc);
#endif
    } else {
        ret = concat_infer_output_size_cpu(inputDesc, outputDesc, axis);
    }    
    return ret;
}

EE concat(std::vector<TensorDesc> inputDesc, std::vector<void*> input, void* inputScale,
          TensorDesc outputDesc, void* output, void* outputScale, I32 axis, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = concat_general(inputDesc, input,
                         outputDesc, output,
                         axis);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = concat_arm(inputDesc, input, inputScale,
                         outputDesc, output, outputScale,
                         axis);
#endif
#ifdef _USE_MALI
    } else if (arch == MALI) {
        ret = concat_mali(extInfo->maliInfo.handle, inputDesc, input, NULL, outputDesc, (GCLMem_t)output, NULL, axis);
#endif
    }
    return ret;
}
