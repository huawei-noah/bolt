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
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

inline EE slice_infer_output_size_cpu(TensorDesc inputDesc, std::vector<TensorDesc>* outputDesc, I32 axis, I32 *slice_point)
{
    if (nullptr == outputDesc)
        CHECK_STATUS(NULL_POINTER);

    U32 num = (*outputDesc).size();
    axis = (axis + inputDesc.nDims) % inputDesc.nDims;
    I32 target_axis = inputDesc.nDims - 1 - axis;
    for (U32 i = 0; i < num; i++) {
        (*outputDesc)[i] = inputDesc;

        I32 prev_point = 0;
        if (i > 0) {
            prev_point = slice_point[i-1];
        }
        I32 next_point = inputDesc.dims[target_axis];
        if (i < num - 1) {
            next_point = slice_point[i];
        }
        if (prev_point < 0) {
            prev_point = prev_point + inputDesc.dims[target_axis];
            if (prev_point < 0)
                prev_point = 0;
        }
        if (next_point < 0) {
            next_point = next_point + inputDesc.dims[target_axis];
            if (next_point < 0)
                next_point = 0;
        }
        (*outputDesc)[i].dims[target_axis] = next_point - prev_point;
    }
    return SUCCESS;
}

EE slice_infer_output_size(TensorDesc inputDesc, std::vector<TensorDesc>* outputDesc, I32 axis, I32* slice_point, Arch arch, ExtInfo_t extInfo)
{
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(slice_infer_output_size_mali(inputDesc, outputDesc, axis, slice_point, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc));
    } else {
#endif
        UNUSED(arch);
        UNUSED(extInfo);
        CHECK_STATUS(slice_infer_output_size_cpu(inputDesc, outputDesc, axis, slice_point));
#ifdef _USE_MALI    
    }    
#endif    
    return SUCCESS;
}

EE slice(TensorDesc inputDesc, void* input, int axis,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output, Arch arch, ExtInfo_t extInfo)
{
#ifndef _USE_MALI
    UNUSED(extInfo);
#endif    
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = slice_general(inputDesc, input, axis, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = slice_arm(inputDesc, input, axis, outputDesc, output);
#endif
#ifdef _USE_MALI            
    } else if (arch == MALI) {
        ret = slice_mali(extInfo->maliInfo.handle, inputDesc, (GCLMem_t)input, axis, outputDesc, output);
#endif    
    }
    return ret;
}
