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
#ifdef _USE_MALI 
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE softmax(TensorDesc inputDesc, const void* input, TensorDesc outputDesc, void* output, Arch arch, ExtInfo_t extInfo)
{
#ifndef _USE_MALI
    UNUSED(extInfo);
#endif    
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = softmax_general(inputDesc, input, outputDesc, output);
            break;
        case ARM_A55:
            ret = softmax_arm(inputDesc, input, outputDesc, output);
            break;
        case ARM_A76:
            ret = softmax_arm(inputDesc, input, outputDesc, output);
            break;
        case ARM_V8:
            ret = softmax_arm(inputDesc, input, outputDesc, output);
            break;
#ifdef _USE_MALI            
        case MALI:
            ret = softmax_mali(extInfo->maliInfo.handle, inputDesc, (GCLMem_t)input, outputDesc, (GCLMem_t)output);
            break;
#endif            
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}

inline EE softmax_infer_output_size_cpu(TensorDesc inputDesc, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) CHECK_STATUS(NULL_POINTER);
    *outputDesc = inputDesc;
    return SUCCESS;
}


EE softmax_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc, Arch arch, ExtInfo_t extInfo)
{
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(softmax_infer_output_size_mali(inputDesc, outputDesc, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc));
    } else {
#endif
        UNUSED(arch);
        UNUSED(extInfo);
        CHECK_STATUS(softmax_infer_output_size_cpu(inputDesc, outputDesc));
#ifdef _USE_MALI    
    }    
#endif    
    return SUCCESS;
}
