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

inline EE scale_infer_output_size_cpu(TensorDesc inputDesc, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc)
        CHECK_STATUS(NULL_POINTER);

    *outputDesc = inputDesc;
    return SUCCESS;
}

EE scale_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc, Arch arch, ExtInfo_t extInfo)
{
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(scale_infer_output_size_mali(inputDesc, outputDesc, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc));
    } else {
#endif
        UNUSED(arch);
        UNUSED(extInfo);
        CHECK_STATUS(scale_infer_output_size_cpu(inputDesc, outputDesc));
#ifdef _USE_MALI
    }
#endif
    return SUCCESS;
}

EE scale(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void* output, Arch arch, ExtInfo_t extInfo)
{
    EE ret = SUCCESS;

#ifndef _USE_MALI
    UNUSED(outputDesc);
    UNUSED(output);
    UNUSED(extInfo);
#endif    
    switch (arch) {
        case CPU_GENERAL:
            ret = scale_general(alpha, beta, inputDesc, input);
            break;
        case ARM_A55:
            ret = scale_arm(alpha, beta, inputDesc, input);
            break;
        case ARM_A76:
            ret = scale_arm(alpha, beta, inputDesc, input);
            break;
        case ARM_V8:
            ret = scale_arm(alpha, beta, inputDesc, input);
            break;
#ifdef _USE_MALI
        case MALI:
            ret = scale_mali(extInfo->maliInfo.handle, (GCLMem_t)alpha, (GCLMem_t)beta, inputDesc, (GCLMem_t)input, outputDesc, (GCLMem_t)output);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}

