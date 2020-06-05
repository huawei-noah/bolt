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
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE embedding_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc, U32 inputDim, U32 numOutput, DataType dt, Arch arch, ExtInfo_t extInfo)
{
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(embedding_infer_output_size_mali(inputDesc, outputDesc, inputDim, numOutput, dt, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc));
    } else {
#endif
        UNUSED(inputDesc);
        UNUSED(outputDesc);
        UNUSED(inputDim);
        UNUSED(numOutput);
        UNUSED(arch);
        UNUSED(extInfo);
        return NOT_SUPPORTED;
#ifdef _USE_MALI    
    }    
#endif    
    return SUCCESS;
}

EE embedding(TensorDesc inputDesc, void* input, TensorDesc weightDesc, void* weight, TensorDesc outputDesc, void *output, U32 inputDim, U32 numOutput, bool transpose, DataType dt, Arch arch, ExtInfo_t extInfo)
{
    EE ret = SUCCESS;
    switch (arch) {
#ifdef _USE_MALI            
        case MALI:
            ret = embedding_mali(extInfo->maliInfo.handle, inputDesc, (GCLMem_t)input, weightDesc, (GCLMem_t)weight, outputDesc, (GCLMem_t)output, inputDim, numOutput, transpose, dt);
            break;
#endif    
        default:
            UNUSED(inputDesc);
            UNUSED(input);
            UNUSED(weightDesc);
            UNUSED(weight);
            UNUSED(outputDesc);
            UNUSED(output);
            UNUSED(inputDim);
            UNUSED(numOutput);
            UNUSED(transpose);
            UNUSED(extInfo);
            ret = NOT_SUPPORTED;
    }
    return ret;
}

