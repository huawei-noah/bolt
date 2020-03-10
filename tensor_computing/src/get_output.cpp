// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <cmath>
#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing.h"
#ifdef _USE_MALI 
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE tensor_computing_get_output_infer_tmpBuf_size(const void* input, TensorDesc hostDesc, U32* tmpBufSize, Arch arch)
{
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(tensor_computing_get_output_infer_tmpBuf_size_mali((const GCLMem_t)input, hostDesc, tmpBufSize));
    } else {
#endif
        UNUSED(input);
        UNUSED(hostDesc);
        UNUSED(tmpBufSize);
        UNUSED(arch);
        return NOT_SUPPORTED;
#ifdef _USE_MALI
    }
#endif
    return SUCCESS;
}

EE tensor_computing_get_output(const void* input, TensorDesc hostDesc, void** hostPtr, void* tmpBuf, bool blocking, Arch arch, ExtInfo_t extInfo)
{
#ifndef _USE_MALI
    UNUSED(input);
    UNUSED(hostDesc);
    UNUSED(hostPtr);
    UNUSED(tmpBuf);
    UNUSED(blocking);
    UNUSED(arch);
    UNUSED(extInfo);
#endif
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            return NOT_SUPPORTED;
            break;
        case ARM_A55:
            return NOT_SUPPORTED;
            break;
        case ARM_A76:
            return NOT_SUPPORTED;
            break;
#ifdef _USE_MALI
        case MALI:
            ret = tensor_computing_get_output_mali(extInfo->maliInfo.handle, (const GCLMem_t)input, hostDesc, (U8**)hostPtr, (GCLMem_t)tmpBuf, blocking);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
