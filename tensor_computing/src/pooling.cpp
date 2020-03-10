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
#include "cpu/general/tensor_computing_general.h"
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_MALI 
#include "gpu/mali/tensor_computing_mali.h"
#endif

inline EE pooling_infer_output_size_cpu(TensorDesc inputDesc, PoolingDesc poolingDesc, TensorDesc* outputDesc){
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    U32 strideH = poolingDesc.stride_h;
    U32 strideW = poolingDesc.stride_w;
    U32 paddingT = poolingDesc.padding_top;
    U32 paddingB = poolingDesc.padding_bottom;
    U32 paddingL = poolingDesc.padding_left;
    U32 paddingR = poolingDesc.padding_right;
    U32 kernelSizeH = poolingDesc.kernelSize_h;
    U32 kernelSizeW = poolingDesc.kernelSize_w;
    RoundMode rm = poolingDesc.rm;
    U32 oh = 0, ow = 0;
    switch (rm) {
        case CEIL: {
            oh = (U32)(ceil((double(ih + paddingT + paddingB - kernelSizeH) / strideH))) + 1;
            ow = (U32)(ceil((double(iw + paddingL + paddingR - kernelSizeW) / strideW))) + 1;
            break;
        }
        case FLOOR: {
            oh = (U32)(floor((double(ih + paddingT + paddingB - kernelSizeH) / strideH))) + 1;
            ow = (U32)(floor((double(iw + paddingL + paddingR - kernelSizeW) / strideW))) + 1;
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);
    return SUCCESS;
}
EE pooling_infer_output_size(TensorDesc inputDesc, PoolingDesc poolingDesc, TensorDesc* outputDesc, Arch arch, ExtInfo_t extInfo)
{
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(pooling_infer_output_size_mali(inputDesc, poolingDesc, outputDesc, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc));
    } else {
#endif
        UNUSED(arch);
        UNUSED(extInfo);
        CHECK_STATUS(pooling_infer_output_size_cpu(inputDesc, poolingDesc, outputDesc));
#ifdef _USE_MALI
    }
#endif
    return SUCCESS;
}

EE pooling(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, const void* scale, TensorDesc outputDesc, void* output, Arch arch, ExtInfo_t extInfo)
{
#ifndef _USE_MALI
    UNUSED(extInfo);
#endif    
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = pooling_general(inputDesc, input,
                                  poolingDesc,
                                  outputDesc, output);
            break;
        case ARM_A55:
            ret = pooling_arm(inputDesc, input,
                              poolingDesc, scale,
                              outputDesc, output);
            break;
        case ARM_A76:
            ret = pooling_arm(inputDesc, input,
                              poolingDesc, scale,
                              outputDesc, output);
            break;
        case ARM_V8:
            ret = pooling_arm(inputDesc, input,
                              poolingDesc, scale,
                              outputDesc, output);
            break;
#ifdef _USE_MALI
        case MALI:
            ret = pooling_mali(extInfo->maliInfo.handle, inputDesc, (const GCLMem_t)input, poolingDesc, scale, outputDesc, (GCLMem_t)output);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
