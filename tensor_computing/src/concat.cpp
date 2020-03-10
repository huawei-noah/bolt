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
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

// Only Support NCHW or NCHWC8 for data concat
// Here concatDim:
//     0 is batch(n)
//     1 is channel(c)
//     2 is height(h)  NOT_SUPPORTED
//     3 is width(w)   NOT_SUPPORTED
inline EE concat_infer_output_size_cpu(std::vector<TensorDesc> inputDesc, TensorDesc* outputDesc, U32 concatDim)
{
    if (inputDesc.size() < 1) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (inputDesc.size() == 1) {
        *outputDesc = inputDesc[0];
        return SUCCESS;
    }
    if (concatDim != 0 && concatDim != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    DataType idt, odt;
    DataFormat idf, odf;
    std::vector<U32> out_dim(4, 0), in_dim(4, 0);

    CHECK_STATUS(tensor4dGet(inputDesc[0], &odt, &odf, &out_dim[0], &out_dim[1], &out_dim[2], &out_dim[3]));
    if (odf != DF_NCHW && odf != DF_NCHWC8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    out_dim[concatDim] = 0;
    for (U32 i = 0; i < inputDesc.size(); i++) {
        CHECK_STATUS(tensor4dGet(inputDesc[i], &idt, &idf, &in_dim[0], &in_dim[1], &in_dim[2], &in_dim[3]));
        if (idf != odf) {
            CHECK_STATUS(NOT_MATCH);
        }
        out_dim[concatDim] += in_dim[concatDim];
    }

    *outputDesc = tensor4df(odt, odf, out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
    return SUCCESS;
}

EE concat_infer_output_size(std::vector<TensorDesc> inputDesc, TensorDesc* outputDesc, U32 concatDim, Arch arch, ExtInfo_t extInfo)
{
#ifdef _USE_MALI
    if(arch == MALI){
        CHECK_STATUS(concat_infer_output_size_mali(inputDesc, outputDesc, concatDim, extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc))
    } else {
#endif
        UNUSED(arch);
        UNUSED(extInfo);
        CHECK_STATUS(concat_infer_output_size_cpu(inputDesc, outputDesc, concatDim));
#ifdef _USE_MALI    
    }    
#endif    
    return SUCCESS;
}

EE concat(std::vector<TensorDesc> inputDesc, std::vector<void*> input, void* inputScale,
          TensorDesc outputDesc, void* output, void* outputScale, U32 concatDim, Arch arch, ExtInfo_t extInfo)
{
#ifndef _USE_MALI
    UNUSED(extInfo);
#endif    
    EE ret = SUCCESS;
    switch (arch) {
        case ARM_A55:
            ret = concat_arm(inputDesc, input, inputScale,
                             outputDesc, output, outputScale,
                             concatDim);
            break;
        case ARM_A76:
            ret = concat_arm(inputDesc, input, inputScale,
                             outputDesc, output, outputScale,
                             concatDim);
            break;
        case ARM_V8:
            ret = concat_arm(inputDesc, input, inputScale,
                             outputDesc, output, outputScale,
                             concatDim);
            break;
#ifdef _USE_MALI            
        case MALI:
            ret = concat_mali(extInfo->maliInfo.handle, inputDesc, input, NULL, outputDesc, (GCLMem_t)output, NULL, concatDim);
            break;
#endif            
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
