// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/fully_connected_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"
inline EE fully_connected_checkpara_mali(GCLHandle_t    handle,
                                         TensorDesc     inputDesc, 
                                         const GCLMem_t input,
                                         TensorDesc     filterDesc, 
                                         const GCLMem_t filter,
                                         const GCLMem_t bias,
                                         TensorDesc     outputDesc, 
                                         GCLMem_t       output){
    if(nullptr == handle || nullptr == input || nullptr == filter || nullptr == output || nullptr == bias) return NULL_POINTER;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 oc;
    CHECK_STATUS(tensorSelectGet(inputDesc,  NULL, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensorSelectGet(outputDesc, NULL, NULL, NULL, &oc, NULL, NULL));
    if(inputDesc.df  != DF_NCHW) return NOT_SUPPORTED;
    if(filterDesc.df != DF_NCHW) return NOT_SUPPORTED;
    if(input->desc.memFormat != DF_NCWHC4)    return NOT_SUPPORTED;
    if(filter->desc.memFormat != DF_NCWHN4C4) return NOT_SUPPORTED;
    if(output->desc.memFormat != DF_NCWHC4)   return NOT_SUPPORTED;
    if(in > 1)               return NOT_SUPPORTED;
    if(fw != iw) return NOT_MATCH;
    if(fh != ih) return NOT_MATCH;
    if(fc != ic) return NOT_MATCH;
    if(fn != oc) return NOT_MATCH;
    return SUCCESS; 
}
EE fully_connected_infer_output_size_mali(TensorDesc   inputDesc,
                                          TensorDesc   filterDesc,
                                          TensorDesc*  outputDesc,
                                          GCLMemDesc_t gclmemInputDesc,
                                          GCLMemDesc_t gclmemOutputDesc) {
    DataType   idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    U32 fn;
    tensorSelectGet(inputDesc,  &idt, &idf, &in,  &ic,  &ih,  &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn,  NULL, NULL, NULL);
    if(idf == DF_NCHW){
        *outputDesc = tensor4df(idt, idf, in, fn, 1, 1);
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, 1, 1, fn, idt, idt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    }
    CHECK_STATUS(NOT_SUPPORTED);
    return NOT_SUPPORTED;
}

EE fully_connected_transform_filter_bytes_mali(TensorDesc   filterDesc, 
                                               GCLMemDesc_t gclmemFilterDesc,
                                               U32*         bytes){
    EE ret = SUCCESS;
    switch(filterDesc.dt){
        case DT_F16:{
           ret = fully_connected_transform_filter_bytes_mali_fp16(filterDesc, gclmemFilterDesc, bytes);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE fully_connected_transform_filter_mali(GCLHandle_t handle,
                                         TensorDesc  filterDesc,
                                         GCLMem_t    filter,
                                         TensorDesc* fltmemDesc,
                                         GCLMem_t    fltmem){
    EE ret = SUCCESS;
    switch(filterDesc.dt){
        case DT_F16:{
           ret = fully_connected_transform_filter_mali_fp16(handle, filterDesc, filter, fltmemDesc, fltmem);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE fully_connected_infer_forward_tmp_bytes_mali(TensorDesc inputDesc, 
                                                TensorDesc filterDesc, 
                                                U32*       bytes){
    EE ret = SUCCESS;
    switch(inputDesc.dt){
        case DT_F16:{
            ret = fully_connected_infer_forward_tmp_bytes_mali_fp16(inputDesc, filterDesc, bytes);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE fully_connected_mali(GCLHandle_t    handle,
                        TensorDesc     inputDesc, 
                        const GCLMem_t input,
                        TensorDesc     filterDesc, 
                        const GCLMem_t filter,
                        TensorDesc     biasDesc, 
                        const GCLMem_t bias,
                        U32            tmpBytes, 
                        GCLMem_t       tmpBuf,
                        TensorDesc     outputDesc, 
                        GCLMem_t       output) {
    EE ret = SUCCESS;
    ret = fully_connected_checkpara_mali(handle, inputDesc, input, filterDesc, filter, bias, outputDesc, output);
    switch(inputDesc.dt){
        case DT_F16:{
            ret = fully_connected_mali_fp16(handle, inputDesc, input, filterDesc, filter, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

