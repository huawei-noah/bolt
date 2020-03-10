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
#include "gpu/mali/fp16/concat_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE concat_infer_output_size_mali(std::vector<TensorDesc> inputDesc,
                                 TensorDesc*             outputDesc,
                                 U32                     concatDim,
                                 GCLMemDesc_t            gclmemInputDesc,
                                 GCLMemDesc_t            gclmemOutputDesc){
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    *outputDesc = inputDesc[0];
    U32 sumDimSize = 0;
    for(U32 i = 0; i < inputDesc.size(); i++) sumDimSize += inputDesc[0].dims[3 - concatDim];
    if(concatDim == 1) {
         (*outputDesc).dims[2] = sumDimSize;
         DataType   idt;
         DataFormat idf;
         U32 iw, ih, ic, in;
         sumDimSize = 0;
         for(U32 i = 0; i < inputDesc.size(); i++){
             tensorSelectGet(inputDesc[i],  &idt, &idf, &in, &ic, &ih, &iw);
             CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, iw, ih, ic, idt, idt, &gclmemInputDesc[i], gclmemOutputDesc));
             sumDimSize += (ic + 3) / 4;
         }
         U32 s0 =  gclmemOutputDesc->stride[0];
         U32 s1 =  gclmemOutputDesc->stride[1];
         U32 s2 =  sumDimSize;
         gclmemOutputDesc->stride[2] = s2;
         gclmemOutputDesc->num = s0 * s1 * s2 * 4;
         gclmemOutputDesc->byteSize = s0 * s1 * s2 * 4 * sizeof(idt);
         return SUCCESS;
    }
    return NOT_SUPPORTED; 
}

inline EE concat_checkpara_mali(GCLHandle_t             handle, 
                                std::vector<TensorDesc> inputDesc,
                                std::vector<void*>      input,
                                TensorDesc              outputDesc,
                                GCLMem_t                output,
                                U32                     concatDim) {

    if(handle == nullptr || nullptr == output) return NULL_POINTER;
    if(input.size() < 1) return NOT_MATCH;
    for(auto it : input) {
        GCLMem_t ptr = (GCLMem_t)it;
        if(ptr == nullptr) return NULL_POINTER;
        if(ptr->desc.memFormat != output->desc.memFormat) return NOT_SUPPORTED;
    }
    U32 sumDimSize = 0;
    if(concatDim != 1) return NOT_SUPPORTED;
    for(auto it : inputDesc) {
        if(it.df != outputDesc.df)           return NOT_SUPPORTED;
        if(it.dims[0] != outputDesc.dims[0]) return NOT_SUPPORTED;
        if(it.dims[1] != outputDesc.dims[1]) return NOT_SUPPORTED;
        if(it.dims[3] != outputDesc.dims[3]) return NOT_SUPPORTED;
        sumDimSize += it.dims[2];
    }
    if(sumDimSize != outputDesc.dims[2]) return NOT_MATCH;
    if(outputDesc.df != DF_NCHW)            return NOT_SUPPORTED;
    if(output->desc.memFormat != DF_NCWHC4) return NOT_SUPPORTED;
    return SUCCESS; 
}

EE concat_mali(GCLHandle_t             handle,
               std::vector<TensorDesc> inputDesc,
               std::vector<void*>      input,
               GCLMem_t                inputScale,
               TensorDesc              outputDesc,
               GCLMem_t                output,
               GCLMem_t                outputScale,
               U32                     concatDim) {
    UNUSED(inputScale);
    UNUSED(outputScale);
    EE ret = SUCCESS;
    CHECK_STATUS(concat_checkpara_mali(handle, inputDesc, input, outputDesc, output, concatDim));
    switch(inputDesc[0].dt){
        case DT_F16:{
            ret = concat_mali_fp16(handle, inputDesc, input, outputDesc, output, concatDim);
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



