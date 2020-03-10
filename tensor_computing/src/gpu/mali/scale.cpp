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
#include "gpu/mali/fp16/scale_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE scale_infer_output_size_mali(TensorDesc   inputDesc,
                                TensorDesc*  outputDesc,
                                GCLMemDesc_t gclmemInputDesc,
                                GCLMemDesc_t gclmemOutputDesc){
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    *outputDesc = inputDesc;

    DataType   idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc,  &idt, &idf, &in, &ic, &ih, &iw);

    if(idf == DF_NCHW) {
        U32 ih_align = (ih + 1) / 2 * 2;
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih_align, ic, 0, 0, iw, ih_align, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
        *gclmemOutputDesc = *gclmemInputDesc;//the input and output mem maybe the same
        return SUCCESS;
    } 
    return NOT_SUPPORTED;
}

inline EE scale_checkpara_mali(GCLHandle_t handle, 
                               GCLMem_t    alpha,
                               TensorDesc  inputDesc,
                               GCLMem_t    input,
                               TensorDesc  outputDesc,
                               GCLMem_t    output) {
    if(handle == nullptr || nullptr == alpha || nullptr == input || nullptr == output)        return NULL_POINTER;
    if(inputDesc.df != outputDesc.df || inputDesc.df != DF_NCHW)                              return NOT_SUPPORTED;
    if(input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCWHC4) return NOT_SUPPORTED;
    return SUCCESS; 
}

EE scale_mali(GCLHandle_t handle,
              GCLMem_t    alpha,
              GCLMem_t    beta,
              TensorDesc  inputDesc,
              GCLMem_t    input,
              TensorDesc  outputDesc,
              GCLMem_t    output) {
    EE ret = SUCCESS;
    CHECK_STATUS(scale_checkpara_mali(handle, alpha, inputDesc, input, outputDesc, output));
    switch(inputDesc.dt){
        case DT_F16:{
            ret = scale_mali_fp16(handle, alpha, beta, inputDesc, input, outputDesc, output);
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



