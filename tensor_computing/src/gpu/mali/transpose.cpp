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
#include "gpu/mali/fp16/transpose_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE transpose_infer_output_size_mali(TensorDesc   inputDesc,
                                    TensorDesc*  outputDesc,
                                    U32*         dim,
                                    GCLMemDesc_t gclmemInputDesc,
                                    GCLMemDesc_t gclmemOutputDesc) {
    U32 dimTran[4] = {1, 1, 1, 1};
    U32 nDims = inputDesc.nDims;
    for(U32 i = 0; i < nDims; ++i) dimTran[nDims - 1 - i] = inputDesc.dims[nDims - 1 - dim[i]];
    if(outputDesc) {
        *outputDesc = inputDesc;
        for(U32 i = 0; i < nDims; ++i) (*outputDesc).dims[i] = dimTran[i];
    }

    if(inputDesc.df == DF_NCHW) {
        DataType   idt;
        U32 iw, ih, ic, in;
        tensorSelectGet(inputDesc,  &idt, NULL, &in, &ic, &ih, &iw);
        U32 iw_align = (iw + 3) / 4 * 4;
        CHECK_STATUS(infer_gclmem_desc_nchw(iw_align, ih, ic, 0, 0, dimTran[0], dimTran[1], dimTran[2], idt, idt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    } 
    return NOT_SUPPORTED;
}

inline EE transpose_checkpara_mali(GCLHandle_t handle,
                                   TensorDesc  inputDesc,
                                   GCLMem_t    input,
                                   TensorDesc  outputDesc,
                                   GCLMem_t    output,
                                   U32*        dim) {
    if(handle == nullptr || input == nullptr || output == nullptr || dim == nullptr) return NULL_POINTER;
    if(inputDesc.df != outputDesc.df || inputDesc.df != DF_NCHW)                            return NOT_SUPPORTED;
    if(input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCHW) return NOT_SUPPORTED;
    if(dim[0] != 0 || dim[1] != 1 || dim[2] != 3 || dim[3] != 2) return NOT_SUPPORTED;
    return SUCCESS; 
}

EE transpose_mali(GCLHandle_t handle,
                  TensorDesc  inputDesc,
                  GCLMem_t    input,
                  TensorDesc  outputDesc,
                  GCLMem_t    output,
                  U32*        dim) {
    EE ret = SUCCESS;
    CHECK_STATUS(transpose_checkpara_mali(handle, inputDesc, input, outputDesc, output, dim));
    switch(inputDesc.dt) {
        case DT_F16:{
            ret = transpose_mali_fp16(handle, inputDesc, input, outputDesc, output, dim);
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



