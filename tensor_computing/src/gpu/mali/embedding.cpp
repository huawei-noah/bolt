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
#include "gpu/mali/fp16/embedding_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE embedding_infer_output_size_mali(TensorDesc   inputDesc,
                                    TensorDesc*  outputDesc,
                                    U32          inputDim,
                                    U32          numOutput,
                                    DataType     dt,
                                    GCLMemDesc_t gclmemInputDesc,
                                    GCLMemDesc_t gclmemOutputDesc) {
    UNUSED(inputDim);
    DataType idt;
    DataFormat df;
    U32 batch, step;
    CHECK_REQUIREMENT(tensorIs2d(inputDesc));
    CHECK_STATUS(tensor2dfGet(inputDesc, &idt, &df, &batch, &step));
    if(outputDesc) *outputDesc = tensor3df(dt, DF_MKT, batch, numOutput, step);

    if(df == DF_NORMAL) {
        U32 iw = step;
        U32 ih = batch;
        U32 ic = 1;
        CHECK_STATUS(infer_gclmem_desc_nchw(iw, ih, ic, 0, 0, 0, 0, 0, idt, dt, gclmemInputDesc, NULL));

        U32 m = 1;
        U32 ow, oh, oc;
        map_nlp_mkt_to_ncwhc4(m, numOutput, step, &ow, &oh, &oc);
        /*oc has been divided 4 in map_nlp_xxx, need to mul 4 for infer_xxx_ncwhc4*/
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(0, 0, 0, 0, 0, ow, oh, oc * 4, idt, dt, NULL, gclmemOutputDesc));
        return SUCCESS;
    } 
    return NOT_SUPPORTED;
}

inline EE embedding_checkpara_mali(GCLHandle_t handle,
                                   GCLMem_t    input,
                                   GCLMem_t    weight,
                                   GCLMem_t    output) {
    if(nullptr == handle || nullptr == input || nullptr == weight || nullptr == output) return NULL_POINTER;
    return SUCCESS; 
}

EE embedding_mali(GCLHandle_t handle,
                  TensorDesc  inputDesc,
                  GCLMem_t    input,
                  TensorDesc  weightDesc,
                  GCLMem_t    weight,
                  TensorDesc  outputDesc,
                  GCLMem_t    output,
                  U32         inputDim, 
                  U32         numOutput, 
                  bool        transpose,
                  DataType    dt) {
    EE ret = SUCCESS;
    CHECK_STATUS(embedding_checkpara_mali(handle, input, weight, output));
    switch(dt) {
        case DT_F16:{
            ret = embedding_mali_fp16(handle, inputDesc, input, weightDesc, weight, outputDesc, output, inputDim, numOutput, transpose);
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



