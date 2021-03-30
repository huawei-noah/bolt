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

#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/embedding_mali_fp16.h"

EE embedding_infer_output_size_mali(TensorDesc inputDesc,
    EmbedParamSpec p,
    DataType odt,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType dt;
    DataFormat df;
    U32 batch, step, nDims;
    nDims = inputDesc.nDims;
    dt = inputDesc.dt;
    if (nDims == 1) {
        batch = 1;
        step = inputDesc.dims[0];
        *outputDesc = tensor2df(odt, DF_NORMAL, step, p.num_output);
    } else if (nDims == 2) {
        batch = inputDesc.dims[1];
        step = inputDesc.dims[0];
        *outputDesc = tensor3df(odt, DF_MTK, batch, step, p.num_output);
    } else {
        return NOT_SUPPORTED;
    }
    CHECK_STATUS(infer_gclmem_desc_nchw(step, batch, 1, 0, 0, p.num_output, step, batch, dt, dt,
        gclmemInputDesc, gclmemOutputDesc));
    return SUCCESS;
}

inline EE embedding_checkpara_mali(
    GCLHandle_t handle, GCLMem_t input, GCLMem_t weight, GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == weight || nullptr == output) {
        return NULL_POINTER;
    }
    return SUCCESS;
}

EE embedding_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc weightDesc,
    GCLMem_t weight,
    EmbedParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(embedding_checkpara_mali(handle, input, weight, output));
    switch (outputDesc.dt) {
        case DT_F16: {
            ret = embedding_mali_fp16(
                handle, inputDesc, input, weightDesc, weight, p, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
