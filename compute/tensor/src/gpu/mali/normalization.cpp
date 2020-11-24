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
#include "types.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/normalization_mali_fp16.h"

EE normalization_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc) {
        *outputDesc = inputDesc;
    }
    if (inputDesc.df == DF_MKT) {
        DataType dt;
        U32 m, k, t;
        U32 w, h, c;
        get_nlp_mkt_val(inputDesc, &dt, &m, &k, &t);
        map_nlp_mkt_to_ncwhc4(m, k, t, &w, &h, &c);
        c = c * 4;
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            w, h, c, 0, 0, w, h, c, dt, dt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

inline EE normalization_checkpara_mali(GCLHandle_t handle,
    GCLMem_t alpha,
    GCLMem_t beta,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == alpha || nullptr == beta || nullptr == input ||
        nullptr == output) {
        return NULL_POINTER;
    }
    if (inputDesc.df != outputDesc.df || inputDesc.df != DF_MKT) {
        return NOT_SUPPORTED;
    }
    if (input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCWHC4) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE layer_normalization_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t alpha,
    GCLMem_t beta,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(
        normalization_checkpara_mali(handle, alpha, beta, inputDesc, input, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = normalization_mali_fp16(handle, alpha, beta, inputDesc, input, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
