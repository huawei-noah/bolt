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
#include "gpu/mali/fp16/normalization_mali_fp16.h"

EE normalization_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    *outputDesc = inputDesc;

    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);

    if (gclmemInputDesc->byteSize == 0 || gclmemInputDesc->memFormat == DF_NCHW) {
        CHECK_STATUS(infer_gclmem_desc_nchw(
            iw, ih, ic, 0, 0, iw, ih, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    } else {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            iw, ih, ic, 0, 0, iw, ih, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    }
    return SUCCESS;
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
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.df != outputDesc.df) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

EE normalization_infer_forward_tmp_bytes_mali(GCLMemDesc gclmemInputDesc, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (gclmemInputDesc.dt) {
        case DT_F16: {
            ret = normalization_infer_forward_tmp_bytes_mali_fp16(gclmemInputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE layer_normalization_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t alpha,
    GCLMem_t beta,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(
        normalization_checkpara_mali(handle, alpha, beta, inputDesc, input, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = normalization_mali_fp16(
                handle, inputDesc, input, alpha, beta, tmpbuf, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
