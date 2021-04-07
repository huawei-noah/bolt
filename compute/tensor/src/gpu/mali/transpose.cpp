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
#include "gpu/mali/fp16/transpose_mali_fp16.h"

EE transpose_infer_output_size_mali(TensorDesc inputDesc,
    TransposeParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 *dim = p.trans_dims;
    U32 dimTran[6] = {1, 1, 1, 1, 1, 1};
    U32 nDims = inputDesc.nDims;
    for (U32 i = 0; i < nDims; ++i) {
        dimTran[nDims - 1 - i] = inputDesc.dims[nDims - 1 - dim[i]];
    }
    *outputDesc = inputDesc;
    for (U32 i = 0; i < nDims; ++i) {
        (*outputDesc).dims[i] = dimTran[i];
    }

    DataType idt;
    DataType odt;
    U32 iw, ih, ic, in, it;
    U32 ow, oh, oc, on, ot;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(*outputDesc, &odt, NULL, &on, &oc, &oh, &ow, &ot);
    if (gclmemInputDesc->byteSize == 0 || gclmemInputDesc->memFormat == DF_NCHW) {
        CHECK_STATUS(infer_gclmem_desc_nchw_3d(
            iw, ih, ic, it, in, 0, 0, 0, 0, 0, 0, 0, idt, odt, gclmemInputDesc, NULL));
    } else {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4_3d(
            iw, ih, ic, it, in, 0, 0, 0, 0, 0, 0, 0, idt, odt, gclmemInputDesc, NULL));
    }
    CHECK_STATUS(infer_gclmem_desc_nchw_3d(
        0, 0, 0, 0, 0, 0, 0, ow, oh, oc, ot, on, idt, odt, NULL, gclmemOutputDesc));
    return SUCCESS;
}

inline EE transpose_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr) {
        return NULL_POINTER;
    }

    if (inputDesc.df != outputDesc.df) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE transpose_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = transpose_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, outputDesc, gclmemInputDesc, gclmemOutputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE transpose_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TransposeParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(transpose_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = transpose_mali_fp16(
                handle, inputDesc, input, outputDesc, output, tmpbuf, p.trans_dims);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
