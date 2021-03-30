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
#include "gpu/mali/fp16/tfslice_mali_fp16.h"

EE tfslice_infer_output_size_mali(TensorDesc inputDesc,
    TfSliceParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 nDims = inputDesc.nDims;
    TensorDesc desc = inputDesc;
    for (U32 i = 0; i < nDims; i++) {
        U32 be = (p.begin_mask[i]) ? 0 : p.begin[i];
        U32 end = (p.end_mask[i]) ? desc.dims[i] : p.end[i];
        desc.dims[nDims - 1 - i] = (end - be) / p.strides[i];
    }
    (*outputDesc) = desc;

    DataType dt;
    U32 iw, ih, ic;
    U32 ow, oh, oc;
    tensorSelectGet(inputDesc, &dt, NULL, NULL, &ic, &ih, &iw);
    tensorSelectGet(desc, NULL, NULL, NULL, &oc, &oh, &ow);

    if (gclmemInputDesc->byteSize == 0 || gclmemInputDesc->memFormat == DF_NCHW) {
        CHECK_STATUS(
            infer_gclmem_desc_nchw(iw, ih, ic, 0, 0, 0, 0, 0, dt, dt, gclmemInputDesc, NULL));
    } else if (gclmemInputDesc->memFormat == DF_NCWHC4) {
        CHECK_STATUS(
            infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, 0, 0, 0, dt, dt, gclmemInputDesc, NULL));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    CHECK_STATUS(infer_gclmem_desc_nchw(0, 0, 0, 0, 0, ow, oh, oc, dt, dt, NULL, gclmemOutputDesc));
    return SUCCESS;
}

inline EE tfslice_checkpara_mali(GCLHandle_t handle, GCLMem_t input, GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    return SUCCESS;
}

EE tfslice_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = tfslice_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, outputDesc, gclmemInputDesc, gclmemOutputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE tfslice_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TfSliceParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(tfslice_checkpara_mali(handle, input, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = tfslice_mali_fp16(handle, inputDesc, input, p, tmpbuf, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
