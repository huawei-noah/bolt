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
#include "gpu/mali/fp16/topk_mali_fp16.h"

EE topk_infer_output_size_mali(TensorDesc inputDesc,
    TopKParamSpec p,
    TensorDesc *outputDesc,
    TensorDesc *outputDescIndices,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    GCLMemDesc_t gclmemOutputIndicesDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    if (outputDesc == nullptr || outputDescIndices == nullptr || gclmemInputDesc == nullptr ||
        gclmemOutputDesc == nullptr || gclmemOutputIndicesDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    int axis = p.axis;
    if (axis < 0) {
        axis += inputDesc.nDims;
    }
    axis = inputDesc.nDims - 1 - axis;
    int topk = p.topk;
    TensorDesc desc = inputDesc;
    desc.dims[axis] = topk;
    (*outputDesc) = desc;
    desc.dt = DT_I32;
    (*outputDescIndices) = desc;

    U32 iw, ih, ic;
    U32 ow, oh, oc;
    U32 inDims = inputDesc.nDims;
    U32 onDims = desc.nDims;
    iw = inputDesc.dims[0];
    ih = (inDims > 1) ? inputDesc.dims[1] : 1;
    ic = (inDims > 2) ? inputDesc.dims[2] : 1;
    ow = desc.dims[0];
    oh = (onDims > 1) ? desc.dims[1] : 1;
    oc = (onDims > 2) ? desc.dims[2] : 1;
    DataType dt = inputDesc.dt;
    CHECK_STATUS(infer_gclmem_desc_nchw(
        iw, ih, ic, 0, 0, ow, oh, oc, dt, dt, gclmemInputDesc, gclmemOutputDesc));
    CHECK_STATUS(infer_gclmem_desc_nchw(
        0, 0, 0, 0, 0, ow, oh, oc, DT_I32, DT_I32, NULL, gclmemOutputIndicesDesc));
    return SUCCESS;
}

inline EE topk_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TopKParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    TensorDesc outputIndicesDesc,
    GCLMem_t outputIndices)
{
    if (handle == nullptr || input == nullptr || output == nullptr || tmpbuf == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (outputIndicesDesc.dt != DT_I32 && outputIndicesDesc.dt != DT_U32) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 totalNum = tensorNumElements(inputDesc);
    I32 axis = p.axis;
    if (axis < 0) {
        axis += inputDesc.nDims;
    }
    axis = inputDesc.nDims - 1 - axis;
    if (inputDesc.dims[axis] != totalNum) {
        CHECK_STATUS(NOT_SUPPORTED);  //only support single axis currently
    }
    if (totalNum > 65536) {
        CHECK_STATUS(NOT_SUPPORTED);  //only support < 65536 in target axis currently
    }
    return SUCCESS;
}

EE topk_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, TopKParamSpec p, TensorDesc outputDesc, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = topk_infer_forward_tmp_bytes_mali_fp16(inputDesc, p, outputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE topk_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TopKParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    TensorDesc outputIndicesDesc,
    GCLMem_t outputIndices)
{
    EE ret = SUCCESS;
    CHECK_STATUS(topk_checkpara_mali(
        handle, inputDesc, input, p, tmpbuf, outputDesc, output, outputIndicesDesc, outputIndices));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = topk_mali_fp16(handle, inputDesc, input, p, tmpbuf, outputDesc, output,
                outputIndicesDesc, outputIndices);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
