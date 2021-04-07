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
#include "gpu/mali/fp16/reshape_mali_fp16.h"

EE reshape_infer_output_size_mali(TensorDesc inputDesc,
    ReshapeParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    I32 *dims = p.shape_dims;
    I32 shapeSize = p.shape_size;
    int inputElementNum = tensorNumElements(inputDesc);
    int outputElementNum = 1;
    for (int i = 0; i < shapeSize; i++) {
        outputElementNum *= dims[i];
    }
    int index_range = ((int)inputDesc.nDims > shapeSize) ? shapeSize : inputDesc.nDims;
    if (inputElementNum > 0 && outputElementNum > 0 && inputElementNum != outputElementNum) {
        for (int i = 0; i < index_range; i++) {
            if ((inputElementNum / (int)inputDesc.dims[inputDesc.nDims - 1 - i]) ==
                (outputElementNum / dims[i])) {
                dims[i] = inputDesc.dims[inputDesc.nDims - 1 - i];
                break;
            }
        }
    }
    *outputDesc = inputDesc;
    (*outputDesc).nDims = shapeSize;
    (*outputDesc).df = getTensorDefaultDataFormat((*outputDesc).nDims);

    U32 factor = 1;
    U32 count = 0;
    for (I32 i = 0; i < shapeSize; i++) {
        I32 value = dims[i];
        if (value == 0) {
            value = inputDesc.dims[inputDesc.nDims - 1 - i];
        }
        if (value == -1) {
            value = 0;
            count++;
        } else {
            factor *= value;
        }
        (*outputDesc).dims[shapeSize - 1 - i] = value;
    }

    if (count > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    for (I32 i = 0; i < 4; i++) {
        if (i < shapeSize) {
            if ((*outputDesc).dims[i] == 0) {
                (*outputDesc).dims[i] = tensorNumElements(inputDesc) / factor;
            }
        } else {
            (*outputDesc).dims[i] = 1;
        }
    }

    DataType idt, odt;
    U32 in, ic, ih, iw, it;
    U32 on, oc, oh, ow, ot;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw, &it);
    tensorSelectGet((*outputDesc), &odt, NULL, &on, &oc, &oh, &ow, &ot);
    if (gclmemInputDesc->memFormat == DF_NCHW || gclmemInputDesc->byteSize == 0) {
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

inline EE reshape_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    return SUCCESS;
}

EE reshape_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = reshape_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, outputDesc, gclmemInputDesc, gclmemOutputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE reshape_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(reshape_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = reshape_mali_fp16(handle, inputDesc, input, outputDesc, output, tmpbuf);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
