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
#include "gpu/mali/fp16/reduction_mali_fp16.h"
#include "tensor_computing.h"
inline EE reduction_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc maskDesc,
    GCLMem_t mask,
    ReductionParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(mask);
    UNUSED(tmp);
    UNUSED(outputDesc);
    if (handle == nullptr || input == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (tensorNumElements(maskDesc) != 0) {
        CHECK_STATUS(NOT_SUPPORTED);  //unsupport currently
    }
    if (p.axes_num > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    int axis = p.axes[0];
    if (axis < 0) {
        axis = inputDesc.nDims + axis;
    }
    axis = inputDesc.nDims - 1 - axis;
    if (axis > 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

EE reduction_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc maskDesc,
    ReductionParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    int axisTran[6];
    TensorDesc tmpDesc = inputDesc;
    for (int i = 0; i < p.axes_num; i++) {
        int axis = p.axes[i];
        if (axis < 0) {
            axis = tmpDesc.nDims + axis;
        }
        axis = tmpDesc.nDims - 1 - axis;
        axisTran[i] = axis;
        if (tensorNumElements(maskDesc) == 0) {
            tmpDesc.dims[axis] = 0;
        } else {
            int num = maskDesc.dims[1] > 1 ? maskDesc.dims[1] : 0;
            tmpDesc.dims[axis] = num;
        }
    }
    if (p.keep_dim) {
        for (U32 i = 0; i < tmpDesc.nDims; i++) {
            if (tmpDesc.dims[i] == 0) {
                tmpDesc.dims[i] = 1;
            }
        }
    } else {
        int index = 0;
        for (U32 i = 0; i < tmpDesc.nDims; i++) {
            if (tmpDesc.dims[i] != 0) {
                tmpDesc.dims[index++] = tmpDesc.dims[i];
            }
        }
        tmpDesc.nDims = index;
    }
    tmpDesc.df = getTensorDefaultDataFormat(tmpDesc.nDims);
    *outputDesc = tmpDesc;

    DataType dt;
    U32 in, ic, ih, iw, it;
    U32 on, oc, oh, ow, ot;
    tensorSelectGet(inputDesc, &dt, NULL, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(tmpDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);
    if (gclmemInputDesc->memFormat == DF_NCHW || gclmemInputDesc->byteSize == 0) {
        iw = ALIGN(iw, 4);
        CHECK_STATUS(infer_gclmem_desc_nchw_3d(
            iw, ih, ic, it, in, 0, 0, 0, 0, 0, 0, 0, dt, dt, gclmemInputDesc, NULL));
    } else {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4_3d(
            iw, ih, ic, it, in, 0, 0, 0, 0, 0, 0, 0, dt, dt, gclmemInputDesc, NULL));
    }

    if (gclmemInputDesc->memFormat == DF_NCWHC4 && p.keep_dim && axisTran[0] < 2) {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4_3d(
            0, 0, 0, 0, 0, 0, 0, ow, oh, oc, on, ot, dt, dt, NULL, gclmemOutputDesc));
    } else {
        CHECK_STATUS(infer_gclmem_desc_nchw_3d(
            0, 0, 0, 0, 0, 0, 0, ow, oh, oc, on, ot, dt, dt, NULL, gclmemOutputDesc));
    }
    return SUCCESS;
}

EE reduction_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    ReductionParamSpec p,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = reduction_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, p, outputDesc, gclmemInputDesc, gclmemOutputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE reduction_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc maskDesc,
    GCLMem_t mask,
    ReductionParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(reduction_checkpara_mali(
        handle, inputDesc, input, maskDesc, mask, p, tmp, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = reduction_mali_fp16(
                handle, inputDesc, input, maskDesc, mask, p, tmp, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
