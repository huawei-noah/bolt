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
    if (p.num_axes > 1) {
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

EE reduction_padding_input_mali(TensorDesc inputDesc,
    TensorDesc maskDesc,
    ReductionParamSpec p,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (outputDesc == nullptr || inputMem == nullptr || outputMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    int axisTran[6];
    TensorDesc tmpDesc = inputDesc;
    for (int i = 0; i < p.num_axes; i++) {
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
    if (inputDesc.df == DF_NCHWC4 && p.keep_dim && axisTran[0] < 2) {
        (*outputDesc).df = DF_NCHWC4;
    }
    if (inputDesc.df != DF_NCHWC4) {
        U32 iw = inputDesc.dims[0];
        U32 iw_align = UNI_ALIGN(iw, 4);
        U32 pr = iw_align - iw;
        inputMem->padding(0, pr, 0, 0);
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
    return reduction_infer_forward_tmp_bytes_mali_fp16(
        inputDesc, p, outputDesc, gclmemInputDesc, gclmemOutputDesc, bytes);
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
    CHECK_STATUS(reduction_checkpara_mali(
        handle, inputDesc, input, maskDesc, mask, p, tmp, outputDesc, output));
    return reduction_mali_fp16(handle, inputDesc, input, maskDesc, mask, p, tmp, outputDesc, output);
}
