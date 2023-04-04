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
#include "gpu/mali/fp16/roialign_mali_fp16.h"

inline EE roialign_checkpara_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDescs,
    std::vector<void *> inputs,
    RoIAlignParamSpec roiAlignParamSpec,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    for (auto &p : inputs) {
        if (p == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
    }
    U32 nDims = inputDescs[0].nDims;
    if (inputDescs[0].dims[nDims - 1] > 1 || inputDescs.size() != 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (outputDesc.dims[0] != roiAlignParamSpec.output_w ||
        outputDesc.dims[1] != roiAlignParamSpec.output_h ||
        outputDesc.dims[2] != inputDescs[0].dims[nDims - 2] ||
        outputDesc.dims[3] != inputDescs[1].dims[1]) {
        CHECK_STATUS(NOT_MATCH)
    }
    if (roiAlignParamSpec.trans_mode != COORDINATE_TRANS_HALF_PIXEL) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

EE roialign_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, GCLMemDesc gclmemInputDesc, TensorDesc outputDesc, U32 *bytes)
{
    return roialign_infer_forward_tmp_bytes_mali_fp16(inputDesc, gclmemInputDesc, outputDesc, bytes);
}

EE roialign_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDescs,
    std::vector<void *> inputs,
    RoIAlignParamSpec roiAlignParamSpec,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(roialign_checkpara_mali(
        handle, inputDescs, inputs, roiAlignParamSpec, tmpbuf, outputDesc, output));
    return roialign_mali_fp16(
        handle, inputDescs, inputs, roiAlignParamSpec, tmpbuf, outputDesc, output);
}
