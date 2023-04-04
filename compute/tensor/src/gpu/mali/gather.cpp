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
#include "gpu/mali/fp16/gather_mali_fp16.h"

inline EE gather_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc indexDesc,
    GCLMem_t index,
    GatherParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == output || nullptr == index) {
        return NULL_POINTER;
    }
    int mode = getGatherMode(p);
    if (mode == 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    } else if (mode == 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    } else {
        U32 nDims = inputDesc.nDims;
        U32 axis = (p.axis + nDims) % nDims;
        axis = nDims - 1 - axis;
        if (tensorNumElements(indexDesc) == 1 && indexDesc.df == DF_SCALAR) {
            //if (outputDesc.nDims != inputDesc.nDims - 1) {
            //    CHECK_STATUS(NOT_MATCH);
            //}
        } else {
            if (outputDesc.nDims != inputDesc.nDims + indexDesc.nDims - 1) {
                CHECK_STATUS(NOT_MATCH);
            }
            for (U32 i = 0, j = axis; i < indexDesc.nDims; i++, j++) {
                if (indexDesc.dims[i] != outputDesc.dims[j]) {
                    CHECK_STATUS(NOT_MATCH);
                }
            }
        }

        for (U32 i = 0; i < axis; i++) {
            if (inputDesc.dims[i] != outputDesc.dims[i]) {
                CHECK_STATUS(NOT_MATCH);
            }
        }

        for (U32 i = inputDesc.nDims - 1, j = outputDesc.nDims - 1; i > axis; i--, j--) {
            if (inputDesc.dims[i] != outputDesc.dims[j]) {
                CHECK_STATUS(NOT_MATCH);
            }
        }
    }
    return SUCCESS;
}

EE gather_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    TensorDesc indexDesc,
    GatherParamSpec p,
    TensorDesc outputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    return gather_infer_forward_tmp_bytes_mali_fp16(
        inputDesc, gclmemInputDesc, indexDesc, p, outputDesc, gclmemOutputDesc, bytes);
}

EE gather_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc indexDesc,
    GCLMem_t index,
    GatherParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(
        gather_checkpara_mali(handle, inputDesc, input, indexDesc, index, p, outputDesc, output));
    return gather_mali_fp16(
        handle, inputDesc, input, indexDesc, index, p, tmpbuf, outputDesc, output);
}
