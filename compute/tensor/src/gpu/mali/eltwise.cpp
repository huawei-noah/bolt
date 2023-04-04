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
#include "gpu/mali/fp16/eltwise_mali_fp16.h"

EE eltwise_padding_input_mali(std::vector<TensorDesc> inputDesc,
    TensorDesc *outputDesc,
    std::vector<OclMemory *> inputMems,
    OclMemory *outputMem)
{
    if (outputDesc == nullptr || outputMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 arrayDimMax = 0;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    *outputDesc = inputDesc[arrayDimMax];

    if (sameDesc) {
        bool useNCHW = true;
        for (U32 i = 0; i < inputDesc.size(); i++) {
            if (inputDesc[i].df == DF_NCHWC4) {
                useNCHW = false;
                break;
            }
        }
        if (!useNCHW) {
            (*outputDesc).df = DF_NCHWC4;
        }
    } else {
        if (inputDesc.size() > 2) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    return SUCCESS;
}

inline EE eltwise_checkpara_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec eltwiseDesc,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    for (auto it : input) {
        GCLMem_t ptr = (GCLMem_t)it;
        if (ptr == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
    }
    EltwiseMode eltwiseMode = eltwiseDesc.mode;
    U32 arrayDimMax = 0;
    bool sameDesc = eltwise_same_desc(inputDesc, &arrayDimMax);
    if (sameDesc) {
        for (auto it : input) {
            if (((GCLMem_t)(it))->desc.memFormat != output->desc.memFormat) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
        for (auto it : inputDesc) {
            U32 nDims = outputDesc.nDims;
            for (U32 i = 0; i < nDims; i++) {
                U32 dv = (i < it.nDims) ? it.dims[i] : 1;
                if (dv != outputDesc.dims[i]) {
                    CHECK_STATUS(NOT_MATCH);
                }
            }
        }
    } else {
        if (inputDesc.size() > 2) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        GCLMem_t iMaxInput = (GCLMem_t)input[arrayDimMax];
        if (iMaxInput->desc.memFormat != output->desc.memFormat) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    if (eltwiseMode != ELTWISE_MAX && eltwiseMode != ELTWISE_MIN && eltwiseMode != ELTWISE_SUM &&
        eltwiseMode != ELTWISE_SUB && eltwiseMode != ELTWISE_PROD && eltwiseMode != ELTWISE_DIV) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

EE eltwise_infer_forward_tmp_bytes_mali(
    std::vector<TensorDesc> inputDesc, std::vector<GCLMemDesc> gclmemInputDesc, U32 *bytes)
{
    return eltwise_infer_forward_tmp_bytes_mali_fp16(inputDesc, gclmemInputDesc, bytes);
}

EE eltwise_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec eltwiseDesc,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(eltwise_checkpara_mali(handle, inputDesc, input, eltwiseDesc, outputDesc, output));
    return eltwise_mali_fp16(handle, inputDesc, input, tmpbuf, outputDesc, output, eltwiseDesc);
}
