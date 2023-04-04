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
#include "gpu/mali/fp16/slice_mali_fp16.h"

EE slice_padding_input_mali(TensorDesc inputDesc,
    SliceParamSpec p,
    std::vector<TensorDesc> *outputDesc,
    OclMemory *inputMem,
    std::vector<OclMemory *> outputMem)
{
    if (outputDesc == nullptr || inputMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.df == DF_NCHWC4) {
        int axis = p.axis;
        U32 target_axis = inputDesc.nDims - 1 - axis;
        bool axisAlign4 = slice_axis_c_align4(target_axis, *outputDesc);
        if (!axisAlign4) {
            for (U32 i = 0; i < outputDesc->size(); i++) {
                (*outputDesc)[i].df = DF_NCHW;
            }
        }
    }
    return SUCCESS;
}

inline EE slice_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SliceParamSpec p,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output)
{
    if (handle == nullptr || input == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    for (auto p : (*output)) {
        if (p == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
    }
    for (auto p : outputDesc) {
        if (p.df != inputDesc.df) {
            return NOT_MATCH;
        }
    }
    return SUCCESS;
}

EE slice_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    SliceParamSpec p,
    std::vector<TensorDesc> outputDesc,
    U32 *bytes)
{
    return slice_infer_forward_tmp_bytes_mali_fp16(inputDesc, gclmemInputDesc, p, outputDesc, bytes);
}

EE slice_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SliceParamSpec p,
    GCLMem_t tmpbuf,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output)
{
    CHECK_STATUS(slice_checkpara_mali(handle, inputDesc, input, p, outputDesc, output));
    return slice_mali_fp16(handle, inputDesc, input, p, tmpbuf, outputDesc, output);
}
