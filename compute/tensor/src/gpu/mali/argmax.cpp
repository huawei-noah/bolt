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
#include "gpu/mali/fp16/argmax_mali_fp16.h"

EE argmax_padding_input_mali(TensorDesc inputDesc,
    ArgMaxParamSpec p,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (outputDesc == nullptr || inputMem == nullptr || outputMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.df == DF_NCHWC4) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    int axis = p.axis;
    if (axis < 0) {
        axis += inputDesc.nDims;
    }
    U32 inDims = inputDesc.nDims;
    U32 iw = inputDesc.dims[0];
    U32 ih = (inDims > 1) ? inputDesc.dims[1] : 1;
    U32 ic = (inDims > 2) ? inputDesc.dims[2] : 1;
    U32 iw_align = (axis == 0) ? (iw + 7) / 8 * 8 : iw;
    U32 ih_align = (axis == 1) ? (ih + 7) / 8 * 8 : ih;
    U32 ic_align = (axis == 2) ? (ic + 7) / 8 * 8 : ic;
    U32 pr = iw_align - iw;
    U32 pb = ih_align - ih;
    U32 pa = ic_align - ic;
    inputMem->padding(0, pr, 0, pb, 0, pa);
    return SUCCESS;
}

inline EE argmax_checkpara_mali(GCLHandle_t handle, GCLMem_t input, GCLMem_t tmpbuf, GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr || tmpbuf == nullptr) {
        return NULL_POINTER;
    }
    if (input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCHW) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE argmax_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, ArgMaxParamSpec p, TensorDesc outputDesc, U32 *bytes)
{
    return argmax_infer_forward_tmp_bytes_mali_fp16(inputDesc, p.axis, outputDesc, bytes);
}

EE argmax_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ArgMaxParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(argmax_checkpara_mali(handle, input, tmpbuf, output));
    return argmax_mali_fp16(handle, inputDesc, input, p.axis, tmpbuf, outputDesc, output);
}
