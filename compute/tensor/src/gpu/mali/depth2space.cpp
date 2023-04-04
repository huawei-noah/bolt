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
#include "gpu/mali/fp16/depth2space_mali_fp16.h"

inline EE depth2space_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    Depth2SpaceParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    return SUCCESS;
}

EE depth2space_padding_input_mali(TensorDesc inputDesc,
    Depth2SpaceParamSpec p,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (outputDesc == nullptr || inputMem == nullptr || outputMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.nDims != 4) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    on = in;
    oc = ic / (p.block_size * p.block_size);
    oh = ih * p.block_size;
    ow = iw * p.block_size;
    if (ic % (p.block_size * p.block_size) != 0) {
        return NOT_MATCH;
    }
    DataFormat odf = idf;
    if ((p.block_size == 2 && oc < 4) || p.block_size != 2) {
        odf = DF_NCHW;
    }
    *outputDesc = tensor4df(idt, odf, on, oc, oh, ow);
    return SUCCESS;
}

EE depth2space_infer_tmpBuf_size_mali(
    TensorDesc inputDesc, Depth2SpaceParamSpec p, TensorDesc outputDesc, U32 *bytes)
{
    return depth2space_infer_tmpBuf_size_mali_fp16(inputDesc, p, outputDesc, bytes);
}

EE depth2space_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    Depth2SpaceParamSpec p,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(depth2space_checkpara_mali(handle, inputDesc, input, p, outputDesc, output));
    return depth2space_mali_fp16(handle, inputDesc, input, p, tmpBuf, outputDesc, output);
}
