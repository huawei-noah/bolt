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
#include "gpu/mali/fp16/softmax_mali_fp16.h"

EE softmax_padding_input_mali(
    TensorDesc inputDesc, TensorDesc *outputDesc, OclMemory *inputMem, OclMemory *outputMem)
{
    if (inputMem == nullptr || outputMem == nullptr || outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 pr = 0;
    if (inputDesc.df != DF_NCHWC4) {
        U32 iw, ih, ic, in;
        tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
        U32 iw_align = (iw + 3) / 4 * 4;
        if ((iw == 1 && ic == 1) || (iw == 1 && ih == 1)) {
            iw_align = 1;
        }
        pr = iw_align - iw;
    }
    inputMem->padding(0, pr, 0, 0);
    outputMem->resize(*outputDesc);
    outputMem->padding(0, pr, 0, 0);  //padding must after resize
    return SUCCESS;
}

inline EE softmax_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SoftmaxParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.df != outputDesc.df) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.dims[0] != outputDesc.dims[0]) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.dims[1] != outputDesc.dims[1]) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.dims[2] != outputDesc.dims[2]) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.dims[3] != outputDesc.dims[3]) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE softmax_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SoftmaxParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(softmax_checkpara_mali(handle, inputDesc, input, p, outputDesc, output));
    return softmax_mali_fp16(handle, inputDesc, input, tmp, p.axis, outputDesc, output);
}

EE softmax_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, GCLMemDesc gclmemInputDesc, SoftmaxParamSpec p, U32 *bytes)
{
    return softmax_infer_forward_tmp_bytes_mali_fp16(inputDesc, gclmemInputDesc, p.axis, bytes);
}
