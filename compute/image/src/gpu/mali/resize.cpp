// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/image_mali.h"
#include "gpu/mali/fp16/resize_mali_fp16.h"

inline EE resize_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

inline EE resize_trans_image_to_buf(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, GCLMem_t tmpbuf, GCLMem_t *inputTran)
{
    if (inputDesc.df == DF_NCHW && input->desc.memType != GCL_MEM_BUF) {
        GCLMemDesc desc;
        DataType idt;
        U32 iw, ih, ic, in;
        tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
        U32 str[3] = {iw, ih, ic * in};
        U32 off[3] = {0, 0, 0};
        MemFlags flag = CL_MEM_READ_WRITE;
        CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, idt, DF_NCHW, GCL_MEM_BUF, flag));
        tmpbuf->desc = desc;
        CHECK_STATUS(ocl_data_trans_form(handle, input, tmpbuf, 0, 0, NCHW_TO_NCHW));
        *inputTran = tmpbuf;
    }
    return SUCCESS;
}

EE resize_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t tmpbuf,
    GCLMem_t output)
{
    CHECK_STATUS(resize_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    GCLMem_t inputTran = input;
    CHECK_STATUS(resize_trans_image_to_buf(handle, inputDesc, input, tmpbuf, &inputTran));
    return resize_mali_fp16(handle, inputDesc, inputTran, p, outputDesc, output);
}
