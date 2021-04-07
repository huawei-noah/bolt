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
#include "gpu/mali/fp16/channel_resize_mali_fp16.h"

EE channel_resize_infer_output_size_mali(TensorDesc inputDesc,
    ChannelResizeParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_REQUIREMENT(((int)ic == p.channel_before));
    if (p.group != 1) {
        return NOT_SUPPORTED;
    }

    *outputDesc = tensor4df(idt, idf, in, p.channel_after, ih, iw);
    if (gclmemInputDesc->memFormat == DF_NCHW || gclmemInputDesc->byteSize == 0) {
        CHECK_STATUS(
            infer_gclmem_desc_nchw(iw, ih, ic, 0, 0, 0, 0, 0, idt, idt, gclmemInputDesc, NULL));
    } else {
        CHECK_STATUS(
            infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, 0, 0, 0, idt, idt, gclmemInputDesc, NULL));
    }
    CHECK_STATUS(infer_gclmem_desc_nchw(
        0, 0, 0, 0, 0, iw, ih, p.channel_after, idt, idt, NULL, gclmemOutputDesc));
    return SUCCESS;
}

inline EE channel_resize_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ChannelResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr) {
        return NULL_POINTER;
    }
    return SUCCESS;
}

EE channel_resize_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ChannelResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(channel_resize_checkpara_mali(handle, inputDesc, input, p, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = channel_resize_mali_fp16(handle, inputDesc, input, p, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
