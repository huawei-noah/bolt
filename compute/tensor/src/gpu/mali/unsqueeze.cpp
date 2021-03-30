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
#include "gpu/mali/fp16/unsqueeze_mali_fp16.h"

EE unsqueeze_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, &idt, NULL, &on, &oc, &oh, &ow);
    CHECK_STATUS(infer_gclmem_desc_nchw(
        iw, ih, ic, 0, 0, ow, oh, oc, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    return SUCCESS;
}

inline EE unsqueeze_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE unsqueeze_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = unsqueeze_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, gclmemInputDesc, outputDesc, gclmemOutputDesc, bytes);
            break;
        }
        case DT_I8: {
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE unsqueeze_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(unsqueeze_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = unsqueeze_mali_fp16(handle, inputDesc, input, tmpbuf, outputDesc, output);
            break;
        }
        case DT_I8: {
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
