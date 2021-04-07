// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/power_mali_fp16.h"

EE power_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    *outputDesc = inputDesc;
    DataType idt;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    if (gclmemInputDesc->memFormat == DF_NCHW || gclmemInputDesc->byteSize == 0) {
        CHECK_STATUS(infer_gclmem_desc_nchw(
            iw, ih, ic, 0, 0, iw, ih, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    } else if (gclmemInputDesc->memFormat == DF_NCWHC4) {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            iw, ih, ic, 0, 0, iw, ih, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    } else {
        return NOT_SUPPORTED;
    }
    *gclmemOutputDesc = *gclmemInputDesc;
    return SUCCESS;
}

inline EE power_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    EE ret = SUCCESS;
    if (handle == nullptr || nullptr == input || nullptr == output) {
        ret = NULL_POINTER;
    }
    if (inputDesc.df != outputDesc.df) {
        ret = NOT_SUPPORTED;
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        ret = NOT_SUPPORTED;
    }
    if (input->desc.memFormat != DF_NCHW && input->desc.memFormat != DF_NCWHC4) {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

EE power_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    PowerParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(power_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = power_mali_fp16(handle, inputDesc, input, p, outputDesc, output);
            break;
        }
        case DT_I32: {
            ret = power_mali_fp16(handle, inputDesc, input, p, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
