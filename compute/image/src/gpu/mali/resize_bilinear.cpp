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
#include "gpu/mali/fp16/resize_bilinear_mali_fp16.h"

EE resize_infer_output_size_mali(TensorDesc inputDesc,
    DataType paramDT,
    void *params,
    TensorDesc *outputDesc,
    U32 *outputBytes,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || outputBytes == nullptr ||
        gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    U32 oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    switch (paramDT) {
        case DT_F32: {
            F32 *scales = (F32 *)params;
            oh = ih * scales[0];
            ow = iw * scales[1];
            break;
        }
        case DT_U32: {
            U32 *len = (U32 *)params;
            oh = len[0];
            ow = len[1];
            break;
        }
        default: {
            return NOT_SUPPORTED;
        }
    }
    *outputDesc = tensor4df(idt, DF_NCHW, in, ic, oh, ow);
    *outputBytes = tensorNumBytes(*outputDesc);
    if ((idf == gclmemInputDesc->byteSize == 0 || gclmemInputDesc->memFormat == DF_NCHW) && ic <= 2) {
        CHECK_STATUS(infer_gclmem_desc_nchw(
            iw, ih, ic, 0, 0, ow, oh, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    } else {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            iw, ih, ic, 0, 0, ow, oh, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    }
    return SUCCESS;
}

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

EE resize_bilinear_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(resize_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = resize_bilinear_mali_fp16(handle, inputDesc, input, outputDesc, output);
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
