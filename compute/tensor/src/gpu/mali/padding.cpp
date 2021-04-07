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
#include "gpu/mali/fp16/padding_mali_fp16.h"

EE padding_infer_output_size_mali(TensorDesc inputDesc,
    PadParamSpec padParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    U32 ow, oh;
    U32 pw, ph, pr, pb;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    pw = padParamSpec.left;
    pr = padParamSpec.right;
    ph = padParamSpec.top;
    pb = padParamSpec.bottom;
    ow = iw + pw + pr;
    oh = ih + ph + pb;
    *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);

    if (gclmemInputDesc->byteSize == 0 || gclmemInputDesc->memFormat == DF_NCHW) {
        CHECK_STATUS(infer_gclmem_desc_nchw(
            iw, ih, ic, 0, 0, ow, oh, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    } else {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            iw, ih, ic, 0, 0, ow, oh, ic, idt, idt, gclmemInputDesc, gclmemOutputDesc));
    }
    return SUCCESS;
}

EE padding_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = padding_mali_fp16(handle, inputDesc, input, padParamSpec, outputDesc, output);
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
