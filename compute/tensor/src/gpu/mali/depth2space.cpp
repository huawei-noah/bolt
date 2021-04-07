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

EE depth2space_infer_output_size_mali(TensorDesc inputDesc,
    Depth2SpaceParamSpec p,
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
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    on = in;
    oc = ic / (p.blockSize * p.blockSize);
    oh = ih * p.blockSize;
    ow = iw * p.blockSize;
    if (ic % (p.blockSize * p.blockSize) != 0) {
        return NOT_MATCH;
    }

    *outputDesc = tensor4df(idt, idf, on, oc, oh, ow);
    if (gclmemInputDesc->byteSize == 0) {
        CHECK_STATUS(infer_gclmem_desc_nchw(
            iw, ih, ic, 0, 0, 0, 0, 0, DT_F16, DT_F16, gclmemInputDesc, NULL));
    } else {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            iw, ih, ic, 0, 0, 0, 0, 0, DT_F16, DT_F16, gclmemInputDesc, NULL));
    }

    if (p.blockSize == 2 && oc < 4) {
        CHECK_STATUS(infer_gclmem_desc_nchw(
            0, 0, 0, 0, 0, ow, oh, oc, DT_F16, DT_F16, NULL, gclmemOutputDesc));
    } else {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            0, 0, 0, 0, 0, ow, oh, oc, DT_F16, DT_F16, NULL, gclmemOutputDesc));
    }
    return SUCCESS;
}

EE depth2space_infer_tmpBuf_size_mali(
    TensorDesc inputDesc, Depth2SpaceParamSpec p, TensorDesc outputDesc, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = depth2space_infer_tmpBuf_size_mali_fp16(inputDesc, p, outputDesc, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depth2space_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    Depth2SpaceParamSpec p,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(depth2space_checkpara_mali(handle, inputDesc, input, p, outputDesc, output));
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = depth2space_mali_fp16(handle, inputDesc, input, p, tmpBuf, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
