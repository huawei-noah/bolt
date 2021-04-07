// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/unsqueeze_mali_fp16.h"

inline EE unsqueeze_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    if (outputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE unsqueeze_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    U32 iw_str, ih_str;
    U32 ow_str, oh_str;

    CHECK_STATUS(gclmem_get_desc_dim(input->desc, NULL, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &on, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, NULL, NULL));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, NULL, NULL));

    bool needTransIn = false;
    bool needPadOut = false;
    if (iw != iw_str || ih != ih_str) {
        needTransIn = true;
    }
    if (ow != ow_str || oh != oh_str) {
        needPadOut = true;
    }
    GCLMem tMem;
    GCLMemDesc desc;
    if (needPadOut) {
        if (needTransIn) {
            tMem.mem = tmpbuf->mem;
        } else {
            tMem.mem = input->mem;
        }
    } else {
        tMem.mem = output->mem;
        if (!needTransIn) {
            needTransIn = true;
        }
    }

    if (needTransIn) {
        desc = input->desc;
        desc.stride[0] = iw;
        desc.stride[1] = ih;
        desc.stride[2] = ic * in;
        desc.offset[0] = 0;
        desc.offset[1] = 0;
        desc.offset[2] = 0;
        desc.memFormat = DF_NCHW;
        tMem.desc = desc;
        CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, NCHW_TO_NCHW));
    }

    if (needPadOut) {
        desc = output->desc;
        desc.stride[0] = ow;
        desc.stride[1] = oh;
        desc.stride[2] = oc * on;
        desc.offset[0] = 0;
        desc.offset[1] = 0;
        desc.offset[2] = 0;
        tMem.desc = desc;
        CHECK_STATUS(ocl_data_trans_form(handle, &tMem, output, 0, 0, NCHW_TO_NCHW));
    }
    return SUCCESS;
}

EE unsqueeze_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    U32 iw, ih, ow, oh;
    U32 iw_str, ih_str, ow_str, oh_str;
    U32 size = 0;
    CHECK_STATUS(gclmem_get_desc_dim(gclmemInputDesc, NULL, NULL, NULL, NULL, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_dim(gclmemOutputDesc, NULL, NULL, NULL, NULL, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(gclmemInputDesc, &iw_str, &ih_str, NULL, NULL, NULL));
    CHECK_STATUS(gclmem_get_desc_padding(gclmemOutputDesc, &ow_str, &oh_str, NULL, NULL, NULL));
    if ((ih != ih_str || iw != iw_str) && (oh != oh_str || ow != ow_str)) {
        size = tensorNumBytes(inputDesc);
    }
    *bytes = size;
    return SUCCESS;
}

EE unsqueeze_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(unsqueeze_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(unsqueeze_core_mali_fp16(handle, inputDesc, input, tmpbuf, outputDesc, output));
    return SUCCESS;
}
