// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/reshape_mali_fp16.h"

inline EE reshape_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE reshape_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf)
{
    DataType idt;
    U32 iw, ih, ic, in, it;
    U32 ow, oh, oc, on, ot;
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    U32 iDims = inputDesc.nDims;
    U32 oDims = outputDesc.nDims;
    DataFormat imf = input->desc.memFormat;
    if (iDims > 5) {  //note the val of ic and it of 5 dims
        for (U32 i = 4; i < iDims; i++) {
            in = in * inputDesc.dims[i];
        }
    }
    if (oDims > 5) {
        for (U32 i = 4; i < oDims; i++) {
            on = on * outputDesc.dims[i];
        }
    }

    bool needTransIn = false;
    bool needPadOut = false;
    if (iw != iw_str || ih != ih_str || imf == DF_NCHWC4 || input->desc.memType != GCL_MEM_BUF) {
        needTransIn = true;
    }
    if (ow != ow_str || oh != oh_str || output->desc.memType != GCL_MEM_BUF) {
        needPadOut = true;
    }
    MemTransFormType type = (imf == DF_NCHWC4) ? NCHWC4_TO_NCHW : NCHW_TO_NCHW;
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

    bool use3dMode = (iDims == 5 && input->desc.memFormat == DF_NCHWC4) ? true : false;
    if (needTransIn) {
        desc = input->desc;
        U32 str[3] = {iw, ih, ic * it * in};
        U32 off[3] = {0, 0, 0};
        MemFlags flag = CL_MEM_READ_WRITE;
        CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, idt, DF_NCHW, GCL_MEM_BUF, flag));
        tMem.desc = desc;
        if (use3dMode) {
            CHECK_STATUS(ocl_data_trans_form_3d(handle, input, &tMem, 0, 0, type));
        } else {
            CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, type));
        }
    }

    if (needPadOut) {
        desc = output->desc;
        U32 str[3] = {ow, oh, oc * ot * on};
        U32 off[3] = {0, 0, 0};
        MemFlags flag = CL_MEM_READ_WRITE;
        CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, idt, DF_NCHW, GCL_MEM_BUF, flag));
        tMem.desc = desc;
        CHECK_STATUS(ocl_data_trans_form(handle, &tMem, output, 0, 0, NCHW_TO_NCHW));
    }
    return SUCCESS;
}

EE reshape_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
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
    if (ih != ih_str || iw != iw_str || gclmemInputDesc.memFormat == DF_NCHWC4 ||
        gclmemInputDesc.memType != GCL_MEM_BUF) {
        size = tensorNumBytes(inputDesc);
    }
    if (oh != oh_str || ow != ow_str || gclmemOutputDesc.memType != GCL_MEM_BUF) {
        size = tensorNumBytes(inputDesc);
    }
    *bytes = size;
    return SUCCESS;
}

EE reshape_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf)
{
    CHECK_STATUS(reshape_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(reshape_core_mali_fp16(handle, inputDesc, input, outputDesc, output, tmpbuf));
    return SUCCESS;
}
