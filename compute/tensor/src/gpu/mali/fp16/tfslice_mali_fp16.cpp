// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/tfslice_mali_fp16.h"

inline EE tfslice_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != DT_F16 && inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE tfslice_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TfSliceParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_dim(input->desc, NULL, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &on, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off));
    U32 be[8], stride[8];
    U32 nDims = inputDesc.nDims;
    for (U32 i = 0; i < 8; i++) {
        be[i] = 0;
        stride[i] = 1;
    }
    for (U32 i = 0; i < nDims; i++) {
        be[nDims - 1 - i] = (p.begin_mask[i]) ? 0 : p.begin[i];
        stride[nDims - 1 - i] = p.strides[i];
    }
    DataFormat imf = input->desc.memFormat;
    DataFormat omf = output->desc.memFormat;

    char kernelName[128];
    Kernel kernel;
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 tf_iw_str = iw_str;
    U32 tf_ih_str = ih_str;
    U32 tf_iw_off = iw_off;
    U32 tf_ih_off = ih_off;
    U32 tf_ow_str = ow_str;
    U32 tf_oh_str = oh_str;
    U32 tf_ow_off = ow_off;
    U32 tf_oh_off = oh_off;
    Mem tf_in = input->mem;
    Mem tf_out = output->mem;
    U32 sub_off = 0;

    if (imf == DF_NCWHC4) {
        tf_iw_str = iw;
        tf_ih_str = ih;
        tf_iw_off = 0;
        tf_ih_off = 0;
        U32 size = tensorNumBytes(inputDesc);
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &tf_in));
        gs[0] = ih;
        gs[1] = (iw + 3) / 4;
        gs[2] = (ic + 3) / 4;
        sprintf(kernelName, "mem_trans_ncwhc4_to_nchw");
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0, iw,
            ih, ic, iw, ih, ic, 0, 0, input->mem, tf_in));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    }

    if (omf == DF_NCWHC4) {
        tf_ow_str = ow;
        tf_oh_str = oh;
        tf_ow_off = 0;
        tf_oh_off = 0;
        U32 size = tensorNumBytes(outputDesc);
        CHECK_STATUS(gcl_create_sub_buffer(size, &sub_off, tmpbuf, &tf_out));
    }

    gs[0] = ow;
    gs[1] = oh;
    gs[2] = oc;
    sprintf(kernelName, "tfslice_nchw");
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, tf_iw_str, tf_ih_str, tf_iw_off, tf_ih_off, tf_ow_str,
        tf_oh_str, tf_ow_off, tf_oh_off, be[0], be[1], be[2], stride[0], stride[1], stride[2],
        gs[0], gs[1], tf_in, tf_out));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif

    if (omf == DF_NCWHC4) {
        gs[0] = (ow + 3) / 4;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4;
        sprintf(kernelName, "mem_trans_nchw_to_ncwhc4");
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ow, oh, 0, 0, ow_str, oh_str, ow_off, oh_off, ow,
            oh, oc, ow, oh, oc, 0, 0, tf_out, output->mem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    }
    return SUCCESS;
}

EE tfslice_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes)
{
    U32 tmpBytes = 0;
    if (gclmemInputDesc.memFormat == DF_NCWHC4) {
        tmpBytes += tensorNumBytes(inputDesc);
    }
    if (gclmemOutputDesc.memFormat == DF_NCWHC4) {
        tmpBytes += tensorNumBytes(outputDesc);
    }
    *bytes = tmpBytes;
    return SUCCESS;
}

EE tfslice_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TfSliceParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(tfslice_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(tfslice_core_mali_fp16(handle, inputDesc, input, p, tmpbuf, outputDesc, output));
    return SUCCESS;
}
