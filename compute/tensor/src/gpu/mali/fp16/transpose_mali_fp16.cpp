// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/transpose_mali_fp16.h"

inline EE transpose_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE transpose_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf,
    U32 *dims)
{
    DataFormat df;
    U32 nDims;
    U32 in, ic, ih, iw, it;
    U32 on, oc, oh, ow, ot;
    nDims = inputDesc.nDims;
    tensorSelectGet(inputDesc, NULL, &df, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);
    DataFormat imf = input->desc.memFormat;
    DataFormat omf = output->desc.memFormat;
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    cl_mem tmp = tmpbuf->mem;
    I32 dimTran[6] = {0, 1, 2, 3, 4, 5};
    for (U32 i = 0; i < nDims; i++) {
        dimTran[nDims - 1 - i] = nDims - 1 - dims[i];
    }
    char kernelName[128];
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (dimTran[2] == 2 && dimTran[3] == 3 && nDims == 4) {
        bool matchCase = false;
        if (imf == DF_NCWHC4 && omf == DF_NCWHC4) {
            if (dimTran[0] == 0 && dimTran[1] == 1) {
                sprintf(kernelName, "mem_trans_ncwhc4_to_ncwhc4");
                gs[0] = oh;
                gs[1] = ow;
                gs[2] = (oc + 3) / 4;
                matchCase = true;
            } else if (dimTran[0] == 1 && dimTran[1] == 0) {
                sprintf(kernelName, "mem_trans_ncwhc4_to_ncwhc4_output_tran");
                gs[0] = ow;
                gs[1] = oh;
                gs[2] = (oc + 3) / 4;
                matchCase = true;
            } else {
                return NOT_SUPPORTED;
            }
        }
        if (imf == DF_NCWHC4 && omf == DF_NCHW) {
            if (dimTran[0] == 0 && dimTran[1] == 1) {
                sprintf(kernelName, "mem_trans_ncwhc4_to_nchw");
                gs[0] = oh;
                gs[1] = (ow + 3) / 4;
                gs[2] = (oc + 3) / 4;
                matchCase = true;
            } else if (dimTran[0] == 1 && dimTran[1] == 0) {
                sprintf(kernelName, "mem_trans_ncwhc4_to_nchw_output_tran");
                gs[0] = (ow + 3) / 4;
                gs[1] = oh;
                gs[2] = (oc + 3) / 4;
                matchCase = true;
            } else {
                return NOT_SUPPORTED;
            }
        }
        if (imf == DF_NCHW && omf == DF_NCWHC4) {
            if (dimTran[0] == 0 && dimTran[1] == 1) {
                sprintf(kernelName, "mem_trans_nchw_to_ncwhc4");
                gs[0] = (ow + 3) / 4;
                gs[1] = oh;
                gs[2] = (oc + 3) / 4;
                matchCase = true;
            } else if (dimTran[0] == 1 && dimTran[1] == 0) {
                sprintf(kernelName, "mem_trans_nchw_to_ncwhc4_output_tran");
                gs[0] = (oh + 3) / 4;
                gs[1] = ow;
                gs[2] = (oc + 3) / 4;
                matchCase = true;
            } else {
                return NOT_SUPPORTED;
            }
        }
        if (matchCase) {
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
                ow_off, oh_off, iw, ih, ic, ow, oh, oc, 0, 0, inbuf, outbuf));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
            return SUCCESS;
        }
    }

    if (imf == DF_NCWHC4) {
        gs[0] = ih;
        gs[1] = (iw + 3) / 4;
        gs[2] = (ic + 3) / 4 * it;
        if (nDims == 5) {
            sprintf(kernelName, "mem_trans_3d_ncwhc4_to_nchw");
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0,
                iw, ih, ic, it, iw, ih, ic, it, 0, 0, inbuf, tmp));
        } else {
            sprintf(kernelName, "mem_trans_ncwhc4_to_nchw");
            CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0,
                iw, ih, ic, iw, ih, ic, 0, 0, inbuf, tmp));
        }
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
        inbuf = tmp;
    }
    U32 ow_str_val = ow_str;
    U32 oh_str_val = oh_str;
    U32 ow_off_val = ow_off;
    U32 oh_off_val = ow_off;

    if (omf == DF_NCWHC4) {
        U32 offset = tensorNumBytes(inputDesc);
        offset = ALIGN(offset, 1024);
        U32 size = tensorNumBytes(outputDesc);
        gcl_create_sub_buffer(size, &offset, tmpbuf, &outbuf);
        ow_str_val = ow;
        oh_str_val = oh;
        ow_off_val = 0;
        oh_off_val = 0;
    }

    gs[0] = (iw + 3) / 4;
    gs[1] = ih;
    gs[2] = ic * it;
    if (nDims == 5) {
        sprintf(kernelName, "transpose_3d_nchw");
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str_val,
            oh_str_val, ow_off_val, oh_off_val, dimTran[0], dimTran[1], dimTran[2], dimTran[3], iw,
            it, ot, gs[0], gs[1], inbuf, outbuf));
    } else {
        sprintf(kernelName, "transpose_nchw");
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str_val,
            oh_str_val, ow_off_val, oh_off_val, dimTran[0], dimTran[1], dimTran[2], iw, gs[0],
            gs[1], inbuf, outbuf));
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    if (omf == DF_NCWHC4) {
        if (nDims == 5) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        sprintf(kernelName, "mem_trans_nchw_to_ncwhc4");
        gs[0] = (ow + 3) / 4;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ow_str_val, oh_str_val, ow_off_val, oh_off_val,
            ow_str, oh_str, ow_off, oh_off, ow, oh, oc, ow, oh, oc, 0, 0, outbuf, output->mem));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    }
    return SUCCESS;
}

EE transpose_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    U32 *bytes)
{
    UNUSED(inputDesc);
    UNUSED(outputDesc);
    U32 input_size = gclmemInputDesc->byteSize;
    input_size = ALIGN(input_size, 1024);
    U32 output_size = gclmemOutputDesc->byteSize;
    *bytes = input_size + output_size;
    return SUCCESS;
}

EE transpose_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t tmpbuf,
    U32 *dim)
{
    CHECK_STATUS(transpose_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(
        transpose_core_mali_fp16(handle, inputDesc, input, outputDesc, output, tmpbuf, dim));
    return SUCCESS;
}
