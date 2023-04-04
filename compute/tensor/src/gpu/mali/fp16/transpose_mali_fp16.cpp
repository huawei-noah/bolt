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
#include "gpu/mali/cl/kernel_option/transpose_opt.h"

inline EE transpose_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
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
    DataType idt, odt;
    U32 in, ic, ih, iw, it;
    U32 on, oc, oh, ow, ot;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, &odt, NULL, &on, &oc, &oh, &ow, &ot);
    U32 iDims = inputDesc.nDims;
    U32 oDims = outputDesc.nDims;
    if (iDims > 8 || oDims > 8) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (iDims != oDims) {
        CHECK_STATUS(NOT_MATCH);
    }
    I32 dimTran[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (U32 i = 0; i < iDims; i++) {
        dimTran[iDims - 1 - i] = iDims - 1 - dims[i];
    }
    U32 iDim[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    U32 oDim[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (U32 i = 0; i < iDims; i++) {
        iDim[i] = inputDesc.dims[i];
    }
    for (U32 i = 0; i < oDims; i++) {
        oDim[i] = outputDesc.dims[i];
    }
    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;

    DataFormat imf = input->desc.memFormat;
    GCLMemType inputMemType = input->desc.memType;
    bool use3dMode = (iDims == 5 && imf == DF_NCHWC4) ? true : false;
    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    U32 rDim = 1;
    for (U32 i = 2; i < iDims; i++) {
        rDim = rDim * iDim[i];
    }
    U32 subMemOff = 0;
    if (imf == DF_NCHWC4) {
        GCLMem tMem;
        GCLMemDesc desc;
        desc = input->desc;
        U32 str[3] = {iDim[0], iDim[1], rDim};
        U32 off[3] = {0, 0, 0};
        MemFlags flag = CL_MEM_READ_WRITE;
        CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, idt, DF_NCHW, GCL_MEM_BUF, flag));
        tMem.desc = desc;
        U32 size = tensorNumBytes(inputDesc);
        CHECK_STATUS(gcl_create_sub_buffer(size, &subMemOff, tmpbuf, &(tMem.mem)));
        if (use3dMode) {
            CHECK_STATUS(ocl_data_trans_form_3d(handle, input, &tMem, 0, 0, NCHWC4_TO_NCHW));
        } else {
            CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, NCHWC4_TO_NCHW));
        }
        iw_str = iDim[0];
        ih_str = iDim[1];
        iw_off = 0;
        ih_off = 0;
        i_off = 0;
        inbuf = tMem.mem;
        inputMemType = GCL_MEM_BUF;
    }

    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (output->desc.memType != GCL_MEM_BUF || output->desc.memFormat != DF_NCHW) {
        U32 size = tensorNumBytes(outputDesc);
        CHECK_STATUS(gcl_create_sub_buffer(size, &subMemOff, tmpbuf, &outbuf));
        ow_str = oDim[0];
        oh_str = oDim[1];
        ow_off = 0;
        oh_off = 0;
        o_off = 0;
    }
    CHECK_STATUS(
        set_transpose_opt_mali(iDims, idt, inputMemType, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    gs[0] = (iDim[0] + 3) / 4;
    gs[1] = iDim[1];
    gs[2] = rDim;
    if (iDims < 4) {
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off,
            dimTran[0], dimTran[1], dimTran[2], iDim[0], gs[0], gs[1], inbuf, outbuf));
    } else {
        switch (iDims) {
            case 4:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off,
                    o_off, dimTran[0], dimTran[1], dimTran[2], dimTran[3], iDim[2], oDim[2],
                    iDim[0], gs[0], gs[1], inbuf, outbuf));
                break;
            case 5:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off,
                    o_off, dimTran[0], dimTran[1], dimTran[2], dimTran[3], iDim[2], oDim[2],
                    dimTran[4], iDim[3], oDim[3], iDim[0], gs[0], gs[1], inbuf, outbuf));
                break;
            case 6:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off,
                    o_off, dimTran[0], dimTran[1], dimTran[2], dimTran[3], iDim[2], oDim[2],
                    dimTran[4], iDim[3], oDim[3], dimTran[5], iDim[4], oDim[4], iDim[0], gs[0],
                    gs[1], inbuf, outbuf));
                break;
            case 7:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off,
                    o_off, dimTran[0], dimTran[1], dimTran[2], dimTran[3], iDim[2], oDim[2],
                    dimTran[4], iDim[3], oDim[3], dimTran[5], iDim[4], oDim[4], dimTran[6], iDim[5],
                    oDim[5], iDim[0], gs[0], gs[1], inbuf, outbuf));
                break;
            case 8:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off,
                    o_off, dimTran[0], dimTran[1], dimTran[2], dimTran[3], iDim[2], oDim[2],
                    dimTran[4], iDim[3], oDim[3], dimTran[5], iDim[4], oDim[4], dimTran[6], iDim[5],
                    oDim[5], dimTran[7], iDim[6], oDim[6], iDim[0], gs[0], gs[1], inbuf, outbuf));
                break;
            default:
                CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    if (output->desc.memType != GCL_MEM_BUF || output->desc.memFormat != DF_NCHW) {
        GCLMem tMem;
        GCLMemDesc desc;
        desc = output->desc;
        rDim = 1;
        for (U32 i = 2; i < oDims; i++) {
            rDim = rDim * oDim[i];
        }
        U32 str[3] = {oDim[0], oDim[1], rDim};
        U32 off[3] = {0, 0, 0};
        MemFlags flag = CL_MEM_READ_WRITE;
        CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, idt, DF_NCHW, GCL_MEM_BUF, flag));
        tMem.desc = desc;
        tMem.mem = outbuf;
        MemTransFormType type = (output->desc.memFormat == DF_NCHW) ? NCHW_TO_NCHW : NCHW_TO_NCHWC4;
        CHECK_STATUS(ocl_data_trans_form(handle, &tMem, output, 0, 0, type));
    }
    return SUCCESS;
}

EE transpose_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    U32 *bytes)
{
    U32 inputSize = 0;
    U32 outputSize = 0;
    if (gclmemInputDesc->memFormat == DF_NCHWC4) {
        inputSize = tensorNumBytes(inputDesc);
        inputSize = UNI_ALIGN(inputSize, BUFFER_ALIGN_BASE);
    }
    if (gclmemOutputDesc->memType != GCL_MEM_BUF) {
        outputSize = tensorNumBytes(outputDesc);
    }
    *bytes = inputSize + outputSize;
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
