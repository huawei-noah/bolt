// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/fully_connected_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"

inline EE fully_connected_checkpara_mali_fp16(
    TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != filterDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_MATCH;
    }
    return SUCCESS;
}

inline U32 getNumberRow(TensorDesc inputDesc, TensorDesc filterDesc)
{
    return tensorNumElements(inputDesc) / filterDesc.dims[0];
}

inline bool gemvNeedTransInput(TensorDesc cpuDesc, GCLMemDesc desc)
{
    U32 iw, ih, ic, in;
    U32 iw_str, ih_str, iw_off, ih_off;
    DataFormat mf = desc.memFormat;
    gclmem_get_desc_dim(desc, NULL, NULL, &in, &ic, &ih, &iw);
    gclmem_get_desc_padding(desc, NULL, &iw_str, &ih_str, &iw_off, &ih_off);
    U32 num = desc.num;
    if (mf == DF_NCHW) {
        if (iw * ih * ic * in == num && iw_off == 0 && ih_off == 0) {
            return false;
        }
    } else if (mf == DF_NCWHC4) {
        if (iw_str == 1 && ih_str == 1 && iw_off == 0 && ih_off == 0) {
            return false;
        }
    }
    return true;
}

inline EE gemvTransFcInput(GCLHandle_t handle, DataFormat imf, GCLMem_t input, GCLMem_t tmpBuf)
{
    GCLMemDesc desc = input->desc;
    U32 iw, ih, ic, in;
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    gclmem_get_desc_dim(desc, NULL, NULL, &in, &ic, &ih, &iw);
    gclmem_get_desc_padding(desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    Mem inbuf = input->mem;
    Mem tmp = tmpBuf->mem;
    char kernelName[128];
    if (imf == DF_NCHW) {
        sprintf(kernelName, "mem_trans_nchw_to_nchw");
        gs[0] = (iw + 3) / 4;
        gs[1] = ih;
        gs[2] = ic;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0, iw,
            ih, ic, iw, ih, ic, 0, 0, inbuf, tmp));
    } else if (imf == DF_NCWHC4) {
        sprintf(kernelName, "mem_trans_ncwhc4_to_nchw");
        gs[0] = ih;
        gs[1] = (iw + 3) / 4;
        gs[2] = (ic + 3) / 4;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, iw, ih, 0, 0, iw,
            ih, ic, iw, ih, ic, 0, 0, inbuf, tmp));
    } else {
        return NOT_SUPPORTED;
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE fully_connected_gemv_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataFormat imf = input->desc.memFormat;
    cl_mem inbuf, fltbuf, biasbuf, outbuf, tmp;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasbuf = bias->mem;
    outbuf = output->mem;
    tmp = tmpBuf->mem;
    if (gemvNeedTransInput(inputDesc, input->desc)) {
        CHECK_STATUS(gemvTransFcInput(handle, imf, input, tmpBuf));
        inbuf = tmpBuf->mem;
    }
    U32 oh_str, ow_str, oh_off, ow_off, ohw_str;
    gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    ohw_str = oh_str * ow_str;
    U32 fc = filterDesc.dims[0];
    U32 fn = filterDesc.dims[1];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 iw_str = 1;
    U32 ih_str = 1;
    U32 iw_off = 0;
    U32 ih_off = 0;
    U32 ihw_str = 1;
    U32 ic_str = (fc + item_c - 1) / item_c;

    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    CHECK_STATUS(set_conv_direct_spe_fwhs1_opt_mali(
        1, 1, 1, 1, item_c, false, true, ACTIVATION_NULL, DT_F16, kernelName, &kernelOpt));
    U32 gs[3] = {fn, 1, 1};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, oh_str, ow_str,
        oh_off, ow_off, fn, 0, 0, gs[0], gs[1], inbuf, fltbuf, biasbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE gemmTransFcInput(
    GCLHandle_t handle, U32 item, GCLMem_t input, GCLMem_t tmpBuf, bool setKernelVec)
{
    GCLMemDesc desc = input->desc;
    U32 iw, ih, ic, in;
    U32 iw_str, ih_str, iw_off, ih_off;
    gclmem_get_desc_dim(desc, NULL, NULL, &in, &ic, &ih, &iw);
    gclmem_get_desc_padding(desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    U32 ow_str = ALIGN(ih, item);
    U32 oh_str = iw;
    U32 ow_off = 0;
    U32 oh_off = 0;
    U32 dimTran[3] = {1, 0, 2};
    U32 gs[3] = {(iw + 3) / 4, ih, ic * in};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Mem inbuf = input->mem;
    Mem outbuf = tmpBuf->mem;
    char kernelName[128];
    Kernel kernel;
    sprintf(kernelName, "transpose_nchw");
    if (setKernelVec) {
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    } else {
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel));
    }
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str, ow_off,
        oh_off, dimTran[0], dimTran[1], dimTran[2], iw, gs[0], gs[1], inbuf, outbuf));
    if (setKernelVec) {
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
    } else {
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    }
    return SUCCESS;
}

inline EE fully_connected_gemm_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 M, N, K;
    cl_mem matrixA, matrixB, biasbuf, matrixC, tmp;
    matrixB = filter->mem;
    biasbuf = bias->mem;
    matrixC = output->mem;
    tmp = tmpBuf->mem;
    U32 item_n = forwardRunInfo->best_w[0];
    U32 item_m = forwardRunInfo->best_k[0];
    CHECK_STATUS(gemmTransFcInput(handle, item_m, input, tmpBuf, true));
    matrixA = tmpBuf->mem;
    M = ALIGN(inputDesc.dims[1], item_m);
    K = filterDesc.dims[0];
    N = ALIGN(filterDesc.dims[1], item_n);

    U32 ow, oh, oc;
    U32 oh_str, ow_str, oh_off, ow_off;
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, NULL, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));

    U32 A_str = M * K;
    U32 B_str = N * K;
    U32 C_str = ow_str * oh_str;
    U32 A_off = 0;
    U32 B_off = 0;
    U32 C_off = oh_off * ow_str + ow_off;

    U32 gs[3] = {(ow + item_n - 1) / item_n, (oh + item_m - 1) / item_m, 1};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    CHECK_STATUS(set_gemm_tn_opt_mali(
        item_m, item_n, false, true, false, ACTIVATION_NULL, DT_F16, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, A_str, B_str, C_str, A_off, B_off, C_off,
        ow_str, ow, oh, oc, gs[0], gs[1], matrixA, matrixB, biasbuf, matrixC));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE fully_connected_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    U32 fc, fn;
    fc = filterDesc.dims[0];
    fn = filterDesc.dims[1];
    U32 item_w = forwardRunInfo->best_w[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    U32 s0 = 0;
    U32 s1 = 0;
    U32 s2 = 0;
    U32 num = 0;
    U32 byteSize;

    if (item_k == 0) {  //gemv
        s0 = fn;
        s1 = (fc + item_c - 1) / item_c;
        s2 = 1;
        DataFormat df = DF_CHWNC4;
        if (item_c == 8) {
            df = DF_CHWNC8;
        }
        if (item_c == 16) {
            df = DF_CHWNC16;
        }
        gclmemFilterDesc->memFormat = df;
        num = s0 * s1 * s2 * item_c;
    } else {
        s0 = ALIGN(fn, item_w);
        s1 = fc;
        s2 = 1;
        gclmemFilterDesc->memFormat = DF_NCHW;
        num = s0 * s1 * s2;
    }
    byteSize = num * bytesOf(DT_F16);
    gclmemFilterDesc->stride[0] = s0;
    gclmemFilterDesc->stride[1] = s1;
    gclmemFilterDesc->stride[2] = s2;
    gclmemFilterDesc->offset[0] = 0;
    gclmemFilterDesc->offset[1] = 0;
    gclmemFilterDesc->offset[2] = 0;
    gclmemFilterDesc->num = num;
    gclmemFilterDesc->byteSize = byteSize;
    gclmemFilterDesc->memType = GCL_MEM_BUF;
    gclmemFilterDesc->flags = CL_MEM_READ_WRITE;
    gclmemFilterDesc->host_ptr = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE fully_connected_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataType fdt;
    DataFormat fdf;
    U32 fc, fn;
    fc = filterDesc.dims[0];
    fn = filterDesc.dims[1];

    char kernelName[128];
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 item_w = forwardRunInfo->best_w[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    if (item_k == 0) {
        sprintf(kernelName, "conv_direct_trans_fltbuf_%d%d", item_c, item_k);
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, fc, fn, filter->mem, fltmem->mem));
        gs[0] = 1;
        gs[1] = (fc + item_c - 1) / item_c;
        gs[2] = fn;
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    } else {
        CHECK_STATUS(gemmTransFcInput(handle, item_w, filter, fltmem, false));
    }
    *fltmemDesc = filterDesc;
    return SUCCESS;
}

EE fully_connected_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, TensorDesc filterDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo)
{
    U32 row = getNumberRow(inputDesc, filterDesc);
    U32 size = 0;
    if (row == 1) {
        size = tensorNumBytes(inputDesc);
    } else {
        DataType dt;
        U32 iw, ih, ic, in;
        tensorSelectGet(inputDesc, &dt, NULL, &in, &ic, &ih, &iw);
        U32 max_h = ih;
        for (U32 i = 1; i <= 8; i++) {
            U32 j = ALIGN(ih, i);
            max_h = (j > max_h) ? j : max_h;
        }
        size = iw * max_h * ic * in * bytesOf(dt);
    }
    *bytes = size;
    return SUCCESS;
}

EE fully_connected_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(fully_connected_checkpara_mali_fp16(inputDesc, filterDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    U32 row = getNumberRow(inputDesc, filterDesc);
    if (row == 1) {
        CHECK_STATUS(fully_connected_gemv_mali_fp16(handle, inputDesc, input, filterDesc, filter,
            biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, forwardRunInfo));
    } else {
        CHECK_STATUS(fully_connected_gemm_mali_fp16(handle, inputDesc, input, filterDesc, filter,
            biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, forwardRunInfo));
    }
    return SUCCESS;
}
