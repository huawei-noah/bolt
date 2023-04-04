// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/cl/kernel_option/gemv_opt.h"
#include "gpu/mali/cl/kernel_option/transpose_opt.h"
#include "gpu/mali/fp16/gemv_mali_fp16.h"

inline bool gemvNeedTransInput(TensorDesc cpuDesc, GCLMemDesc desc)
{
    U32 iw, ih, ic, in;
    U32 iw_str, ih_str, iw_off, ih_off;
    DataFormat mf = desc.memFormat;
    gclmem_get_desc_dim(desc, NULL, NULL, &in, &ic, &ih, &iw);
    gclmem_get_desc_padding(desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    U32 num = desc.num;
    if (mf == DF_NCHW) {
        if (iw * ih * ic * in == num && iw_off == 0 && ih_off == 0) {
            return false;
        }
    } else if (mf == DF_NCHWC4) {
        if (iw_str == 1 && ih_str == 1 && iw_off == 0 && ih_off == 0) {
            return false;
        }
    }
    return true;
}

inline EE gemvTransInput(GCLHandle_t handle, GCLMem_t input, GCLMem_t inputTran)
{
    DataType idt;
    U32 iw, ih, ic, in;
    gclmem_get_desc_dim(input->desc, &idt, NULL, &in, &ic, &ih, &iw);
    GCLMemDesc desc = input->desc;
    U32 str[3] = {iw, ih, ic * in};
    U32 off[3] = {0, 0, 0};
    MemFlags flag = CL_MEM_READ_WRITE;
    CHECK_STATUS(gclmem_set_desc_padding(&desc, str, off, idt, DF_NCHW, GCL_MEM_BUF, flag));
    inputTran->desc = desc;
    MemTransFormType type = (input->desc.memFormat == DF_NCHWC4) ? NCHWC4_TO_NCHW : NCHW_TO_NCHW;
    CHECK_STATUS(ocl_data_trans_form(handle, input, inputTran, 0, 0, type));
    return SUCCESS;
}

inline EE gemv_build_run_info_core(GCLHandle_t handle,
    U32 item_c,
    U32 row,
    U32 pitch,
    ActivationParamSpec activeMode,
    bool useBias,
    bool useOutputNchwc4,
    DataType dt,
    U32 *tmpOff,
    GCLMem_t tmpBuf,
    Mem *subTmpBuf,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    bool useReduceMode = false;
    if (item_c > 16) {
        item_c = item_c >> 4;
        useReduceMode = true;
        U32 size = 32 * row * pitch * bytesOf(dt);
        CHECK_STATUS(gcl_create_sub_buffer(size, tmpOff, tmpBuf, subTmpBuf));
    }
    CHECK_STATUS(set_gemv_opt(
        item_c, activeMode, useBias, useReduceMode, useOutputNchwc4, dt, kernelName, kernelOpt));
    return SUCCESS;
}

inline EE gemv_run_core(GCLHandle_t handle,
    U32 item_c,
    U32 row,
    U32 col,
    U32 pitch,
    U32 ow_str,
    U32 oh_str,
    U32 on_str,
    U32 o_off,
    Mem vec,
    Mem mat,
    Mem bias,
    Mem tmp,
    Mem out,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    U32 gs[3] = {row, pitch, 1};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 2;
    bool useReduceMode = (item_c > 16) ? true : false;
    if (useReduceMode) {
        gs[0] = 32;
        gs[1] = row;
        gs[2] = pitch;
        ls[0] = 32;
        ls[1] = 1;
        ls[2] = 1;
        dim = 3;
    }
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, row, col, ow_str, oh_str, on_str, o_off, gs[0], gs[1], vec, mat, bias, tmp, out));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE gemv_build_run_info(GCLHandle_t handle,
    U32 item_c,
    U32 row,
    U32 pitch,
    ActivationParamSpec activeMode,
    bool useBias,
    bool useOutputNchwc4,
    DataType dt,
    U32 *tmpOff,
    GCLMem_t tmpBuf,
    Mem *subTmpBuf,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    return gemv_build_run_info_core(handle, item_c, row, pitch, activeMode, useBias,
        useOutputNchwc4, dt, tmpOff, tmpBuf, subTmpBuf, kernelName, kernelOpt);
}

EE gemv_run(GCLHandle_t handle,
    U32 item_c,
    U32 row,
    U32 col,
    U32 pitch,
    U32 ow_str,
    U32 oh_str,
    U32 on_str,
    U32 o_off,
    Mem vec,
    Mem mat,
    Mem bias,
    Mem tmp,
    Mem out,
    char *kernelName,
    KernelOpt *kernelOpt)
{
    return gemv_run_core(handle, item_c, row, col, pitch, ow_str, oh_str, on_str, o_off, vec, mat,
        bias, tmp, out, kernelName, kernelOpt);
}

EE gemv(GCLHandle_t handle,
    TensorDesc vecDesc,
    TensorDesc outputDesc,
    ActivationParamSpec activeMode,
    bool useOutputNchwc4,
    U32 *tmpOff,
    GCLMem_t tmpBuf,
    GCLMem_t vec,
    GCLMem_t bias,
    GCLMem_t mat,
    GCLMem_t out,
    ForwardRunInfoMali_t forwardRunInfo)
{
    GCLMem vecTran = *vec;
    U32 tmpOffVal = *tmpOff;
    DataType dt = vecDesc.dt;
    if (gemvNeedTransInput(vecDesc, vec->desc)) {
        U32 size = tensorNumBytes(vecDesc);
        CHECK_STATUS(gcl_create_sub_buffer(size, &tmpOffVal, tmpBuf, &vecTran.mem));
        CHECK_STATUS(gemvTransInput(handle, vec, &vecTran));
    }
    GCLMemType vmt = vec->desc.memType;
    GCLMemType omt = out->desc.memType;
    if (vmt != GCL_MEM_BUF || omt != GCL_MEM_BUF) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 item_c = forwardRunInfo->best_c[0];
    bool useBias = (bias) ? true : false;
    Mem vecMem, biasMem, matMem, tmpMem, outMem;
    vecMem = vecTran.mem;
    biasMem = (useBias) ? bias->mem : vecMem;
    tmpMem = vecMem;
    matMem = mat->mem;
    outMem = out->mem;

    U32 row, pitch, on_str;
    U32 ow_str = out->desc.stride[0];
    U32 oh_str = out->desc.stride[1];
    U32 ow_off = out->desc.offset[0];
    U32 oh_off = out->desc.offset[1];
    U32 o_off = oh_off * ow_str + ow_off;
    if (useOutputNchwc4) {
        row = outputDesc.dims[outputDesc.nDims - 2];
        pitch = outputDesc.dims[outputDesc.nDims - 1];
        on_str = ow_str * oh_str * UNI_ALIGN(row, 4);
    } else {
        row = outputDesc.dims[0];
        pitch = (outputDesc.nDims > 1) ? outputDesc.dims[1] : 1;
        on_str = ow_str;
    }
    U32 col = tensorNumElements(vecDesc) / pitch;

    char kernelName[128];
    KernelOpt kernelOpt;
    CHECK_STATUS(gemv_build_run_info_core(handle, item_c, row, pitch, activeMode, useBias,
        useOutputNchwc4, dt, &tmpOffVal, tmpBuf, &tmpMem, kernelName, &kernelOpt));
    CHECK_STATUS(gemv_run_core(handle, item_c, row, col, pitch, ow_str, oh_str, on_str, o_off,
        vecMem, matMem, biasMem, tmpMem, outMem, kernelName, &kernelOpt));
    *tmpOff = tmpOffVal;
    return SUCCESS;
}

TensorDesc gemv_transform_filter_desc(TensorDesc filterDesc, U32 item_h, U32 item_c, U32 item_k)
{
    U32 fc = filterDesc.dims[filterDesc.nDims - 2];
    U32 fn = filterDesc.dims[filterDesc.nDims - 1];
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = filterDesc.dt;
    desc.nDims = 4;
    desc.dims[3] = 1;
    desc.dims[2] = 1;
    if (item_c > 16) {
        item_c = item_c >> 4;
        desc.dims[0] = UNI_ALIGN(fc, item_c);
        desc.dims[1] = fn;
    } else {
        desc.dims[0] = fn * item_c;
        desc.dims[1] = (fc + item_c - 1) / item_c;
    }
    return desc;
}

inline EE gemv_transform_filter_core(
    GCLHandle_t handle, U32 fc, U32 fn, U32 item_c, DataType dt, Mem filter, Mem filterTran)
{
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[2];
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    bool useReduceMode = false;
    if (item_c > 16) {
        useReduceMode = true;
        item_c = item_c >> 4;
    }
    U32 fcAlign = UNI_ALIGN(fc, item_c);
    gs[0] = fcAlign >> 2;
    gs[1] = fn;
    CHECK_STATUS(set_gemv_trans_mat_opt(item_c, useReduceMode, dt, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fn, fc, filter, filterTran));
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    return SUCCESS;
}

EE gemv_transform_filter_run(
    GCLHandle_t handle, U32 fc, U32 fn, U32 item_c, DataType dt, Mem filter, Mem filterTran)
{
    return gemv_transform_filter_core(handle, fc, fn, item_c, dt, filter, filterTran);
}

EE gemv_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    DataType fdt = filterDesc.dt;
    U32 fc = filterDesc.dims[filterDesc.nDims - 2];
    U32 fn = filterDesc.dims[filterDesc.nDims - 1];
    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    CHECK_STATUS(gemv_transform_filter_core(handle, fc, fn, item_c, fdt, filter->mem, fltmem->mem));
    *fltmemDesc = gemv_transform_filter_desc(filterDesc, item_h, item_c, item_k);
    return SUCCESS;
}

EE gemv_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, TensorDesc outputDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo)
{
    U32 size = tensorNumBytes(inputDesc);
    U32 item_c = forwardRunInfo->best_c[0];
    U32 row = 1;
    U32 pitch = 1;
    for (U32 i = 0; i < outputDesc.nDims; i++) {
        if (outputDesc.dims[i] > 1) {
            row = outputDesc.dims[i];
            pitch = (i + 1 <= outputDesc.nDims - 1) ? outputDesc.dims[i + 1] : 1;
            break;
        }
    }
    if (item_c > 16) {
        size = UNI_ALIGN(size, BUFFER_ALIGN_BASE) + row * pitch * 32 * bytesOf(inputDesc.dt);
    }
    *bytes = size;
    return SUCCESS;
}
