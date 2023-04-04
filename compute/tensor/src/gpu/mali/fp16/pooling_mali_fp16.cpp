// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/pooling_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/pooling_opt.h"

inline EE pooling_checkpara_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.df != outputDesc.df) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.dims[2] != outputDesc.dims[2] || inputDesc.dims[3] != outputDesc.dims[3]) {
        return NOT_SUPPORTED;
    }
    if (poolingParamSpec.pad_top >= poolingParamSpec.kernel_h) {
        return NOT_SUPPORTED;
    }
    if (poolingParamSpec.pad_bottom >= poolingParamSpec.kernel_w) {
        return NOT_SUPPORTED;
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE pooling_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t temp)
{
    DataType dt;
    DataFormat df;
    U32 iw, ih, ic, in, it;
    U32 ow, oh, oc, on, ot;
    tensorSelectGet(inputDesc, &dt, &df, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);

    cl_mem inbuf, outbuf, tmpbuf;
    inbuf = input->mem;
    outbuf = output->mem;
    tmpbuf = temp->mem;
    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;

    U32 sw, sh, st, pw, ph, pt, kw, kh, kt;
    sw = poolingParamSpec.stride_w;
    sh = poolingParamSpec.stride_h;
    st = poolingParamSpec.stride_t;
    pw = poolingParamSpec.pad_left;
    ph = poolingParamSpec.pad_top;
    pt = poolingParamSpec.pad_before;
    kw = poolingParamSpec.kernel_w;
    kh = poolingParamSpec.kernel_h;
    kt = poolingParamSpec.kernel_t;

    if (inputDesc.nDims < 5) {
        st = 1;
        pt = 0;
        kt = 1;
    }
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim;
    char kernelName[128];
    KernelOpt kernelOpt;
    PoolingMode mode = poolingParamSpec.mode;
    if (df == DF_NCHWC4 && oh == 1 && ow == 1 && iw > 7) {
        if (ot > 1 || mode != POOLING_MEAN) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        gs[0] = iw;
        gs[1] = (oc + 3) / 4 * on;
        dim = 2;
        CHECK_STATUS(set_common_opt(
            dt, input->desc.memType, GCL_MEM_BUF, "pooling_global_mean_h", kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(
            kernel, iw_str, ih_str * iw_str, i_off, iw, ih, gs[0], gs[1], inbuf, tmpbuf));
        CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
        CHECK_STATUS(set_common_opt(dt, GCL_MEM_BUF, output->desc.memType, "pooling_global_mean_w",
            kernelName, &kernelOpt));
        gs[0] = (oc + 3) / 4 * on;
        dim = 1;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, iw, ow_str, oh_str * ow_str, o_off, gs[0], tmpbuf, outbuf));
        CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
    } else {
        if (st != 1 || pt != 0 || kt != 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        gs[0] = ow;
        gs[1] = oh;
        if (df == DF_NCHWC4) {
            gs[2] = (oc + 3) / 4 * ot * on;
        } else {
            gs[2] = oc * ot * on;
        }
        dim = 3;
        CHECK_STATUS(set_pooling_opt_mali(
            mode, dt, df, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str,
            o_off, iw, ih, ow, oh, sw, sh, pw, ph, kw, kh, (int)poolingParamSpec.count_include_pad,
            inbuf, outbuf));
        CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName));
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
    }
    return SUCCESS;
}

EE pooling_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t temp)
{
    CHECK_STATUS(
        pooling_checkpara_mali_fp16(handle, inputDesc, input, poolingParamSpec, outputDesc, output));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(pooling_core_mali_fp16(
        handle, inputDesc, input, poolingParamSpec, outputDesc, output, temp));
    return SUCCESS;
}

EE pooling_infer_forward_tmp_bytes_mali_fp16(
    TensorDesc inputDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo)
{
    DataType idt;
    U32 in, ic, ih, iw;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    *bytes = iw * ((ic + 3) / 4 * 4) * bytesOf(idt);
    return SUCCESS;
}
