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
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.df != outputDesc.df) {
        return NOT_SUPPORTED;
    }
    if (inputDesc.dims[2] != outputDesc.dims[2] || inputDesc.dims[3] != outputDesc.dims[3]) {
        return NOT_SUPPORTED;
    }
    if (poolingParamSpec.padding_top >= poolingParamSpec.kernel_h) {
        return NOT_SUPPORTED;
    }
    if (poolingParamSpec.padding_bottom >= poolingParamSpec.kernel_w) {
        return NOT_SUPPORTED;
    }
    if (input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCWHC4) {
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
    DataFormat df;
    U32 iw, ih, ic, in, it;
    U32 ow, oh, oc, on, ot;
    tensorSelectGet(inputDesc, NULL, &df, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);

    cl_mem inbuf, outbuf, tmpbuf;
    inbuf = input->mem;
    outbuf = output->mem;
    tmpbuf = temp->mem;
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);

    U32 sw, sh, st, pw, ph, pt, kw, kh, kt;
    sw = poolingParamSpec.stride_w;
    sh = poolingParamSpec.stride_h;
    st = poolingParamSpec.stride_t;
    pw = poolingParamSpec.padding_left;
    ph = poolingParamSpec.padding_top;
    pt = poolingParamSpec.padding_before;
    kw = poolingParamSpec.kernel_w;
    kh = poolingParamSpec.kernel_h;
    kt = poolingParamSpec.kernel_t;

    if (df == DF_NCHW) {
        st = 1;
        pt = 0;
        kt = 1;
    }
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    char kernelname[128];
    switch (poolingParamSpec.mode) {
        case POOLING_MAX: {
            gs[0] = oh;
            gs[1] = ow;
            gs[2] = (oc + 3) / 4 * ot * on;
            if (st == 1 && pt == 0 && kt == 1) {
                sprintf(kernelname, "pooling_max");
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_off, iw_off, ih_str, iw_str, oh,
                    ow, oh_off, ow_off, oh_str, ow_str, sh, sw, ph, pw, kh, kw, inbuf, outbuf));
            } else {
                return NOT_SUPPORTED;
            }
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
            handle->t_total += handle->t_execute;
#endif
            break;
        }
        case POOLING_MEAN: {
            if (oh == 1 && ow == 1 && iw > 7) {
                sprintf(kernelname, "pooling_global_mean_w");
                gs[0] = ih;
                gs[1] = (oc + 3) / 4 * on;
                dim = 2;
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ih_str * iw_str, ih_off, iw_off, ih,
                    iw, gs[0], gs[1], inbuf, tmpbuf));
                CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
                handle->t_total += handle->t_execute;
#endif
                sprintf(kernelname, "pooling_global_mean_h");
                gs[0] = (oc + 3) / 4 * on;
                dim = 1;
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(
                    kernel, ih, oh_str, oh_str * ow_str, oh_off, ow_off, gs[0], tmpbuf, outbuf));
                CHECK_STATUS(gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#endif
            } else {
                sprintf(kernelname, "pooling_mean");
                CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_off, iw_off, ih_str, iw_str, oh,
                    ow, oh_off, ow_off, oh_str, ow_str, sh, sw, ph, pw, kh, kw, inbuf, outbuf));

                gs[0] = oh;
                gs[1] = ow;
                gs[2] = (oc + 3) / 4 * on;
                dim = 3;
                gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
                handle->t_total += handle->t_execute;
#endif
            }
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
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
    UNUSED(forwardRunInfo);
    DataType idt;
    U32 in, ic, ih, iw;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    *bytes = ih * ((ic + 3) / 4 * 4) * bytesOf(idt);
    return SUCCESS;
}
