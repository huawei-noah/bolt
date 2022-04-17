// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/resize_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/resize_opt.h"

inline EE resize_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    if (outputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE resize_bilinear_core_mali_fp16(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    F32 ratiow = (F32)iw / (F32)ow;
    F32 ratioh = (F32)ih / (F32)oh;

    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    bool useNchw = (input->desc.df == DF_NCHW) ? true : false;
    GCLMemType inputMemType = input->desc.memType;
    GCLMemType outputMemType = output->desc.memType;
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;

    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (useNchw) {
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = oc * on;
    } else {
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4 * on;
    }

    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    CHECK_STATUS(set_resize_bilinear_opt_mali(
        useNchw, DT_F16, inputMemType, outputMemType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, i_off, ow_str, oh_str, o_off, iw, ih,
        ow, oh, ratiow, ratioh, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

inline EE resize_nearest_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    bool useNchw = (input->desc.df == DF_NCHW) ? true : false;
    GCLMemType inputMemType = input->desc.memType;
    GCLMemType outputMemType = output->desc.memType;

    F32 ratiow, ratioh;
    if (p.trans_mode == COORDINATE_TRANS_ALIGN_CORNERS) {
        ratiow = (iw - 1.0) / (ow - 1.0);
        ratioh = (ih - 1.0) / (oh - 1.0);
    } else {
        ratiow = iw * 1.0 / ow;
        ratioh = ih * 1.0 / oh;
    }
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;

    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (useNchw) {
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = oc * on;
    } else {
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4 * on;
    }

    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    CHECK_STATUS(set_resize_nearest_opt_mali(
        p, useNchw, DT_F16, inputMemType, outputMemType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, i_off, ow_str, oh_str, o_off, ow, oh,
        gs[0], gs[1], ratiow, ratioh, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "resize_bilinear"));
#endif
    return SUCCESS;
}

EE resize_bilinear_mali_fp16(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    CHECK_STATUS(resize_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(resize_bilinear_core_mali_fp16(handle, inputDesc, input, outputDesc, output));
    return SUCCESS;
}

EE resize_nearest_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(resize_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(resize_nearest_core_mali_fp16(handle, inputDesc, input, p, outputDesc, output));
    return SUCCESS;
}
