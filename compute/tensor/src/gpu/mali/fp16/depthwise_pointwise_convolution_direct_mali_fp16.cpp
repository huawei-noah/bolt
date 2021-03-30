// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/depthwise_pointwise_convolution_direct_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_depthwise_opt.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"

inline EE depthwise_pointwise_direct_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    const GCLMem_t dwFilter,
    const GCLMem_t pwFilter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc dwBiasDesc,
    TensorDesc pwBiasDesc,
    const GCLMem_t dwBias,
    const GCLMem_t pwBias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode)
{
    UNUSED(inputDesc);
    UNUSED(dwBiasDesc);
    UNUSED(pwBiasDesc);
    UNUSED(tmpBytes);

    cl_mem inbuf, dwBiasimg, pwBiasimg, outbuf, dwFltbuf, pwFltbuf, tmp;
    inbuf = input->mem;
    dwFltbuf = dwFilter->mem;
    pwFltbuf = pwFilter->mem;
    dwBiasimg = dwBias->mem;
    pwBiasimg = pwBias->mem;
    outbuf = output->mem;
    tmp = tmpBuf->mem;
    U32 fw, fh, sw, sh, pw, ph, dw, dh, fc;
    U32 ow, oh, oc, on;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    ph = convParamSpec.padding_top;
    pw = convParamSpec.padding_left;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    tensorSelectGet(dwFilterDesc, NULL, NULL, NULL, &fc, &fh, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    U32 iw_str, ih_str, ihw_str, ic_str, ih_off, iw_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    iw_off -= pw;
    ih_off -= ph;
    ihw_str = iw_str * ih_str;

    U32 th_str, tw_str, th_off, tw_off, thw_str;
    U32 w_align, item_wd, item_wp;
    item_wd = forwardRunInfo->best_w[0];
    item_wp = forwardRunInfo->best_w[1];
    w_align = (ow + item_wp - 1) / item_wp * item_wp;
    th_str = oh;
    tw_str = w_align;
    th_off = 0;
    tw_off = 0;
    thw_str = th_str * tw_str;

    U32 ow_str, oh_str, ow_off, oh_off, ohw_str;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    ohw_str = oh_str * ow_str;

    U32 gs[3] = {oh, (ow + item_wd - 1) / item_wd, (fc + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    char kernelName[128];
    KernelOpt kernelOpt;
    if (dw > 1 || dh > 1) {
        CHECK_STATUS(set_conv_depthwise_dila_opt_mali(fw, fh, sw, dw, item_wd,
            depthwiseActivationMode, false, DT_F16, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, th_str,
            tw_str, thw_str, th_off, tw_off, ow, sh, dw, dh, gs[0], gs[1], inbuf, dwFltbuf,
            dwBiasimg, tmp));
    } else {
        CHECK_STATUS(set_conv_depthwise_opt_mali(
            fw, fh, sw, item_wd, depthwiseActivationMode, false, DT_F16, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, th_str,
            tw_str, thw_str, th_off, tw_off, ow, sh, gs[0], gs[1], inbuf, dwFltbuf, dwBiasimg, tmp));
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    fw = 1;
    sw = 1;
    U32 item_kp = forwardRunInfo->best_k[1];
    item_kp = item_kp >> 2;
    U32 gsp[3] = {1, 1, 1};
    U32 lsp[3] = {0, 0, 0};
    U32 dimp = 3;
    if ((item_wp >> 8) > 0) {
        U32 item_h = item_wp >> 8;
        CHECK_STATUS(set_conv_direct_reuse_h_opt_mali(
            1, 1, 1, 1, item_h, item_kp, pointwiseActivationMode, DT_F16, kernelName, &kernelOpt));
        gsp[0] = (oh + item_h - 1) / item_h;
        gsp[1] = ow;
        gsp[2] = (oc + 3) / 4 * on / item_kp;
    } else {
        CHECK_STATUS(set_conv_direct_opt_mali(1, 1, 1, sw, item_wp, item_kp,
            pointwiseActivationMode, DT_F16, kernelName, &kernelOpt));
        gsp[0] = oh;
        gsp[1] = (ow + item_wp - 1) / item_wp;
        gsp[2] = (oc + 3) / 4 * on / item_kp;
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, th_str, thw_str, ic_str, th_off, tw_off, oh_str,
        ohw_str, oh_off, ow_off, ow, oc, 1, 0, 0, gsp[0], gsp[1], tmp, pwFltbuf, pwBiasimg, outbuf));
    gcl_set_kernelVec(handle, kernel, dimp, gsp, lsp, kernelName);

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimp, gsp, lsp, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE depthwise_pointwise_convolution_direct_transform_filter_bytes_mali_fp16(TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemDwFilterDesc,
    GCLMemDesc_t gclmemPwFilterDesc,
    U32 *bytes)
{
    U32 fw, fh, fc, fn;
    tensorSelectGet(dwFilterDesc, NULL, NULL, NULL, &fc, &fh, &fw);
    tensorSelectGet(pwFilterDesc, NULL, NULL, &fn, NULL, NULL, NULL);
    U32 item_kd = forwardRunInfo->best_k[0];
    U32 item_kp = forwardRunInfo->best_k[1];
    U32 item_c = forwardRunInfo->best_c[1];
    U32 s0, s1, s2;
    U32 num, byteSize;
    s0 = fw * fh;
    s1 = (fc + item_kd - 1) / item_kd;
    s2 = 1;
    num = s0 * s1 * s2 * item_kd;
    byteSize = num * bytesOf(DT_F16);
    gclmemDwFilterDesc->stride[0] = s0;
    gclmemDwFilterDesc->stride[1] = s1;
    gclmemDwFilterDesc->stride[2] = s2;
    gclmemDwFilterDesc->offset[0] = 0;
    gclmemDwFilterDesc->offset[1] = 0;
    gclmemDwFilterDesc->offset[2] = 0;
    gclmemDwFilterDesc->num = num;
    gclmemDwFilterDesc->byteSize = byteSize;
    gclmemDwFilterDesc->memType = GCL_MEM_BUF;
    gclmemDwFilterDesc->memFormat = DF_NHWCN4;
    gclmemDwFilterDesc->flags = CL_MEM_READ_WRITE;
    gclmemDwFilterDesc->host_ptr = NULL;

    s0 = item_kp >> 2;
    s1 = (fc + item_c - 1) / item_c;
    s2 = (fn + item_kp - 1) / item_kp;
    num = s0 * s1 * s2 * item_c * item_kp / (item_kp >> 2);
    byteSize = num * bytesOf(DT_F16);
    gclmemPwFilterDesc->stride[0] = s0;
    gclmemPwFilterDesc->stride[1] = s1;
    gclmemPwFilterDesc->stride[2] = s2;
    gclmemPwFilterDesc->offset[0] = 0;
    gclmemPwFilterDesc->offset[1] = 0;
    gclmemPwFilterDesc->offset[2] = 0;
    gclmemPwFilterDesc->num = num;
    gclmemPwFilterDesc->byteSize = byteSize;
    gclmemPwFilterDesc->memType = GCL_MEM_BUF;
    gclmemPwFilterDesc->memFormat = DF_NCHWN4C4;
    gclmemPwFilterDesc->flags = CL_MEM_READ_WRITE;
    gclmemPwFilterDesc->host_ptr = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE depthwise_pointwise_convolution_direct_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    GCLMem_t dwFilter,
    GCLMem_t pwFilter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *dwFltmemDesc,
    TensorDesc *pwFltmemDesc,
    GCLMem_t dwFltmem,
    GCLMem_t pwFltmem)
{
    U32 dfw, dfh, dfc;
    U32 pfc, pfn;
    tensorSelectGet(dwFilterDesc, NULL, NULL, NULL, &dfc, &dfh, &dfw);
    tensorSelectGet(pwFilterDesc, NULL, NULL, &pfn, &pfc, NULL, NULL);
    U32 dfwh = dfw * dfh;
    U32 item_kd = forwardRunInfo->best_k[0];
    U32 item_kp = forwardRunInfo->best_k[1];
    U32 item_c = forwardRunInfo->best_c[1];
    char kernelName[128];
    Kernel kernel;
    sprintf(kernelName, "conv_depthwise_trans_fltbuf_%d", item_kd);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, dfwh, dfc, dwFilter->mem, dwFltmem->mem));
    U32 gs[2] = {dfwh, (dfc + item_kd - 1) / item_kd};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    *dwFltmemDesc = dwFilterDesc;

    sprintf(kernelName, "conv_direct_trans_fltbuf_%d%d", item_c, item_kp);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, pfc, pfn, pwFilter->mem, pwFltmem->mem));
    U32 gsc[3] = {1, (pfc + item_c - 1) / item_c, (pfn + item_kp - 1) / item_kp * item_kp};
    U32 lsc[3] = {0, 0, 0};
    U32 dimc = 3;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimc, gsc, lsc, kernelName));
    *pwFltmemDesc = pwFilterDesc;
    return SUCCESS;
}

EE depthwise_pointwise_convolution_direct_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    UNUSED(inputDesc);
    UNUSED(pwFilterDesc);
    UNUSED(convParamSpec);
    DataType odt;
    U32 oh, ow, fc;
    tensorSelectGet(dwFilterDesc, NULL, NULL, NULL, &fc, NULL, NULL);
    tensorSelectGet(outputDesc, &odt, NULL, NULL, NULL, &oh, &ow);

    U32 w_align;
    U32 item_w = forwardRunInfo->best_w[1];
    w_align = (ow + item_w - 1) / item_w * item_w;
    *bytes = oh * w_align * ((fc + 3) / 4) * 4 * bytesOf(odt);
    return SUCCESS;
}

EE depthwise_pointwise_convolution_direct_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    const GCLMem_t dwFilter,
    const GCLMem_t pwFilter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc dwBiasDesc,
    TensorDesc pwBiasDesc,
    const GCLMem_t dwBias,
    const GCLMem_t pwBias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode)
{
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(depthwise_pointwise_direct_core_mali_fp16(handle, inputDesc, input, dwFilterDesc,
        pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo, dwBiasDesc, pwBiasDesc,
        dwBias, pwBias, tmpBytes, tmpBuf, outputDesc, output, depthwiseActivationMode,
        pointwiseActivationMode));
    return SUCCESS;
}
