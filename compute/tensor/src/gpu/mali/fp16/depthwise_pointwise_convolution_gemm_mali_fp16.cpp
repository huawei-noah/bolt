// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"
#include "error.h"
#include "types.h"
#include "gpu/mali/fp16/depthwise_pointwise_convolution_gemm_mali_fp16.h"

inline EE depthwise_pointwise_gemm_core_mali_fp16(GCLHandle_t handle,
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

    cl_mem inbuf, dwBiasimg, pwBiasbuf, outbuf, dwFltbuf, pwFltbuf, tmp;
    inbuf = input->mem;
    dwFltbuf = dwFilter->mem;
    pwFltbuf = pwFilter->mem;
    dwBiasimg = dwBias->mem;
    pwBiasbuf = pwBias->mem;
    outbuf = output->mem;
    tmp = tmpBuf->mem;
    U32 fw, sw, pw, ph, fc;
    U32 ow, oh, oc, on;
    sw = convParamSpec.stride_w;
    ph = convParamSpec.padding_top;
    pw = convParamSpec.padding_left;
    tensorSelectGet(dwFilterDesc, NULL, NULL, NULL, &fc, NULL, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    U32 iw_str, ih_str, ihw_str, ic_str, ih_off, iw_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    iw_off -= pw;
    ih_off -= ph;
    ihw_str = iw_str * ih_str;

    U32 th_str, tw_str, th_off, tw_off, thw_str;
    U32 item_wd, item_whp, item_kp;
    item_wd = forwardRunInfo->best_w[0];
    item_whp = forwardRunInfo->best_w[1];
    item_kp = forwardRunInfo->best_k[1];
    th_str = oh;
    tw_str = ow;
    th_off = 0;
    tw_off = 0;
    thw_str = ALIGN(th_str * tw_str, item_whp);

    U32 ow_str, oh_str, ow_off, oh_off, ohw_str;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    ohw_str = oh_str * ow_str;

    U32 gs[3] = {oh, ALIGN(ow, item_wd) / item_wd, ALIGN(fc, 4) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    char kernelname[128];
    Kernel kernel;
    if (depthwiseActivationMode == ACTIVATION_NULL) {
        sprintf(kernelname, "conv_depthwise_s%d_ncwh_%d%d", sw, fw, item_wd);
    } else if (depthwiseActivationMode == ACTIVATION_RELU) {
        sprintf(kernelname, "conv_depthwise_s%d_relu_ncwh_%d%d", sw, fw, item_wd);
    } else if (depthwiseActivationMode == ACTIVATION_RELU6) {
        sprintf(kernelname, "conv_depthwise_s%d_relu6_ncwh_%d%d", sw, fw, item_wd);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, th_str, tw_str,
        thw_str, th_off, tw_off, ow, gs[0], gs[1], inbuf, dwFltbuf, dwBiasimg, tmp));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    handle->t_total += handle->t_execute;
#endif
    if (pointwiseActivationMode == ACTIVATION_NULL) {
        sprintf(kernelname, "gemm_tn_ncwhc4_%d%d", item_kp, item_whp);
    } else if (pointwiseActivationMode == ACTIVATION_RELU) {
        sprintf(kernelname, "gemm_tn_relu_ncwhc4_%d%d", item_kp, item_whp);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }

    U32 M, N, K;
    M = ALIGN(oc, item_kp);
    N = thw_str;
    K = fc;
    U32 gsp[3] = {N / item_whp, M / item_kp};
    U32 lsp[3] = {0, 0};
    U32 dimp = 2;
    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, oh, ow, oc, oh_str, ow_str, ohw_str, oh_off,
        ow_off, gsp[0], gsp[1], pwFltbuf, tmp, pwBiasbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dimp, gsp, lsp, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimp, gsp, lsp, kernelname));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE depthwise_pointwise_convolution_gemm_transform_filter_bytes_mali_fp16(TensorDesc dwFilterDesc,
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
    s1 = ALIGN(fc, item_kd) / item_kd;
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

    s0 = ALIGN(fn, item_kp);
    s1 = ALIGN(fc, item_c);
    s2 = 1;
    num = s0 * s1 * s2;
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
    gclmemPwFilterDesc->memFormat = DF_HWCN;
    gclmemPwFilterDesc->flags = CL_MEM_READ_WRITE;
    gclmemPwFilterDesc->host_ptr = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE depthwise_pointwise_convolution_gemm_transform_filter_mali_fp16(GCLHandle_t handle,
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
    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "conv_depthwise_trans_fltbuf_%d", item_kd);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, dfwh, dfc, dwFilter->mem, dwFltmem->mem));
    U32 gs[2] = {dfwh, (dfc + item_kd - 1) / item_kd};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    *dwFltmemDesc = dwFilterDesc;

    U32 fn_align = ALIGN(pfn, item_kp);
    sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d", item_c, 0);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, pfc, fn_align, pwFilter->mem, pwFltmem->mem));
    U32 gsc[3] = {1, ALIGN(pfc, item_c) / item_c, fn_align};
    U32 lsc[3] = {0, 0, 0};
    U32 dimc = 3;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimc, gsc, lsc, kernelname));
    *pwFltmemDesc = pwFilterDesc;
    return SUCCESS;
}

EE depthwise_pointwise_convolution_gemm_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
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

    U32 N;
    U32 item_wh = forwardRunInfo->best_w[1];
    N = ALIGN(oh * ow, item_wh);
    *bytes = N * ALIGN(fc, 4) * bytesOf(odt);
    return SUCCESS;
}

EE depthwise_pointwise_convolution_gemm_mali_fp16(GCLHandle_t handle,
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
    CHECK_STATUS(depthwise_pointwise_gemm_core_mali_fp16(handle, inputDesc, input, dwFilterDesc,
        pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo, dwBiasDesc, pwBiasDesc,
        dwBias, pwBias, tmpBytes, tmpBuf, outputDesc, output, depthwiseActivationMode,
        pointwiseActivationMode));

    return SUCCESS;
}
