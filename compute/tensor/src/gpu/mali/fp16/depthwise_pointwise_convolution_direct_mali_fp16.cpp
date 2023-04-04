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
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec)
{
    cl_mem inbuf, dwBiasimg, pwBiasimg, outbuf, dwFltbuf, pwFltbuf, tmp;
    inbuf = input->mem;
    dwFltbuf = dwFilter->mem;
    pwFltbuf = pwFilter->mem;
    dwBiasimg = dwBias->mem;
    pwBiasimg = pwBias->mem;
    outbuf = output->mem;
    tmp = tmpBuf->mem;
    DataType idt;
    U32 iw, ih, ic, in;
    U32 fw, fh, sw, sh, pw, ph, dw, dh, fc;
    U32 ow, oh, oc, on;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    ph = convParamSpec.pad_top;
    pw = convParamSpec.pad_left;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;

    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    fc = ic;
    U32 item_hd = forwardRunInfo->best_h[0];
    U32 item_hp = forwardRunInfo->best_h[1];

    U32 iw_str, ih_str, ic_str, ihw_str, in_str;
    I32 iw_off, ih_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, (U32 *)&iw_off, (U32 *)&ih_off);
    iw_off -= pw;
    ih_off -= ph;
    ihw_str = iw_str * ih_str;
    ic_str = (ic + 3) / 4;
    in_str = ic_str * ihw_str;

    U32 th_str, tw_str, tc_str, th_off, tw_off, thw_str, tn_str, t_off;
    U32 h_align = UNI_ALIGN(oh, item_hp);
    if ((item_hp >> 8) > 0 || tmpBuf->desc.memType != GCL_MEM_BUF) {
        h_align = oh;
    }
    tw_str = ow;
    th_str = h_align;
    tw_off = 0;
    th_off = 0;
    t_off = 0;
    thw_str = th_str * tw_str;
    tc_str = (ic + 3) / 4;
    tn_str = tc_str * thw_str;

    U32 ow_str, oh_str, oc_str, ow_off, oh_off, ohw_str, on_str, o_off;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    ohw_str = oh_str * ow_str;
    o_off = oh_off * ow_str + ow_off;
    oc_str = (oc + 3) / 4;
    on_str = oc_str * ohw_str;

    U32 gs[3] = {ow, (oh + item_hd - 1) / item_hd, (fc + 3) / 4 * in};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    char kernelName[128];
    KernelOpt kernelOpt;
    if (dw > 1 || dh > 1) {
        CHECK_STATUS(
            set_conv_depthwise_dila_opt_mali(fw, fh, sh, dh, item_hd, depthwiseActivationParamSpec,
                false, idt, input->desc.memType, tmpBuf->desc.memType, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, tw_str,
            thw_str, t_off, oh, ic, sw, dw, dh, in_str, tn_str, gs[0], gs[1], inbuf, dwFltbuf,
            dwBiasimg, tmp));
    } else {
        CHECK_STATUS(set_conv_depthwise_opt_mali(fw, fh, sh, item_hd, depthwiseActivationParamSpec,
            false, idt, input->desc.memType, tmpBuf->desc.memType, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, tw_str, thw_str,
                t_off, oh, ic, sw, in_str, tn_str, gs[0], gs[1], inbuf, dwFltbuf, dwBiasimg, tmp));
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//    handle->t_total += handle->t_execute;
#endif
    U32 item_kp = forwardRunInfo->best_k[1];
    item_kp = item_kp >> 2;
    U32 gsp[3] = {1, 1, 1};
    U32 lsp[3] = {0, 0, 0};
    U32 dimp = 3;
    if ((item_hp >> 8) > 0) {
        U32 item_w = item_hp >> 8;
        CHECK_STATUS(set_conv_direct_reuse_w_opt_mali(1, 1, 1, 1, item_w, item_kp, false,
            pointwiseActivationParamSpec, idt, GCL_MEM_BUF, output->desc.memType, kernelName, &kernelOpt));
        gsp[0] = (ow + item_w - 1) / item_w;
        gsp[1] = oh;
        gsp[2] = (oc + 3) / 4 * on / item_kp;
    } else {
        CHECK_STATUS(
            set_conv_direct_opt_mali(1, 1, 1, 1, item_hp, item_kp, false, pointwiseActivationParamSpec,
                idt, tmpBuf->desc.memType, output->desc.memType, kernelName, &kernelOpt));
        gsp[0] = ow;
        gsp[1] = (oh + item_hp - 1) / item_hp;
        gsp[2] = (oc + 3) / 4 * on / item_kp;
    }

    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, tw_str, thw_str, ic_str, tw_off, th_off, ow_str, ohw_str,
        o_off, oh, oc, 1, tn_str, on_str, gsp[0], gsp[1], tmp, pwFltbuf, pwBiasimg, outbuf));
    gcl_set_kernelVec(handle, kernel, dimp, gsp, lsp, kernelName);

#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimp, gsp, lsp, kernelName));
//    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline void transform_filter_desc(TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    U32 item_kd,
    U32 item_kp,
    U32 item_c,
    TensorDesc *dwFtmDesc,
    TensorDesc *pwFtmDesc)
{
    DataType fdt;
    U32 fw, fh, fc, fn;
    tensorSelectGet(dwFilterDesc, &fdt, NULL, NULL, &fc, &fh, &fw);
    tensorSelectGet(pwFilterDesc, NULL, NULL, &fn, NULL, NULL, NULL);
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = fdt;
    desc.nDims = 4;
    desc.dims[3] = 1;
    desc.dims[0] = fw * fh * item_kd;
    desc.dims[1] = (fc + item_kd - 1) / item_kd;
    desc.dims[2] = 1;
    *dwFtmDesc = desc;

    desc.dims[0] = item_kp * item_c;
    desc.dims[1] = (fc + item_c - 1) / item_c;
    desc.dims[2] = (fn + item_kp - 1) / item_kp;
    *pwFtmDesc = desc;
}

EE depthwise_pointwise_convolution_direct_transform_filter_bytes_mali_fp16(TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *dwFtmDesc,
    TensorDesc *pwFtmDesc)
{
    U32 item_kd = forwardRunInfo->best_k[0];
    U32 item_kp = forwardRunInfo->best_k[1];
    U32 item_c = forwardRunInfo->best_c[1];
    transform_filter_desc(
        dwFilterDesc, pwFilterDesc, item_kd, item_kp, item_c, dwFtmDesc, pwFtmDesc);
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
    DataType fdt;
    U32 dfw, dfh, dfc;
    U32 pfc, pfn;
    tensorSelectGet(dwFilterDesc, &fdt, NULL, NULL, &dfc, &dfh, &dfw);
    tensorSelectGet(pwFilterDesc, NULL, NULL, &pfn, &pfc, NULL, NULL);
    U32 dfwh = dfw * dfh;
    U32 item_kd = forwardRunInfo->best_k[0];
    U32 item_kp = forwardRunInfo->best_k[1];
    U32 item_c = forwardRunInfo->best_c[1];
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    CHECK_STATUS(set_conv_depthwise_trans_flt(item_kd, fdt, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, dfw, dfh, dfwh, dfc, dwFilter->mem, dwFltmem->mem));
    U32 gs[2] = {dfwh, (dfc + item_kd - 1) / item_kd};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    *dwFltmemDesc = dwFilterDesc;

    CHECK_STATUS(
        set_conv_direct_trans_flt(item_c, item_kp, false, fdt, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, 1, 1, 1, pfc, pfn, pwFilter->mem, pwFltmem->mem));
    U32 gsc[3] = {1, (pfc + item_c - 1) / item_c, (pfn + item_kp - 1) / item_kp * item_kp};
    U32 lsc[3] = {0, 0, 0};
    U32 dimc = 3;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimc, gsc, lsc, kernelName));
    transform_filter_desc(
        dwFilterDesc, pwFilterDesc, item_kd, item_kp, item_c, dwFltmemDesc, pwFltmemDesc);
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
    DataType odt;
    U32 oh, ow, ic, in;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, NULL, NULL);
    tensorSelectGet(outputDesc, &odt, NULL, NULL, NULL, &oh, &ow);
    bool useImg = check_qualcomm_device();
    if (useImg) {
        U32 width = ow;
        U32 height = oh;
        U32 depth = (ic + 3) / 4 * in;
        if (CHECK_MEET_IMAGE_LIMITS(width, height, depth)) {
            bytes[4] = width;
            bytes[5] = height;
            bytes[6] = depth;
        } else {
            useImg = false;
        }
    }
    if (!useImg) {
        U32 item_h = forwardRunInfo->best_h[1];
        U32 h_align = UNI_ALIGN(oh, item_h);
        U32 ic_align = UNI_ALIGN(ic, 4);
        bytes[0] = ow * h_align * ic_align * in * bytesOf(odt);
    }
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
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec)
{
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(depthwise_pointwise_direct_core_mali_fp16(handle, inputDesc, input, dwFilterDesc,
        pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo, dwBiasDesc, pwBiasDesc,
        dwBias, pwBias, tmpBytes, tmpBuf, outputDesc, output, depthwiseActivationParamSpec,
        pointwiseActivationParamSpec));
    return SUCCESS;
}
