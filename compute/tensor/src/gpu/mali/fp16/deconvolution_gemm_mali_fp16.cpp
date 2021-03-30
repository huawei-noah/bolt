// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/deconvolution_mali_fp16.h"
#include "gpu/mali/fp16/deconvolution_gemm_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"

inline EE deconv_gemm_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    cl_mem inbuf, biasmem, outbuf, fltbuf, tmp;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasmem = bias->mem;
    outbuf = output->mem;
    tmp = tmpBuf->mem;
    U32 iw, ih, ic;
    U32 fn, fw, fh, fc, sw, sh, pw, ph;
    U32 ow, oh, oc, on;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    ph = convParamSpec.padding_top;
    pw = convParamSpec.padding_left;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    U32 ihw_str = ih_str * iw_str;
    U32 ohw_str = oh_str * ow_str;

    U32 item_w = forwardRunInfo->best_w[0];
    U32 item_c = forwardRunInfo->best_c[0];
    item_c = item_c >> 2;
    char kernelName[128];
    KernelOpt kernelOpt;
    char modeName[16];
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    switch (activationMode) {
        case ACTIVATION_RELU:
            strcpy(modeName, "relu_");
            break;
        case ACTIVATION_NULL:
            strcpy(modeName, "");
            break;
        default:
            return NOT_SUPPORTED;
    }

    if (fw == 2 && fh == 2 && sw == 2 && sh == 2) {
        if ((item_w >> 8) > 0) {
            U32 item_h = item_w >> 8;
            sprintf(kernelName, "deconv_gemm_f2s2_h_%s%d%d", modeName, item_h, item_c);
            gs[0] = ((oh + 1) / 2 + item_h - 1) / item_h;
            gs[1] = (ow + 1) / 2;
            gs[2] = (fc * fw * fh + 3) / 4 / item_c;
        } else {
            sprintf(kernelName, "deconv_gemm_f2s2_%s%d%d", modeName, item_w, item_c);
            gs[0] = (oh + 1) / 2;
            gs[1] = ((ow + 1) / 2 + item_w - 1) / item_w;
            gs[2] = (fc * fw * fh + 3) / 4 / item_c;
        }
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, oh_str,
            ohw_str, oh_off, ow_off, ow, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
    } else {
        return NOT_SUPPORTED;
        U32 th_str = ih;
        U32 tw_str = iw;
        U32 th_off = 0;
        U32 tw_off = 0;
        U32 th = ih;
        U32 tw = iw;
        U32 tc = fw * fh * fc;
        U32 thw_str = th_str * tw_str;
        if ((item_w >> 8) > 0) {
            U32 item_h = item_w >> 8;
            CHECK_STATUS(set_conv_direct_reuse_h_opt_mali(
                1, 1, 1, 1, item_h, item_c, ACTIVATION_NULL, DT_F16, kernelName, &kernelOpt));
            gs[0] = (th + item_h - 1) / item_h;
            gs[1] = tw;
            gs[2] = (tc + 3) / 4 / item_c;
        } else {
            CHECK_STATUS(set_conv_direct_opt_mali(
                1, 1, 1, 1, item_w, item_c, ACTIVATION_NULL, DT_F16, kernelName, &kernelOpt));
            gs[0] = th;
            gs[1] = (tw + item_w - 1) / item_w;
            gs[2] = (tc + 3) / 4 / item_c;
        }

        bool has_bias = false;
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, th_str,
            thw_str, th_off, tw_off, tw, tc, 1, 0, 0, gs[0], gs[1], has_bias, inbuf, fltbuf,
            biasmem, tmp));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif

        gs[0] = oh * ow * (oc + 3) / 4;
        ls[0] = 0;
        dim = 1;
        sprintf(kernelName, "col2im");
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, th, tw, tc, fw, fh, pw, ph, sw, sh, oh_str, ow_str,
            oh_off, ow_off, oh, ow, gs[0], biasmem, tmp, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        CHECK_STATUS(gcl_print_memory<F16>(handle, bias, "deconv_col2im_bias"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, output, "deconv_col2im_output"));
        handle->t_total += handle->t_execute;
#endif
    }
    return SUCCESS;
}

EE deconvolution_gemm_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes)
{
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    U32 s0 = 0;
    U32 s1 = 0;
    U32 s2 = 0;
    U32 num = 0;
    U32 byteSize;
    s0 = item_c >> 2;
    s1 = (fn + item_k - 1) / item_k;
    s2 = (fc * fw * fh + item_c - 1) / item_c;
    gclmemFilterDesc->memFormat = DF_NCHWN4C4;
    num = s1 * s1 * s2 * item_c * item_k / (item_c >> 2);
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

EE deconvolution_gemm_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem)
{
    DataType fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    U32 fwh = fw * fh;
    U32 fwhc = fwh * fc;
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    char kernelName[128];
    Kernel kernel;
    sprintf(kernelName, "deconv_gemm_trans_fltbuf_%d%d", (item_c >> 2), item_k);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel));

    CHECK_STATUS(gcl_set_kernelArgs(kernel, fw, fwh, fwhc, fc, fn, filter->mem, fltmem->mem));
    U32 gs[2] = {fwh * ((fc + 3) / 4), (fn + 3) / 4};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "deconv_gemm_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, fltmem, "deconv_gemm_filter_tran"));
#endif
    return SUCCESS;
}

EE deconvolution_gemm_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    UNUSED(outputDesc);
    UNUSED(convParamSpec);
    UNUSED(forwardRunInfo);
    U32 iw, ih;
    U32 fw, fh, fc;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, NULL, &fc, &fh, &fw);
    *bytes = iw * ih * fw * fh * fc * bytesOf(inputDesc.dt);
    return SUCCESS;
}

EE deconvolution_gemm_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(
        deconv_gemm_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, convParamSpec,
            forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
    return SUCCESS;
}
