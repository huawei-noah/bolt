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
#include "gpu/mali/cl/kernel_option/deconv_opt.h"
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
    ActivationParamSpec activationMode)
{
    cl_mem inbuf, biasmem, outbuf, fltbuf, tmp;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasmem = bias->mem;
    outbuf = output->mem;
    tmp = tmpBuf->mem;
    DataType dt;
    U32 iw, ih, ic;
    U32 fn, fw, fh, fc, sw, sh, pw, ph;
    U32 ow, oh, oc, on;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    ph = convParamSpec.pad_top;
    pw = convParamSpec.pad_left;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;
    tensorSelectGet(inputDesc, &dt, NULL, NULL, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    fc = oc;
    fn = ic;

    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    U32 ihw_str = ih_str * iw_str;
    U32 ohw_str = oh_str * ow_str;
    U32 o_off = oh_off * ow_str + ow_off;
    GCLMemType imt = input->desc.memType;
    GCLMemType omt = output->desc.memType;

    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[0];
    item_c = item_c >> 2;
    char kernelName[128];
    KernelOpt kernelOpt;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    bool reuseOnW = false;
    U32 edge = oh;
    if (fw == 2 && fh == 2 && sw == 2 && sh == 2) {
        if ((item_h >> 8) > 0) {
            item_h = item_h >> 8;
            reuseOnW = true;
            edge = ow;
            gs[0] = ((ow + 1) / 2 + item_h - 1) / item_h;
            gs[1] = (oh + 1) / 2;
            gs[2] = (fc + 3) / 4 * fw * fh / item_c;
        } else {
            gs[0] = (ow + 1) / 2;
            gs[1] = ((oh + 1) / 2 + item_h - 1) / item_h;
            gs[2] = (fc + 3) / 4 * fw * fh / item_c;
        }
        CHECK_STATUS(set_deconv_gemm_f2s2_opt(
            item_c, item_h, reuseOnW, activationMode, dt, imt, omt, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, ow_str,
            ohw_str, o_off, edge, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//        handle->t_total += handle->t_execute;
#endif
    } else {
        U32 th_str = ih;
        U32 tw_str = iw;
        U32 th_off = 0;
        U32 tw_off = 0;
        U32 th = ih;
        U32 tw = iw;
        U32 tc = fw * fh * ((fc + 3) / 4 * 4);
        U32 thw_str = th_str * tw_str;
        if ((item_h >> 8) > 0) {
            U32 item_w = item_h >> 8;
            CHECK_STATUS(set_conv_direct_reuse_w_opt_mali(1, 1, 1, 1, item_w, item_c, true,
                {}, dt, imt, tmpBuf->desc.memType, kernelName, &kernelOpt));
            gs[0] = (tw + item_w - 1) / item_w;
            gs[1] = th;
            gs[2] = (tc + 3) / 4 / item_c;
        } else {
            CHECK_STATUS(set_conv_direct_opt_mali(1, 1, 1, 1, item_h, item_c, true, {},
                dt, imt, tmpBuf->desc.memType, kernelName, &kernelOpt));
            gs[0] = tw;
            gs[1] = (th + item_h - 1) / item_h;
            gs[2] = (tc + 3) / 4 / item_c;
        }

        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, tw_str,
            thw_str, o_off, th, tc, 1, 0, 0, gs[0], gs[1], inbuf, fltbuf, biasmem, tmp));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//        handle->t_total += handle->t_execute;
#endif
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4;
        dim = 3;
        CHECK_STATUS(
            set_common_opt(dt, tmpBuf->desc.memType, omt, "col2im", kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, tw, th, fw, fh, pw, ph, sw, sh, ow_str, oh_str,
            o_off, ow, oh, gs[0], gs[1], tmp, biasmem, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//        handle->t_total += handle->t_execute;
#endif
    }
    return SUCCESS;
}

inline TensorDesc transform_filter_desc(TensorDesc filterDesc, U32 item_c, U32 item_k)
{
    DataType fdt;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, NULL, &fn, &fc, &fh, &fw);
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = fdt;
    desc.nDims = 4;
    item_c = item_c >> 2;
    desc.dims[3] = 1;
    desc.dims[0] = 4 * item_k * item_c;
    desc.dims[1] = (fn + item_k - 1) / item_k;
    desc.dims[2] = (fc + 3) / 4 * ((fw * fh + item_c - 1) / item_c);
    return desc;
}

EE deconvolution_gemm_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    *ftmDesc = transform_filter_desc(filterDesc, item_c, item_k);
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
    KernelOpt kernelOpt;
    CHECK_STATUS(
        set_deconv_gemm_trans_fltbuf(item_c, item_k, fdt, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fw, fh, fwh, fwhc, fc, fn, filter->mem, fltmem->mem));
    U32 gs[2] = {fwh * ((fc + 3) / 4), (fn + 3) / 4};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    *fltmemDesc = transform_filter_desc(filterDesc, item_c, item_k);
    return SUCCESS;
}

EE deconvolution_gemm_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    U32 iw, ih;
    U32 fw, fh, fc;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw);
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;
    fc = outputDesc.dims[outputDesc.nDims - 2];
    *bytes = iw * ih * fw * fh * ((fc + 3) / 4 * 4) * bytesOf(inputDesc.dt);
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
    ActivationParamSpec activationMode)
{
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(
        deconv_gemm_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, convParamSpec,
            forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
    return SUCCESS;
}
