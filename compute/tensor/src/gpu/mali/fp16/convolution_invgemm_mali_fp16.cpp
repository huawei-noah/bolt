// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/convolution_mali_fp16.h"
#include "gpu/mali/fp16/convolution_invgemm_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_invgemm_opt.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"

inline EE invgemm_core_mali_fp16(GCLHandle_t handle,
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
    U32 iw, ih, ic, in;
    U32 fw, fh, sw, sh, pl, pt;
    U32 ow, oh, oc, on;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    pl = convParamSpec.pad_left;
    pt = convParamSpec.pad_top;
    tensorSelectGet(inputDesc, &dt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    item_k = item_k >> 2;

    U32 iw_str, ih_str, ihw_str, ic_str, iw_off, ih_off, in_str;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    U32 i_off = ih_off * iw_str + iw_off;
    ihw_str = ih_str * iw_str;
    ic_str = (ic + item_c - 1) / item_c;
    in_str = ihw_str * ic_str;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    o_off = oh_off * ow_str + ow_off;

    char kernelName[128];
    KernelOpt kernelOpt;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim;
    Kernel kernel;
    if (sw == 1 && sh == 1) {
        U32 tw = iw;
        U32 th = ih;
        U32 tc = fw * fh * ((oc + 3) / 4 * 4);
        U32 tw_str = iw;
        U32 th_str = ih;
        U32 t_off = 0;
        U32 thw_str = tw_str * th_str;
        U32 tn_str = thw_str * ((tc + 3) / 4);
        gs[0] = tw;
        gs[1] = (th + item_h - 1) / item_h;
        gs[2] = tc / 4 / item_k * on;
        dim = 3;
        CHECK_STATUS(set_conv_direct_opt_mali(1, 1, 1, 1, item_h, item_k, true, {}, dt,
            input->desc.memType, GCL_MEM_BUF, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, tw_str,
            thw_str, t_off, th, tc, sw, in_str, tn_str, gs[0], gs[1], inbuf, fltbuf, biasmem, tmp));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4 * on;
        I32 pw = fw - 1 - pl;
        I32 ph = fh - 1 - pt;
        CHECK_STATUS(set_conv_invgemm_col2img_opt(
            activationMode, dt, GCL_MEM_BUF, output->desc.memType, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, ih, fw, fh, pw, ph, ow_str, oh_str, o_off, oc,
            gs[0], gs[1], tmp, biasmem, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        handle->t_total += handle->t_execute;
#endif
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
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
    desc.dims[0] = item_k * item_c;
    desc.dims[1] = (fc + item_c - 1) / item_c;
    desc.dims[2] = (fn + item_k - 1) / item_k * fw * fh;
    desc.dims[3] = 1;
    return desc;
}

EE convolution_invgemm_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    *ftmDesc = transform_filter_desc(filterDesc, item_c, item_k);
    return SUCCESS;
}

EE convolution_invgemm_transform_filter_mali_fp16(GCLHandle_t handle,
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
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    CHECK_STATUS(set_conv_invgemm_trans_flt_opt(item_k, fdt, kernelName, &kernelOpt));
    gs[0] = fwh;
    gs[1] = (fc + item_c - 1) / item_c;
    gs[2] = (fn + item_k - 1) / item_k * item_k;
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fw, fh, fwh, fc, fn, filter->mem, fltmem->mem));
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    *fltmemDesc = transform_filter_desc(filterDesc, item_c, item_k);
    return SUCCESS;
}

EE convolution_invgemm_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    DataType dt = inputDesc.dt;
    U32 iw = inputDesc.dims[0];
    U32 ih = inputDesc.dims[1];
    U32 fw = convParamSpec.kernel_w;
    U32 fh = convParamSpec.kernel_h;
    U32 oc = outputDesc.dims[outputDesc.nDims - 2];
    U32 on = outputDesc.dims[outputDesc.nDims - 1];
    U32 bufSize = 0;
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];

    U32 tw = iw;
    U32 th = ih;
    U32 tc = fw * fh * ((oc + 3) / 4 * 4);
    U32 tn = on;
    bufSize = tw * th * tc * tn * bytesOf(dt);
    *bytes = bufSize;
    return SUCCESS;
}

EE convolution_invgemm_mali_fp16(GCLHandle_t handle,
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
    CHECK_STATUS(invgemm_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, convParamSpec,
        forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
    return SUCCESS;
}
