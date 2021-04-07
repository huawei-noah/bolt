// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/depthwise_convolution_direct_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_depthwise_opt.h"

inline EE depthwise_core_mali_fp16(GCLHandle_t handle,
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
    ActivationMode depthwiseActivationMode)
{
    UNUSED(inputDesc);
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);

    cl_mem inbuf, biasimg, outbuf, fltbuf;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasimg = bias->mem;
    outbuf = output->mem;
    U32 fw, fh, sw, sh, pw, ph, dw, dh;
    U32 ow, oh, oc, on;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    pw = convParamSpec.padding_left;
    ph = convParamSpec.padding_top;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;

    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, &fh, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    U32 iw_str, ih_str, ihw_str, ic_str, ih_off, iw_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    iw_off -= pw;
    ih_off -= ph;
    ihw_str = iw_str * ih_str;

    U32 ow_str, oh_str, ow_off, oh_off, ohw_str;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    ohw_str = oh_str * ow_str;

    U32 item_w = forwardRunInfo->best_w[0];
    U32 gs[3] = {oh, (ow + item_w - 1) / item_w, (oc + 3) / 4 * on};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    char kernelname[128];
    KernelOpt kernelOpt;
    if (dw > 1 || dh > 1) {
        CHECK_STATUS(set_conv_depthwise_dila_opt_mali(fw, fh, sw, dw, item_w,
            depthwiseActivationMode, false, DT_F16, kernelname, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, oh_str,
            ow_str, ohw_str, oh_off, ow_off, ow, sh, dw, dh, gs[0], gs[1], inbuf, fltbuf, biasimg,
            outbuf));
    } else {
        CHECK_STATUS(set_conv_depthwise_opt_mali(
            fw, fh, sw, item_w, depthwiseActivationMode, false, DT_F16, kernelname, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, oh_str,
            ow_str, ohw_str, oh_off, ow_off, ow, sh, gs[0], gs[1], inbuf, fltbuf, biasimg, outbuf));
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE depthwise_convolution_direct_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes)
{
    U32 fw, fh, fc;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, &fc, &fh, &fw);
    U32 item_k = forwardRunInfo->best_k[0];
    U32 s0, s1, s2;
    U32 num, byteSize;
    s0 = fw * fh;
    s1 = (fc + item_k - 1) / item_k;
    s2 = 1;
    num = s0 * s1 * s2 * item_k;
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
    gclmemFilterDesc->memFormat = DF_NHWCN4;
    gclmemFilterDesc->flags = CL_MEM_READ_WRITE;
    gclmemFilterDesc->host_ptr = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE depthwise_convolution_direct_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem)
{
    DataType fdt;
    DataFormat fdf;
    U32 fw, fh, fc;
    tensorSelectGet(filterDesc, &fdt, &fdf, NULL, &fc, &fh, &fw);
    U32 fwh = fw * fh;
    U32 item_k = forwardRunInfo->best_k[0];
    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "conv_depthwise_trans_fltbuf_%d", item_k);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, filter->mem, fltmem->mem));
    U32 gs[3] = {fwh, (fc + item_k - 1) / item_k};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    *fltmemDesc = tensor4df(fdt, fdf, 1, fc, fh, fw);
    return SUCCESS;
}

EE depthwise_convolution_direct_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    UNUSED(inputDesc);
    UNUSED(filterDesc);
    UNUSED(outputDesc);
    UNUSED(convParamSpec);
    UNUSED(forwardRunInfo);
    *bytes = 0;
    return SUCCESS;
}

EE depthwise_convolution_direct_mali_fp16(GCLHandle_t handle,
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
    ActivationMode depthwiseActivationMode)
{
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(depthwise_core_mali_fp16(handle, inputDesc, input, filterDesc, filter,
        convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
        depthwiseActivationMode));
    return SUCCESS;
}
