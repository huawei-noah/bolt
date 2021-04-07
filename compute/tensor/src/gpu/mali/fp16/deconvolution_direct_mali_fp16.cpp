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
#include "gpu/mali/fp16/deconvolution_direct_mali_fp16.h"

inline EE deconv_direct_core_mali_fp16(GCLHandle_t handle,
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
    UNUSED(tmpBuf);
    cl_mem inbuf, biasmem, outbuf, fltbuf;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasmem = bias->mem;
    outbuf = output->mem;
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

    U32 iw_str, ih_str, ihw_str, ic_str, iw_off, ih_off;
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ic_str = input->desc.stride[2];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];
    ihw_str = ih_str * iw_str;

    U32 ow_str, oh_str, ohw_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];
    ohw_str = oh_str * ow_str;

    char kernelname[128];
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim;
    Kernel kernel;
    sprintf(kernelname, "deconv_direct");
    gs[0] = oh;
    gs[1] = ow;
    gs[2] = (oc + 3) / 4;
    dim = 3;
    U32 in_channel_blocks = (ic + 3) / 4;
    U32 out_channel_blocks = gs[2];

    pw = fw - pw - 1;
    ph = fh - ph - 1;
    U32 align_h = sh - 1 - ph;
    U32 align_w = sw - 1 - pw;

    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, inbuf, fltbuf, outbuf, biasmem, iw, iw_str, iw_off, ih,
        ih_str, ih_off, fw, fh, fc, fn, sw, sh, pw, ph, ow, ow_str, ow_off, oh, oh_str, oh_off, ic,
        oc, align_h, align_w, in_channel_blocks, out_channel_blocks));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);

#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE deconvolution_direct_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
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
    if (item_c == 4) {
        s0 = fw * fh;
        s1 = (fc + item_c - 1) / item_c;
        s2 = (fn + item_k - 1) / item_k;
        gclmemFilterDesc->memFormat = DF_NCHWN4C4;
        num = s0 * s1 * s2 * item_c * item_k;
    } else {
        CHECK_STATUS(NOT_MATCH);
    }
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

EE deconvolution_direct_transform_filter_mali_fp16(GCLHandle_t handle,
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
    U32 item_k = forwardRunInfo->best_k[0];
    if (item_k != 4) {
        CHECK_STATUS(NOT_MATCH);
    }
    //   if(item_k == 0) item_k = fn;
    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "deconv_direct_trans_fltbuf");
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, filter->mem, fltmem->mem));
    U32 gs[3] = {fwh, (fc + 3) / 4, (fn + 3) / 4 * 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "deconv_direct_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, fltmem, "deconv_direct_filter_tran"));
#endif
    return SUCCESS;
}

EE deconvolution_direct_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
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

EE deconvolution_direct_mali_fp16(GCLHandle_t handle,
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
    U32 fw, fh, ih, iw;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, &fh, &fw);
    tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw);
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    if (inputDesc.df == DF_NCHW || (fw == 1 && fw == 1 && ih == 1 && iw == 1)) {
        CHECK_STATUS(deconv_direct_core_mali_fp16(handle, inputDesc, input, filterDesc, filter,
            convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
            activationMode));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}
