// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/depthwise_convolution_mali_fp16.h"
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
    ActivationParamSpec depthwiseActivationParamSpec)
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
    DataType idt;
    U32 iw, ih, ic, in;
    U32 fw, fh, sw, sh, pw, ph, dw, dh;
    U32 ow, oh, oc, on;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    pw = convParamSpec.pad_left;
    ph = convParamSpec.pad_top;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;

    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    U32 item_h = forwardRunInfo->best_h[0];

    U32 iw_str, ih_str, ic_str, ihw_str, in_str;
    I32 iw_off, ih_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, (U32 *)&iw_off, (U32 *)&ih_off);
    iw_off -= pw;
    ih_off -= ph;
    ihw_str = iw_str * ih_str;
    ic_str = (ic + 3) / 4;
    in_str = ic_str * ihw_str;

    U32 ow_str, oh_str, oc_str, ow_off, oh_off, ohw_str, on_str, o_off;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    ohw_str = oh_str * ow_str;
    oc_str = (oc + 3) / 4;
    on_str = oc_str * ohw_str;
    o_off = oh_off * ow_str + ow_off;

    U32 gs[3] = {ow, (oh + item_h - 1) / item_h, (oc + 3) / 4 * on};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    char kernelName[128];
    KernelOpt kernelOpt;
    if (dw > 1 || dh > 1) {
        CHECK_STATUS(
            set_conv_depthwise_dila_opt_mali(fw, fh, sh, dh, item_h, depthwiseActivationParamSpec, false,
                idt, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, ow_str,
            ohw_str, o_off, oh, oc, sw, dw, dh, in_str, on_str, gs[0], gs[1], inbuf, fltbuf,
            biasimg, outbuf));
    } else {
        CHECK_STATUS(set_conv_depthwise_opt_mali(fw, fh, sh, item_h, depthwiseActivationParamSpec, false,
            idt, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
        CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, ow_str, ohw_str,
                o_off, oh, oc, sw, in_str, on_str, gs[0], gs[1], inbuf, fltbuf, biasimg, outbuf));
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);

#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline TensorDesc transform_filter_desc(TensorDesc filterDesc, U32 item_k)
{
    DataType fdt;
    U32 fw, fh, fc;
    tensorSelectGet(filterDesc, &fdt, NULL, NULL, &fc, &fh, &fw);
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = fdt;
    desc.nDims = 4;
    desc.dims[3] = 1;
    desc.dims[0] = fw * fh * item_k;
    desc.dims[1] = (fc + item_k - 1) / item_k;
    desc.dims[2] = 1;
    return desc;
}

EE depthwise_convolution_direct_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    U32 item_k = forwardRunInfo->best_k[0];
    *ftmDesc = transform_filter_desc(filterDesc, item_k);
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
    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    CHECK_STATUS(set_conv_depthwise_trans_flt(item_k, fdt, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fw, fh, fwh, fc, filter->mem, fltmem->mem));
    U32 gs[3] = {fwh, (fc + item_k - 1) / item_k};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    *fltmemDesc = transform_filter_desc(filterDesc, item_k);
    return SUCCESS;
}

GCLMemDesc depthwise_convolution_get_input_nchwc4_desc(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc outputDesc,
    U32 item_h)
{
    GCLMemDesc desc;
    U32 oh = outputDesc.dims[1];
    U32 ih_align = UNI_ALIGN(oh, item_h);
    U32 pl, pr, pt, pb;
    calDepthwisePaddingVal(inputDesc, convParamSpec, ih_align, &pl, &pr, &pt, &pb);
    inputDesc.df = DF_NCHWC4;
    bool useImg = check_qualcomm_device();
    if (useImg) {
        OclMemoryImg mem;
        mem.resize(inputDesc);
        U32 str[3] = {0};
        mem.stride(str);
        if (CHECK_MEET_IMAGE_LIMITS(str[0], str[1], str[2])) {
            mem.padding(pl, pr, pt, pb, 0, 0);
            desc = mem.get_desc();
        } else {
            useImg = false;
        }
    }
    if (!useImg) {
        OclMemory mem;
        mem.resize(inputDesc);
        mem.padding(pl, pr, pt, pb);
        desc = mem.get_desc();
    }
    return desc;
}

EE depthwise_convolution_trans_input_to_nchwc4(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    GCLMem_t input,
    ConvolutionParamSpec convParamSpec,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    U32 item_h,
    GCLMemDesc *transDesc,
    U32 *tmpSubOff)
{
    GCLMemDesc desc = depthwise_convolution_get_input_nchwc4_desc(
        inputDesc, filterDesc, convParamSpec, outputDesc, item_h);
    GCLMem tMem;
    if (desc.memType != tmpBuf->desc.memType) {
        CHECK_STATUS(NOT_MATCH);
    }
    tMem.mem = tmpBuf->mem;
    tMem.desc = desc;
    CHECK_STATUS(ocl_fill_memory_zero(handle, &tMem, 0));
    CHECK_STATUS(ocl_data_trans_form(handle, input, &tMem, 0, 0, NCHW_TO_NCHWC4));
    *transDesc = desc;
    if (desc.memType == GCL_MEM_BUF) {
        U32 size = desc.byteSize;
        (*tmpSubOff) += UNI_ALIGN(size, BUFFER_ALIGN_BASE);
    }
    return SUCCESS;
}

EE depthwise_convolution_direct_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    U32 size = 0;
    if (inputDesc.df == DF_NCHW) {
        GCLMemDesc desc = depthwise_convolution_get_input_nchwc4_desc(
            inputDesc, filterDesc, convParamSpec, outputDesc, forwardRunInfo->best_h[0]);
        if (desc.memType == GCL_MEM_IMG_3D) {
            bytes[1] = desc.stride[0];
            bytes[2] = desc.stride[1];
            bytes[3] = desc.stride[2];
        } else {
            bytes[0] = desc.byteSize;
        }
    }
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
    ActivationParamSpec depthwiseActivationParamSpec)
{
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(depthwise_core_mali_fp16(handle, inputDesc, input, filterDesc, filter,
        convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
        depthwiseActivationParamSpec));
    return SUCCESS;
}
