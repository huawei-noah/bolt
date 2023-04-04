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
#include "gpu/mali/fp16/convolution_direct_mali_fp16.h"
#include "gpu/mali/fp16/gemv_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"
#include "gpu/mali/cl/kernel_option/gemv_opt.h"

inline TensorDesc get_nchw_desc_for_img(TensorDesc inputDesc, ConvolutionParamSpec convParamSpec)
{
    TensorDesc desc = inputDesc;
    desc.dims[0] += convParamSpec.pad_left + convParamSpec.pad_right;
    desc.dims[1] += convParamSpec.pad_bottom;
    return desc;
}

inline EE trans_input_nchw_to_img(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ConvolutionParamSpec convParamSpec,
    GCLMem_t tmp,
    U32 *iw_str,
    U32 *ih_str,
    I32 *iw_off,
    I32 *ih_off)
{
    TensorDesc descNchwImg = get_nchw_desc_for_img(inputDesc, convParamSpec);
    GCLMem inputTran = *input;
    inputTran.desc.dims[0] = descNchwImg.dims[0];  //move left padding zero into img
    inputTran.desc.dims[1] = descNchwImg.dims[1];
    inputTran.desc.offset[0] -= convParamSpec.pad_left;
    if ((I32)inputTran.desc.offset[0] < 0) {
        CHECK_STATUS(NOT_MATCH);
    }
    GCLMem inputImg;
    OclMemoryImg mem;
    mem.resize(descNchwImg);
    inputImg.desc = mem.get_desc();
    inputImg.mem = tmp->mem;
    if (inputDesc.nDims == 5) {
        CHECK_STATUS(ocl_data_trans_form_3d(handle, &inputTran, &inputImg, 0, 0, NCHW_TO_NCHW));
    } else {
        CHECK_STATUS(ocl_data_trans_form(handle, &inputTran, &inputImg, 0, 0, NCHW_TO_NCHW));
    }
    *iw_str = inputImg.desc.stride[0];
    *ih_str = inputImg.desc.stride[1];
    *iw_off = 0;
    *ih_off = -convParamSpec.pad_top;
    return SUCCESS;
}

inline EE direct_core_nchw_to_nchwc4_mali_fp16(GCLHandle_t handle,
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
    cl_mem inbuf, biasmem, outbuf, fltbuf;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasmem = bias->mem;
    outbuf = output->mem;
    GCLMemType imt = input->desc.memType;
    GCLMemType omt = output->desc.memType;
    DataType dt;
    DataFormat df;
    U32 iw, ih, it, ic;
    U32 fw, fh, fn, ft, sw, sh, st, pw, ph, pt;
    U32 ow, oh, oc, on, ot;
    U32 inDims;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;
    ft = convParamSpec.kernel_t;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    st = convParamSpec.stride_t;
    ph = convParamSpec.pad_top;
    pw = convParamSpec.pad_left;
    pt = convParamSpec.pad_before;

    tensorSelectGet(inputDesc, &dt, &df, NULL, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);
    fn = oc;
    inDims = inputDesc.nDims;
    U32 iw_str, ih_str, iwh_str, ic_str;
    I32 iw_off, ih_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, (U32 *)&iw_off, (U32 *)&ih_off);
    iw_off -= pw;
    ih_off -= ph;
    iwh_str = iw_str * ih_str;
    ic_str = ic;
    U32 ow_str, oh_str, owh_str, ow_off, oh_off, o_off;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    o_off = oh_off * ow_str + ow_off;

    if (tmpBuf->desc.memType != GCL_MEM_BUF) {
        CHECK_STATUS(trans_input_nchw_to_img(
            handle, inputDesc, input, convParamSpec, tmpBuf, &iw_str, &ih_str, &iw_off, &ih_off));
        iwh_str = iw_str * ih_str;
        inbuf = tmpBuf->mem;
        imt = tmpBuf->desc.memType;
    }

    U32 item_w = forwardRunInfo->best_h[0];  //for nchw, reuse on w
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    gs[0] = (ow + item_w - 1) / item_w;
    gs[1] = oh;
    gs[2] = (oc + 3) / 4 * on * ot;
    if (ot > 1 && on > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    CHECK_STATUS(set_conv_direct_nchw_to_nchwc4_opt_mali(
        fw, fh, ft, sw, item_w, activationMode, dt, imt, omt, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    if (ot > 1) {
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, iwh_str, ic_str, iw_off, ih_off, ow_str,
            oh_str, o_off, ow, ot, it, pt, sh, st, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
    } else {
        U32 in_str = iwh_str * ic;
        U32 on_str = ow_str * oh_str * ((oc + 3) / 4);
        CHECK_STATUS(
            gcl_set_kernelArgs(kernel, iw_str, iwh_str, ic_str, iw_off, ih_off, ow_str, oh_str,
                o_off, ow, oc, sh, in_str, on_str, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);

#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE direct_core_fn_spe(GCLHandle_t handle,
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
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);
    cl_mem inbuf, biasmem, outbuf, fltbuf;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasmem = bias->mem;
    outbuf = output->mem;
    DataType dt;
    U32 iw, ih;
    U32 fw, fh, fn, sw, sh, pw, ph;
    U32 ow, oh, oc, on;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    ph = convParamSpec.pad_top;
    pw = convParamSpec.pad_left;
    tensorSelectGet(inputDesc, &dt, NULL, NULL, NULL, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    fn = oc;

    U32 iw_str, ih_str, ihw_str, ic_str;
    I32 iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off, ohw_str, o_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, (U32 *)&iw_off, (U32 *)&ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    o_off = oh_off * ow_str + ow_off;
    iw_off -= pw;
    ih_off -= ph;
    ihw_str = iw_str * ih_str;
    ohw_str = ow_str * oh_str;

    U32 item_h = forwardRunInfo->best_h[0];
    if (output->desc.memFormat != DF_NCHW) {
        CHECK_STATUS(NOT_MATCH);
    }
    bool useNchw = true;
    char kernelName[128];
    KernelOpt kernelOpt;
    CHECK_STATUS(set_conv_direct_sh1_fn_spe_opt_mali(fw, fh, item_h, useNchw, activationMode, dt,
        input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
    U32 gs[3] = {ow, (oh + item_h - 1) / item_h, 1};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 2;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, ow_str, o_off,
        oh, sw, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);

#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE direct_core_gemv(GCLHandle_t handle,
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
    U32 tmpOff = 0;
    CHECK_STATUS(gemv(handle, inputDesc, outputDesc, activationMode, true, &tmpOff, tmpBuf, input,
        bias, filter, output, forwardRunInfo));
    return SUCCESS;
}

inline EE direct_core_mali_fp16(GCLHandle_t handle,
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
    cl_mem inbuf, biasmem, outbuf, fltbuf;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasmem = bias->mem;
    outbuf = output->mem;
    DataType dt;
    U32 iw, ih, ic, in, it;
    U32 fw, fh, ft, sw, sh, st, pw, ph, pt;
    U32 ow, oh, oc, on, ot;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;
    ft = convParamSpec.kernel_t;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    st = convParamSpec.stride_t;
    ph = convParamSpec.pad_top;
    pw = convParamSpec.pad_left;
    pt = convParamSpec.pad_before;
    tensorSelectGet(inputDesc, &dt, NULL, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow, &ot);
    if (on > 1 && ot > 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    U32 iw_str, ih_str, ihw_str, ic_str, in_str;
    U32 ow_str, oh_str, ohw_str, oc_str, ow_off, oh_off, on_str, o_off;
    I32 ih_off, iw_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, (U32 *)&iw_off, (U32 *)&ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    o_off = oh_off * ow_str + ow_off;
    ih_off -= ph;
    iw_off -= pw;
    ihw_str = ih_str * iw_str;
    ohw_str = oh_str * ow_str;
    ic_str = (ic + item_c - 1) / item_c;
    oc_str = (oc + 3) / 4;
    in_str = ihw_str * ic_str;
    on_str = ohw_str * oc_str;
    char kernelName[128];
    KernelOpt kernelOpt;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim;
    Kernel kernel;
    U32 sv = sw;
    if ((item_h >> 8) > 0) {
        U32 item_w = item_h >> 8;
        item_k = item_k >> 2;
        CHECK_STATUS(set_conv_direct_reuse_w_opt_mali(fw, fh, ft, sw, item_w, item_k, false,
            activationMode, dt, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
        gs[0] = (ow + item_w - 1) / item_w;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4 * on / item_k;
        dim = 3;
        sv = sh;
    } else if ((item_h >> 4) > 0) {
        U32 n = item_h >> 4;
        U32 h = item_h & 15;
        U32 k = item_k >> 2;
        CHECK_STATUS(set_conv_direct_multi_batch_opt_mali(fw, fh, ft, sh, h, k, n, activationMode,
            dt, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
        gs[0] = ow;
        gs[1] = (oh + h - 1) / h;
        gs[2] = oc_str / k * ((on + n - 1) / n);
        dim = 3;
    } else {
        item_k = item_k >> 2;
        CHECK_STATUS(set_conv_direct_opt_mali(fw, fh, ft, sh, item_h, item_k, false, activationMode,
            dt, input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
        gs[0] = ow;
        gs[1] = (oh + item_h - 1) / item_h;
        gs[2] = (oc + 3) / 4 * on / item_k * ot;
        dim = 3;
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    if ((item_h >> 4) > 0 && (item_h >> 8 == 0)) {
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, ow_str,
            ohw_str, o_off, oh, oc, on, sw, in_str, on_str, gs[0], gs[1], inbuf, fltbuf, biasmem,
            outbuf));
    } else {
        if (ot > 1) {
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, ow_str,
                ohw_str, o_off, oh, oc, ot, it, pt, sv, st, gs[0], gs[1], inbuf, fltbuf, biasmem,
                outbuf));
        } else {
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, ow_str,
                ohw_str, o_off, oh, oc, sv, in_str, on_str, gs[0], gs[1], inbuf, fltbuf, biasmem,
                outbuf));
        }
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    //CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    //handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE direct_dila_core_mali_fp16(GCLHandle_t handle,
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
    if (input->desc.memFormat != DF_NCHWC4) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (input->desc.nDims != 4) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    cl_mem inbuf, biasmem, outbuf, fltbuf;
    inbuf = input->mem;
    fltbuf = filter->mem;
    biasmem = bias->mem;
    outbuf = output->mem;
    DataType dt;
    U32 iw, ih, ic, in;
    U32 fw, fh, sw, sh, pw, ph, dw, dh;
    U32 ow, oh, oc, on;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    pw = convParamSpec.pad_left;
    ph = convParamSpec.pad_top;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    tensorSelectGet(inputDesc, &dt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    U32 iw_str, ih_str, ihw_str, ic_str, in_str;
    U32 ow_str, oh_str, ohw_str, oc_str, ow_off, oh_off, on_str, o_off;
    I32 iw_off, ih_off;
    CHECK_STATUS(gclmem_get_desc_padding(
        input->desc, &iw_str, &ih_str, &ic_str, (U32 *)&iw_off, (U32 *)&ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off));
    ih_off -= ph;
    iw_off -= pw;
    ihw_str = ih_str * iw_str;
    ohw_str = oh_str * ow_str;
    ic_str = (ic + item_c - 1) / item_c;
    oc_str = (oc + 3) / 4;
    in_str = ihw_str * ic_str;
    on_str = ohw_str * oc_str;
    o_off = oh_off * ow_str + ow_off;

    char kernelName[128];
    KernelOpt kernelOpt;
    item_k = item_k >> 2;
    U32 gs[3] = {ow, (oh + item_h - 1) / item_h, (oc + 3) / 4 / item_k * on};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    CHECK_STATUS(set_conv_direct_dila_opt_mali(fw, fh, sh, dh, item_h, item_k, activationMode, dt,
        input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ihw_str, ic_str, iw_off, ih_off, ow_str, ohw_str,
        o_off, oh, oc, sw, dw, dh, in_str, on_str, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
//    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline TensorDesc transform_filter_desc(TensorDesc filterDesc, U32 item_c, U32 item_k)
{
    DataType fdt;
    U32 fw, fh, fc, fn, ft;
    tensorSelectGet(filterDesc, &fdt, NULL, &fn, &fc, &fh, &fw, &ft);
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = fdt;
    desc.nDims = 4;
    desc.dims[3] = 1;
    if (item_k == 0) {
        return gemv_transform_filter_desc(filterDesc, 0, item_c, item_k);
    } else {
        desc.dims[0] = fw * fh * ft * item_c * item_k;
        desc.dims[1] = (fc + item_c - 1) / item_c;
        desc.dims[2] = (fn + item_k - 1) / item_k;
    }
    return desc;
}

EE convolution_direct_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    *ftmDesc = transform_filter_desc(filterDesc, item_c, item_k);
    return SUCCESS;
}

EE convolution_direct_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem)
{
    DataType fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn, ft;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw, &ft);
    U32 fwht = fw * fh * ft;
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (item_k == 0) {
        CHECK_STATUS(gemv_transform_filter_mali_fp16(
            handle, filterDesc, filter, fltmemDesc, fltmem, forwardRunInfo));
    } else {
        bool transWH = true;
        if (item_c == 1 || fw == 1 || fh == 1) {
            transWH = false;
        }
        CHECK_STATUS(set_conv_direct_trans_flt(
            item_c, item_k, transWH, fdt, fltmem->desc.memType, kernelName, &kernelOpt));
        gs[0] = fwht;
        gs[1] = (fc + item_c - 1) / item_c;
        gs[2] = (fn + item_k - 1) / item_k * item_k;
        CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, fw, fh, fwht, fc, fn, filter->mem, fltmem->mem));
        *fltmemDesc = transform_filter_desc(filterDesc, item_c, item_k);
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    }
    return SUCCESS;
}

inline GCLMemDesc convolution_get_input_nchwc4_desc(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc outputDesc,
    bool useNchwMode,
    ForwardRunInfoMali_t forwardRunInfo)
{
    GCLMemDesc desc;
    U32 pl, pr, pt, pb, pa, pf;
    U32 item_h = forwardRunInfo->best_h[0];
    U32 item_n = 1;
    if ((item_h >> 8) > 0) {
        item_h = 1;
    } else if ((item_h >> 4) > 0) {
        item_n = item_h >> 4;
        item_h = item_h & 15;
    }
    U32 in = inputDesc.dims[inputDesc.nDims - 1];
    U32 ow = outputDesc.dims[0];
    U32 oh = outputDesc.dims[1];
    U32 w_align = ow;
    U32 h_align = UNI_ALIGN(oh, item_h);
    U32 in_align = UNI_ALIGN(in, item_n);
    calPaddingVal(inputDesc, filterDesc, convParamSpec, w_align, h_align, in_align, useNchwMode,
        &pl, &pr, &pt, &pb, &pa, &pf);
    inputDesc.df = DF_NCHWC4;
    bool useImg = check_qualcomm_device();
    if (useImg) {
        OclMemoryImg mem;
        mem.resize(inputDesc);
        U32 str[3] = {0};
        mem.stride(str);
        if (CHECK_MEET_IMAGE_LIMITS(str[0], str[1], str[2])) {
            mem.padding(pl, pr, pt, pb, pa, pf);
            desc = mem.get_desc();
        } else {
            useImg = false;
        }
    }
    if (!useImg) {
        OclMemory mem;
        mem.resize(inputDesc);
        mem.padding(pl, pr, pt, pb, pa, pf);
        desc = mem.get_desc();
    }
    return desc;
}

EE convolution_direct_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    for (U32 i = 0; i < 4; i++) {
        bytes[i] = 0;
    }
    U32 fw = convParamSpec.kernel_w;
    U32 sw = convParamSpec.stride_w;
    U32 sh = convParamSpec.stride_h;
    U32 dw = convParamSpec.dilatedRate_w;
    U32 dh = convParamSpec.dilatedRate_h;
    U32 ic = inputDesc.dims[inputDesc.nDims - 2];
    DataFormat idf = inputDesc.df;
    bool useGemvMode = useGemvCalMode(inputDesc, convParamSpec, GCL_MEM_BUF, GCL_MEM_BUF);
    bool useNchwMode = useNchwCalMode(idf, fw, ic, dw, dh);
    if (useGemvMode) {
        CHECK_STATUS(
            gemv_infer_forward_tmp_bytes_mali_fp16(inputDesc, outputDesc, bytes, forwardRunInfo));
    } else if (useNchwMode) {
        bool useImg = check_qualcomm_device();
        if (useImg) {
            TensorDesc descNchwImg = get_nchw_desc_for_img(inputDesc, convParamSpec);
            U32 width = (descNchwImg.dims[0] + 3) / 4;
            U32 height = descNchwImg.dims[1];
            U32 depth = 1;
            for (U32 i = 2; i < descNchwImg.nDims; i++) {
                depth *= descNchwImg.dims[i];
            }
            if (CHECK_MEET_IMAGE_LIMITS(width, height, depth)) {
                bytes[1] = width;
                bytes[2] = height;
                bytes[3] = depth;
            }
        }
    } else if (idf == DF_NCHW) {  //use tran c1 to c4
        GCLMemDesc desc = convolution_get_input_nchwc4_desc(
            inputDesc, filterDesc, convParamSpec, outputDesc, useNchwMode, forwardRunInfo);
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

EE convolution_direct_mali_fp16(GCLHandle_t handle,
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
    U32 fw, sw, sh, dw, dh, ic;
    fw = convParamSpec.kernel_w;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    ic = inputDesc.dims[inputDesc.nDims - 2];
    GCLMemType imt = input->desc.memType;
    GCLMemType omt = output->desc.memType;
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    bool useNchwMode = (useNchwCalMode(inputDesc.df, fw, ic, dw, dh));
    bool useGemvMode = useGemvCalMode(inputDesc, convParamSpec, imt, omt);

    if (useGemvCalMode(inputDesc, convParamSpec, imt, omt)) {
        CHECK_STATUS(direct_core_gemv(handle, inputDesc, input, filterDesc, filter, convParamSpec,
            forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
    } else if (useNchwMode) {
        CHECK_STATUS(direct_core_nchw_to_nchwc4_mali_fp16(handle, inputDesc, input, filterDesc,
            filter, convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc,
            output, activationMode));
    } else if (outputDesc.df == DF_NCHW) {
        CHECK_STATUS(direct_core_fn_spe(handle, inputDesc, input, filterDesc, filter, convParamSpec,
            forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
    } else {
        GCLMem inputTran = *input;
        if (inputDesc.df == DF_NCHW) {
            GCLMemDesc desc = convolution_get_input_nchwc4_desc(
                inputDesc, filterDesc, convParamSpec, outputDesc, useNchwMode, forwardRunInfo);
            inputTran.mem = tmpBuf->mem;
            inputTran.desc = desc;
            CHECK_STATUS(fill_output_zero(handle, &inputTran, inputDesc));
            if (inputDesc.nDims == 5) {
                CHECK_STATUS(
                    ocl_data_trans_form_3d(handle, input, &inputTran, 0, 0, NCHW_TO_NCHWC4));
            } else {
                CHECK_STATUS(ocl_data_trans_form(handle, input, &inputTran, 0, 0, NCHW_TO_NCHWC4));
            }
            inputDesc.df = DF_NCHWC4;
        }
        if (dw > 1 || dh > 1) {
            CHECK_STATUS(direct_dila_core_mali_fp16(handle, inputDesc, &inputTran, filterDesc,
                filter, convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc,
                output, activationMode));
        } else {
            CHECK_STATUS(direct_core_mali_fp16(handle, inputDesc, &inputTran, filterDesc, filter,
                convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
                activationMode));
        }
    }
    return SUCCESS;
}
