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
#include "gpu/mali/fp16/convolution_wino_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_wino_opt.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"

TensorDesc getInputPreProcessDesc(
    TensorDesc inputDesc, ConvolutionParamSpec convParamSpec, U32 wino_w, U32 wino_h)
{
    U32 fw = convParamSpec.kernel_w;
    U32 fh = convParamSpec.kernel_h;
    U32 pl = convParamSpec.pad_left;
    U32 pr = convParamSpec.pad_right;
    U32 pt = convParamSpec.pad_top;
    U32 pb = convParamSpec.pad_bottom;
    TensorDesc desc = inputDesc;
    desc.df = DF_NCHW;
    desc.dims[0] = wino_w * 4;
    desc.dims[0] += ((fw / 2 < pl) ? pl : (fw / 2));
    desc.dims[0] += ((fw / 2 < pr) ? pr : (fw / 2));
    desc.dims[1] = wino_h * 4;
    desc.dims[1] += ((fh / 2 < pt) ? pt : (fh / 2));
    desc.dims[1] += ((fh / 2 < pb) ? pb : (fh / 2));
    return desc;
}

TensorDesc getPicTranDesc(DataType dt, U32 wino_w, U32 wino_h, U32 wino_num, U32 ic, U32 item_n)
{
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = dt;
    desc.nDims = 4;
    desc.dims[0] = UNI_ALIGN(wino_w * wino_h, item_n);
    desc.dims[1] = ic;
    desc.dims[2] = wino_num * wino_num;
    desc.dims[3] = 1;
    return desc;
}

TensorDesc getGemmOutDesc(DataType dt, U32 M, U32 N, U32 wino_num)
{
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = dt;
    desc.nDims = 4;
    desc.dims[0] = N;
    desc.dims[1] = M;
    desc.dims[2] = wino_num * wino_num;
    desc.dims[3] = 1;
    return desc;
}

inline EE wino_preprocess_input(GCLHandle_t handle,
    DataType dt,
    DataFormat df,
    U32 iw_str,
    U32 ih_str,
    U32 i_off,
    U32 ow_str,
    U32 oh_str,
    U32 iw,
    U32 ih,
    U32 ic,
    U32 pw,
    U32 ph,
    GCLMemType imt,
    GCLMemType omt,
    Mem in,
    Mem out)
{
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    bool useNchwFormat = (df == DF_NCHW) ? true : false;
    CHECK_STATUS(
        set_conv_wino_preprocess_input_opt(dt, useNchwFormat, imt, omt, kernelName, &kernelOpt));
    U32 gs[3] = {(ow_str + 3) / 4, oh_str, (ic + 3) / 4};
    U32 ls[3] = {0};
    U32 dim = 3;
    if (useNchwFormat) {
        gs[0] = (iw + 3) / 4 + ow_str - iw;
        gs[2] = ic;
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, iw_str, ih_str, i_off, ow_str, oh_str, iw, ih, ic, pw, ph, gs[0], gs[1], in, out));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE wino_trans_pic_nchw(GCLHandle_t handle,
    DataType dt,
    U32 wino_w,
    U32 wino_h,
    U32 ic,
    U32 iw_str,
    U32 ih_str,
    U32 i_off,
    U32 pw_str,
    U32 pwh_str,
    GCLMemType imt,
    Mem in,
    Mem out)
{
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    CHECK_STATUS(
        set_common_opt(dt, imt, GCL_MEM_BUF, "conv_wino_trans_picbuf_nchw", kernelName, &kernelOpt));
    U32 gs[3] = {wino_w, wino_h, ic};
    U32 ls[3] = {0};
    U32 dim = 3;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(
        gcl_set_kernelArgs(kernel, iw_str, ih_str, i_off, pw_str, pwh_str, gs[0], gs[1], in, out));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE wino_trans_pic_img(GCLHandle_t handle, TensorDesc picTranDesc, Mem picTran, Mem picTranImg)
{
    OclMemory tmpMem;
    OclMemoryImg tmpMemImg;
    GCLMem picBuf;
    GCLMem picImg;
    tmpMem.resize(picTranDesc);
    tmpMemImg.resize(picTranDesc);
    picBuf.desc = tmpMem.get_desc();
    picImg.desc = tmpMemImg.get_desc();
    picBuf.mem = picTran;
    picImg.mem = picTranImg;
    CHECK_STATUS(ocl_data_trans_form(handle, &picBuf, &picImg, 0, 0, NCHW_TO_NCHW));
    return SUCCESS;
}

inline EE wino_gemm(GCLHandle_t handle,
    DataType dt,
    U32 M,
    U32 N,
    U32 K,
    U32 item_m,
    U32 item_n,
    U32 wino_num,
    GCLMemType ma,
    GCLMemType mb,
    Mem A,
    Mem B,
    Mem C)
{
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    CHECK_STATUS(set_gemm_tn_opt_mali(item_m, item_n, NO_BIAS, false, {}, dt, ma, mb,
        GCL_MEM_BUF, kernelName, &kernelOpt));
    U32 gs[3] = {N / item_n, M / item_m, wino_num * wino_num};
    U32 ls[3] = {0};
    U32 dim = 3;
    U32 A_str = M * K;
    U32 B_str = N * K;
    U32 C_str = N * M;
    U32 cw_str = N;
    U32 cw = N;
    U32 ch = M;
    U32 cc = wino_num * wino_num;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, A_str, B_str, C_str, 0, 0, 0, cw_str, cw, ch,
        cc, gs[0], gs[1], A, B, C, C));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE wino_trans_out(GCLHandle_t handle,
    DataType dt,
    U32 wino_w,
    U32 wino_h,
    U32 pw_str,
    U32 pwh_str,
    U32 ow_str,
    U32 oh_str,
    U32 ow_off,
    U32 oh_off,
    U32 ow,
    U32 oh,
    U32 oc,
    GCLMemType omt,
    ActivationParamSpec activationMode,
    Mem bias,
    Mem gemm_out,
    Mem output)
{
    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    bool useAlign = false;
    if ((oh & 3) == 0 && (ow & 3) == 0) {
        useAlign = true;
    }
    CHECK_STATUS(set_conv_wino_trans_outbuf_opt(
        useAlign, activationMode, dt, GCL_MEM_BUF, omt, kernelName, &kernelOpt));
    U32 gs[3] = {wino_w, wino_h, (oc + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 o_off = oh_off * ow_str + ow_off;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, wino_w, wino_h, pw_str, pwh_str, ow_str, oh_str, o_off,
        ow, oh, oc, bias, gemm_out, output));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline TensorDesc transform_filter_desc(TensorDesc filterDesc, U32 item_k)
{
    DataType fdt;
    U32 fw, fh, fc, fn;
    U32 winoTransNum = 36;
    tensorSelectGet(filterDesc, &fdt, NULL, &fn, &fc, &fh, &fw);
    TensorDesc desc;
    desc.df = DF_NCHW;
    desc.dt = fdt;
    desc.nDims = 4;
    desc.dims[0] = (fn + item_k - 1) / item_k * item_k;
    desc.dims[1] = fc;
    desc.dims[2] = winoTransNum;
    desc.dims[3] = 1;
    return desc;
}

EE convolution_wino_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    U32 item_k = forwardRunInfo->best_k[0];
    *ftmDesc = transform_filter_desc(filterDesc, item_k);
    return SUCCESS;
}

EE convolution_wino_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    GCLMem_t tmp)
{
    DataType fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    U32 item_k = forwardRunInfo->best_k[0];
    U32 fn_align = (fn + item_k - 1) / item_k * item_k;
    U32 fwhc = fw * fh * fc;
    U32 fnc = fn_align * fc;

    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    CHECK_STATUS(set_conv_wino_rotate_flt(fw, fh, fdt, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwhc, fnc, fn_align, fn, filter->mem, tmp->mem));
    U32 gs[2] = {fwhc, fn_align};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));

    Mem fltTranMem = fltmem->mem;
    TensorDesc fltTranDesc = transform_filter_desc(filterDesc, item_k);
    if (fltmem->desc.memType != GCL_MEM_BUF) {
        U32 bytes = tensorNumBytes(fltTranDesc);
        U32 offset = UNI_ALIGN(fn_align * fwhc * bytesOf(fdt), BUFFER_ALIGN_BASE);
        CHECK_STATUS(gcl_create_sub_buffer(bytes, &offset, tmp, &fltTranMem));
    }
    CHECK_STATUS(set_common_opt(
        fdt, GCL_MEM_BUF, GCL_MEM_BUF, "conv_wino_trans_fltbuf_3x3", kernelName, &kernelOpt));
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fn_align, fc, fnc, tmp->mem, fltTranMem));
    gs[0] = fn_align;
    gs[1] = fc;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    if (fltmem->desc.memType != GCL_MEM_BUF) {
        GCLMem fltTranBuf;
        fltTranBuf.mem = fltTranMem;
        OclMemory tmpMem;
        tmpMem.resize(fltTranDesc);
        fltTranBuf.desc = tmpMem.get_desc();
        CHECK_STATUS(ocl_data_trans_form(handle, &fltTranBuf, fltmem, 0, 0, NCHW_TO_NCHW, false));
    }
    *fltmemDesc = fltTranDesc;
    return SUCCESS;
}

EE convolution_wino_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    DataType idt = inputDesc.dt;
    U32 ic = inputDesc.dims[inputDesc.nDims - 2];
    U32 ow = outputDesc.dims[0];
    U32 oh = outputDesc.dims[1];
    U32 fn = outputDesc.dims[2];
    U32 bufSize = 0;
    U32 wino_num = 6;
    U32 wino_w = (ow + 3) / 4;
    U32 wino_h = (oh + 3) / 4;
    U32 item_n = forwardRunInfo->best_h[0];
    U32 item_m = forwardRunInfo->best_k[0];

    TensorDesc inputNchwDesc = getInputPreProcessDesc(inputDesc, convParamSpec, wino_w, wino_h);
    if (inputDesc.df != DF_NCHW) {
        bool useImg = check_qualcomm_device();
        if (useImg) {
            U32 width = (inputNchwDesc.dims[0] + 3) / 4;
            U32 height = inputNchwDesc.dims[1];
            U32 depth = inputNchwDesc.dims[2] * inputNchwDesc.dims[3];
            if (CHECK_MEET_IMAGE_LIMITS(width, height, depth)) {
                bytes[1] = width;
                bytes[2] = height;
                bytes[3] = depth;
            } else {
                useImg = false;
            }
        }
        if (!useImg) {
            bufSize += UNI_ALIGN(tensorNumBytes(inputNchwDesc), BUFFER_ALIGN_BASE);
        }
    } else {  //for input is NCHW and memType is image
        bufSize += UNI_ALIGN(tensorNumBytes(inputNchwDesc), BUFFER_ALIGN_BASE);
    }

    TensorDesc picTranDesc = getPicTranDesc(idt, wino_w, wino_h, wino_num, ic, item_n);
    bufSize += UNI_ALIGN(tensorNumBytes(picTranDesc), BUFFER_ALIGN_BASE);
    bool useImg = check_qualcomm_device();
    if (useImg) {
        U32 width = (picTranDesc.dims[0] + 3) / 4;
        U32 height = picTranDesc.dims[1];
        U32 depth = picTranDesc.dims[2] * picTranDesc.dims[3];
        if (CHECK_MEET_IMAGE_LIMITS(width, height, depth)) {
            bytes[4] = width;
            bytes[5] = height;
            bytes[6] = depth;
        }
    }

    U32 M = UNI_ALIGN(fn, item_m);
    U32 N = picTranDesc.dims[0];
    TensorDesc gemmOutDesc = getGemmOutDesc(idt, M, N, wino_num);
    bufSize += UNI_ALIGN(tensorNumBytes(gemmOutDesc), BUFFER_ALIGN_BASE);

    DataType fdt = filterDesc.dt;
    U32 fw = convParamSpec.kernel_w;
    U32 fh = convParamSpec.kernel_h;
    U32 fc = ic;
    U32 item_k = item_m;
    U32 fn_align = (fn + item_k - 1) / item_k * item_k;
    U32 tempBufNum = fn_align * fc * fw * fh;
    U32 fltTempBufSize = UNI_ALIGN(tempBufNum * bytesOf(fdt), BUFFER_ALIGN_BASE);
    useImg = check_qualcomm_device();
    if (useImg) {
        TensorDesc fltTranDesc = transform_filter_desc(filterDesc, item_k);
        fltTempBufSize += tensorNumBytes(fltTranDesc);
    }

    if (bufSize < fltTempBufSize) {
        bufSize = fltTempBufSize;
    }
    *bytes = bufSize;
    return SUCCESS;
}

EE convolution_wino_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    std::vector<GCLMem_t> tmp,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationParamSpec activationMode)
{
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    U32 wino_num = 6;
    DataType idt;
    U32 iw, ih, ic;
    U32 fw, fh, fc, fn, pw, ph;
    U32 ow, oh, oc, on;
    fw = convParamSpec.kernel_w;
    fh = convParamSpec.kernel_h;
    pw = convParamSpec.pad_left;
    ph = convParamSpec.pad_top;
    tensorSelectGet(inputDesc, &idt, NULL, NULL, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    fc = ic;
    fn = oc;
    Mem inMem = input->mem;
    U32 iw_str, ih_str;
    I32 iw_off, ih_off, i_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, (U32 *)&iw_off, (U32 *)&ih_off);
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);

    GCLMemType imt = input->desc.memType;
    U32 wino_w = (ow + 3) / 4;
    U32 wino_h = (oh + 3) / 4;
    U32 item_n = forwardRunInfo->best_h[0];
    U32 item_m = forwardRunInfo->best_k[0];
    U32 offset = 0;

    if (inputDesc.df != DF_NCHW || input->desc.memType == GCL_MEM_IMG_3D) {
        TensorDesc desc = getInputPreProcessDesc(inputDesc, convParamSpec, wino_w, wino_h);
        Mem inputPre;
        GCLMemType omt;
        bool useImg = (tmp[1]) ? true : false;
        if (inputDesc.df == DF_NCHW) {  //for padding input(must be image), have to set data to buffer
            useImg = false;
        }
        if (useImg) {
            inputPre = tmp[1]->mem;
            omt = GCL_MEM_IMG_3D;
        } else {
            U32 bytes = tensorNumBytes(desc);
            CHECK_STATUS(gcl_create_sub_buffer(bytes, &offset, tmp[0], &inputPre));
            omt = GCL_MEM_BUF;
        }
        U32 tw_str = desc.dims[0];
        U32 th_str = desc.dims[1];
        i_off = ih_off * iw_str + iw_off;
        CHECK_STATUS(wino_preprocess_input(handle, desc.dt, input->desc.df, iw_str, ih_str, i_off,
            tw_str, th_str, iw, ih, ic, pw, ph, imt, omt, inMem, inputPre));
        inMem = inputPre;
        iw_str = tw_str;
        ih_str = th_str;
        i_off = 0;
        imt = omt;
    } else {
        i_off = (ih_off - ph) * iw_str + iw_off - pw;
    }

    TensorDesc picTranDesc = getPicTranDesc(idt, wino_w, wino_h, wino_num, ic, item_n);
    U32 picTranSize = tensorNumBytes(picTranDesc);
    Mem picTran;
    GCLMemType picTranType = GCL_MEM_BUF;
    CHECK_STATUS(gcl_create_sub_buffer(picTranSize, &offset, tmp[0], &picTran));
    U32 pw_str = picTranDesc.dims[0];
    U32 pwh_str = pw_str * picTranDesc.dims[1];
    CHECK_STATUS(wino_trans_pic_nchw(handle, picTranDesc.dt, wino_w, wino_h, ic, iw_str, ih_str,
        i_off, pw_str, pwh_str, imt, inMem, picTran));
    if (tmp[2]) {
        CHECK_STATUS(wino_trans_pic_img(handle, picTranDesc, picTran, tmp[2]->mem));
        picTran = tmp[2]->mem;
        picTranType = GCL_MEM_IMG_3D;
    }

    U32 M = UNI_ALIGN(fn, item_m);
    U32 N = picTranDesc.dims[0];
    U32 K = ic;
    TensorDesc gemmOutDesc = getGemmOutDesc(idt, M, N, wino_num);
    U32 gemmOutSize = tensorNumBytes(gemmOutDesc);
    Mem gemmOut;
    Mem fltTran = filter->mem;
    GCLMemType fltTranType = filter->desc.memType;

    CHECK_STATUS(gcl_create_sub_buffer(gemmOutSize, &offset, tmp[0], &gemmOut));
    CHECK_STATUS(wino_gemm(handle, idt, M, N, K, item_m, item_n, wino_num, fltTranType, picTranType,
        fltTran, picTran, gemmOut));
    Mem biasbuf = bias->mem;
    Mem outbuf = output->mem;

    CHECK_STATUS(wino_trans_out(handle, idt, wino_w, wino_h, N, N * M, ow_str, oh_str, ow_off,
        oh_off, ow, oh, oc, output->desc.memType, activationMode, biasbuf, gemmOut, outbuf));
    return SUCCESS;
}
