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

#define calPicTranRDesc(                                                                           \
    wino_h, wino_w, wino_num, ic, fh, ph, dt, prh, prw, prc, prn, prh_off, prw_off, prhwc, prsize) \
    {                                                                                              \
        U32 ext_h = (fh / 2 < ph) ? ph : fh / 2;                                                   \
        prh = wino_h * 4 + 2 * ext_h;                                                              \
        prw = ((wino_w + 3) / 4 * 4);                                                              \
        prc = ic;                                                                                  \
        prn = wino_num;                                                                            \
        prhwc = prh * prw * prc;                                                                   \
        prsize = prhwc * prn * bytesOf(dt);                                                        \
        prh_off = ph;                                                                              \
        prw_off = 0;                                                                               \
    }

#define calPtrTranRLDesc(                                                                     \
    wino_h, wino_w, wino_num, ic, item_n, dt, prlh, prlw, prlc, prln, prlhw, prlhwc, prlsize) \
    {                                                                                         \
        prlh = wino_h;                                                                        \
        prlw = wino_w;                                                                        \
        prlc = ic;                                                                            \
        prln = wino_num * wino_num;                                                           \
        prlhw = (wino_h * wino_w + item_n - 1) / item_n * item_n;                             \
        prlhwc = prlhw * ic;                                                                  \
        prlsize = prlhwc * prln * bytesOf(dt);                                                \
    }

#define calGemmOutDesc(wino_num, fn, phw, ic, item_m, dt, M, N, C, MC, NC, MN, gSize) \
    {                                                                                 \
        M = (fn + item_m - 1) / item_m * item_m;                                      \
        N = prlhw_str;                                                                \
        C = ic;                                                                       \
        MC = M * C;                                                                   \
        NC = N * C;                                                                   \
        MN = M * N;                                                                   \
        gSize = MN * wino_num * wino_num * bytesOf(dt);                               \
    }
inline EE wino_trans_pic(GCLHandle_t handle,
    U32 ih_str,
    U32 iw_str,
    U32 ih_off,
    U32 iw_off,
    U32 ic_str,
    U32 prh_str,
    U32 prw_str,
    U32 prc_str,
    U32 prhwc_str,
    U32 prh_off,
    U32 prw_off,
    U32 prlh_str,
    U32 prlw_str,
    U32 prlc_str,
    U32 prlhw_str,
    U32 prlhwc_str,
    Mem pic,
    Mem picTranR,
    Mem picTranRL)

{
    UNUSED(prw_str);
    UNUSED(prw_off);
    Kernel kernel;
    char kernelname[128];
    U32 ih_str4 = ih_str * 4;
    U32 ih_off4 = ih_off * 4;
    U32 prh_off4 = prh_off * 4;
    U32 gs[3] = {prh_str * 4, (prw_str / 4 + 3) / 4 * 4, ic_str};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    sprintf(kernelname, "conv_wino_trans_picbuf_right");
    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str4, iw_str, ih_off4, iw_off, prh_str, prw_str,
        prhwc_str, prh_off4, gs[0], gs[1], pic, picTranR));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    handle->t_total += handle->t_execute;
#endif
    U32 item_h = 1;
    if (prlh_str % 2 == 0) {
        item_h = 2;
    }
    if (prlh_str % 3 == 0) {
        item_h = 3;
    }
    if (prlh_str % 4 == 0) {
        item_h = 4;
    }
    gs[0] = (prlh_str / item_h + 3) / 4 * 4;
    gs[1] = prlw_str;
    gs[2] = prlc_str * 6;
    sprintf(kernelname, "conv_wino_trans_picbuf_left_%d", item_h);
    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, prh_str, prw_str, prc_str, prlh_str, prlw_str,
        prlhw_str, prlhwc_str, gs[0], gs[1], picTranR, picTranRL));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE wino_gemm(GCLHandle_t handle,
    U32 M,
    U32 N,
    U32 C,
    U32 item_m,
    U32 item_n,
    U32 flttran_str,
    U32 pictran_str,
    U32 out_str,
    U32 wino_num,
    Mem flttran,
    Mem pictran,
    Mem out)
{
    Kernel kernel;
    wino_num = wino_num * wino_num;
    char kernelname[128];
    sprintf(kernelname, "conv_wino_gemm%d_tn_%d%d", wino_num, item_m, item_n);
    U32 gs[2] = {(N + item_n - 1) / item_n, (M + item_m - 1) / item_m};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    for (U32 i = 0; i < wino_num; i++) {
        CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, C, i * flttran_str, i * pictran_str,
            i * out_str, gs[0], gs[1], flttran, pictran, out));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
        handle->t_total += handle->t_execute;
#endif
    }
    return SUCCESS;
}

inline EE wino_trans_out(GCLHandle_t handle,
    U32 wino_h,
    U32 wino_w,
    U32 pw_str,
    U32 pwh_str,
    U32 oh_str,
    U32 ow_str,
    U32 oh_off,
    U32 ow_off,
    U32 oh,
    U32 ow,
    U32 oc,
    ActivationMode activationMode,
    Mem bias,
    Mem gemm_out,
    Mem output)
{
    Kernel kernel;
    char kernelname[128];
    char modeName[16];
    switch (activationMode) {
        case ACTIVATION_RELU:
            strcpy(modeName, "_relu");
            break;
        case ACTIVATION_NULL:
            strcpy(modeName, "");
            break;
        default:
            return NOT_SUPPORTED;
    }
    sprintf(kernelname, "conv_wino_trans_outbuf%s", modeName);
    if ((oh & 3) == 0 && (ow & 3) == 0) {
        sprintf(kernelname, "conv_wino_trans_outbuf%s_align", modeName);
    }
    U32 gs[3] = {(wino_h + 3) / 4 * 4, (wino_w + 3) / 4 * 4, oc / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    CHECK_STATUS(gcl_create_kernel(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, wino_h, wino_w, pw_str, pwh_str, oh_str, ow_str, oh_off,
        ow_off, oh, ow, bias, gemm_out, output));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE convolution_wino_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes)
{
    U32 item_k = forwardRunInfo->best_k[0];
    U32 fw, fh, fc, fn;
    U32 winoTransNum = 36;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    U32 s0 = (fn + item_k - 1) / item_k * item_k;
    U32 s1 = fc;
    U32 s2 = winoTransNum;
    U32 num = s0 * s1 * s2;
    U32 byteSize = num * bytesOf(DT_F16);
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
    gclmemFilterDesc->memFormat = DF_HWCN;
    gclmemFilterDesc->host_ptr = NULL;
    *bytes = fn * fc * fh * fw * bytesOf(DT_F16);
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
    UNUSED(forwardRunInfo);
    DataType fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    U32 item_k = forwardRunInfo->best_k[0];
    U32 fn_align = (fn + item_k - 1) / item_k * item_k;
    U32 fwhc = fw * fh * fc;
    U32 fnc = fn_align * fc;

    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "conv_wino_rotate_fltbuf_%d", fw);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwhc, fnc, fn, filter->mem, tmp->mem));
    U32 gs[2] = {fwhc, fn_align};
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "conv_wino_filter_org"));
    CHECK_STATUS(
        gcl_print_buffer<F16>(handle, tmp->mem, fn_align * fc * fw * fh, "conv_wino_filter_tmp"));
#endif
    sprintf(kernelname, "conv_wino_trans_fltbuf_3x3");
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fn_align, fc, fnc, tmp->mem, fltmem->mem));
    gs[0] = fn_align;
    gs[1] = fc;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, fltmem, "conv_wino_filter_tran"));
#endif
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
    return SUCCESS;
}

EE convolution_wino_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    UNUSED(inputDesc);
    UNUSED(outputDesc);
    UNUSED(convParamSpec);
    DataType fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    U32 item_k = forwardRunInfo->best_k[0];
    U32 fn_align = (fn + item_k - 1) / item_k * item_k;
    U32 tempBufNum = fn_align * fc * fw * fh;
    U32 fltTempBufSize = tempBufNum * bytesOf(fdt);

    DataType odt;
    U32 ow, oh, oc, on;
    tensorSelectGet(outputDesc, &odt, NULL, &on, &oc, &oh, &ow);
    U32 ph = convParamSpec.padding_top;
    U32 wino_num = 6;
    U32 wino_h = (oh + 3) / 4;
    U32 wino_w = (ow + 3) / 4;
    U32 prh_str, prw_str, prc_str, prn_str, prh_off, prw_off, prhwc_str, prSize;
    calPicTranRDesc(wino_h, wino_w, wino_num, fc, fh, ph, odt, prh_str, prw_str, prc_str, prn_str,
        prh_off, prw_off, prhwc_str, prSize);

    U32 item_n = forwardRunInfo->best_w[0];
    U32 item_m = forwardRunInfo->best_k[0];
    U32 prlh_str, prlw_str, prlc_str, prln_str, prlhw_str, prlhwc_str, prlSize;
    calPtrTranRLDesc(wino_h, wino_w, wino_num, fc, item_n, odt, prlh_str, prlw_str, prlc_str,
        prln_str, prlhw_str, prlhwc_str, prlSize);

    U32 M, N, C, MC, NC, MN, gemmOutSize;
    calGemmOutDesc(wino_num, fn, prlhw_str, fc, item_m, odt, M, N, C, MC, NC, MN, gemmOutSize);

    U32 tempBufSize = (prSize + 1023) / 1024 * 1024;
    tempBufSize += (prlSize + 1023) / 1024 * 1024;
    tempBufSize += gemmOutSize;
    if (tempBufSize < fltTempBufSize) {
        tempBufSize = fltTempBufSize;
    }
    *bytes = tempBufSize;
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
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    U32 wino_num = 6;
    DataType idt;
    U32 iw, ih, ic;
    U32 fw, fh, fc, fn, pw, ph;
    U32 ow, oh, oc, on;
    ph = convParamSpec.padding_top;
    pw = convParamSpec.padding_left;
    tensorSelectGet(inputDesc, &idt, NULL, NULL, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    U32 iw_str, ih_str, ih_str4, ic_str, iw_off, ih_off;
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ic_str = input->desc.stride[2];
    ih_off = input->desc.offset[0];  // input have not pad in h axis
    iw_off = input->desc.offset[1] - pw;
    ih_str4 = ih_str * 4;

    U32 ow_str, oh_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];

    Mem pic = input->mem;
    Mem picTranR, picTranRL, gemmOut;
    U32 wino_h = (oh + 3) / 4;
    U32 wino_w = (ow + 3) / 4;
    U32 offset = 0;
    U32 prh_str, prw_str, prc_str, prn_str, prh_off, prw_off, prhwc_str, prSize;
    calPicTranRDesc(wino_h, wino_w, wino_num, ic, fh, ph, idt, prh_str, prw_str, prc_str, prn_str,
        prh_off, prw_off, prhwc_str, prSize);
    CHECK_STATUS(gcl_create_sub_buffer(prSize, &offset, tmpBuf, &picTranR));

    U32 item_n = forwardRunInfo->best_w[0];
    U32 item_m = forwardRunInfo->best_k[0];
    U32 prlh_str, prlw_str, prlc_str, prln_str, prlhw_str, prlhwc_str, prlSize;
    calPtrTranRLDesc(wino_h, wino_w, wino_num, ic, item_n, idt, prlh_str, prlw_str, prlc_str,
        prln_str, prlhw_str, prlhwc_str, prlSize);
    CHECK_STATUS(gcl_create_sub_buffer(prlSize, &offset, tmpBuf, &picTranRL));

    U32 M, N, C, MC, NC, MN, gemmOutSize;
    calGemmOutDesc(wino_num, fn, prlhw_str, ic, item_m, idt, M, N, C, MC, NC, MN, gemmOutSize);
    CHECK_STATUS(gcl_create_sub_buffer(gemmOutSize, &offset, tmpBuf, &gemmOut));

    CHECK_STATUS(wino_trans_pic(handle, ih_str, iw_str, ih_off, iw_off, ic_str, prh_str, prw_str,
        prc_str, prhwc_str, prh_off, prw_off, prlh_str, prlw_str, prlc_str, prlhw_str, prlhwc_str,
        pic, picTranR, picTranRL));
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, input, "conv_wino_input"));
    CHECK_STATUS(
        gcl_print_buffer<F16>(handle, picTranR, prSize / bytesOf(idt), "conv_wino_pictran_right"));
    CHECK_STATUS(
        gcl_print_buffer<F16>(handle, picTranRL, prlSize / bytesOf(idt), "conv_wino_pictran_left"));
#endif

    Mem fltTran = filter->mem;
    CHECK_STATUS(wino_gemm(
        handle, M, N, C, item_m, item_n, MC, NC, MN, wino_num, fltTran, picTranRL, gemmOut));
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "conv_wino_flttran"));
    CHECK_STATUS(
        gcl_print_buffer<F16>(handle, gemmOut, gemmOutSize / bytesOf(idt), "conv_wino_gemm_out"));
#endif

    Mem biasbuf = bias->mem;
    Mem outbuf = output->mem;
    CHECK_STATUS(wino_trans_out(handle, wino_h, wino_w, N, MN, oh_str, ow_str, oh_off, ow_off, oh,
        ow, oc, activationMode, biasbuf, gemmOut, outbuf));
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "conv_wino_output"));
#endif
    return SUCCESS;
}
