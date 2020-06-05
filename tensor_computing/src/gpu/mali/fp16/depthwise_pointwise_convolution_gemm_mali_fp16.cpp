// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "sys.h"
#include "tensor_desc.h"
#include "type.h"
#include "error.h"
#include "tensor_computing_type.h"
#include "gpu/mali/fp16/depthwise_pointwise_convolution_gemm_mali_fp16.h"


inline EE depthwise_pointwise_gemm_core_mali_fp16(GCLHandle_t          handle,
                                                  TensorDesc           inputDesc, 
                                                  const GCLMem_t       input,
                                                  TensorDesc           filterDesc, 
                                                  const GCLMem_t       filter,
                                                  ConvolutionDesc      convDesc,
                                                  ForwardRunInfoMali_t forwardRunInfo,
                                                  TensorDesc           biasDesc, 
                                                  const GCLMem_t       bias,
                                                  U32                  tmpBytes, 
                                                  GCLMem_t             tmpBuf,
                                                  TensorDesc           outputDesc, 
                                                  GCLMem_t             output,
                                                  ActivationMode       depthwiseActivationMode,
                                                  ActivationMode       pointwiseActivationMode) {
    UNUSED(inputDesc);
    UNUSED(forwardRunInfo);
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);

    cl_mem inbuf, biasimg, biasbuf, outbuf, fltbuf0, fltbuf1, tmp;
    inbuf   = input->mem;
    fltbuf0 = filter[0].mem;
    fltbuf1 = filter[1].mem;
    biasimg = bias[0].mem;
    biasbuf = bias[1].mem;
    outbuf  = output->mem;
    tmp     = tmpBuf->mem;
    U32 fw, sw, pw, ph, fc;
    U32 ow, oh, oc, on;
    sw = convDesc.stride_w;
    ph = convDesc.padding_bottom;
    pw = convDesc.padding_left;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, &fc, NULL, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on,  &oc, &oh,  &ow);

    U32 iw_str, ih_str, ihw_str, ic_str, ih_off, iw_off;
    ih_str = input->desc.stride[0]; 
    iw_str = input->desc.stride[1]; 
    ic_str = input->desc.stride[2]; 
    ih_off = input->desc.offset[0] - ph;
    iw_off = input->desc.offset[1] - pw;
    ihw_str = ih_str * iw_str;

    U32 th_str, tw_str, th_off, tw_off, thw_str;
    U32 item_wd, item_whp, item_kp;
    item_wd  = forwardRunInfo->best_w[0];
    item_whp = forwardRunInfo->best_w[1];
    item_kp = forwardRunInfo->best_k[1];
    th_str  = oh;
    tw_str  = ow;
    th_off  = 0;
    tw_off  = 0;
    thw_str = ALIGN(th_str * tw_str, item_whp);

    U32 ow_str, oh_str, ow_off, oh_off, ohw_str;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];
    ohw_str = oh_str * ow_str;

    U32 gs[3] = {oh, ALIGN(ow, item_wd) / item_wd, ALIGN(fc, 4) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim   = 3;
    char kernelname[128];
    Kernel kernel;
    if(depthwiseActivationMode == ACTIVATION_NULL) {
        sprintf(kernelname, "conv_depthwise_s%d_ncwh_%d%d",sw, fw, item_wd);
    } else if (depthwiseActivationMode == ACTIVATION_RELU) {
        sprintf(kernelname, "conv_depthwise_s%d_relu_ncwh_%d%d",sw, fw, item_wd);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, th_str, tw_str, thw_str, th_off, tw_off, ow, gs[0], gs[1], inbuf, fltbuf0, biasimg, tmp)); 
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
    
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,      "conv_depthwise_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &filter[0], "conv_depthwise_filter"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &bias[0],   "conv_depthwise_bias"));
    CHECK_STATUS(gcl_print_buffer<F16>(handle, tmp, thw_str * fc, "conv_depthwise_output_tmp"));
    handle->t_total += handle->t_execute;
#endif
    if(pointwiseActivationMode == ACTIVATION_NULL) {
        sprintf(kernelname, "gemm_tn_ncwhc4_%d%d", item_kp, item_whp);
    } else if (pointwiseActivationMode == ACTIVATION_RELU) {
        sprintf(kernelname, "gemm_tn_relu_ncwhc4_%d%d", item_kp, item_whp);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }

    U32 M, N, K;
    M = ALIGN(oc, item_kp);;
    N = thw_str;
    K = fc;
    U32 gsp[3] = {N / item_whp, M / item_kp};
    U32 lsp[3] = {0, 0};
    U32 dimp   = 2;
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, M, N, K, oh, ow, oc, oh_str, ow_str, ohw_str, oh_off, ow_off, gsp[0], gsp[1], fltbuf1, tmp, biasbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dimp, gsp, lsp, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimp, gsp, lsp, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &filter[1], "conv_direct_filter"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &bias[1],   "conv_direct_bias"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output,     "conv_direct_output"));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

EE depthwise_pointwise_convolution_gemm_transform_filter_bytes_mali_fp16(TensorDesc            filterDesc, 
                                                                         ForwardRunInfoMali_t  forwardRunInfo,
                                                                         GCLMemDesc_t          gclmemFilterDesc,
                                                                         U32*                  bytes)
{
    UNUSED(forwardRunInfo);
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    U32 item_kd = forwardRunInfo->best_k[0];
    U32 item_kp = forwardRunInfo->best_k[1];
    U32 item_c  = forwardRunInfo->best_c[1];
    U32 s0, s1, s2;
    U32 num, byteSize;
    s0 = fw * fh;
    s1 = ALIGN(fc, item_kd) / item_kd;
    s2 = 1;
    num = s0 * s1 * s2 * item_kd;
    byteSize = num * bytesOf(DT_F16);
    gclmemFilterDesc[0].stride[0] = s0;
    gclmemFilterDesc[0].stride[1] = s1;
    gclmemFilterDesc[0].stride[2] = s2;
    gclmemFilterDesc[0].offset[0] = 0;
    gclmemFilterDesc[0].offset[1] = 0;
    gclmemFilterDesc[0].offset[2] = 0;
    gclmemFilterDesc[0].num       = num;
    gclmemFilterDesc[0].byteSize  = byteSize;
    gclmemFilterDesc[0].memType   = GCL_MEM_BUF;
    gclmemFilterDesc[0].memFormat = DF_NHWCN4;
    gclmemFilterDesc[0].flags     = CL_MEM_READ_WRITE;
    gclmemFilterDesc[0].host_ptr  = NULL;
    
    s0 = ALIGN(fn, item_kp);
    s1 = ALIGN(fc, item_c);
    s2 = 1;
    num = s0 * s1 * s2;
    byteSize = num * bytesOf(DT_F16);
    gclmemFilterDesc[1].stride[0] = s0;
    gclmemFilterDesc[1].stride[1] = s1;
    gclmemFilterDesc[1].stride[2] = s2;
    gclmemFilterDesc[1].offset[0] = 0;
    gclmemFilterDesc[1].offset[1] = 0;
    gclmemFilterDesc[1].offset[2] = 0;
    gclmemFilterDesc[1].num       = num;
    gclmemFilterDesc[1].byteSize  = byteSize;
    gclmemFilterDesc[1].memType   = GCL_MEM_BUF;
    gclmemFilterDesc[1].memFormat = DF_HWCN;
    gclmemFilterDesc[1].flags     = CL_MEM_READ_WRITE;
    gclmemFilterDesc[1].host_ptr  = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE depthwise_pointwise_convolution_gemm_transform_filter_mali_fp16(GCLHandle_t          handle,
                                                                   TensorDesc           filterDesc,
                                                                   GCLMem_t             filter,
                                                                   ForwardRunInfoMali_t forwardRunInfo,
                                                                   TensorDesc*          fltmemDesc,
                                                                   GCLMem_t             fltmem)
{
    UNUSED(forwardRunInfo);
    DataType   fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    U32 fwh = fw * fh;
    U32 item_kd = forwardRunInfo->best_k[0];
    U32 item_kp = forwardRunInfo->best_k[1];
    U32 item_c  = forwardRunInfo->best_c[1];
    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "conv_depthwise_trans_fltbuf_%d", item_kd);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, filter[0].mem, fltmem[0].mem));
    U32 gs[3] = {fwh, ALIGN(fc, item_kd) / item_kd};
    U32 ls[3] = {0, 0, 0};
    U32 dim   = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, &filter[0], "conv_depthwise_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &fltmem[0], "conv_depthwise_filter_tran"));
#endif
    
    fwh = 1;
    U32 fn_align = ALIGN(fn, item_kp);
    sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d",item_c, 0);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn_align, filter[1].mem, fltmem[1].mem));
    U32 gsc[3] = {fwh, ALIGN(fc, item_c) / item_c, fn_align};
    U32 lsc[3] = {0, 0, 0};
    U32 dimc   = 3;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimc, gsc, lsc, kernelname));
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, &filter[1], "conv_direct_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &fltmem[1], "conv_direct_filter_tran"));
#endif
    return SUCCESS;
}

EE depthwise_pointwise_convolution_gemm_infer_forward_tmp_bytes_mali_fp16(TensorDesc            inputDesc, 
                                                                          TensorDesc            filterDesc, 
                                                                          TensorDesc            outputDesc,
                                                                          ConvolutionDesc       convDesc, 
                                                                          ForwardRunInfoMali_t  forwardRunInfo,
                                                                          U32*                  bytes) 
{
    DataType odt;
    U32 oh, ow, fc;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, &fc,  NULL, NULL);
    tensorSelectGet(outputDesc, &odt, NULL, NULL, NULL, &oh,  &ow);
    UNUSED(inputDesc); 
    UNUSED(convDesc); 
    
    U32 N;
    U32 item_wh = forwardRunInfo->best_w[1];
    N = ALIGN(oh * ow, item_wh);
    *bytes = N * ALIGN(fc, 4) * bytesOf(odt);
    return SUCCESS;
}

EE depthwise_pointwise_convolution_gemm_mali_fp16(GCLHandle_t          handle,
                                                  TensorDesc           inputDesc, 
                                                  const GCLMem_t       input,
                                                  TensorDesc           filterDesc, 
                                                  const GCLMem_t       filter,
                                                  ConvolutionDesc      convDesc,
                                                  ForwardRunInfoMali_t forwardRunInfo,
                                                  TensorDesc           biasDesc, 
                                                  const GCLMem_t       bias,
                                                  U32                  tmpBytes, 
                                                  GCLMem_t             tmpBuf,
                                                  TensorDesc           outputDesc, 
                                                  GCLMem_t             output,
                                                  ActivationMode       depthwiseActivationMode,
                                                  ActivationMode       pointwiseActivationMode) {
    CHECK_STATUS(depthwise_pointwise_gemm_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo,
                biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, depthwiseActivationMode, pointwiseActivationMode));
                                                              
    return SUCCESS;
}
