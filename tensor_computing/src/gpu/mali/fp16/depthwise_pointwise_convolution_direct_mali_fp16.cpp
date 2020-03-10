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
#include "gpu/mali/fp16/depthwise_pointwise_convolution_direct_mali_fp16.h"

inline EE depthwise_pointwise_core_mali_fp16(GCLHandle_t          handle,
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
                                             ActivationMode       pointwiseActivationMode){
    UNUSED(inputDesc);
    UNUSED(forwardRunInfo);
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);

    cl_mem inbuf, biasimg0, biasimg1, outbuf, fltbuf0, fltbuf1, tmp;
    inbuf    = input->mem;
    fltbuf0  = filter[0].mem;
    fltbuf1  = filter[1].mem;
    biasimg0 = bias[0].mem;
    biasimg1 = bias[1].mem;
    outbuf   = output->mem;
    tmp      = tmpBuf->mem;
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
    U32 w_align, item_w;
    item_w = 8;
    if(ow < item_w) item_w = ow;
    w_align = (ow + item_w - 1) / item_w * item_w;
    th_str  = oh;
    tw_str  = w_align;
    th_off  = 0;
    tw_off  = 0;
    thw_str = th_str * tw_str;

    U32 ow_str, oh_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];

    if(sw == 1) item_w = (fw == 5) ? 4 : 8;
    if(sw == 2) item_w = 4;
    if(ow < item_w) item_w = ow;

    char kernelname[128];
    Kernel kernel;
    if(depthwiseActivationMode == ACTIVATION_NULL){
        sprintf(kernelname, "conv_depthwise_s%d_%d%d",sw, fw, item_w);
    } else if (depthwiseActivationMode == ACTIVATION_RELU) {
        sprintf(kernelname, "conv_depthwise_s%d_relu_%d%d",sw, fw, item_w);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, th_str, tw_str, th_off, tw_off, ow, inbuf, fltbuf0, biasimg0, tmp)); 
    U32 gs[3] = {oh, (ow + item_w - 1) / item_w, (fc + 3) / 4};
    U32 ls[3] = {16, 16, 1};
    U32 dim   = 3;
    gcl_set_kernelVec(handle, kernel, dim, gs, ls);
    
#ifdef _DEBUG
    
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,      "conv_depthwise_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &filter[0], "conv_depthwise_filter"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &bias[0],   "conv_depthwise_bias"));
    CHECK_STATUS(gcl_print_buffer<F16>(handle, tmp, th_str * tw_str * fc, "conv_depthwise_output_tmp"));
#endif
    fw = 1;
    sw = 1;
    item_w = 8;
    if(ow < item_w) item_w = ow;
    if(pointwiseActivationMode == ACTIVATION_NULL){
        sprintf(kernelname, "conv_direct_s%d_%d%d",sw, fw, item_w);
    } else if (pointwiseActivationMode == ACTIVATION_RELU) {
        sprintf(kernelname, "conv_direct_s%d_relu_%d%d",sw, fw, item_w);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }

    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, th_str, thw_str, ic_str, th_off, tw_off, oh_str, ow_str, oh_off, ow_off, ow, tmp, fltbuf1, biasimg1, outbuf));
    U32 gsp[3] = {oh, (ow + item_w - 1) / item_w, (oc + 3) / 4 * on};
    U32 lsp[3] = {16, 16, 1};
    U32 dimp   = 3;
    gcl_set_kernelVec(handle, kernel, dimp, gsp, lsp);
    
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimp, gsp, lsp, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &filter[1], "conv_direct_filter"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &bias[1],   "conv_direct_bias"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output,     "conv_direct_output"));
#endif
    return SUCCESS;
}


EE depthwise_pointwise_convolution_direct_infer_forward_algorithm_mali_fp16(GCLHandle_t          handle,
                                                                            TensorDesc           inputDesc, 
                                                                            TensorDesc           filterDesc, 
                                                                            ConvolutionDesc      convDesc,
                                                                            TensorDesc           outputDesc,
                                                                            ConvolutionPolicy    policy, 
                                                                            ActivationMode       depthwiseActivationMode,
                                                                            ForwardRunInfoMali_t forwardRunInfo) 
{
    UNUSED(handle);
    UNUSED(inputDesc);
    UNUSED(filterDesc); 
    UNUSED(convDesc);
    UNUSED(outputDesc);
    UNUSED(policy); 
    UNUSED(depthwiseActivationMode);
    UNUSED(forwardRunInfo); 
    return SUCCESS;
}

EE depthwise_pointwise_convolution_direct_transform_filter_bytes_mali_fp16(TensorDesc            filterDesc, 
                                                                           ForwardRunInfoMali_t  forwardRunInfo,
                                                                           GCLMemDesc_t          gclmemFilterDesc,
                                                                           U32*                  bytes)
{
    UNUSED(forwardRunInfo);
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    U32 item_k = 4;
    U32 s0, s1, s2;
    U32 num, byteSize;
    s0 = fw * fh;
    s1 = (fc + item_k - 1) / item_k;
    s2 = 1;
    num = s0 * s1 * s2 * item_k;
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
    
    U32 item_c = 4;
    s0 = item_k;
    s1 = (fc + item_c - 1) / item_c;
    s2 = (fn + item_k - 1) / item_k;
    num = s0 * s1 * s2 * item_c;
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
    gclmemFilterDesc[1].memFormat = DF_NCHWN4C4;
    gclmemFilterDesc[1].flags     = CL_MEM_READ_WRITE;
    gclmemFilterDesc[1].host_ptr  = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE depthwise_pointwise_convolution_direct_transform_filter_mali_fp16(GCLHandle_t          handle,
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
    U32 item_k = 4;
    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "conv_depthwise_trans_fltbuf_%d", item_k);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, filter[0].mem, fltmem[0].mem));
    U32 gs[3] = {fwh, (fc + item_k - 1) / item_k};
    U32 ls[3] = {16, 16, 1};
    U32 dim   = 2;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, &filter[0], "conv_depthwise_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &fltmem[0], "conv_depthwise_filter_tran"));
#endif
    
    U32 item_c = 4;
    fwh = 1;
    sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d",item_c, item_k);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, filter[1].mem, fltmem[1].mem));
    U32 gsc[3] = {fwh, (fc + item_c - 1) / item_c, (fn + item_k - 1) / item_k * item_k};
    U32 lsc[3] = {16, 16, 1};
    U32 dimc   = 3;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dimc, gsc, lsc, kernelname));
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, &filter[1], "conv_direct_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, &fltmem[1], "conv_direct_filter_tran"));
#endif
    return SUCCESS;
}

EE depthwise_pointwise_convolution_direct_infer_forward_tmp_bytes_mali_fp16(TensorDesc            inputDesc, 
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
    UNUSED(forwardRunInfo);
    
    U32 w_align;
    U32 item_w = 8;
    if(ow < item_w) item_w = ow;
    w_align = (ow + item_w - 1) / item_w * item_w;
    *bytes = oh * w_align * ((fc + 3) / 4) * 4 * bytesOf(odt);
    return SUCCESS;
}

EE depthwise_pointwise_convolution_direct_mali_fp16(GCLHandle_t          handle,
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
                                                    ActivationMode       pointwiseActivationMode){
    CHECK_STATUS(depthwise_pointwise_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo,
                biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, depthwiseActivationMode, pointwiseActivationMode));
                                                              
    return SUCCESS;
}
