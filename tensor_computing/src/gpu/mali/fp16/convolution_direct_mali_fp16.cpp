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

#include "gpu/mali/fp16/convolution_mali_fp16.h"
#include "gpu/mali/fp16/convolution_direct_mali_fp16.h"

inline EE direct_core_nchw_to_ncwhc4_mali_fp16(GCLHandle_t          handle,
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
                                               ActivationMode       activationMode){
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);
    cl_mem inbuf, biasmem, outbuf, fltbuf;
    inbuf   = input->mem;
    fltbuf  = filter->mem;
    biasmem = bias->mem;
    outbuf  = output->mem;
    U32 iw, ih;
    U32 fw, fh, fn, sw, pw, ph;
    U32 ow, oh, oc, on;
    sw = convDesc.stride_w;
    ph = convDesc.padding_bottom;
    pw = convDesc.padding_left;
    tensorSelectGet(inputDesc,  NULL, NULL, NULL, NULL, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn,  NULL, &fh, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on,  &oc,  &oh, &ow);
    U32 iw_str, ih_str, iwh_str, ic_str, iw_off, ih_off;
    iw_str = input->desc.stride[0]; 
    ih_str = input->desc.stride[1]; 
    ic_str = input->desc.stride[2];
    iw_off = input->desc.offset[0] - pw;
    ih_off = input->desc.offset[1] - ph;
    iwh_str = iw_str * ih_str;

    U32 ow_str, oh_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];

    U32 item_w = forwardRunInfo->best_w[0];
    char kernelname[128];
    char modeName[16];
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim;
    Kernel kernel;
    switch(activationMode) {
        case ACTIVATION_RELU:
            strcpy(modeName, "relu_");
            break;
        case ACTIVATION_NULL:
            strcpy(modeName, "");
            break;
        default:
            return NOT_SUPPORTED;
    }
    sprintf(kernelname, "conv_direct_s%d_nchw_to_ncwhc4_%s%d%d",sw, modeName, fw, item_w);
    gs[0] = (ow + item_w - 1) / item_w;
    gs[1] = oh; 
    gs[2] = (oc + 3) / 4 * on;
    dim   = 3;
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, iwh_str, ic_str, iw_off, ih_off, oh_str, ow_str, oh_off, ow_off, ow, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
    
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "conv_direct_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "conv_direct_filter"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, bias,   "conv_direct_bias"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "conv_direct_output"));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}

inline EE direct_core_mali_fp16(GCLHandle_t          handle,
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
                                ActivationMode       activationMode){
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);
    cl_mem inbuf, biasmem, outbuf, fltbuf;
    inbuf   = input->mem;
    fltbuf  = filter->mem;
    biasmem = bias->mem;
    outbuf  = output->mem;
    U32 iw, ih;
    U32 fw, fh, fn, sw, pw, ph;
    U32 ow, oh, oc, on;
    sw = convDesc.stride_w;
    ph = convDesc.padding_bottom;
    pw = convDesc.padding_left;
    tensorSelectGet(inputDesc,  NULL, NULL, NULL, NULL, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn,  NULL, &fh, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on,  &oc,  &oh, &ow);

    U32 iw_str, ih_str, ihw_str, ic_str, iw_off, ih_off;
    ih_str = input->desc.stride[0]; 
    iw_str = input->desc.stride[1]; 
    ic_str = input->desc.stride[2];
    ih_off = input->desc.offset[0] - ph;
    iw_off = input->desc.offset[1] - pw;
    ihw_str = ih_str * iw_str;

    U32 ow_str, oh_str, ohw_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];
    ohw_str = oh_str * ow_str;

    U32 item_w = forwardRunInfo->best_w[0];
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    char kernelname[128];
    char modeName[16];
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim;
    Kernel kernel;
    switch(activationMode) {
        case ACTIVATION_RELU:
            strcpy(modeName, "relu_");
            break;
        case ACTIVATION_NULL:
            strcpy(modeName, "");
            break;
        default:
            return NOT_SUPPORTED;
    }
    if(item_k == 0) {
        if((ih_str > 1 || iw_str > 1) && (item_c != 4)) CHECK_STATUS(NOT_SUPPORTED);
        sprintf(kernelname, "conv_direct_spe_fwhs1_%s%d", modeName, item_c);
        ic_str = filter->desc.stride[1];
        ow = fn;
        gs[0] = fn;
        gs[1] = 1;
        gs[2] = 1;
        dim   = 1;
    } else {
        item_k = item_k >> 2;
        sprintf(kernelname, "conv_direct_s%d_%s%d%d%d",sw, modeName, fw, item_w, item_k);
        gs[0] = oh; 
        gs[1] = (ow + item_w - 1) / item_w;
        gs[2] = (oc + 3) / 4 * on / item_k;
        dim   = 3;
    }
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, oh_str, ohw_str, oh_off, ow_off, ow, gs[0], gs[1], inbuf, fltbuf, biasmem, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
    
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "conv_direct_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "conv_direct_filter"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, bias,   "conv_direct_bias"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "conv_direct_output"));
    handle->t_total += handle->t_execute;
#endif
    return SUCCESS;
}


EE convolution_direct_transform_filter_bytes_mali_fp16(TensorDesc            filterDesc, 
                                                       ForwardRunInfoMali_t  forwardRunInfo,
                                                       GCLMemDesc_t          gclmemFilterDesc,
                                                       U32*                  bytes)
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
    if(item_k == 0) {
        s0 = fn;
        s1 = (fc + item_c - 1) / item_c;
        s2 = 1;
        DataFormat df = DF_CHWNC4;
        if(item_c == 8)  df = DF_CHWNC8;
        if(item_c == 16) df = DF_CHWNC16;
        gclmemFilterDesc->memFormat = df;
        num = s0 * s1 * s2 * item_c;
    } else if(item_c == 4) {
        s0 = fw * fh * (item_k >> 2);
        s1 = (fc + item_c - 1) / item_c;
        s2 = (fn + item_k - 1) / item_k;
        gclmemFilterDesc->memFormat = DF_NCHWN4C4;
        num = s0 * s1 * s2 * item_c * item_k / (item_k >> 2);
    } else if(item_c == 1) {
        s0 = fw * fh;
        s1 = fc;
        s2 = (fn + item_k - 1) / item_k;
        gclmemFilterDesc->memFormat = DF_NCHWN4;
        num = s0 * s1 * s2 * item_k;
    }
    byteSize = num * bytesOf(DT_F16);
    gclmemFilterDesc->stride[0] = s0;
    gclmemFilterDesc->stride[1] = s1;
    gclmemFilterDesc->stride[2] = s2;
    gclmemFilterDesc->offset[0] = 0;
    gclmemFilterDesc->offset[1] = 0;
    gclmemFilterDesc->offset[2] = 0;
    gclmemFilterDesc->num       = num;
    gclmemFilterDesc->byteSize  = byteSize;
    gclmemFilterDesc->memType   = GCL_MEM_BUF;
    gclmemFilterDesc->flags     = CL_MEM_READ_WRITE;
    gclmemFilterDesc->host_ptr  = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE convolution_direct_transform_filter_mali_fp16(GCLHandle_t          handle,
                                                 TensorDesc           filterDesc,
                                                 GCLMem_t             filter,
                                                 ForwardRunInfoMali_t forwardRunInfo,
                                                 TensorDesc*          fltmemDesc,
                                                 GCLMem_t             fltmem)
{
    DataType   fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    U32 fwh = fw * fh;
    U32 item_c = forwardRunInfo->best_c[0];
    U32 item_k = forwardRunInfo->best_k[0];
    U32 nk = item_k;
    if(item_k == 0) item_k = fn;
    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d",item_c, nk);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, filter->mem, fltmem->mem));
    U32 gs[3] = {fwh, (fc + item_c - 1) / item_c, (fn + item_k - 1) / item_k * item_k};
    U32 ls[3] = {0, 0, 0};
    U32 dim   = 3;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "conv_direct_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, fltmem, "conv_direct_filter_tran"));
#endif
    return SUCCESS;
}

EE convolution_direct_infer_forward_tmp_bytes_mali_fp16(TensorDesc            inputDesc, 
                                                        TensorDesc            filterDesc, 
                                                        TensorDesc            outputDesc,
                                                        ConvolutionDesc       convDesc, 
                                                        ForwardRunInfoMali_t  forwardRunInfo,
                                                        U32*                  bytes) 
{
    UNUSED(inputDesc); 
    UNUSED(filterDesc); 
    UNUSED(outputDesc);
    UNUSED(convDesc); 
    UNUSED(forwardRunInfo);
    *bytes = 0;
    return SUCCESS;
}

EE convolution_direct_mali_fp16(GCLHandle_t          handle,
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
                                ActivationMode       activationMode){
    U32 fw, fh, ih, iw;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, &fh, &fw);
    tensorSelectGet(inputDesc,  NULL, NULL, NULL, NULL, &ih, &iw);
    if(inputDesc.df == DF_NCHW || (fw == 1 && fw == 1 && ih == 1 && iw == 1)) {
        CHECK_STATUS(direct_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo,
            biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
    } else if(inputDesc.df == DF_NCHW_ORG_MALI){
        CHECK_STATUS(direct_core_nchw_to_ncwhc4_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo,
            biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}
