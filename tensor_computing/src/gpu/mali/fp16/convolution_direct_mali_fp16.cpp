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
    
    UNUSED(inputDesc);
    UNUSED(forwardRunInfo);
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);
    cl_mem inbuf, biasimg, outbuf, fltbuf;
    inbuf   = input->mem;
    fltbuf  = filter->mem;
    biasimg = bias->mem;
    outbuf  = output->mem;
    U32 fw, sw, pw, ph;
    U32 ow, oh, oc, on;
    sw = convDesc.stride_w;
    ph = convDesc.padding_bottom;
    pw = convDesc.padding_left;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, NULL, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on,  &oc,  &oh,  &ow);

    U32 iw_str, ih_str, ihw_str, ic_str, iw_off, ih_off;
    ih_str = input->desc.stride[0]; 
    iw_str = input->desc.stride[1]; 
    ic_str = input->desc.stride[2];
    ih_off = input->desc.offset[0] - ph;
    iw_off = input->desc.offset[1] - pw;
    ihw_str = ih_str * iw_str;

    U32 ow_str, oh_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];

    U32 item_w;
    if(sw == 1) item_w = (fw == 5) ? 4 : 8;
    if(sw == 2) item_w = (fw == 1) ? 8 : 4;
    if(ow < item_w) item_w = ow;
     
    char kernelname[128];
    Kernel kernel;
    if(activationMode == ACTIVATION_NULL){
        sprintf(kernelname, "conv_direct_s%d_%d%d",sw, fw, item_w);
    } else if (activationMode == ACTIVATION_RELU) {
        sprintf(kernelname, "conv_direct_s%d_relu_%d%d",sw, fw, item_w);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, ihw_str, ic_str, ih_off, iw_off, oh_str, ow_str, oh_off, ow_off, ow, inbuf, fltbuf, biasimg, outbuf));
    U32 gs[3] = {oh, (ow + item_w - 1) / item_w, (oc + 3) / 4 * on};
    U32 ls[3] = {16, 16, 1};
    U32 dim   = 3;
    gcl_set_kernelVec(handle, kernel, dim, gs, ls);
    
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "conv_direct_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "conv_direct_filter"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, bias,   "conv_direct_bias"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "conv_direct_output"));
#endif
    return SUCCESS;
}


EE convolution_direct_infer_forward_algorithm_mali_fp16(GCLHandle_t          handle,
                                                        TensorDesc           inputDesc, 
                                                        TensorDesc           filterDesc, 
                                                        ConvolutionDesc      convDesc,
                                                        TensorDesc           outputDesc,
                                                        ConvolutionPolicy    policy, 
                                                        ActivationMode       activationMode,
                                                        ForwardRunInfoMali_t forwardRunInfo) 
{
    UNUSED(handle);
    UNUSED(inputDesc); 
    UNUSED(filterDesc); 
    UNUSED(convDesc);
    UNUSED(outputDesc);
    UNUSED(policy); 
    UNUSED(activationMode);
    UNUSED(forwardRunInfo); 
    return SUCCESS;
}

EE convolution_direct_transform_filter_bytes_mali_fp16(TensorDesc            filterDesc, 
                                                       ForwardRunInfoMali_t  forwardRunInfo,
                                                       GCLMemDesc_t          gclmemFilterDesc,
                                                       U32*                  bytes)
{
    UNUSED(forwardRunInfo);
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    U32 item_c = 4;
    U32 item_k = 4;
    U32 s0, s1, s2;
    U32 num, byteSize;
    s0 = fw * fh;
    s1 = (fc + item_c - 1) / item_c;
    s2 = (fn + item_k - 1) / item_k;
    num = s0 * s1 * s2 * item_c * item_k;
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
    gclmemFilterDesc->memFormat = DF_NCHWN4C4;
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
    UNUSED(forwardRunInfo);
    DataType   fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    U32 fwh = fw * fh;
    U32 item_c = 4;
    U32 item_k = 4;
    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "conv_direct_trans_fltbuf_%d%d",item_c, item_k);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fwh, fc, fn, filter->mem, fltmem->mem));
    U32 gs[3] = {fwh, (fc + item_c - 1) / item_c, (fn + item_k - 1) / item_k * item_k};
    U32 ls[3] = {16, 16, 1};
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
    CHECK_STATUS(direct_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo,
                                                   biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
                                                              
    return SUCCESS;
}
