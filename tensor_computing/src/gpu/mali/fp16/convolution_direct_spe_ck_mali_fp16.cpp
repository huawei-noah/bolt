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
#include "gpu/mali/fp16/convolution_direct_spe_ck_mali_fp16.h"

inline EE direct_spe_ck_core_mali_fp16(GCLHandle_t          handle,
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
    UNUSED(bias);
    UNUSED(tmpBytes);
    UNUSED(tmpBuf);
    UNUSED(activationMode);

    cl_mem inbuf, outbuf, fltbuf;
    inbuf   = input->mem;
    fltbuf  = filter->mem;
    outbuf  = output->mem;
    U32 fn, fc, fw, sw;
    U32 ow, oh, oc, on;
    sw = convDesc.stride_w;
    tensorSelectGet(filterDesc, NULL, NULL, &fn,  &fc,  NULL, &fw);
    tensorSelectGet(outputDesc, NULL, NULL, &on,  &oc,  &oh,  &ow);
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {16, 16, 1};
    U32 dim;
    char kernelname[128];

    if(fn == 1 && fc == 4 && fw == 1){//fc = orgfc + fn
        U32 iw_str, ih_str;
        iw_str = input->desc.stride[0]; 
        ih_str = input->desc.stride[1]; 
        U32 ow_str, oh_str, ow_off, oh_off;
        ow_str = output->desc.stride[0];
        oh_str = output->desc.stride[1];
        ow_off = output->desc.offset[0];
        oh_off = output->desc.offset[1];
        if(output->desc.memFormat != DF_NCHW) return NOT_SUPPORTED;
        U32 item_w = 2;
        U32 item_h = 1;
        U32 ew = ow % item_w;
        sprintf(kernelname, "conv_direct_s%d_spe_f1c3k1_%d", sw, ew);
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ow_str, ow_off, oh_off, ow >> 1, inbuf, fltbuf, outbuf));//c = 3 k = 1, bias val has been set in fltbuf
        gs[0] = (ow + item_w - 1) / item_w;
        gs[1] = (oh + item_h - 1) / item_h;
        dim   = 2;
        gcl_set_kernelVec(handle, kernel, dim, gs, ls);
    } else {
        return NOT_SUPPORTED;
    }
    
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "conv_direct_spe_ck_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "conv_direct_spe_ck_filter"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "conv_direct_spe_ck_output"));
#endif
    return SUCCESS;
}


EE convolution_direct_spe_ck_infer_forward_algorithm_mali_fp16(GCLHandle_t          handle,
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

EE convolution_direct_spe_ck_transform_filter_bytes_mali_fp16(TensorDesc            filterDesc, 
                                                              ForwardRunInfoMali_t  forwardRunInfo,
                                                              GCLMemDesc_t          gclmemFilterDesc,
                                                              U32*                  bytes)
{
    UNUSED(forwardRunInfo);
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    U32 s0, s1, s2;
    U32 num, byteSize;
    if(fn == 1 && fc == 3 && fw == 1){
        s0 = fw * fh;
        s1 = fc + fn;//set bias val in flt
        s2 = fn;
        gclmemFilterDesc->memFormat = DF_NCHW;
    } else {
        return NOT_SUPPORTED;
    }
    num = s0 * s1 * s2;
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
    gclmemFilterDesc->flags     = CL_MEM_READ_ONLY;
    gclmemFilterDesc->host_ptr  = NULL;
    *bytes = 0;
    return SUCCESS;
}

EE convolution_direct_spe_ck_transform_filter_mali_fp16(GCLHandle_t          handle,
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
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc + fn, fh, fw);//set bias val in flt
    U32 size = tensorNumBytes(*fltmemDesc);
    CHECK_STATUS(gcl_trans_memory(handle, filter, fltmem, &size, DEVICE_BUF_TO_BUF, CL_FALSE));
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "conv_direct_spe_ck_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, fltmem, "conv_direct_spe_ck_filter_tran"));
#endif
    return SUCCESS;
}

EE convolution_direct_spe_ck_infer_forward_tmp_bytes_mali_fp16(TensorDesc            inputDesc, 
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

EE convolution_direct_spe_ck_mali_fp16(GCLHandle_t          handle,
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
    CHECK_STATUS(direct_spe_ck_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo,
                                                   biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode));
                                                              
    return SUCCESS;
}
