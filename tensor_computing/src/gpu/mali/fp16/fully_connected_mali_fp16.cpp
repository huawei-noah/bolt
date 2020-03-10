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
#include "gpu/mali/fp16/fully_connected_mali_fp16.h"
#define IC 4
#define IK 4
#define CAL_ITEM_W(w, h, iw) {\
    iw = (64 + h - 1) / h;\
    iw = (iw > w) ? w : iw;\
}
inline EE fully_connected_checkpara_mali_fp16(TensorDesc     inputDesc, 
                                              TensorDesc     filterDesc, 
                                              TensorDesc     outputDesc) { 
    if(inputDesc.dt != outputDesc.dt || inputDesc.dt != filterDesc.dt || inputDesc.dt != DT_F16) return NOT_MATCH;
    return SUCCESS;
}


inline EE fully_connected_core_mali_fp16(GCLHandle_t          handle,
                                         TensorDesc           inputDesc, 
                                         const GCLMem_t       input,
                                         TensorDesc           filterDesc, 
                                         const GCLMem_t       filter,
                                         TensorDesc           biasDesc, 
                                         const GCLMem_t       bias,
                                         U32                  tmpBytes, 
                                         GCLMem_t             tmpBuf,
                                         TensorDesc           outputDesc, 
                                         GCLMem_t             output) {
    UNUSED(biasDesc);
    UNUSED(tmpBytes);
    UNUSED(outputDesc);

    U32 iw, ih, ih_str, iw_str, ih_off, iw_off, ihy_str, ihw_str;
    U32 oh_str, ow_str, oh_off, ow_off;
    U32 fw, fh, fc, fn, fhy_str, fhw_str, fwc_str;
    cl_mem inbuf, fltbuf, biasimg, outbuf, tmp;
    inbuf   = input->mem;
    fltbuf  = filter->mem;
    biasimg = bias->mem;
    outbuf  = output->mem;
    tmp     = tmpBuf->mem;

    tensorSelectGet(inputDesc,  NULL, NULL, NULL, NULL, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn,  &fc,  &fh, &fw);
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];
    U32 item_w;
    CAL_ITEM_W(iw, ih, item_w); 
    ihy_str = ih_str * item_w;
    ihw_str = ih_str * iw_str;
    fc      = (fc + IC - 1) / IC;
    fn      = (fn + IK - 1) / IK;
    fhy_str = fh * item_w;
    fhw_str = fh * fw;
    fwc_str = fw * fc;

    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel_binary(handle, "fc_p1", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, item_w, ih_str, iw_str, ih_off, iw_off, ihy_str, ihw_str, fh, fw, fc, fn, fhy_str, fhw_str, fwc_str, fltbuf, inbuf, tmp));
    U32 gs[3] = {fh, item_w, fn};
    U32 ls[3] = {16, 16, 1};
    U32 dim   = 3;
    gcl_set_kernelVec(handle, kernel, dim, gs, ls);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "fc_p1"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,                      "fc_p1_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter,                     "fc_p1_filter"));
    CHECK_STATUS(gcl_print_buffer<F16>(handle, tmp, fh * item_w * fn * IK, "fc_p1_output"));
#endif
    CHECK_STATUS(gcl_create_kernel_binary(handle, "fc_p2", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fh * item_w, fn, oh_str, ow_str, oh_off, ow_off, tmp, biasimg, outbuf));
    U32 gs2 = fn;
    U32 ls2 = 256;
    dim = 1;
    gcl_set_kernelVec(handle, kernel, dim, &gs2, &ls2);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs2, &ls2, "fc_p2"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, bias,   "fc_p2_bias"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "fc_p2_output"));
#endif
    return SUCCESS;
}

EE fully_connected_transform_filter_bytes_mali_fp16(TensorDesc            filterDesc, 
                                                    GCLMemDesc_t          gclmemFilterDesc,
                                                    U32*                  bytes){
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    U32 s0, s1, s2;
    U32 num, byteSize;
    s0 = fh;
    s1 = fw;
    s2 = ((fc + IC - 1) / IC) * ((fn + IK - 1) / IK);
    num = s0 * s1 * s2 * IC * IK;
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
    gclmemFilterDesc->memFormat = DF_NCWHN4C4;
    gclmemFilterDesc->flags     = CL_MEM_READ_WRITE;
    gclmemFilterDesc->host_ptr  = NULL;
    
    *bytes = 0;
    return SUCCESS;
}

EE fully_connected_transform_filter_mali_fp16(GCLHandle_t          handle,
                                              TensorDesc           filterDesc,
                                              GCLMem_t             filter,
                                              TensorDesc*          fltmemDesc,
                                              GCLMem_t             fltmem){
    DataType   fdt;
    DataFormat fdf;
    U32 fw, fh, fc, fn;
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    U32 item_c = IC;
    U32 item_k = IK;
    U32 fwh = fw * fh;
    char kernelname[128];
    Kernel kernel;
    sprintf(kernelname, "fc_trans_fltbuf_%d%d",item_c, item_k);
    CHECK_STATUS(gcl_get_kernel_from_map(handle, kernelname, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, fw, fh, fwh, fc, fn, filter->mem, fltmem->mem));
    U32 gs[3] = {fw, fh, (fc + IC - 1) / IC * ((fn + IK - 1) / IK) * IK};
    U32 ls[3] = {16, 16, 1};
    U32 dim   = 3;
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    *fltmemDesc = tensor4df(fdt, fdf, fn, fc, fh, fw);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, filter, "fc_filter_org"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, fltmem, "fc_filter_tran"));
#endif
    return SUCCESS;
}

EE fully_connected_infer_forward_tmp_bytes_mali_fp16(TensorDesc            inputDesc, 
                                                     TensorDesc            filterDesc, 
                                                     U32*                  bytes){
    DataType dt;                                                     
    U32 ic, ih, iw, fn, item_w;
    tensorSelectGet(inputDesc,  &dt,  NULL, NULL, &ic,  &ih,  &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn,  NULL, NULL, NULL);
    CAL_ITEM_W(iw, ih, item_w); 
    *bytes = ih * item_w * ((fn + IK - 1) / IK * IK) * bytesOf(dt);  
    return SUCCESS;
}

EE fully_connected_mali_fp16(GCLHandle_t          handle,
                             TensorDesc           inputDesc, 
                             const GCLMem_t       input,
                             TensorDesc           filterDesc, 
                             const GCLMem_t       filter,
                             TensorDesc           biasDesc, 
                             const GCLMem_t       bias,
                             U32                  tmpBytes, 
                             GCLMem_t             tmpBuf,
                             TensorDesc           outputDesc, 
                             GCLMem_t             output) {
    CHECK_STATUS(fully_connected_checkpara_mali_fp16(inputDesc, filterDesc, outputDesc));
    CHECK_STATUS(fully_connected_core_mali_fp16(handle, inputDesc, input, filterDesc, filter, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output));
    return SUCCESS;
}
