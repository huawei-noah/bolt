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
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

inline EE depth2space_checkpara_mali(GCLHandle_t handle, 
                                     TensorDesc  inputDesc,
                                     GCLMem_t    input,
                                     TensorDesc  outputDesc,
                                     GCLMem_t    output) {
    if(handle == nullptr || nullptr == input || nullptr == output) return NULL_POINTER;
    if(input->desc.memFormat != DF_NCWHC4) return NOT_SUPPORTED;
    if(output->desc.memFormat != DF_NCHW)    return NOT_SUPPORTED;
    if(inputDesc.df != DF_NCHW)  return NOT_SUPPORTED;
    if(outputDesc.df != DF_NCHW) return NOT_SUPPORTED;
    if(inputDesc.dt != DT_F16)  return NOT_SUPPORTED;
    if(outputDesc.dt != DT_U8) return NOT_SUPPORTED;
    if(inputDesc.dims[0] * 4  != outputDesc.dims[0])  return NOT_SUPPORTED;
    if(inputDesc.dims[1] * 4  != outputDesc.dims[1])  return NOT_SUPPORTED;
    if(inputDesc.dims[2] / 16 != outputDesc.dims[2]) return NOT_SUPPORTED;
    if(inputDesc.dims[3] != outputDesc.dims[3]) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE depth2space_core_mali_fp16(GCLHandle_t handle, 
                                     TensorDesc  inputDesc,
                                     GCLMem_t    input,
                                     TensorDesc  outputDesc,
                                     GCLMem_t    output) {
    UNUSED(outputDesc);
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc,  NULL, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, iw_off, ih_off;
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];
    U32 ow_str, oh_str, ow_off, oh_off;
    ow_str = output->desc.stride[0];
    oh_str = output->desc.stride[1];
    ow_off = output->desc.offset[0];
    oh_off = output->desc.offset[1];

    cl_mem inbuf, outbuf;
    inbuf  = input->mem;
    outbuf = output->mem;

    U32 gs[2] = {ih, iw * 4};
    U32 ls[2] = {0, 0};
    U32 dim   = 2;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel_binary(handle, "depth2space", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, ih, iw_str, ih_str, iw_off, ih_off,
        ow_str, oh_str, ow_off, oh_off, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, "depth2space");
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "depth2space"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "depth2space_input"));
    CHECK_STATUS(gcl_print_memory<U8>(handle, output, "depth2space_output"));
#endif
    return SUCCESS; 
}

EE depth2space_infer_output_size_mali(TensorDesc   inputDesc,
                                      TensorDesc*  outputDesc,
                                      GCLMemDesc_t gclmemInputDesc,
                                      GCLMemDesc_t gclmemOutputDesc){
    /*tensorDesc record cpu org data format info*/
    /*gclmemDesc record gpu trans data format info*/
    if(outputDesc) *outputDesc = inputDesc;

    DataType   idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc,  &idt, &idf, &in, &ic, &ih, &iw);
    if(idt != DT_F16) return NOT_SUPPORTED;
    if(ic  != 16)    return NOT_SUPPORTED;
    on = in;
    oc = 1;
    oh = ih * 4;
    ow = iw * 4;

    if(idf == DF_NCHW) {
        *outputDesc = tensor4df(DT_U8, idf, on, oc, oh, ow);
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, 0,  0,  0,  DT_F16, DT_F16, gclmemInputDesc,  NULL));
        CHECK_STATUS(infer_gclmem_desc_nchw(  0,  0,  0,  0, 0, ow, oh, oc, DT_U8,  DT_U8,  NULL, gclmemOutputDesc));
        return SUCCESS;
    } 
    return NOT_SUPPORTED;
}

EE depth2space_mali(GCLHandle_t handle, 
                    TensorDesc  inputDesc,
                    GCLMem_t    input,
                    TensorDesc  outputDesc,
                    GCLMem_t    output) {
    EE ret = SUCCESS;
    CHECK_STATUS(depth2space_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    CHECK_STATUS(depth2space_core_mali_fp16(handle, inputDesc, input, outputDesc, output));
    return ret;
}



