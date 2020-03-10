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
#include "tensor_computing_type.h"
#include "gpu/mali/fp16/scale_mali_fp16.h"

inline EE scale_checkpara_mali_fp16(TensorDesc inputDesc,
                                    TensorDesc outputDesc) {
    if(inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE scale_core_mali_fp16(GCLHandle_t handle,
                               GCLMem_t    alpha,
                               GCLMem_t    beta,
                               TensorDesc  inputDesc,
                               GCLMem_t    input,
                               TensorDesc  outputDesc,
                               GCLMem_t    output) {
    UNUSED(outputDesc);
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc,  NULL, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];
    cl_mem inbuf, outbuf, albuf, bebuf;
    inbuf  = input->mem;
    outbuf = output->mem;
    albuf  = alpha->mem;
    bebuf  = (beta) ? beta->mem : albuf;

    char modeName[16];
    if(beta){
        strcpy(modeName, "beta");
    } else {
        strcpy(modeName, "nobeta");
    }
    char kernelName[128];
    sprintf(kernelName, "scale_%s", modeName);    
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, ih_str, iw_str, ih_off, iw_off, oh_str, ow_str, oh_off, ow_off, albuf, bebuf, inbuf, outbuf));
    U32 gs[3] = {(ih + 1) / 2, iw, (ic + 3) / 4};
    U32 ls[3] = {16, 16, 1};
    U32 dim   = 3;
    gcl_set_kernelVec(handle, kernel, dim, gs, ls);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "scale_input"));
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    CHECK_STATUS(gcl_print_memory<F16>(handle, alpha,  "scale_alpha"));
    if(beta) 
    CHECK_STATUS(gcl_print_memory<F16>(handle, beta,  "scale_beta"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "scale_output"));
#endif
    return SUCCESS; 
}


EE scale_mali_fp16(GCLHandle_t handle,
                   GCLMem_t    alpha,
                   GCLMem_t    beta,
                   TensorDesc  inputDesc,
                   GCLMem_t    input,
                   TensorDesc  outputDesc,
                   GCLMem_t    output) {
    CHECK_STATUS(scale_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(scale_core_mali_fp16(handle, alpha, beta, inputDesc, input, outputDesc, output));
    return SUCCESS; 
}

