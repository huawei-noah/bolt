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
#include "gpu/mali/fp16/embedding_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

inline EE embedding_checkpara_mali_fp16(TensorDesc weightDesc,
                                        TensorDesc outputDesc) {
    if(weightDesc.dt != outputDesc.dt || weightDesc.dt != DT_F16) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE embedding_core_mali_fp16(GCLHandle_t handle,
                                   TensorDesc  inputDesc,
                                   GCLMem_t    input,
                                   TensorDesc  weightDesc,
                                   GCLMem_t    weight,
                                   TensorDesc  outputDesc,
                                   GCLMem_t    output,
                                   U32         inputDim, 
                                   U32         numOutput, 
                                   bool        transpose) {
    UNUSED(weightDesc);
    UNUSED(outputDesc);
    UNUSED(inputDim);
    UNUSED(numOutput);
    U32 step = inputDesc.dims[0];
    U32 on   = numOutput;
    U32 oh_str = output->desc.stride[0];
    U32 ow_str = output->desc.stride[1];
    U32 oc_str = output->desc.stride[2];
    U32 oh_off = output->desc.offset[0];
    U32 ow_off = output->desc.offset[1];
    if(ow_str != 1 || oh_off != 0 || ow_off != 0) CHECK_STATUS(NOT_SUPPORTED);
    cl_mem inbuf, weibuf, outbuf;
    inbuf  = input->mem;
    weibuf = weight->mem;
    outbuf = output->mem;

    if(!transpose) {
        U32 gs[2] = {oc_str, step};
        U32 ls[2] = {0, 0};
        U32 dim = 2;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel_binary(handle, "embedding", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, step, on, oc_str, oh_str, oh_off, ow_off, inbuf, weibuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "embedding");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "embedding"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, output, "embedding_output"));
#endif
        return SUCCESS; 
    } else {
        return NOT_SUPPORTED;
    }
}


EE embedding_mali_fp16(GCLHandle_t handle,
                       TensorDesc  inputDesc,
                       GCLMem_t    input,
                       TensorDesc  weightDesc,
                       GCLMem_t    weight,
                       TensorDesc  outputDesc,
                       GCLMem_t    output,
                       U32         inputDim, 
                       U32         numOutput, 
                       bool        transpose) {
    CHECK_STATUS(embedding_checkpara_mali_fp16(weightDesc, outputDesc));
    CHECK_STATUS(embedding_core_mali_fp16(handle, inputDesc, input, weightDesc, weight, outputDesc, output, inputDim, numOutput, transpose));
    return SUCCESS; 
}

