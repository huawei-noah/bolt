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
#include "gpu/mali/fp16/transpose_mali_fp16.h"

inline EE transpose_checkpara_mali_fp16(TensorDesc inputDesc,
                                        TensorDesc outputDesc) {
    if(inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE transpose_core_mali_fp16(GCLHandle_t handle,
                                   TensorDesc  inputDesc,
                                   GCLMem_t    input,
                                   TensorDesc  outputDesc,
                                   GCLMem_t    output,
                                   U32*        dim) {
    UNUSED(inputDesc);                                  
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    if(input->desc.memFormat == DF_NCHW) {
        iw_str = input->desc.stride[0];
        ih_str = input->desc.stride[1];
        iw_off = input->desc.offset[0];
        ih_off = input->desc.offset[1];
        ow_str = output->desc.stride[0];
        oh_str = output->desc.stride[1];
        ow_off = output->desc.offset[0];
        oh_off = output->desc.offset[1];
        cl_mem inbuf  = input->mem;
        cl_mem outbuf = output->mem;
        U32 gs[3] = {0, 0, 0};
        U32 ls[3] = {0, 0, 0};
        U32 kdim = 3;
        Kernel kernel;
        if(dim[0] == 0 && dim[1] == 1 && dim[2] == 3 && dim[3] == 2) {
            U32 ow, oh, oc, on;
            tensorSelectGet(outputDesc,  NULL, NULL, &on, &oc, &oh, &ow);
            gs[0] =(oh + 3) / 4;
            gs[1] = ow;
            gs[2] = oc * on;
            CHECK_STATUS(gcl_create_kernel_binary(handle, "transpose_nchw_0132", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off, oh_str, ow_str, oh_off, ow_off, oh, gs[0], gs[1], inbuf, outbuf));
            gcl_set_kernelVec(handle, kernel, kdim, gs, ls, "tranpose_nchw_0132");
#ifdef _DEBUG
            CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "transpose_nchw_0132_input"));
            CHECK_STATUS(gcl_run_kernel(handle, kernel, kdim, gs, ls, "transpose_nchw_0132"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, output, "transpose_nchw_0132_output"));
#endif
            return SUCCESS;
        }
        return NOT_SUPPORTED;
    }
    return NOT_SUPPORTED; 
}


EE transpose_mali_fp16(GCLHandle_t handle,
                       TensorDesc  inputDesc,
                       GCLMem_t    input,
                       TensorDesc  outputDesc,
                       GCLMem_t    output,
                       U32*        dim) {
    CHECK_STATUS(transpose_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(transpose_core_mali_fp16(handle, inputDesc, input, outputDesc, output, dim));
    return SUCCESS; 
}

