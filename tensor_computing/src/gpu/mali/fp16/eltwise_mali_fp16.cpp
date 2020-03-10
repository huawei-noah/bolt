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
#include "gpu/mali/fp16/eltwise_mali_fp16.h"

inline EE eltwise_checkpara_mali_fp16(std::vector<TensorDesc> inputDesc,
                                      std::vector<void*>      input,
                                      TensorDesc              outputDesc) {
    for(auto it : inputDesc) {
        if(it.dt != outputDesc.dt) return NOT_SUPPORTED;
    }
    U32 num = input.size();
    if(num > 8) return NOT_SUPPORTED;
    if(outputDesc.dt != DT_F16) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE eltwise_core_mali_fp16(GCLHandle_t             handle, 
                                 std::vector<TensorDesc> inputDesc,
                                 std::vector<void*>      input,
                                 TensorDesc              outputDesc,
                                 GCLMem_t                output,
                                 EltwiseMode             eltwiseMode) {
    UNUSED(outputDesc);
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc[0],  NULL, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 num = input.size();
    GCLMem_t inputMem[8];
    for(U32 i = 0; i < num; ++i) inputMem[i] = (GCLMem_t)input[i];
    ih_str = inputMem[0]->desc.stride[0];
    iw_str = inputMem[0]->desc.stride[1];
    ih_off = inputMem[0]->desc.offset[0];
    iw_off = inputMem[0]->desc.offset[1];
    U32 ow_str, oh_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];

    cl_mem outbuf;
    outbuf = output->mem;

    char modeName[16];
    if(eltwiseMode == ELTWISE_MAX)  strcpy(modeName, "max");
    if(eltwiseMode == ELTWISE_SUM)  strcpy(modeName, "sum");
    if(eltwiseMode == ELTWISE_PROD) strcpy(modeName, "prod");

    char kernelName[128];
    sprintf(kernelName, "eltwise_%s%d", modeName, num);
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel_binary(handle, kernelName, &kernel));
    switch(num) {
        case 1:
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, ih_str, iw_str, ih_off, iw_off,
                 oh_str, ow_str, oh_off, ow_off, inputMem[0]->mem, outbuf));break;
        case 2:
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, ih_str, iw_str, ih_off, iw_off,
                 oh_str, ow_str, oh_off, ow_off, inputMem[0]->mem, inputMem[1]->mem, outbuf));break;
        case 3: 
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, ih_str, iw_str, ih_off, iw_off,
                 oh_str, ow_str, oh_off, ow_off, inputMem[0]->mem, inputMem[1]->mem, inputMem[2]->mem, outbuf));break;
        case 4: 
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, ih_str, iw_str, ih_off, iw_off,
                 oh_str, ow_str, oh_off, ow_off, inputMem[0]->mem, inputMem[1]->mem, inputMem[2]->mem, 
                 inputMem[3]->mem, outbuf));break;
        case 5: 
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, ih_str, iw_str, ih_off, iw_off,
                 oh_str, ow_str, oh_off, ow_off, inputMem[0]->mem, inputMem[1]->mem, inputMem[2]->mem, 
                 inputMem[3]->mem, inputMem[4]->mem, outbuf));break;
        case 6: 
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, ih_str, iw_str, ih_off, iw_off,
                 oh_str, ow_str, oh_off, ow_off, inputMem[0]->mem, inputMem[1]->mem, inputMem[2]->mem, 
                 inputMem[3]->mem, inputMem[4]->mem, inputMem[5]->mem, outbuf));break;
        case 7: 
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, ih_str, iw_str, ih_off, iw_off,
                 oh_str, ow_str, oh_off, ow_off, inputMem[0]->mem, inputMem[1]->mem, inputMem[2]->mem, 
                 inputMem[3]->mem, inputMem[4]->mem, inputMem[5]->mem, inputMem[6]->mem, outbuf));break;
        case 8: 
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ic, ih_str, iw_str, ih_off, iw_off,
                 oh_str, ow_str, oh_off, ow_off, inputMem[0]->mem, inputMem[1]->mem, inputMem[2]->mem, 
                 inputMem[3]->mem, inputMem[4]->mem, inputMem[5]->mem, inputMem[6]->mem, inputMem[7]->mem, outbuf));break;
        default:
             return NOT_SUPPORTED;
    }
    

    U32 gs[3] = {(ih + 1) / 2, iw, (ic + 3) / 4 * in};
    U32 ls[3] = {16, 16, 1};
    U32 dim   = 3;
    gcl_set_kernelVec(handle, kernel, dim, gs, ls);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    for(U32 i = 0; i < num; ++i){
        std::cout << "eltwise_input " << i << " " << std::endl;
        CHECK_STATUS(gcl_print_memory<F16>(handle, inputMem[i],  "eltwise_input"));
    }
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "eltwise_output"));
#endif
    return SUCCESS; 
}


EE eltwise_mali_fp16(GCLHandle_t             handle,
                     std::vector<TensorDesc> inputDesc,
                     std::vector<void*>      input,
                     TensorDesc              outputDesc,
                     GCLMem_t                output,
                     EltwiseMode             eltwiseMode) {
    CHECK_STATUS(eltwise_checkpara_mali_fp16(inputDesc, input, outputDesc));
    CHECK_STATUS(eltwise_core_mali_fp16     (handle, inputDesc, input, outputDesc, output, eltwiseMode));
    return SUCCESS; 
}

