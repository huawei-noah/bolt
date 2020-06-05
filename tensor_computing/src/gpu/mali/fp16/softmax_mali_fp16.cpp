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
#include "gpu/mali/fp16/softmax_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

inline EE softmax_checkpara_mali_fp16(TensorDesc inputDesc,
                                      TensorDesc outputDesc) {
    if(inputDesc.dt != outputDesc.dt) return NOT_SUPPORTED;
    if(outputDesc.dt != DT_F16) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE softmax_core_mali_fp16(GCLHandle_t handle, 
                                 TensorDesc  inputDesc,
                                 GCLMem_t    input,
                                 int         axis,
                                 TensorDesc  outputDesc,
                                 GCLMem_t    output) {
    UNUSED(axis);
    UNUSED(outputDesc);
    U32 iw, ih, ic, in;
    if(inputDesc.df == DF_NCHW) { 
        tensorSelectGet(inputDesc,  NULL, NULL, &in, &ic, &ih, &iw);
    } else if(inputDesc.df == DF_MKT) {
        get_nlp_mkt_val(inputDesc, NULL, NULL, &ic, &ih);
        iw = 1;
        in = 1;
    } else {
        return NOT_SUPPORTED;
    }
    U32 iw_str, ih_str, iw_off, ih_off, ihw_str;
    U32 ow_str, oh_str, ow_off, oh_off, ohw_str;
    get_gclmem_dim(input->desc,  &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    ihw_str = ih_str * iw_str;
    ohw_str = oh_str * ow_str;

    cl_mem inbuf, outbuf;
    inbuf  = input->mem;
    outbuf = output->mem;
    U32 gs[2];
    U32 ls[2] = {0, 0};
    U32 dim = 2;
    Kernel kernel;
    char kernelname[128];
    if(input->desc.memFormat == DF_NCWHC4) {
        if(axis != 1) CHECK_STATUS(NOT_SUPPORTED);
        gs[0] = ih;
        gs[1] = iw;
        I32 icd4 = (ic + 3) >> 2;
        I32 ice4 = ((ic & 3) == 0) ? 4 : (ic & 3);
        sprintf(kernelname, "softmax");    
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, icd4, ice4, ih_str, ihw_str, ih_off, iw_off,
            oh_str, ohw_str, oh_off, ow_off, gs[0], gs[1], inbuf, outbuf));
    } else if(input->desc.memFormat == DF_NCHW) {
        I32 axisTran = (axis + 4) % 4;
        if(axisTran == 1) {//on c axis
            gs[0] = (iw + 3) / 4;
            gs[1] = ih;
            sprintf(kernelname, "softmax_nchw_c");    
            CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ic, iw_str, ihw_str, iw_off, ih_off,
                ow_str, ohw_str, ow_off, oh_off, gs[0], gs[1], inbuf, outbuf));
        }
        if(axisTran == 3) {//on w axis
            gs[0] = ih;
            gs[1] = ic;
            I32 iwd4 = (iw + 3) >> 2;
            I32 iwe4 = ((iw & 3) == 0) ? 4 : (iw & 3);
            sprintf(kernelname, "softmax_nchw_w");    
            CHECK_STATUS(gcl_create_kernel_binary(handle, kernelname, &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iwd4, iwe4, iw_str, ih_str, iw_off, ih_off,
                ow_str, oh_str, ow_off, oh_off, gs[0], gs[1], inbuf, outbuf));
        }
    } else {
        return NOT_SUPPORTED;
    }
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelname);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelname));
    CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "softmax_nchw_input"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "softmax_nchw_output"));
#endif
    return SUCCESS; 
}


EE softmax_mali_fp16(GCLHandle_t handle,
                     TensorDesc  inputDesc,
                     GCLMem_t    input,
                     int         axis,
                     TensorDesc  outputDesc,
                     GCLMem_t    output) {

    CHECK_STATUS(softmax_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(softmax_core_mali_fp16(handle, inputDesc, input, axis, outputDesc, output));
    return SUCCESS; 
}

