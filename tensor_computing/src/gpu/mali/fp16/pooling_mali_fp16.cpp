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
#include "gpu/mali/fp16/pooling_mali_fp16.h"

inline EE pooling_checkpara_mali_fp16(GCLHandle_t    handle, 
                                      TensorDesc     inputDesc, 
                                      const GCLMem_t input, 
                                      PoolingDesc    poolingDesc, 
                                      TensorDesc     outputDesc, 
                                      GCLMem_t       output){
    if (handle == nullptr || nullptr == input || nullptr == output)                            return NULL_POINTER;
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16)                               return NOT_SUPPORTED;
    if (inputDesc.df != outputDesc.df || inputDesc.df != DF_NCHW)                              return NOT_SUPPORTED;
    if (inputDesc.dims[2] != outputDesc.dims[2] || inputDesc.dims[3] != outputDesc.dims[3])    return NOT_SUPPORTED;
    if (poolingDesc.padding_top >= poolingDesc.kernelSize_h)                                   return NOT_SUPPORTED;
    if (poolingDesc.padding_bottom >= poolingDesc.kernelSize_w)                                return NOT_SUPPORTED;
    if (input->desc.memFormat != output->desc.memFormat || input->desc.memFormat != DF_NCWHC4) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE pooling_core_mali_fp16(GCLHandle_t    handle, 
                                 TensorDesc     inputDesc, 
                                 const GCLMem_t input, 
                                 PoolingDesc    poolingDesc, 
                                 TensorDesc     outputDesc, 
                                 GCLMem_t       output){


    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc,  NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    cl_mem inbuf, outbuf;
    inbuf  = input->mem;
    outbuf = output->mem;

    U32 iw_str, ih_str, iw_off, ih_off;
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];

    U32 ow_str, oh_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];

    U32 sw, sh, pw, ph, kw, kh;
    sw = poolingDesc.stride_w;
    sh = poolingDesc.stride_h;
    pw = poolingDesc.padding_left;
    ph = poolingDesc.padding_top;
    kw = poolingDesc.kernelSize_w;
    kh = poolingDesc.kernelSize_h;

    Kernel kernel;
    switch(poolingDesc.pm){
        case POOLING_MAX:{
	     CHECK_STATUS(gcl_create_kernel_binary(handle, "pooling_max", &kernel));
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_off, iw_off, ih_str, iw_str,
                                                                 oh, ow, oh_off, ow_off, oh_str, ow_str,
                                                                 sh, sw, ph, pw, kh, kw, inbuf, outbuf));

             U32 gs[3] = {oh, ow, (oc + 3) / 4 * on};
             U32 ls[3] = {16, 16, 1};
             U32 dim   = 3;
             gcl_set_kernelVec(handle, kernel, dim, gs, ls);
#ifdef _DEBUG
             CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "bolt_pooling_max"));
             CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "pooling_max_input"));
             CHECK_STATUS(gcl_print_memory<F16>(handle, output, "pooling_max_output"));
#endif
             break;
        }
        case POOLING_MEAN:{
	     CHECK_STATUS(gcl_create_kernel_binary(handle, "pooling_mean", &kernel));
	     CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_off, iw_off, ih_str, iw_str,
                                                                 oh, ow, oh_off, ow_off, oh_str, ow_str,
                                                                 sh, sw, ph, pw, kh, kw, inbuf, outbuf));

             U32 gs[3] = {oh, ow, (oc + 3) / 4 * on};
             U32 ls[3] = {16, 16, 1};
             U32 dim   = 3;
             gcl_set_kernelVec(handle, kernel, dim, gs, ls);
#ifdef _DEBUG
             CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "bolt_pooling_mean"));
             CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "pooling_mean_input"));
             CHECK_STATUS(gcl_print_memory<F16>(handle, output, "pooling_mean_output"));
#endif
             break;
        }
        default:
	     {CHECK_STATUS(NOT_SUPPORTED);}
    }
    return SUCCESS; 
     
}

EE pooling_mali_fp16(GCLHandle_t    handle,
                     TensorDesc     inputDesc, 
                     const GCLMem_t input, 
                     PoolingDesc    poolingDesc, 
                     TensorDesc     outputDesc, 
                     GCLMem_t       output){
    CHECK_STATUS(pooling_checkpara_mali_fp16(handle, inputDesc, input, poolingDesc, outputDesc, output));
    CHECK_STATUS(pooling_core_mali_fp16     (handle, inputDesc, input, poolingDesc, outputDesc, output));
    return SUCCESS; 
}

