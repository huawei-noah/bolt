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
#include "gpu/mali/fp16/reshape_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

inline EE reshape_checkpara_mali_fp16(TensorDesc inputDesc,
                                      TensorDesc outputDesc) {
    if(inputDesc.dt != outputDesc.dt) return NOT_SUPPORTED;
    if(outputDesc.dt != DT_F16) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE reshape_core_mali_fp16(GCLHandle_t handle, 
                                 TensorDesc  inputDesc,
                                 GCLMem_t    input,
                                 TensorDesc  outputDesc,
                                 GCLMem_t    output) {
    DataFormat idf, odf;
    idf = inputDesc.df;
    odf = outputDesc.df;
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    get_gclmem_dim(input->desc,  &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    if(idf == DF_NCHW) {
        U32 iw, ih, ic, in;
        tensorSelectGet(inputDesc,  NULL, NULL, &in, &ic, &ih, &iw);
        if(odf == DF_NCHW) {
            if(inbuf == outbuf) return SUCCESS;
            U32 gs[3] = {ih, iw, (ic + 3) / 4};
            U32 ls[3] = {0, 0, 0};
            U32 dim   = 3;
            Kernel kernel;
            CHECK_STATUS(gcl_create_kernel_binary(handle, "reshape", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, ih_str, iw_str, ih_off, iw_off, oh_str, ow_str, oh_off, ow_off, gs[0], gs[1], inbuf, outbuf));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, "reshape");
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "reshape"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "reshape_input"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, output, "reshape_output"));
#endif
            return SUCCESS;
        }
        if(odf == DF_MKT) {
            iw_str = input->desc.stride[0];
            ih_str = input->desc.stride[1];
            iw_off = input->desc.offset[0];
            ih_off = input->desc.offset[1];
            U32 m, k, t;
            get_nlp_mkt_val(outputDesc, NULL, &m, &k, &t);
            U32 gs[2] = {t, (k + 3) / 4};
            U32 ls[2] = {0, 0};
            U32 dim   = 2;
            Kernel kernel;
            CHECK_STATUS(gcl_create_kernel_binary(handle, "reshape_nchw_to_mkt", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ih, k, oh_str, ow_str, oh_off, ow_off, gs[0], gs[1], inbuf, outbuf));
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, "reshape_nchw_to_mkt");
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "reshape_nchw_to_mkt"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "reshape_nchw_to_mkt_input"));
            CHECK_STATUS(gcl_print_memory<F16>(handle, output, "reshape_nchw_to_mkt_output"));
#endif
                return SUCCESS;
        }
    }

    if(idf == DF_MKT && odf == DF_NCHW) {
        U32 m, k, t;
        U32 oh;
        get_nlp_mkt_val(inputDesc, NULL, &m, &k, &t);
        tensorSelectGet(outputDesc, NULL, NULL, NULL, NULL, &oh, NULL);
        U32 gs[2] = {t, (k + 3) / 4};
        U32 ls[2] = {0, 0};
        U32 dim   = 2;
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel_binary(handle, "reshape_mkt_to_nchw", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off, ow_str, oh_str, ow_off, oh_off, oh, gs[0], gs[1], inbuf, outbuf));
        gcl_set_kernelVec(handle, kernel, dim, gs, ls, "reshape_mkt_to_nchw");
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "reshape_mkt_to_nchw"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "reshape_mkt_to_nchw_input"));
        CHECK_STATUS(gcl_print_memory<F16>(handle, output, "reshape_mkt_to_nchw_output"));
#endif
        return SUCCESS;
    }
    return NOT_SUPPORTED; 
}

EE reshape_mali_fp16(GCLHandle_t handle,
                     TensorDesc  inputDesc,
                     GCLMem_t    input,
                     TensorDesc  outputDesc,
                     GCLMem_t    output) {

    CHECK_STATUS(reshape_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(reshape_core_mali_fp16(handle, inputDesc, input, outputDesc, output));
    return SUCCESS; 
}

