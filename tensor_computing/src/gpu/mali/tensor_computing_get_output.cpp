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

EE tensor_computing_get_output_infer_tmpBuf_size_mali(const GCLMem_t input, 
                                                      TensorDesc     hostDesc, 
                                                      U32*           tmpBufSize) {
    UNUSED(input);
    UNUSED(hostDesc);
    *tmpBufSize = 0;
//    if(input->desc.memFormat == DF_NCWHC4) {*tmpBufSize = tensorNumBytes(hostDesc);}
    return SUCCESS;
}

EE tensor_computing_get_output_mali(GCLHandle_t    handle, 
                                    const GCLMem_t input, 
                                    TensorDesc     hostDesc, 
                                    U8**           hostPtr, 
                                    GCLMem_t       tmpBuf, 
                                    bool           blocking) {
    UNUSED(hostPtr);
    UNUSED(tmpBuf);
    GCLMemDesc desc = input->desc;
    Kernel kernel;
    DataType host_dt;
    DataFormat host_df, device_df;
    U32 ow, oh, oc, on;
    U32 iw, ih, ic, pw, ph;
    if(hostDesc.df == DF_NCHW) {
        tensorSelectGet(hostDesc, &host_dt, &host_df, &on, &oc, &oh, &ow);
    } else if (hostDesc.df == DF_MKT) {
        get_nlp_mkt_val(hostDesc, &host_dt, &on, &oc, &oh);
        ow = 1;
        host_df = DF_MKT;
    }
    U32 size = tensorNumBytes(hostDesc);
    U32 offset = 0;
    ih = desc.stride[0];
    iw = desc.stride[1];
    ic = desc.stride[2];
    ph = desc.offset[0]; 
    pw = desc.offset[1]; 
    device_df = desc.memFormat;
    if(pw != 0 || ph != 0) CHECK_STATUS(NOT_SUPPORTED);
    if(desc.byteSize < size) CHECK_STATUS(NOT_MATCH);
    if(desc.use_map == false) CHECK_STATUS(NOT_MATCH);

    if(device_df == DF_NCWHC4 && host_df == DF_NCHW && 
       host_dt == DT_F16 && (ih != 1 || iw != 1)) {
        if(desc.byteSize < size * 2) CHECK_STATUS(NOT_MATCH);
        U32 owh_str = ow * oh;
        offset = iw * ih * ic * 4;
        CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_ncwhc4_to_nchw", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, ih, pw, ph, ow, oh, oc, owh_str, offset, input->mem, input->mem));
        U32 gs[3] = {oh, (ow + 3) >> 2, (oc + 3) / 4 * on};
        U32 ls[3] = {0, 0, 0};
        U32 dim   = 3;
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_ncwhc4_to_nchw"));
#ifdef _DEBUG
        CHECK_STATUS(gcl_print_memory<F16>(handle, input, "input"));
#endif
        offset = offset * bytesOf(host_dt);
    }

    if(device_df == DF_NCWHC4 && host_df == DF_MKT) {
        if(desc.byteSize < size * 2) CHECK_STATUS(NOT_MATCH);
        offset = iw * ih * ic * 4;
        U32 gs[2] = {oh, (oc + 3) / 4};
        U32 ls[2] = {0, 0};
        U32 dim   = 2;
        CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_ncwhc4_to_mtk", &kernel));
        CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ph, pw, oc, offset, gs[0], gs[1], input->mem, input->mem));
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_ncwhc4_to_mtk"));
#ifdef _DEBUG
        CHECK_STATUS(gcl_print_memory<F16>(handle, input, "input"));
#endif
        offset = offset * bytesOf(host_dt);
    }
    
    CHECK_STATUS(gcl_map_memory(handle, input, &offset, &size, CL_MAP_READ, blocking));
    return SUCCESS;
}



