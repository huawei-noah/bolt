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

EE tensor_computing_get_output_infer_tmpBuf_size_mali(const GCLMem_t input, 
                                                      TensorDesc     hostDesc, 
                                                      U32*           tmpBufSize){
    *tmpBufSize = 0;
    if(input->desc.memFormat == DF_NCWHC4){*tmpBufSize = tensorNumBytes(hostDesc);}
    return SUCCESS;
}

EE tensor_computing_get_output_mali(GCLHandle_t    handle, 
                                    const GCLMem_t input, 
                                    TensorDesc     hostDesc, 
                                    U8**           hostPtr, 
                                    GCLMem_t       tmpBuf, 
                                    bool           blocking){
    GCLMemDesc desc = input->desc;
    Kernel kernel;
    DataType host_dt;
    DataFormat host_df, device_df;
    U32 ow, oh, oc, on;
    U32 iw, ih, pw, ph;
    tensorSelectGet(hostDesc, &host_dt, &host_df, &on, &oc, &oh, &ow);
    U32 size = tensorNumBytes(hostDesc);

    device_df = desc.memFormat;
    if(host_df == device_df){
        ph = desc.offset[0]; 
        pw = desc.offset[1]; 
        if(pw == 0 && ph == 0){
            if(desc.use_map){
                CHECK_STATUS(gcl_map_memory(handle, input, &size, CL_MAP_READ, blocking));
                *hostPtr = (U8*)input->desc.map_ptr;
                return SUCCESS;
            } else {
                gcl_trans_memory(handle, (void*)input, (void*)*hostPtr, &size, DEVICE_BUF_TO_HOST, blocking);
                return SUCCESS;
            }
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
            return NOT_SUPPORTED;
        }
        
    } else {
        if(host_dt != DT_F16) return NOT_SUPPORTED;
        if(device_df == DF_NCWHC4 && host_df == DF_NCHW){
            ih = desc.stride[0];
            iw = desc.stride[1];
            ph = desc.offset[0]; 
            pw = desc.offset[1]; 
            U32 owh_str = ow * oh;
            CHECK_STATUS(gcl_get_kernel_from_map(handle, "mem_trans_ncwhc4_to_nchw", &kernel));
            CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, ih, pw, ph, ow, oh, oc, owh_str, input->mem, tmpBuf->mem));
            U32 gs[3] = {oh, (ow + 3) >> 2, (oc + 3) / 4 * on };
            U32 ls[3] = {16, 16, 1};
            U32 dim   = 3;
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "mem_trans_ncwhc4_to_nchw"));
#ifdef _DEBUG
            CHECK_STATUS(gcl_print_memory<F16>(handle, input, "input"));
#endif
            gcl_trans_memory(handle, (void*)tmpBuf, (void*)*hostPtr, &size, DEVICE_BUF_TO_HOST, blocking);
            return SUCCESS;
        }
    }
    return NOT_SUPPORTED;
}



