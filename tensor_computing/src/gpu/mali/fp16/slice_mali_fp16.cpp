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
#include "gpu/mali/fp16/slice_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"
#define MAX_SLICE_NUM 2

inline EE slice_checkpara_mali_fp16(TensorDesc              inputDesc,
                                    std::vector<TensorDesc> outputDesc) {
    if(inputDesc.dt != DT_F16) return NOT_SUPPORTED;
    for(auto p : outputDesc) {
        if(p.dt != DT_F16) return NOT_SUPPORTED;
    }
    return SUCCESS; 
}

inline EE slice_core_mali_fp16(GCLHandle_t             handle,
                               TensorDesc              inputDesc,
                               GCLMem_t                input,
                               I32                     axis,
                               std::vector<TensorDesc> outputDesc,
                               std::vector<void*>*     output) {
    if(inputDesc.df == DF_MKT) {
        U32 m, k, t;
        U32 gw, gh, gc;
        get_nlp_mkt_val(inputDesc, NULL, &m, &k, &t);
        map_nlp_mkt_to_ncwhc4(m, k, t, &gw, &gh, &gc);
        if(axis == 2) {
            U32 iw_str, ih_str, iw_off, ih_off;
            ih_str = input->desc.stride[0];
            iw_str = input->desc.stride[1];
            ih_off = input->desc.offset[0];
            iw_off = input->desc.offset[1];
            U32 ow_str[MAX_SLICE_NUM];
            U32 oh_str[MAX_SLICE_NUM];
            U32 ow_off[MAX_SLICE_NUM];
            U32 oh_off[MAX_SLICE_NUM];
            cl_mem outbuf[MAX_SLICE_NUM];
            U32 sliceEnd[MAX_SLICE_NUM];
            U32 sliceNum = (*output).size();
            if(sliceNum > MAX_SLICE_NUM) CHECK_STATUS(NOT_SUPPORTED);
            U32 j = 0;
            std::vector<void*> outputArray = *output;
            for(U32 i = 0; i < sliceNum; ++i) {
                oh_str[i] = ((GCLMem_t)outputArray[i])->desc.stride[0];
                ow_str[i] = ((GCLMem_t)outputArray[i])->desc.stride[1];
                oh_off[i] = ((GCLMem_t)outputArray[i])->desc.offset[0];
                ow_off[i] = ((GCLMem_t)outputArray[i])->desc.offset[1];
                outbuf[i] = ((GCLMem_t)outputArray[i])->mem;
                get_nlp_mkt_val(outputDesc[i], NULL, NULL, NULL, &t);
                j += t;
                sliceEnd[i] = j;
            }
            char kernelName[128];
            sprintf(kernelName, "slice_h_%d", sliceNum);
            Kernel kernel;
            CHECK_STATUS(gcl_create_kernel_binary(handle, kernelName, &kernel));
            U32 gs[3] = {gh, gw, gc};
            U32 ls[3] = {0, 0, 0};
            U32 dim   = 3;
            switch(sliceNum) {
                case 2:
                    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off, gs[0], gs[1], input->mem, 
                        oh_str[0], ow_str[0], oh_off[0], ow_off[0], sliceEnd[0], outbuf[0],
                        oh_str[1], ow_str[1], oh_off[1], ow_off[1], sliceEnd[1], outbuf[1]));
                    break;
                default:
                    return NOT_SUPPORTED;
            }
            gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
            CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
            CHECK_STATUS(gcl_print_memory<F16>(handle, input,  "slice_input"));
            for(U32 i = 0; i < sliceNum; ++i) CHECK_STATUS(gcl_print_memory<F16>(handle, (GCLMem_t)(outputArray[i]), "slice_output"));
#endif
            return SUCCESS;
        }
        return NOT_SUPPORTED;
    }
    return NOT_SUPPORTED; 
}


EE slice_mali_fp16(GCLHandle_t             handle,
                   TensorDesc              inputDesc,
                   GCLMem_t                input,
                   I32                     axis,
                   std::vector<TensorDesc> outputDesc,
                   std::vector<void*>*     output) {
    CHECK_STATUS(slice_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(slice_core_mali_fp16(handle, inputDesc, input, axis, outputDesc, output));
    return SUCCESS; 
}

