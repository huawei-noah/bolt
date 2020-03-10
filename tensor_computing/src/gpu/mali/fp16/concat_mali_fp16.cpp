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
#include "gpu/mali/fp16/concat_mali_fp16.h"

inline EE concat_checkpara_mali_fp16(std::vector<TensorDesc> inputDesc,
                                     TensorDesc              outputDesc) {
    for(auto it : inputDesc) {
        if(it.dt != outputDesc.dt) return NOT_SUPPORTED;
    }
    if(outputDesc.dt != DT_F16) return NOT_SUPPORTED;
    return SUCCESS; 
}

inline EE concat_core_mali_fp16(GCLHandle_t             handle, 
                                std::vector<TensorDesc> inputDesc,
                                std::vector<void*>      input,
                                TensorDesc              outputDesc,
                                GCLMem_t                output,
                                U32                     concatDim) {
    UNUSED(inputDesc);
    U32 ow, oh;
    tensorSelectGet(outputDesc,  NULL, NULL, NULL, NULL, &oh, &ow);
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, oc_str, ow_off, oh_off;
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oc_str = output->desc.stride[2];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];
    U32 num = input.size();
    GCLMem_t inputMem[8];
    cl_mem inbuf[8];
    cl_mem outbuf = output->mem;
    U32 c[7];
    U32 bn  = (num + 7) / 8;
    U32 en;
    U32 nmax;
    U32 cmax;
    U32 out_size = 0; 
    inputMem[0] = (GCLMem_t)input[0];
    ih_str = inputMem[0]->desc.stride[0];
    iw_str = inputMem[0]->desc.stride[1];
    ih_off = inputMem[0]->desc.offset[0];
    iw_off = inputMem[0]->desc.offset[1];
    for(U32 i = 0; i < bn; i++) {
        en = (i * 8 + 8 <= num) ? 8 : (num & 7);
        cmax = 0;
        nmax = en - 1;
        for(U32 j = 0; j < en; ++j) {
            inputMem[j] = (GCLMem_t)input[i * 8 + j];
            inbuf[j] = inputMem[j]->mem;
        }
        for(U32 j = 0; j < nmax; ++j) {
            c[j] = inputMem[j]->desc.stride[2];
            cmax += c[j];
        }
        char kernelName[128];
        sprintf(kernelName, "concat_%d%d", concatDim, en);
        Kernel kernel;
        CHECK_STATUS(gcl_create_kernel_binary(handle, kernelName, &kernel));
        switch(en) {
            case 1:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off,
                             oh_str, ow_str, oh_off, ow_off, cmax, nmax, out_size, inbuf[0], outbuf));break;
            case 2:
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off,
                             oh_str, ow_str, oh_off, ow_off, cmax, nmax, out_size, inbuf[0], c[0], inbuf[1], outbuf));break;
            case 3: 
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off,
                             oh_str, ow_str, oh_off, ow_off, cmax, nmax, out_size, inbuf[0], c[0], inbuf[1], c[1], inbuf[2], 
                             outbuf));break;
            case 4: 
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off,
                             oh_str, ow_str, oh_off, ow_off, cmax, nmax, out_size, inbuf[0], c[0], inbuf[1], c[1], inbuf[2], 
                             c[2], inbuf[3], outbuf));break;
            case 5: 
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off,
                             oh_str, ow_str, oh_off, ow_off, cmax, nmax, out_size, inbuf[0], c[0], inbuf[1], c[1], inbuf[2], 
                             c[2], inbuf[3], c[3], inbuf[4], outbuf));break;
            case 6: 
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off,
                             oh_str, ow_str, oh_off, ow_off, cmax, nmax, out_size, inbuf[0], c[0], inbuf[1], c[1], inbuf[2], 
                             c[2], inbuf[3], c[3], inbuf[4], c[4], inbuf[5], outbuf));break;
            case 7: 
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off,
                             oh_str, ow_str, oh_off, ow_off, cmax, nmax, out_size, inbuf[0], c[0], inbuf[1], c[1], inbuf[2], 
                             c[2], inbuf[3], c[3], inbuf[4], c[4], inbuf[5], c[5], inbuf[6], outbuf));break;
            case 8: 
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ih_off, iw_off,
                             oh_str, ow_str, oh_off, ow_off, cmax, nmax, out_size, inbuf[0], c[0], inbuf[1], c[1], inbuf[2], 
                             c[2], inbuf[3], c[3], inbuf[4], c[4], inbuf[5], c[5], inbuf[6], c[6], inbuf[7], outbuf));break;
            default:
                return NOT_SUPPORTED;
        }
        U32 gs[3] = {oh, ow, cmax + inputMem[nmax]->desc.stride[2]};
        U32 ls[3] = {16, 16, 1};
        U32 dim   = 3;
        gcl_set_kernelVec(handle, kernel, dim, gs, ls);
        out_size += ow_str * oh_str * gs[2] * 4;
#ifdef _DEBUG
        CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
        for(U32 i = 0; i < en; ++i){
            std::cout << "concat_input " << i << " " << std::endl;
            CHECK_STATUS(gcl_print_memory<F16>(handle, inputMem[i],  "concat_input"));
        }
#endif
    }
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "concat_output"));
#endif
    return SUCCESS; 
}


EE concat_mali_fp16(GCLHandle_t              handle,
                    std::vector<TensorDesc> inputDesc,
                    std::vector<void*>      input,
                    TensorDesc              outputDesc,
                    GCLMem_t                output,
                    U32                     concatDim) {
    CHECK_STATUS(concat_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(concat_core_mali_fp16(handle, inputDesc, input, outputDesc, output, concatDim));
    return SUCCESS; 
}

