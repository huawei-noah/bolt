// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/prelu_mali_fp16.h"

inline EE prelu_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE prelu_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(outputDesc);
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];
    oh_str = output->desc.stride[0];
    ow_str = output->desc.stride[1];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];
    cl_mem inbuf, outbuf, webuf;
    inbuf = input->mem;
    outbuf = output->mem;
    webuf = weight->mem;

    char modeName[16];
    char kernelName[128];
    if (preluDesc.propagate_down) {
        strcpy(modeName, "prop");
    } else {
        strcpy(modeName, "noprop");
    }
    sprintf(kernelName, "prelu_%s", modeName);

    U32 gs[3] = {ih, iw, (ic + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih, iw, ih_str, iw_str, ih_off, iw_off, ih, iw, oh_str,
        ow_str, oh_off, ow_off, webuf, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, input, "prelu_input"));
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
    CHECK_STATUS(gcl_print_memory<F16>(handle, weight, "prelu_weight"));
#endif
    return SUCCESS;
}

EE prelu_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(prelu_checkpara_mali_fp16(inputDesc, outputDesc));
    if (input->mem != output->mem) {
        CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    }
    CHECK_STATUS(
        prelu_core_mali_fp16(handle, inputDesc, input, weight, preluDesc, outputDesc, output));
    return SUCCESS;
}
