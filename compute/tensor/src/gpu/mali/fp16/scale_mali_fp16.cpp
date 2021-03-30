// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/scale_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/scale_opt.h"

inline EE scale_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE scale_core_mali_fp16(GCLHandle_t handle,
    GCLMem_t alpha,
    GCLMem_t beta,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(outputDesc);
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    cl_mem inbuf, outbuf, albuf, bebuf;
    inbuf = input->mem;
    outbuf = output->mem;
    albuf = (alpha) ? alpha->mem : outbuf;
    bebuf = (beta) ? beta->mem : albuf;
    bool useAlpha = (alpha) ? true : false;
    bool useBeta = (beta) ? true : false;
    bool useNchwFormat = false;

    char kernelName[128];
    KernelOpt kernelOpt;
    U32 gs[3] = {ih, iw, (ic + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (input->desc.memFormat == DF_NCHW) {
        gs[0] = (iw + 3) / 4;
        gs[1] = ih;
        gs[2] = ic;
        useNchwFormat = true;
    }
    CHECK_STATUS(
        set_scale_opt_mali(useAlpha, useBeta, useNchwFormat, DT_F16, kernelName, &kernelOpt));

    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw, ih_str, iw_str, ih_off, iw_off, oh_str, ow_str,
        oh_off, ow_off, gs[0], gs[1], albuf, bebuf, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE scale_mali_fp16(GCLHandle_t handle,
    GCLMem_t alpha,
    GCLMem_t beta,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(scale_checkpara_mali_fp16(inputDesc, outputDesc));
    if (input->mem != output->mem) {
        CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    }
    CHECK_STATUS(scale_core_mali_fp16(handle, alpha, beta, inputDesc, input, outputDesc, output));
    return SUCCESS;
}
