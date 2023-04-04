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
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE scale_core_mali_fp16(GCLHandle_t handle,
    GCLMem_t alpha,
    GCLMem_t beta,
    ScaleParamSpec p,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType idt;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    cl_mem inbuf, outbuf, albuf, bebuf;
    inbuf = input->mem;
    outbuf = output->mem;
    albuf = (alpha) ? alpha->mem : beta->mem;
    bebuf = (beta) ? beta->mem : albuf;
    bool useAlpha = (alpha) ? true : false;
    bool useBeta = (beta) ? true : false;
    bool useNchwFormat = false;
    I32 axis = p.axis;
    U32 nDims = inputDesc.nDims;
    axis = (axis + nDims) % nDims;
    axis = nDims - 1 - axis;
    bool useBroadCast = false;
    if (outputDesc.dims[axis] != inputDesc.dims[axis]) {
        useBroadCast = true;
    }
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[3] = {ow, oh, (oc + 3) / 4 * on};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    U32 i_off = ih_off * iw_str + iw_off;
    U32 o_off = oh_off * ow_str + ow_off;
    if (input->desc.memFormat == DF_NCHW) {
        gs[0] = (ow + 3) / 4 * on;
        gs[1] = oh;
        gs[2] = oc;
        useNchwFormat = true;
    }
    CHECK_STATUS(set_scale_opt_mali(useAlpha, useBeta, useNchwFormat, useBroadCast, axis, idt,
        input->desc.memType, output->desc.memType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, iw, ih,
        ic, ow, oh, oc, gs[0], gs[1], albuf, bebuf, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE scale_mali_fp16(GCLHandle_t handle,
    GCLMem_t alpha,
    GCLMem_t beta,
    ScaleParamSpec p,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(scale_checkpara_mali_fp16(inputDesc, outputDesc));
    if (input->mem != output->mem) {
        CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    }
    CHECK_STATUS(scale_core_mali_fp16(handle, alpha, beta, p, inputDesc, input, outputDesc, output));
    return SUCCESS;
}
