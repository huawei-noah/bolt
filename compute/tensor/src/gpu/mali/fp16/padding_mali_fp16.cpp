// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/padding_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/padding_opt.h"

inline EE padding_checkpara_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    if (padParamSpec.pad_mode == PAD_REFLECT &&
        (padParamSpec.top >= inputDesc.dims[1] || padParamSpec.bottom >= inputDesc.dims[1])) {
        return NOT_SUPPORTED;
    }
    if (padParamSpec.pad_mode == PAD_SYMMETRIC &&
        (padParamSpec.left > inputDesc.dims[0] || padParamSpec.right > inputDesc.dims[0])) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE padding_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType idt;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;

    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;

    U32 pl, pr, pt, pb, pf, pa;
    pl = padParamSpec.left;
    pr = padParamSpec.right;
    pt = padParamSpec.top;
    pb = padParamSpec.bottom;
    pf = padParamSpec.front;
    pa = padParamSpec.back;
    if (pf > 0 || pa > 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    char kernelName[128];
    Kernel kernel;
    KernelOpt kernelOpt;
    bool useNchw = (inputDesc.df == DF_NCHWC4) ? false : true;
    CHECK_STATUS(set_padding_opt_mali(
        useNchw, padParamSpec.pad_mode, idt, GCL_MEM_BUF, GCL_MEM_BUF, kernelName, &kernelOpt));

    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (input->desc.memFormat == DF_NCHW) {
        gs[0] = (ow + 3) / 4;
        gs[1] = oh;
        gs[2] = oc * on;
    } else if (input->desc.memFormat == DF_NCHWC4) {
        gs[0] = ow;
        gs[1] = oh;
        gs[2] = (oc + 3) / 4 * on;
    } else {
        CHECK_STATUS(NOT_SUPPORTED)
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, ow_str, oh_str, i_off, o_off, iw, ih,
        ow, oh, pl, pr, pt, pb, gs[0], gs[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE padding_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(
        padding_checkpara_mali_fp16(handle, inputDesc, input, padParamSpec, outputDesc, output));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(padding_core_mali_fp16(handle, inputDesc, input, padParamSpec, outputDesc, output));
    return SUCCESS;
}
