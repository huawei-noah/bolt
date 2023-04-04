// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/resize_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/resize_opt.h"

inline EE resize_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE resize_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType idt, odt;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, &idt, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, &odt, NULL, &on, &oc, &oh, &ow);

    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    cl_mem inbuf = input->mem;
    cl_mem outbuf = output->mem;
    GCLMemType inputMemType = input->desc.memType;
    GCLMemType outputMemType = output->desc.memType;
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    float r0_w = iw / (float)ow;
    float r0_h = ih / (float)oh;
    float r1_w = (iw - 1.0f) / (ow - 1.0f);
    float r1_h = (ih - 1.0f) / (oh - 1.0f);

    U32 dim = 3;
    U32 gs[3] = {ow, oh, 0};
    U32 ls[3] = {0, 0, 0};
    if (input->desc.df == DF_NCHWC4) {
        gs[2] = (oc + 3) / 4 * on;
    } else if (input->desc.df == DF_NHWC) {
        gs[2] = on;
    } else {
        gs[2] = oc * on;
    }

    Kernel kernel;
    KernelOpt kernelOpt;
    char kernelName[128];
    CHECK_STATUS(set_resize_opt_mali(
        p, input->desc.df, idt, odt, inputMemType, outputMemType, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, i_off, iw, ih, ow_str, oh_str, o_off,
        ow, oh, r0_w, r0_h, r1_w, r1_h, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE resize_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(resize_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    return resize_core_mali_fp16(handle, inputDesc, input, p, outputDesc, output);
}
