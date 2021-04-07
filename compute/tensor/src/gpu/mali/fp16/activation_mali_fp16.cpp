// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/activation_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/activation_opt.h"

inline EE activation_checkpara_mali_fp16(TensorDesc inputDesc)
{
    if (inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE activation_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    UNUSED(inputDesc);
    UNUSED(outputDesc);
    U32 ow, oh, oc, on;
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &on, &oc, &oh, &ow));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;
    bool useNchwFormat = (input->desc.memFormat == DF_NCHW) ? true : false;
    char kernelName[128];
    KernelOpt kernelOpt;
    CHECK_STATUS(set_activation_opt_mali(useNchwFormat, activationMode, DT_F16, kernelName, &kernelOpt));
    Kernel kernel;
    U32 gs[3] = {1, 1, 1};
    U32 ls[3] = {16, 1, 1};
    U32 dim = 3;
    U32 i_off = 0;
    U32 o_off = 0;
    if (useNchwFormat) {
        i_off = ih_off * iw_str + iw_off;
        o_off = oh_off * ow_str + ow_off;
        gs[0] = (ow + 3) / 4;
        gs[1] = oh;
        gs[2] = oc;
    } else {
        i_off = iw_off * ih_str + ih_off;
        o_off = ow_off * oh_str + oh_off;
        gs[0] = oh;
        gs[1] = ow;
        gs[2] = (oc + 3) / 4;
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ow, oh, oc, iw_str, ih_str, ow_str,
        oh_str, i_off, o_off, gs[0], gs[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
    return SUCCESS;
}

EE activation_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode)
{
    CHECK_STATUS(activation_checkpara_mali_fp16(inputDesc));
    if (input->mem != output->mem) {
        CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    }
    CHECK_STATUS(
        activation_core_mali_fp16(handle, inputDesc, input, outputDesc, output, activationMode));
    return SUCCESS;
}
