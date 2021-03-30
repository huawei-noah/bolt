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
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    if (padParamSpec.pad_mode == Pad_Reflect &&
        (padParamSpec.top >= inputDesc.dims[1] || padParamSpec.bottom >= inputDesc.dims[1])) {
        return NOT_SUPPORTED;
    }
    if (padParamSpec.pad_mode == Pad_Symmetric &&
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
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, &on, &oc, &oh, &ow);

    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;

    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);

    U32 pl, pr, pt, pb;
    pl = padParamSpec.left;
    pr = padParamSpec.right;
    pt = padParamSpec.top;
    pb = padParamSpec.bottom;

    char kernelName[128];
    char formatName[128];
    char modeName[128];
    U32 gs[3] = {0, 0, 0};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (input->desc.memFormat == DF_NCHW) {
        strcpy(formatName, "nchw_");
        gs[0] = (ow + 3) / 4;
        gs[1] = oh;
        gs[2] = oc * on;
    } else if (input->desc.memFormat == DF_NCWHC4) {
        strcpy(formatName, "");
        gs[0] = oh;
        gs[1] = ow;
        gs[2] = (oc + 3) / 4 * on;
    } else {
        CHECK_STATUS(NOT_SUPPORTED)
    }
    switch (padParamSpec.pad_mode) {
        case Pad_Constant:
            strcpy(modeName, "constant");
            break;
        case Pad_Edge:
            strcpy(modeName, "edge");
            break;
        case Pad_Reflect:
            strcpy(modeName, "reflect");
            break;
        case Pad_Symmetric:
            strcpy(modeName, "symmetric");
            break;
        default:
            return NOT_SUPPORTED;
    }
    sprintf(kernelName, "padding_%s%s", formatName, modeName);
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str, ow_off,
        oh_off, iw, ih, ow, oh, pt, pb, pl, pr, 0, 0, gs[0], gs[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, "padding_constant"));
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
