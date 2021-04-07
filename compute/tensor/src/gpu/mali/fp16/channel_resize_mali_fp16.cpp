// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/channel_resize_mali_fp16.h"

inline EE channel_resize_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE channel_resize_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ChannelResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 iw, ih, ic, in, oc;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    tensorSelectGet(outputDesc, NULL, NULL, NULL, &oc, NULL, NULL);
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    get_gclmem_dim(input->desc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    get_gclmem_dim(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off);
    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;
    DataFormat imf = input->desc.memFormat;
    DataFormat omf = output->desc.memFormat;
    U32 dim = 3;
    Kernel kernel;
    char kernelName[128];
    U32 gs[3] = {ih, iw, (U32)(p.channel_after + 3) / 4};
    U32 ls[3] = {0, 0, 0};
    sprintf(kernelName, "channel_resize");
    if (imf == DF_NCHW && omf == DF_NCHW) {
        gs[0] = (iw + 3) / 4;
        gs[1] = ih;
        gs[2] = p.channel_after;
        sprintf(kernelName, "channel_resize_nchw");
    }

    if (imf == DF_NCHW && omf == DF_NCWHC4) {
        gs[0] = (iw + 3) / 4;
        gs[1] = ih;
        gs[2] = (p.channel_after + 3) / 4;
        sprintf(kernelName, "channel_resize_nchw_ncwhc4");
    }

    if (imf == DF_NCWHC4 && omf == DF_NCHW) {
        gs[0] = ih;
        gs[1] = (iw + 3) / 4;
        gs[2] = (p.channel_after + 3) / 4;
        sprintf(kernelName, "channel_resize_ncwhc4_nchw");
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iw_str, ic_str, ih_off, iw_off, oh_str, ow_str,
        oh_off, ow_off, p.channel_before, p.channel_after, iw, gs[0], gs[1], inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE channel_resize_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ChannelResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(channel_resize_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(channel_resize_core_mali_fp16(handle, inputDesc, input, p, outputDesc, output));
    return SUCCESS;
}
