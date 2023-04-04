// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/cl/kernel_option/prelu_opt.h"
#include "gpu/mali/fp16/prelu_mali_fp16.h"

inline EE prelu_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt) {
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
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw);
    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    Mem inMem = input->mem;
    Mem outMem = output->mem;
    Mem weiMem = weight->mem;
    if (in > 1 && !preluDesc.propagate_down) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (inputDesc.nDims > 4) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 weightNum = weight->desc.dims[0];
    ReluAxis reluAxis = RELU_ON_C;
    if (!preluDesc.propagate_down) {
        if (weightNum == iw) {
            reluAxis = RELU_ON_W;
        } else if (weightNum == ih) {
            reluAxis = RELU_ON_H;
        }
    }

    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    bool useNchw = (input->desc.memFormat == DF_NCHW) ? true : false;
    CHECK_STATUS(set_prelu_opt_mali(preluDesc.propagate_down, useNchw, reluAxis, input->desc.dt,
        input->desc.memType, output->desc.memType, kernelName, &kernelOpt));

    U32 gs[3] = {iw, ih, ((ic + 3) / 4) * in};
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    if (useNchw) {
        gs[0] = (iw + 3) / 4;
        gs[1] = ih;
        gs[2] = ic * in;
    }
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, i_off, ow_str, oh_str, o_off, iw, gs[0],
        gs[1], weiMem, inMem, outMem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
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
