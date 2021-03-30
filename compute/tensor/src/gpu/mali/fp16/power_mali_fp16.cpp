// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/power_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/power_opt.h"

inline EE power_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    EE ret = SUCCESS;
    if (inputDesc.dt != outputDesc.dt || (inputDesc.dt != DT_F16 && inputDesc.dt != DT_I32)) {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

inline EE power_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    PowerParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType dt;
    U32 iw, ih, ic, in;
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;
    CHECK_STATUS(gclmem_get_desc_dim(input->desc, &dt, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    cl_mem inbuf, outbuf;
    inbuf = input->mem;
    outbuf = output->mem;
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {16, 1, 1};
    U32 dim = 3;
    char kernelName[128];
    KernelOpt kernelOpt;
    bool useNchwFormat = (input->desc.memFormat == DF_NCHW) ? true : false;
    CHECK_STATUS(set_power_opt_mali(useNchwFormat, dt, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    if (useNchwFormat) {
        gs[0] = (iw + 3) / 4;
        gs[1] = ih;
        gs[2] = ic;
    } else {
        gs[0] = ih;
        gs[1] = iw;
        gs[2] = (ic + 3) / 4;
    }
    U32 has_power = (p.power == (F32)1.0) ? 0 : 1;
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str, ow_off,
        oh_off, iw, gs[0], gs[1], has_power, p.scale, p.shift, p.power, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
    return SUCCESS;
}

EE power_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    PowerParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(power_checkpara_mali_fp16(inputDesc, outputDesc));
    if (input->mem != output->mem) {
        CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    }
    CHECK_STATUS(power_core_mali_fp16(handle, inputDesc, input, p, outputDesc, output));
    return SUCCESS;
}
