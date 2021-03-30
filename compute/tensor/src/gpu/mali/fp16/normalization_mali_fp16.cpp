// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/normalization_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/normalization_opt.h"

inline EE normalization_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE normalization_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t alpha,
    GCLMem_t beta,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    U32 iw, ih, ic, in;
    U32 iw_str, ih_str, iw_off, ih_off;
    U32 ow_str, oh_str, ow_off, oh_off;

    CHECK_STATUS(gclmem_get_desc_dim(input->desc, NULL, NULL, &in, &ic, &ih, &iw));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    U32 axis_len = iw;  //normalization on axis 0;
    cl_mem alpbuf, betbuf, inbuf, outbuf, tmp;
    alpbuf = alpha->mem;
    betbuf = beta->mem;
    inbuf = input->mem;
    outbuf = output->mem;
    tmp = tmpbuf->mem;

    bool useNchw = (input->desc.memFormat == DF_NCHW) ? true : false;
    char kernelName[128];
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    Kernel kernel;
    KernelOpt kernelOpt;
    set_normalization_opt_mali(useNchw, DT_F16, kernelName, &kernelOpt);
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));

    if (useNchw) {
        gs[0] = 16;
        gs[1] = ih;
        gs[2] = ic * in;
        ls[0] = 16;
        ls[1] = 1;
        ls[2] = 1;
    } else {
        gs[0] = ih;
        gs[1] = iw;
        gs[2] = (ic + 3) / 4 * in;
    }

    float para = 1.0 / axis_len;
    CHECK_STATUS(gcl_set_kernelArgs(kernel, iw_str, ih_str, iw_off, ih_off, ow_str, oh_str, ow_off,
        oh_off, axis_len, gs[0], gs[1], para, tmp, alpbuf, betbuf, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE normalization_infer_forward_tmp_bytes_mali_fp16(GCLMemDesc gclmemInputDesc, U32 *bytes)
{
    U32 size = 0;
    if (gclmemInputDesc.memFormat == DF_NCHW) {
        DataType dt;
        U32 ih, ic, in;
        CHECK_STATUS(gclmem_get_desc_dim(gclmemInputDesc, &dt, NULL, &in, &ic, &ih, NULL));
        size = in * ic * ih * 16 * bytesOf(DT_F32) * 2;
    }
    *bytes = size;
    return SUCCESS;
}
EE normalization_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t alpha,
    GCLMem_t beta,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(normalization_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(normalization_core_mali_fp16(
        handle, inputDesc, input, alpha, beta, tmpbuf, outputDesc, output));
    return SUCCESS;
}
