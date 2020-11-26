// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"
#include "error.h"
#include "types.h"
#include "gpu/mali/fp16/normalization_mali_fp16.h"

inline EE normalization_checkpara_mali_fp16(TensorDesc inputDesc, TensorDesc outputDesc)
{
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != DT_F16) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE normalization_core_mali_fp16(GCLHandle_t handle,
    GCLMem_t alpha,
    GCLMem_t beta,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(outputDesc);
    U32 step = inputDesc.dims[0];
    U32 numOutput = inputDesc.dims[1];
    U32 iw_str, ih_str, ic_str, iw_off, ih_off;
    U32 oh_str, ow_off, oh_off;
    ih_str = input->desc.stride[0];
    iw_str = input->desc.stride[1];
    ic_str = input->desc.stride[2];
    ih_off = input->desc.offset[0];
    iw_off = input->desc.offset[1];
    oh_str = output->desc.stride[0];
    oh_off = output->desc.offset[0];
    ow_off = output->desc.offset[1];
    if (iw_str != 1 || ih_off != 0 || iw_off != 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    cl_mem alpbuf, betbuf, inbuf, outbuf;
    alpbuf = alpha->mem;
    betbuf = beta->mem;
    inbuf = input->mem;
    outbuf = output->mem;

    U32 gs = step;
    U32 ls = 0;
    U32 dim = 1;
    float para = 1.0 / numOutput;
    Kernel kernel;
    CHECK_STATUS(gcl_create_kernel(handle, "normalization", &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, step, ih_str, ic_str, ih_off, iw_off, oh_str, oh_off,
        ow_off, para, alpbuf, betbuf, inbuf, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, "normalization");
#ifdef _DEBUG
    CHECK_STATUS(gcl_print_memory<F16>(handle, input, "normalization_input"));
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, "normalization"));
    CHECK_STATUS(gcl_print_memory<F16>(handle, output, "normalization_output"));
#endif
    return SUCCESS;
}

EE normalization_mali_fp16(GCLHandle_t handle,
    GCLMem_t alpha,
    GCLMem_t beta,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(normalization_checkpara_mali_fp16(inputDesc, outputDesc));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(
        normalization_core_mali_fp16(handle, alpha, beta, inputDesc, input, outputDesc, output));
    return SUCCESS;
}
