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

#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/cl/kernel_option/cast_opt.h"

inline EE cast_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.nDims != outputDesc.nDims) {
        CHECK_STATUS(NOT_MATCH);
    }
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        if (inputDesc.dims[i] != outputDesc.dims[i]) {
            CHECK_STATUS(NOT_MATCH);
        }
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

inline EE cast_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    CastParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    UNUSED(p);
    U32 n, c, h, w;
    U32 iw_str, ih_str, iw_off, ih_off, i_off;
    U32 ow_str, oh_str, ow_off, oh_off, o_off;
    CHECK_STATUS(gclmem_get_desc_dim(output->desc, NULL, NULL, &n, &c, &h, &w));
    CHECK_STATUS(gclmem_get_desc_padding(input->desc, &iw_str, &ih_str, NULL, &iw_off, &ih_off));
    CHECK_STATUS(gclmem_get_desc_padding(output->desc, &ow_str, &oh_str, NULL, &ow_off, &oh_off));
    i_off = ih_off * iw_str + iw_off;
    o_off = oh_off * ow_str + ow_off;
    DataType idt = inputDesc.dt;
    DataType odt = outputDesc.dt;
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 gs[3];
    U32 ls[3] = {0, 0, 0};
    U32 dim = 3;
    bool useNchw = (input->desc.memFormat != DF_NCHWC4) ? true : false;
    if (useNchw) {
        gs[0] = (w + 3) / 4;
        gs[1] = h;
        gs[2] = n * c;
    } else {
        gs[0] = w;
        gs[1] = h;
        gs[2] = (c + 3) / 4 * n;
    }
    CHECK_STATUS(
        set_cast_opt_mali(useNchw, idt, odt, GCL_MEM_BUF, GCL_MEM_BUF, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, w, iw_str, ih_str, i_off, ow_str, oh_str, o_off, gs[0],
        gs[1], input->mem, output->mem));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE cast_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    CastParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(cast_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    CHECK_STATUS(fill_output_zero(handle, output, outputDesc));
    CHECK_STATUS(cast_core_mali_fp16(handle, inputDesc, input, p, outputDesc, output));
    return SUCCESS;
}
