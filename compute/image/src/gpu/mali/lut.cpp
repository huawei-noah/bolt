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
#include "gpu/mali/image_mali.h"
#include "gpu/mali/cl/kernel_option/lut_opt.h"

inline EE lut_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc lutDesc,
    GCLMem_t lut,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || lut == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.nDims != outputDesc.nDims) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (input->desc.memFormat != output->desc.memFormat) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

inline EE lut_core_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc lutDesc,
    GCLMem_t lut,
    LutParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    DataType idt, ldt;
    DataFormat idf, ldf;
    U32 in, ic, ih, iw;
    U32 ln, lc, lr, lg, lb;
    if (inputDesc.df == DF_NHWC) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    } else {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ih, &iw, &ic));
    }
    CHECK_STATUS(tensor4dGet(lutDesc, &ldt, &ldf, &lc, &lr, &lg, &lb));
    CHECK_REQUIREMENT(lr == lg && lr == lb);
    CHECK_REQUIREMENT(lc == 3);
    U32 height = ih / 3 * 2;
    U32 width = iw;
    int src_step = width;
    int dim = lr;
    int shift = dim * dim * dim;
    float binsize = 1.0001 / (dim - 1);

    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 kdim = 3;
    U32 gs[3] = {width / 2, height / 2, in};
    U32 ls[3] = {0, 0, 0};
    CHECK_STATUS(set_lut_opt_mali(p, ldt, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, input->mem, lut->mem, output->mem, dim, shift, binsize,
        width, height, in, tensorNumElements(inputDesc) / in, tensorNumElements(outputDesc) / in));
    gcl_set_kernelVec(handle, kernel, kdim, gs, ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE lut_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc lutDesc,
    GCLMem_t lut,
    LutParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    CHECK_STATUS(lut_checkpara_mali(handle, inputDesc, input, lutDesc, lut, outputDesc, output));
    CHECK_STATUS(lut_core_mali_fp16(handle, inputDesc, input, lutDesc, lut, p, outputDesc, output));
    return SUCCESS;
}
