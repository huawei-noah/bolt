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
#include "gpu/mali/cl/kernel_option/lut_preprocess_opt.h"

inline EE lut_preprocess_checkpara_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || input == nullptr || output == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.nDims != outputDesc.nDims) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

inline EE lut_preprocess_core_mali_fp16(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    DataType odt;
    DataFormat odf;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 height = oh * 2;
    U32 width = ow * 2;
    int src_step = width;
    int dst_step = width * height / 4;
    int src_row = height;
    int dst_col = width / 2;
    char kernelName[128];
    KernelOpt kernelOpt;
    Kernel kernel;
    U32 dim = 3;
    U32 gs[3] = {width / 2, height / 2, on};
    U32 ls[3] = {0, 0, 0};
    CHECK_STATUS(set_lut_preprocess_opt_mali(odt, kernelName, &kernelOpt));
    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel, &kernelOpt));
    CHECK_STATUS(gcl_set_kernelArgs(
        kernel, input->mem, output->mem,
        tensorNumElements(inputDesc) / on, src_row, src_step,
        on, oh, ow, oc * oh * ow, oh * ow));
    gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
#ifdef _DEBUG
//    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, gs, ls, kernelName));
#endif
    return SUCCESS;
}

EE lut_preprocess_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output)
{
    CHECK_STATUS(lut_preprocess_checkpara_mali(handle, inputDesc, input, outputDesc, output));
    CHECK_STATUS(lut_preprocess_core_mali_fp16(handle, inputDesc, input, outputDesc, output));
    return SUCCESS;
}
