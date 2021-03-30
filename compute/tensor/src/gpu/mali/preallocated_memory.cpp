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

inline EE preallocated_memory_checkpara_mali(
    GCLHandle_t handle, TensorDesc outputDesc, GCLMem_t output)
{
    if (handle == nullptr || nullptr == output) {
        return NULL_POINTER;
    }
    if (output->desc.memFormat != DF_NCHW) {
        return NOT_SUPPORTED;
    }
    if (outputDesc.dt != DT_F16 && outputDesc.dt != DT_I32) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline EE preallocated_memory_core_mali_fp16(
    GCLHandle_t handle, TensorDesc outputDesc, GCLMem_t output)
{
    DataType dt = outputDesc.dt;
    U32 numElements = output->desc.num;
    cl_mem outbuf = output->mem;
    U32 gs = numElements;
    U32 ls = 0;
    U32 dim = 1;
    Kernel kernel;
    char dataType[16];
    if (dt == DT_I32) {
        strcpy(dataType, "i32");
    }
    if (dt == DT_F16) {
        strcpy(dataType, "f16");
    }
    char kernelName[128];
    sprintf(kernelName, "fill_memory_zero_%s", dataType);

    CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
    CHECK_STATUS(gcl_set_kernelArgs(kernel, numElements, outbuf));
    gcl_set_kernelVec(handle, kernel, dim, &gs, &ls, kernelName);
#ifdef _DEBUG
    CHECK_STATUS(gcl_run_kernel(handle, kernel, dim, &gs, &ls, kernelName));
    CHECK_STATUS(gcl_print_memory<U8>(handle, output, "preallocated_memory_output"));
#endif
    return SUCCESS;
}

EE preallocated_memory_infer_output_size_mali(TensorDesc *outputDesc, GCLMemDesc_t gclmemOutputDesc)
{
    U32 w, h, c, n;
    TensorDesc desc = *outputDesc;
    U32 ndims = desc.nDims;
    DataType dt = desc.dt;
    if (ndims < 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    w = desc.dims[0];
    h = (ndims > 1) ? desc.dims[1] : 1;
    c = (ndims > 2) ? desc.dims[2] : 1;
    n = (ndims > 3) ? desc.dims[3] : 1;
    if (dt != DT_F16 && dt != DT_I32) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (n != 1) {
        CHECK_STATUS(NOT_SUPPORTED)
        CHECK_STATUS(infer_gclmem_desc_nchw(0, 0, 0, 0, 0, w, h, c, dt, dt, NULL, gclmemOutputDesc));
    }
    return SUCCESS;
}

EE preallocated_memory_mali(GCLHandle_t handle, TensorDesc outputDesc, GCLMem_t output)
{
    EE ret = SUCCESS;
    CHECK_STATUS(preallocated_memory_checkpara_mali(handle, outputDesc, output));
    CHECK_STATUS(preallocated_memory_core_mali_fp16(handle, outputDesc, output));
    return ret;
}
