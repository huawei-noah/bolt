// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _OCL_DATA_TRANS
#define _OCL_DATA_TRANS

#include "tensor_desc.h"
#include "gcl_common.h"
typedef enum { NCHW_TO_NCWHC4 = 0, NCWHC4_TO_NCHW = 1, NCHW_TO_NCHW = 2 } DataTransFormType;

EE ocl_data_trans_form(GCLHandle_t handle,
    GCLMem_t input,
    GCLMem_t output,
    U32 in_off,
    U32 out_off,
    DataTransFormType type,
    bool setKernelVec = true);

EE ocl_set_input(GCLHandle_t handle,
    GCLMem_t input,
    TensorDesc hostDesc,
    const U8 *hostPtr,
    GCLMem_t tmpBuf,
    bool blocking);

EE ocl_get_output(GCLHandle_t handle, const GCLMem_t input, TensorDesc hostDesc, bool blocking);

EE ocl_trans_mem(
    GCLHandle_t handle, GCLMem_t src, GCLMemDesc srcDesc, GCLMem_t dst, GCLMemDesc dstDesc);

EE ocl_map_mem_write(
    GCLHandle_t handle, GCLMem_t gclMem, GCLMemDesc desc, TensorDesc hostDesc, U8 *host_ptr);

EE ocl_map_mem_read(GCLHandle_t handle, GCLMem_t gclMem, GCLMemDesc desc);
#endif
