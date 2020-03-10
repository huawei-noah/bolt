// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _FC_MALI_FP16
#define _FC_MALI_FP16
#include "sys.h"
#include "tensor_desc.h"
#include "type.h"
#include "error.h"
#include "tensor_computing_type.h"

EE fully_connected_transform_filter_bytes_mali_fp16(TensorDesc            filterDesc, 
                                                    GCLMemDesc_t          gclmemFilterDesc,
                                                    U32*                  bytes);

EE fully_connected_transform_filter_mali_fp16(GCLHandle_t          handle,
                                              TensorDesc           filterDesc,
                                              GCLMem_t             filter,
                                              TensorDesc*          fltmemDesc,
                                              GCLMem_t             fltmem);

EE fully_connected_infer_forward_tmp_bytes_mali_fp16(TensorDesc            inputDesc, 
                                                     TensorDesc            filterDesc, 
                                                     U32*                  bytes);

EE fully_connected_mali_fp16(GCLHandle_t          handle,
                             TensorDesc           inputDesc, 
                             const GCLMem_t       input,
                             TensorDesc           filterDesc, 
                             const GCLMem_t       filter,
                             TensorDesc           biasDesc, 
                             const GCLMem_t       bias,
                             U32                  tmpBytes, 
                             GCLMem_t             tmpBuf,
                             TensorDesc           outputDesc, 
                             GCLMem_t             output);
#endif                             
