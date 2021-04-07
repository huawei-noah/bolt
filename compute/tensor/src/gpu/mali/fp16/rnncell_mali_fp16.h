// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _RNNCELL_MALI_FP16
#define _RNNCELL_MALI_FP16

#include "gpu/mali/fp16/tensor_computing_fp16.h"

EE rnncell_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnncellDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE rnncell_mali_fp16(GCLHandle_t handle,
    TensorDesc xDesc,
    const GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    GCLMem_t state,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    RNNParamSpec rnncellDesc,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo);
#endif
