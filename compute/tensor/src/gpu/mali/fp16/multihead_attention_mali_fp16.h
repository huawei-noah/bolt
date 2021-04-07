// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MULTIHEAD_ATTENTION_MALI_FP16
#define _MULTIHEAD_ATTENTION_MALI_FP16

#include "gpu/mali/fp16/tensor_computing_fp16.h"

EE multihead_attention_transform_filter_bytes_mali_fp16(std::vector<TensorDesc> filterDesc,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE multihead_attention_transform_filter_mali_fp16(GCLHandle_t handle,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> *fltmemDesc,
    std::vector<void *> fltmem,
    ForwardRunInfoMali_t forwardRunInfo);

EE multihead_attention_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    std::vector<bool> eltwiseWithLayerNormIn,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE multihead_attention_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> biasDesc,
    std::vector<void *> bias,
    std::vector<void *> layerNormAlpha,
    std::vector<void *> layerNormBeta,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo);
#endif
