// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _DEPTHWISE_POINTWISE_CONVOLUTION_GEMM_MALI_FP16
#define _DEPTHWISE_POINTWISE_CONVOLUTION_GEMM_MALI_FP16

#include "gpu/mali/fp16/tensor_computing_fp16.h"

EE depthwise_pointwise_convolution_gemm_transform_filter_bytes_mali_fp16(TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemDwFilterDesc,
    GCLMemDesc_t gclmemPwFilterDesc,
    U32 *bytes);

EE depthwise_pointwise_convolution_gemm_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    GCLMem_t dwFilter,
    GCLMem_t pwFilter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *dwFltmemDesc,
    TensorDesc *pwFltmemDesc,
    GCLMem_t dwFltmem,
    GCLMem_t pwFltmem);

EE depthwise_pointwise_convolution_gemm_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes);

EE depthwise_pointwise_convolution_gemm_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    const GCLMem_t dwFilter,
    const GCLMem_t pwFilter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc dwBiasDesc,
    TensorDesc pwBiasDesc,
    const GCLMem_t dwBias,
    const GCLMem_t pwBias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode);
#endif
