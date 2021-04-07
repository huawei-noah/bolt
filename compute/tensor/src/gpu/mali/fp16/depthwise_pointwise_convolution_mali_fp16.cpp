// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/depthwise_pointwise_convolution_mali_fp16.h"
#include "gpu/mali/fp16/depthwise_pointwise_convolution_direct_mali_fp16.h"
#include "gpu/mali/fp16/depthwise_pointwise_convolution_gemm_mali_fp16.h"

inline EE depthwise_pointwise_convolution_checkpara_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    const GCLMem_t dwFilter,
    const GCLMem_t pwFilter,
    const GCLMem_t dwBias,
    const GCLMem_t pwBias,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == dwFilter || nullptr == pwFilter ||
        nullptr == output || nullptr == dwBias || nullptr == pwBias || nullptr == tmpBuf) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != dwFilterDesc.dt ||
        inputDesc.dt != pwFilterDesc.dt || inputDesc.dt != DT_F16) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 ic, fn, fh, fw, oc;
    U32 dfc, pfc;
    CHECK_STATUS(tensorSelectGet(inputDesc, NULL, NULL, NULL, &ic, NULL, NULL));
    CHECK_STATUS(tensorSelectGet(dwFilterDesc, NULL, NULL, NULL, &dfc, &fh, &fw));
    CHECK_STATUS(tensorSelectGet(pwFilterDesc, NULL, NULL, &fn, &pfc, NULL, NULL));
    CHECK_STATUS(tensorSelectGet(outputDesc, NULL, NULL, NULL, &oc, NULL, NULL));
    if (ic != dfc || ic != pfc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (fn != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

EE depthwise_pointwise_convolution_transform_filter_bytes_mali_fp16(TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemDwFilterDesc,
    GCLMemDesc_t gclmemPwFilterDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct_transform_filter_bytes_mali_fp16(
                dwFilterDesc, pwFilterDesc, forwardRunInfo, gclmemDwFilterDesc, gclmemPwFilterDesc,
                bytes);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM:
            ret = depthwise_pointwise_convolution_gemm_transform_filter_bytes_mali_fp16(dwFilterDesc,
                pwFilterDesc, forwardRunInfo, gclmemDwFilterDesc, gclmemPwFilterDesc, bytes);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_pointwise_convolution_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    GCLMem_t dwFilter,
    GCLMem_t pwFilter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *dwFltmemDesc,
    TensorDesc *pwFltmemDesc,
    GCLMem_t dwFltmem,
    GCLMem_t pwFltmem)
{
    EE ret = SUCCESS;
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct_transform_filter_mali_fp16(handle,
                dwFilterDesc, pwFilterDesc, dwFilter, pwFilter, forwardRunInfo, dwFltmemDesc,
                pwFltmemDesc, dwFltmem, pwFltmem);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM:
            ret = depthwise_pointwise_convolution_gemm_transform_filter_mali_fp16(handle,
                dwFilterDesc, pwFilterDesc, dwFilter, pwFilter, forwardRunInfo, dwFltmemDesc,
                pwFltmemDesc, dwFltmem, pwFltmem);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = SUCCESS;
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct_infer_forward_tmp_bytes_mali_fp16(inputDesc,
                dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM:
            ret = depthwise_pointwise_convolution_gemm_infer_forward_tmp_bytes_mali_fp16(inputDesc,
                dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_pointwise_convolution_mali_fp16(GCLHandle_t handle,
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
    ActivationMode pointwiseActivationMode)
{
    EE ret = SUCCESS;
    CHECK_STATUS(depthwise_pointwise_convolution_checkpara_mali_fp16(handle, inputDesc, input,
        dwFilterDesc, pwFilterDesc, dwFilter, pwFilter, dwBias, pwBias, tmpBuf, outputDesc, output));
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct_mali_fp16(handle, inputDesc, input,
                dwFilterDesc, pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo,
                dwBiasDesc, pwBiasDesc, dwBias, pwBias, tmpBytes, tmpBuf, outputDesc, output,
                depthwiseActivationMode, pointwiseActivationMode);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM:
            ret = depthwise_pointwise_convolution_gemm_mali_fp16(handle, inputDesc, input,
                dwFilterDesc, pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo,
                dwBiasDesc, pwBiasDesc, dwBias, pwBias, tmpBytes, tmpBuf, outputDesc, output,
                depthwiseActivationMode, pointwiseActivationMode);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
