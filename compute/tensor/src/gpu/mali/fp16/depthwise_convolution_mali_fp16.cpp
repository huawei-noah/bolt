// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/depthwise_convolution_mali_fp16.h"
#include "gpu/mali/fp16/depthwise_convolution_direct_mali_fp16.h"

inline EE depthwise_convolution_checkpara_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    const GCLMem_t bias,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == filter || nullptr == output ||
        nullptr == bias) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != filterDesc.dt || inputDesc.dt != DT_F16) {
        CHECK_STATUS(NOT_MATCH);
    }

    DataFormat fdf;
    U32 ic, fc, fh, fw, oc;
    CHECK_STATUS(tensorSelectGet(inputDesc, NULL, NULL, NULL, &ic, NULL, NULL));
    CHECK_STATUS(tensorSelectGet(filterDesc, NULL, &fdf, NULL, &fc, &fh, &fw));
    CHECK_STATUS(tensorSelectGet(outputDesc, NULL, NULL, NULL, &oc, NULL, NULL));
    if (input->desc.memFormat == DF_NCWHC4) {
        if (filter->desc.memFormat != DF_NHWCN4) {
            CHECK_STATUS(NOT_MATCH);
        }
        if (output->desc.memFormat != DF_NCWHC4) {
            CHECK_STATUS(NOT_MATCH);
        }
    }
    if (fdf == DF_NCHW && ic != fc) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (fc != oc) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

EE depthwise_convolution_transform_filter_bytes_mali_fp16(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_convolution_direct_transform_filter_bytes_mali_fp16(
                filterDesc, forwardRunInfo, gclmemFilterDesc, bytes);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem)
{
    EE ret = SUCCESS;
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_convolution_direct_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = SUCCESS;
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_convolution_direct_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode)
{
    EE ret = SUCCESS;
    CHECK_STATUS(depthwise_convolution_checkpara_mali_fp16(
        handle, inputDesc, input, filterDesc, filter, bias, outputDesc, output));
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_convolution_direct_mali_fp16(handle, inputDesc, input, filterDesc,
                filter, convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc,
                output, depthwiseActivationMode);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
