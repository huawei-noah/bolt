// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/convolution_mali_fp16.h"
#include "gpu/mali/fp16/convolution_direct_mali_fp16.h"
#include "gpu/mali/fp16/convolution_wino_mali_fp16.h"
#include "gpu/mali/fp16/convolution_invgemm_mali_fp16.h"

inline EE convolution_checkpara_mali_fp16(GCLHandle_t handle,
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
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != filterDesc.dt) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 oc = outputDesc.dims[outputDesc.nDims - 2];
    if (input->desc.memFormat == DF_NCHWC4) {
        if (output->desc.memFormat == DF_NCHW) {
            if (oc != 1) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        } else if (output->desc.memFormat != DF_NCHWC4) {
            CHECK_STATUS(NOT_MATCH);
        }
    }
    return SUCCESS;
}

EE convolution_transform_filter_bytes_mali_fp16(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    EE ret = SUCCESS;
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            ret = convolution_direct_transform_filter_bytes_mali_fp16(
                filterDesc, forwardRunInfo, ftmDesc);
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ret = NOT_SUPPORTED;
            break;
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ret = convolution_wino_transform_filter_bytes_mali_fp16(
                filterDesc, forwardRunInfo, ftmDesc);
            break;
        case CONVOLUTION_ALGORITHM_INVGEMM:
            ret = convolution_invgemm_transform_filter_bytes_mali_fp16(
                filterDesc, forwardRunInfo, ftmDesc);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_transform_filter_mali_fp16(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    GCLMem_t tmp)
{
    EE ret = SUCCESS;
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            ret = convolution_direct_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem);
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ret = NOT_SUPPORTED;
            break;
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ret = convolution_wino_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem, tmp);
            break;
        case CONVOLUTION_ALGORITHM_INVGEMM:
            ret = convolution_invgemm_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_infer_forward_tmp_bytes_mali_fp16(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = SUCCESS;
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            ret = convolution_direct_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ret = NOT_SUPPORTED;
            break;
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ret = convolution_wino_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        case CONVOLUTION_ALGORITHM_INVGEMM:
            ret = convolution_invgemm_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_mali_fp16(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    std::vector<GCLMem_t> tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationParamSpec activationMode)
{
    CHECK_STATUS(convolution_checkpara_mali_fp16(
        handle, inputDesc, input, filterDesc, filter, bias, outputDesc, output));
    EE ret = SUCCESS;
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            ret = convolution_direct_mali_fp16(handle, inputDesc, input, filterDesc, filter,
                convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf[0], outputDesc,
                output, activationMode);
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ret = NOT_SUPPORTED;
            break;
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ret = convolution_wino_mali_fp16(handle, inputDesc, input, filterDesc, filter,
                convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
                activationMode);
            break;
        case CONVOLUTION_ALGORITHM_INVGEMM:
            ret = convolution_invgemm_mali_fp16(handle, inputDesc, input, filterDesc, filter,
                convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf[0], outputDesc,
                output, activationMode);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
