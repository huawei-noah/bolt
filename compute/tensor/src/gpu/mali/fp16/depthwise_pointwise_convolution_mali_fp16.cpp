// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gpu/mali/fp16/depthwise_convolution_direct_mali_fp16.h"
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
    const GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == dwFilter || nullptr == pwFilter ||
        nullptr == output || nullptr == dwBias || nullptr == pwBias || nullptr == tmp) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.dt != outputDesc.dt || inputDesc.dt != dwFilterDesc.dt ||
        inputDesc.dt != pwFilterDesc.dt) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

EE depthwise_pointwise_convolution_transform_filter_bytes_mali_fp16(TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *dwFtmDesc,
    TensorDesc *pwFtmDesc)
{
    EE ret = SUCCESS;
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    switch (algorithm) {
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct_transform_filter_bytes_mali_fp16(
                dwFilterDesc, pwFilterDesc, forwardRunInfo, dwFtmDesc, pwFtmDesc);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM:
            ret = depthwise_pointwise_convolution_gemm_transform_filter_bytes_mali_fp16(
                dwFilterDesc, pwFilterDesc, forwardRunInfo, dwFtmDesc, pwFtmDesc);
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
    if (inputDesc.df == DF_NCHW) {
        U32 size = 0;
        GCLMemDesc desc = depthwise_convolution_get_input_nchwc4_desc(
            inputDesc, dwFilterDesc, convParamSpec, outputDesc, forwardRunInfo->best_h[0]);
        size = UNI_ALIGN(desc.byteSize, BUFFER_ALIGN_BASE);
        if (desc.memType == GCL_MEM_IMG_3D) {
            bytes[1] = desc.stride[0];
            bytes[2] = desc.stride[1];
            bytes[3] = desc.stride[2];
        } else {
            bytes[0] += desc.byteSize;
        }
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
    std::vector<GCLMem_t> tmp,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec)
{
    EE ret = SUCCESS;
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    GCLMem inputTran;
    GCLMem tmpMem;
    GCLMem_t inputPtr = input;
    bool useImg = (tmp[2] && algorithm == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) ? true
                                                                                            : false;
    GCLMem_t tmpPtr = (useImg) ? tmp[2] : tmp[0];
    if (inputDesc.df == DF_NCHW) {
        U32 tmpBufOff = 0;
        tmpPtr = (tmp[1]) ? tmp[1] : tmp[0];
        GCLMemDesc transDesc;
        CHECK_STATUS(depthwise_convolution_trans_input_to_nchwc4(handle, inputDesc, dwFilterDesc,
            input, convParamSpec, tmpPtr, outputDesc, forwardRunInfo->best_h[0], &transDesc,
            &tmpBufOff));
        inputDesc.df = DF_NCHWC4;
        inputTran.desc = transDesc;
        inputTran.mem = tmpPtr->mem;
        inputPtr = &inputTran;

        if (useImg) {
            tmpPtr = tmp[2];
        } else {
            tmpPtr = tmp[0];
            if (tmpBufOff > 0) {
                U32 bytes[7] = {0};
                switch (algorithm) {
                    case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
                        CHECK_STATUS(
                            depthwise_pointwise_convolution_direct_infer_forward_tmp_bytes_mali_fp16(
                                inputDesc, dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec,
                                forwardRunInfo, bytes));
                        break;
                    case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM:
                        CHECK_STATUS(
                            depthwise_pointwise_convolution_gemm_infer_forward_tmp_bytes_mali_fp16(
                                inputDesc, dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec,
                                forwardRunInfo, bytes));
                        break;
                    default:
                        CHECK_STATUS(NOT_SUPPORTED);
                        break;
                }
                if (bytes[4] > 0 || bytes[5] > 0 || bytes[6] > 0) {
                    CHECK_STATUS(NOT_MATCH);  //suppose to use Img for depthwise out
                }
                CHECK_STATUS(gcl_create_sub_buffer(bytes[0], &tmpBufOff, tmp[0], &tmpMem.mem));
                tmpMem.desc = tmp[0]->desc;
                tmpPtr = &tmpMem;
            }
        }
    }
    CHECK_STATUS(depthwise_pointwise_convolution_checkpara_mali_fp16(handle, inputDesc, input,
        dwFilterDesc, pwFilterDesc, dwFilter, pwFilter, dwBias, pwBias, tmpPtr, outputDesc, output));
    switch (algorithm) {
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct_mali_fp16(handle, inputDesc, inputPtr,
                dwFilterDesc, pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo,
                dwBiasDesc, pwBiasDesc, dwBias, pwBias, tmpBytes, tmpPtr, outputDesc, output,
                depthwiseActivationParamSpec, pointwiseActivationParamSpec);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM:
            ret = depthwise_pointwise_convolution_gemm_mali_fp16(handle, inputDesc, inputPtr,
                dwFilterDesc, pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo,
                dwBiasDesc, pwBiasDesc, dwBias, pwBias, tmpBytes, tmpPtr, outputDesc, output,
                depthwiseActivationParamSpec, pointwiseActivationParamSpec);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
