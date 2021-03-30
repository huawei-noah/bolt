// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#endif
#include "tensor_transpose.h"

EE depthwise_pointwise_convolution_transform_filter_x86(TensorDesc dwFilterDesc,
    const void *dwFilter,
    TensorDesc pwFilterDesc,
    const void *pwFilter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *dwFtmDesc,
    void *dwFilterTransformed,
    TensorDesc *pwFtmDesc,
    void *pwFilterTransformed)
{
    EE ret = SUCCESS;
    switch (dwFilterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = depthwise_pointwise_convolution_transform_filter_fp32(dwFilterDesc,
                (F32 *)dwFilter, pwFilterDesc, (F32 *)pwFilter, algorithm, dwFtmDesc,
                (F32 *)dwFilterTransformed, pwFtmDesc, (F32 *)pwFilterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_pointwise_convolution_x86(TensorDesc inputDesc,
    void *input,
    TensorDesc dwFilterDesc,
    const void *dwFilter,
    TensorDesc pwFilterDesc,
    const void *pwFilter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc dwBiasDesc,
    const void *dwBias,
    TensorDesc pwBiasDesc,
    const void *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    Arch arch)
{
    TensorDesc newInputDesc = inputDesc;
    void *newInput = input;
    if (inputDesc.df != DF_NCHWC8) {
        newInputDesc.df = DF_NCHWC8;
        newInput = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(inputDesc);
        tmpBytes -= tensorNumBytes(inputDesc);
        transformNCHWToNCHWC8(inputDesc, input, newInputDesc, newInput);
    }
    EE ret = SUCCESS;
    switch (dwFilterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = depthwise_pointwise_convolution_fp32(newInputDesc, (F32 *)newInput, dwFilterDesc,
                (const F32 *)dwFilter, pwFilterDesc, (const F32 *)pwFilter, convParamSpec,
                algorithm, dwBiasDesc, (const F32 *)dwBias, pwBiasDesc, (const F32 *)pwBias,
                tmpBytes, tmp, outputDesc, (F32 *)output, depthwiseActivationParamSpec,
                pointwiseActivationParamSpec, arch);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
