// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#include "cpu/tensor_computing_cpu.h"

EE depthwise_pointwise_convolution_cpu(TensorDesc inputDesc,
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
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = depthwise_pointwise_convolution_general(inputDesc, input, dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, convParamSpec, dwBiasDesc, dwBias, pwBiasDesc, pwBias, tmpBytes,
            tmp, outputDesc, output, depthwiseActivationParamSpec, pointwiseActivationParamSpec);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = depthwise_pointwise_convolution_x86(inputDesc, input, dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, convParamSpec, algorithm, dwBiasDesc, dwBias, pwBiasDesc,
            pwBias, tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
            pointwiseActivationParamSpec, arch);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_pointwise_convolution_arm(inputDesc, input, dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, convParamSpec, algorithm, dwBiasDesc, dwBias, pwBiasDesc,
            pwBias, tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
            pointwiseActivationParamSpec, arch);
#endif
    }
    return ret;
}