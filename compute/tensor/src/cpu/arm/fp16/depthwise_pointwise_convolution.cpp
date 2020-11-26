// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp16/tensor_computing_fp16.h"
#include "cpu/arm/fp16/depthwise_pointwise_convolution_direct.h"
#include "cpu/arm/fp16/depthwise_pointwise_convolution_direct_no_padding.h"
#include "cpu/arm/fp16/depthwise_pointwise_convolution_3x3s1p1.h"

EE depthwise_pointwise_convolution_fp16(TensorDesc inputDesc,
    F16 *input,
    TensorDesc dwFilterDesc,
    const F16 *dwFilter,
    TensorDesc pwFilterDesc,
    const F16 *pwFilter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc dwBiasDesc,
    const F16 *dwBias,
    TensorDesc pwBiasDesc,
    const F16 *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F16 *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    Arch arch)
{
    if (nullptr == input || nullptr == dwFilter || nullptr == output || nullptr == dwBias ||
        nullptr == tmp) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (!(idt == DT_F16 && fdt == DT_F16 && odt == DT_F16)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(idf == DF_NCHWC8 && odf == DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (ic != fc) {
        CHECK_STATUS(NOT_MATCH);
    }

    EE ret = SUCCESS;
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct(inputDesc, input, dwFilterDesc, dwFilter,
                pwFilterDesc, pwFilter, convParamSpec, dwBiasDesc, dwBias, pwBiasDesc, pwBias,
                tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
                pointwiseActivationParamSpec, arch);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct(inputDesc, input, dwFilterDesc, dwFilter,
                pwFilterDesc, pwFilter, convParamSpec, dwBiasDesc, dwBias, pwBiasDesc, pwBias,
                tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
                pointwiseActivationParamSpec, arch);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING:
            ret = depthwise_pointwise_convolution_direct_no_padding(inputDesc, input, dwFilterDesc,
                dwFilter, pwFilterDesc, pwFilter, convParamSpec, dwBiasDesc, dwBias, pwBiasDesc,
                pwBias, tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
                pointwiseActivationParamSpec, arch);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1:
            ret = depthwise_pointwise_convolution_3x3s1p1(inputDesc, input, dwFilterDesc, dwFilter,
                pwFilterDesc, pwFilter, convParamSpec, dwBiasDesc, dwBias, pwBiasDesc, pwBias,
                tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
                pointwiseActivationParamSpec, arch);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
