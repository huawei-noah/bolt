// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"

EE depthwise_pointwise_convolution_fp32(TensorDesc inputDesc,
    F32 *input,
    TensorDesc dwFilterDesc,
    const F32 *dwFilter,
    TensorDesc pwFilterDesc,
    const F32 *pwFilter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc dwBiasDesc,
    const F32 *dwBias,
    TensorDesc pwBiasDesc,
    const F32 *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    Arch arch)
{
    UNUSED(arch);
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

    if (!(idt == DT_F32 && fdt == DT_F32 && odt == DT_F32)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(idf == DF_NCHWC8 && odf == DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (ic != fc) {
        CHECK_STATUS(NOT_MATCH);
    }

    EE ret = NOT_MATCH;
    if (algorithm == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT ||
        algorithm == DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT) {
        ret = depthwise_convolution_direct(inputDesc, input, dwFilterDesc, dwFilter, pwFilterDesc,
            pwFilter, convParamSpec, dwBiasDesc, dwBias, pwBiasDesc, pwBias, tmpBytes, tmp,
            outputDesc, output, depthwiseActivationParamSpec, pointwiseActivationParamSpec);
    }
    return ret;
}
