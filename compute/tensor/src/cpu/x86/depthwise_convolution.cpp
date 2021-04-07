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

EE depthwise_convolution_transform_filter_x86(TensorDesc filterDesc,
    const void *filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = depthwise_convolution_transform_filter_fp32(
                filterDesc, (F32 *)filter, algorithm, ftmDesc, (F32 *)filterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_infer_forward_tmp_bytes_x86(TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    EE ret = SUCCESS;
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = ic * ih_pad * iw_pad;
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = ic * ih_pad * iw_pad + ic * oh * ow;
            break;
        default: {
            ret = NOT_MATCH;
            *bytes = 0;
            break;
        }
    }
    *bytes *= bytesOf(idt);
    if (idf != DF_NCHWC8) {
        *bytes += tensorNumBytes(inputDesc);
    }
    *bytes += 32;
    return ret;
}

EE depthwise_convolution_x86(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    Arch arch)
{
    TensorDesc blankTensorDesc;
    ActivationParamSpec blankActivationParamSpec;
    return depthwise_pointwise_convolution_x86(inputDesc, input, filterDesc, filter, blankTensorDesc,
        nullptr, convParamSpec, algorithm, blankTensorDesc, bias, biasDesc, nullptr, tmpBytes, tmp,
        outputDesc, output, depthwiseActivationParamSpec, blankActivationParamSpec, arch);
}
