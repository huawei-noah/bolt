// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif
#ifdef _USE_INT8
#include "cpu/arm/int8/tensor_computing_int8.h"
#endif
#include "tensor_transpose.h"

EE depthwise_convolution_transform_filter_arm(TensorDesc filterDesc,
    const void *filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed)
{
    DataFormat ftmDataFormat;
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ftmDataFormat = DF_NCHWC8;
            break;
        default:
            return NOT_MATCH;
    }
    *ftmDesc = filterDesc;
    ftmDesc->df = ftmDataFormat;
    EE ret = NOT_SUPPORTED;
    if (filterDesc.df == ftmDataFormat) {
        memcpy(filterTransformed, filter, tensorNumBytes(filterDesc));
        ret = SUCCESS;
    } else if (filterDesc.df == DF_NCHW) {
        if (ftmDataFormat == DF_NCHWC8) {
            ret = transformNCHWToNCHWC8(filterDesc, filter, *ftmDesc, filterTransformed);
        }
    }
    return ret;
}

EE depthwise_convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
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
        default: {
            ret = NOT_MATCH;
            *bytes = 0;
            break;
        }
    }
    *bytes *= bytesOf(idt);

    switch (filterDesc.dt) {
#ifdef _USE_INT8
        case DT_I8: {
            *bytes += ic * oh * ow * sizeof(I32);
            break;
        }
#endif
        default:
            break;
    }
    if (idf != DF_NCHWC8) {
        *bytes += tensorNumBytes(inputDesc);
    }
    *bytes += 32;
    return ret;
}

EE depthwise_convolution_arm(TensorDesc inputDesc,
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
    return depthwise_pointwise_convolution_arm(inputDesc, input, filterDesc, filter, blankTensorDesc,
        nullptr, convParamSpec, algorithm, blankTensorDesc, bias, biasDesc, nullptr, tmpBytes, tmp,
        outputDesc, output, depthwiseActivationParamSpec, blankActivationParamSpec, arch);
}
