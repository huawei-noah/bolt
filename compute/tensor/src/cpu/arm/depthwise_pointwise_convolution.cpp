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

EE depthwise_pointwise_convolution_infer_forward_algorithm_arm(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    DepthwiseConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType)
{
    UNUSED(policy);
    if (nullptr == algorithm) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT;
    if (convParamSpec.dilatedRate_h != 1 || convParamSpec.dilatedRate_w != 1) {
        return ret;
    }

    switch (targetDataType) {
        case DT_F16: {
            U32 strideH = convParamSpec.stride_h;
            U32 strideW = convParamSpec.stride_w;
            U32 paddingT = convParamSpec.padding_top;
            U32 paddingB = convParamSpec.padding_bottom;
            U32 paddingL = convParamSpec.padding_left;
            U32 paddingR = convParamSpec.padding_right;

            if (fh == 3 && fw == 3 && strideH == 1 && strideW == 1 && paddingT == 1 &&
                paddingB == 1 && paddingL == 1 && paddingR == 1 && ow % 4 == 0 && ow >= 12) {
                *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1;
            }
            break;
        }
        default: {
            break;
        }
    }
    return ret;
}

EE depthwise_pointwise_convolution_transform_filter_arm(TensorDesc dwFilterDesc,
    const void *dwFilter,
    TensorDesc pwFilterDesc,
    const void *pwFilter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *dwFtmDesc,
    void *dwFilterTransformed,
    TensorDesc *pwFtmDesc,
    void *pwFilterTransformed)
{
    EE ret = depthwise_convolution_transform_filter_arm(dwFilterDesc, dwFilter,
        DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT, dwFtmDesc, dwFilterTransformed);
    if (ret == SUCCESS) {
        convParamSpec.group = 1;
        ret = convolution_transform_filter_arm(pwFilterDesc, pwFilter, convParamSpec,
            CONVOLUTION_ALGORITHM_GEMM, pwFtmDesc, pwFilterTransformed);
    }
    return ret;
}

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
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
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = ic * ih_pad * iw_pad + ic * oh * ow;
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING:
            *bytes = ic * oh * ow;
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1:
            *bytes = ic * oh * ow + ic * 8;
            break;
        default: {
            ret = NOT_MATCH;
            *bytes = 0;
            break;
        }
    }
    *bytes *= bytesOf(idt);

    switch (dwFilterDesc.dt) {
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

EE depthwise_pointwise_convolution_arm(TensorDesc inputDesc,
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
#ifdef _USE_FP16
        case DT_F16: {
            ret = depthwise_pointwise_convolution_fp16(newInputDesc, (F16 *)newInput, dwFilterDesc,
                (const F16 *)dwFilter, pwFilterDesc, (const F16 *)pwFilter, convParamSpec,
                algorithm, dwBiasDesc, (const F16 *)dwBias, pwBiasDesc, (const F16 *)pwBias,
                tmpBytes, tmp, outputDesc, (F16 *)output, depthwiseActivationParamSpec,
                pointwiseActivationParamSpec, arch);
            break;
        }
#endif
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
#ifdef _USE_INT8
        case DT_I8: {
            ret = depthwise_pointwise_convolution_int8(newInputDesc, (INT8 *)newInput, dwFilterDesc,
                (const INT8 *)dwFilter, pwFilterDesc, (const INT8 *)pwFilter, convParamSpec,
                algorithm, dwBiasDesc, (const I32 *)dwBias, pwBiasDesc, (const I32 *)pwBias,
                tmpBytes, tmp, outputDesc, (I32 *)output, depthwiseActivationParamSpec,
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
