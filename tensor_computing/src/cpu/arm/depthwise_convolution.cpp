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

EE depthwise_convolution_infer_forward_algorithm_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm, DataType targetDataType)
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
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    switch (fdf)
    {
        case DF_NCHW: {
            *algorithm = DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT;
            if (convDesc.dilatedRate_h != 1 || convDesc.dilatedRate_w != 1) {
                return ret;
            }
            break;
        }
        case DF_CHW_NC: {
            *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT;
            if (convDesc.dilatedRate_h != 1 || convDesc.dilatedRate_w != 1) {
                return ret;
            }
            break;
        }
        default:
            return NOT_MATCH;
    }

    switch (targetDataType) {
        case DT_F16: {
            if (fdf == DF_NCHW) {
                break;
            }
            U32 strideH = convDesc.stride_h;
            U32 strideW = convDesc.stride_w;
            U32 paddingT = convDesc.padding_top;
            U32 paddingB = convDesc.padding_bottom;
            U32 paddingL = convDesc.padding_left;
            U32 paddingR = convDesc.padding_right;

            if (fh == 3 && fw == 3 && strideH == 1 && strideW == 1 && paddingT == 1 && paddingB == 1 && paddingL == 1 && paddingR == 1 && ow % 4 == 0 && ow >= 12) {
                *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1;
            }
            else if (fh == 3 && fw == 3 && strideH == 2 && strideW == 2 && ow >= 28) {
                *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING;
            }
            break;
        }
        default: {
            break;
        }
    }
    return ret;
}

EE depthwise_convolution_transform_filter_bytes_arm(TensorDesc filterDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32* bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    U32 fhfw = fh * fw;
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = fc * fhfw;
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = fc * fhfw + fn * fc;
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING:
            *bytes = fc * fhfw + fn * fc;
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1:
            *bytes = fc * fhfw + fn * fc;
            break;
        default:
            return NOT_SUPPORTED;
    }
    *bytes *= bytesOf(fdt);
    *bytes += 32;
    return SUCCESS;
}

EE depthwise_convolution_transform_filter_arm(TensorDesc filterDesc, const void* filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = depthwise_convolution_transform_filter_fp16(filterDesc, (F16*)filter, algorithm, ftmDesc, (F16*)filterTransformed);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = depthwise_convolution_transform_filter_fp32(filterDesc, (F32*)filter, algorithm, ftmDesc, (F32*)filterTransformed);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = depthwise_convolution_transform_filter_int8(filterDesc, (INT8*)filter, algorithm, ftmDesc, (INT8*)filterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes)
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
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;

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
    *bytes += 32;
    return ret;
}

EE depthwise_convolution_arm(TensorDesc inputDesc, void* input,
    TensorDesc filterDesc, const void* filter,
    ConvolutionDesc convDesc,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = depthwise_convolution_fp16(inputDesc, (F16*)input,
                                   filterDesc, (const F16*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (const F16*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (F16*)output,
                                   depthwiseActivationMode,
                                   pointwiseActivationMode,
                                   arch);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = depthwise_convolution_fp32(inputDesc, (F32*)input,
                                   filterDesc, (const F32*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (const F32*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (F32*)output,
                                   depthwiseActivationMode,
                                   pointwiseActivationMode,
                                   arch);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = depthwise_convolution_int8(inputDesc, (INT8*)input,
                                   filterDesc, (INT8*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (I32*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (I32*)output,
                                   depthwiseActivationMode,
                                   pointwiseActivationMode,
                                   arch);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
