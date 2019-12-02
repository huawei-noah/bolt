// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "sys.h"
#include "tensor_desc.h"
#include "type.h"
#include "error.h"
#include "tensor_computing_type.h"

#include "cpu/arm/fp16/depthwise_convolution_fp16.h"
#include "cpu/arm/fp16/depthwise_convolution_direct.h"
#include "cpu/arm/fp16/depthwise_pointwise_convolution_direct.h"
#include "cpu/arm/fp16/depthwise_pointwise_convolution_direct_no_padding.h"
#include "cpu/arm/fp16/depthwise_pointwise_convolution_3x3s1p1.h"

EE depthwise_convolution_infer_forward_algorithm_fp16(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm)
{
    UNUSED(policy);

    if (nullptr == algorithm)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 stride = convDesc.stride;
    U32 padding = convDesc.padding;

    switch (fdf) {
        case DF_NCHW:
            *algorithm = DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT;
            break;
        case DF_CHW_NC: {
            if (fh == 3 && fw == 3 && stride == 1 && padding == 1 && ow%4 == 0 && ow >= 12) {
                *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1;
            } else if (fh == 3 && fw == 3 && stride == 2 && ow >= 28) {
                *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING;
            } else {
                *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT;
            }
            break;
        }
        default:
            return NOT_MATCH;
    }
    return SUCCESS;
}

EE depthwise_convolution_infer_forward_tmp_bytes_fp16(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 padding = convDesc.padding;

    U32 ih_pad = ih + 2*padding;
    U32 iw_pad = iw + 2*padding;
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
    return ret;
}

EE depthwise_convolution_fp16(TensorDesc inputDesc, F16* input,
    TensorDesc filterDesc, const F16* filter,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const F16* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch)
{
    if(nullptr == input || nullptr == filter || nullptr == output || nullptr == bias || nullptr == tmp)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS_WITH_RETURN(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (!(idt == DT_F16 && fdt == DT_F16 && odt == DT_F16))
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    if (fh != fw)
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    if (!(idf == DF_NCHWC8 && odf == DF_NCHWC8))
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);

    EE ret = SUCCESS;
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_convolution_direct(inputDesc, input,
                                               filterDesc, filter,
                                               convDesc,
                                               biasDesc, bias,
                                               tmpBytes, tmp,
                                               outputDesc, output,
                                               depthwiseActivationMode,
                                               arch);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ret = depthwise_pointwise_convolution_direct(inputDesc, input,
                                                         filterDesc, filter,
                                                         convDesc,
                                                         biasDesc, bias,
                                                         tmpBytes, tmp,
                                                         outputDesc, output,
                                                         depthwiseActivationMode,
                                                         pointwiseActivationMode,
                                                         arch);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT_NO_PADDING:
            ret = depthwise_pointwise_convolution_direct_no_padding(inputDesc, input,
                                                                    filterDesc, filter,
                                                                    convDesc,
                                                                    biasDesc, bias,
                                                                    tmpBytes, tmp,
                                                                    outputDesc, output,
                                                                    depthwiseActivationMode,
                                                                    pointwiseActivationMode,
                                                                    arch);
            break;
        case DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_3X3S1P1:
            ret = depthwise_pointwise_convolution_3x3s1p1(inputDesc, input,
                                                         filterDesc, filter,
                                                         convDesc,
                                                         biasDesc, bias,
                                                         tmpBytes, tmp,
                                                         outputDesc, output,
                                                         depthwiseActivationMode,
                                                         pointwiseActivationMode,
                                                         arch);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
