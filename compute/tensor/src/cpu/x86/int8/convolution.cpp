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
#include "cpu/x86/int8/tensor_computing_int8.h"

EE convolution_infer_forward_tmp_bytes_int8(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
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
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;

    U32 icAlignSize = 16;
    U32 icPadding = (ic + icAlignSize - 1) / icAlignSize * icAlignSize;

    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = icPadding * ih_pad * iw_pad;
            break;
        case CONVOLUTION_ALGORITHM_POINTWISE: {
            U32 strideH = convParamSpec.stride_h;
            U32 strideW = convParamSpec.stride_w;
            *bytes = oc + 32;
            if (strideH > 1 || strideW > 1) {
                U32 noStrideH = (ih_pad + strideH - 1) / strideH;
                U32 noStrideW = (iw_pad + strideW - 1) / strideW;
                *bytes += icPadding * noStrideW * noStrideH;
            }
            if (idf != DF_NCHWC16) {
                *bytes += icPadding * ih_pad * iw_pad;
            }
            if (paddingT > 1 || paddingB > 1 || paddingL > 1 || paddingR > 1) {
                *bytes += oc * 4;
            }
            break;
        }
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            *bytes = 0;
            break;
        default:
            ret = NOT_MATCH;
            break;
    }

    if (fdt == DT_I8) {
        *bytes += oc * bytesOf(DT_I32);
        if (idt != DT_U8_Q) {
            *bytes += icPadding * ih_pad * iw_pad;
        }
    }
    if (odt == DT_U8_Q) {
        // scaled bias + results before quantization
        *bytes += on * oc * oh * ow * bytesOf(DT_I32);
    }

    // pre data processing space for not complete NCHWC8 group convolution input
    U32 icGroupSize = ic / convParamSpec.group;
    if (idf == DF_NCHWC8 && icGroupSize % 8 != 0) {
        *bytes += tensorNumBytes(inputDesc);
    }

    *bytes += 32;
    return ret;
}

EE convolution_int8(TensorDesc inputDesc,
    UINT8 *input,
    F32 *eltwiseInput,
    TensorDesc filterDesc,
    const INT8 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const F32 *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    F32 *scale,
    ActivationParamSpec activationDesc,
    Arch arch)
{
    if (nullptr == input || nullptr == filter || nullptr == output || nullptr == bias ||
        nullptr == tmp) {
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

    if (!(odf == DF_NCHWC16 || odf == DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }
    // if (!(ic == fc && oc == fn)) {
    //     CHECK_STATUS(NOT_MATCH);
    // }

    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT: {
            ret = convolution_direct(inputDesc, input, eltwiseInput, filterDesc, filter, convParamSpec, biasDesc,
                bias, tmpBytes, tmp, outputDesc, output, scale, activationDesc);
            break;
        }
        case CONVOLUTION_ALGORITHM_POINTWISE:
            ret = convolution_1x1_direct(inputDesc, input, eltwiseInput, filterDesc, filter, convParamSpec,
                biasDesc, bias, tmpBytes, tmp, outputDesc, output, scale, activationDesc);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
