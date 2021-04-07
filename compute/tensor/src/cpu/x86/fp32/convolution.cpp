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
#include "error.h"

#include "cpu/x86/fp32/tensor_computing_fp32.h"

EE convolution_infer_forward_tmp_bytes_fp32(TensorDesc inputDesc,
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
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;

    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;

    U32 icAlignSize = 8;
    U32 icPadding = (ic + icAlignSize - 1) / icAlignSize * icAlignSize;

    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = icPadding * ih_pad * iw_pad;
            break;
        case CONVOLUTION_ALGORITHM_POINTWISE:
            *bytes = oc;
            break;
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            *bytes = 0;
            break;
        default:
            ret = NOT_MATCH;
            break;
    }

    // pre data processing space for not complete NCHWC8 group convolution input
    U32 icGroupSize = ic / convParamSpec.group;
    if (idf == DF_NCHWC8 && icGroupSize % 8 != 0) {
        *bytes += tensorNumBytes(inputDesc);
    }

    *bytes *= bytesOf(idt);
    *bytes += 32;
    return ret;
}

EE convolution_fp32(TensorDesc inputDesc,
    F32 *input,
    F32 *eltwiseInput,
    TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const F32 *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *output,
    ActivationParamSpec activationDesc,
    Arch arch)
{
    UNUSED(arch);
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

    if (!(idt == DT_F32 && fdt == DT_F32 && odt == DT_F32)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(odf == DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(ic == fc && oc == fn)) {
        CHECK_STATUS(NOT_MATCH);
    }

    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            ret = convolution_direct(inputDesc, input, eltwiseInput, filterDesc, filter,
                convParamSpec, biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc);
            break;
        case CONVOLUTION_ALGORITHM_POINTWISE:
            ret = convolution_1x1_direct(inputDesc, input, eltwiseInput, filterDesc, filter,
                convParamSpec, bias, tmpBytes, tmp, outputDesc, output, activationDesc);
            break;
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            ret = convolution_direct_nchw(inputDesc, input, filterDesc, filter, convParamSpec,
                biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
