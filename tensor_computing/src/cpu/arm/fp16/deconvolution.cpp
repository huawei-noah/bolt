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
#include "cpu/arm/fp16/convolution_winograd.h"
#include "cpu/arm/fp16/convolution_gemm.h"
#include <cstring>

EE deconvolution_infer_forward_tmp_bytes_fp16(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));

    U32 strideH = convDesc.stride_h;
    U32 strideW = convDesc.stride_w;
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;

    if (fh < paddingT + 2 || fh < paddingB + 2 || fw < paddingL + 2 || fw < paddingR + 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 tPadding = fh - 1 - paddingT;
    U32 bPadding = fh - 1 - paddingB;
    U32 lPadding = fw - 1 - paddingL;
    U32 rPadding = fw - 1 - paddingR;

    ConvolutionDesc transposedCD;
    transposedCD.stride_h = 1;
    transposedCD.stride_w = 1;
    transposedCD.padding_top = 0;
    transposedCD.padding_bottom = 0;
    transposedCD.padding_left = 0;
    transposedCD.padding_right = 0;
    transposedCD.dilatedRate_h = 1;
    transposedCD.dilatedRate_w = 1;

    if (CONVOLUTION_ALGORITHM_WINOGRAD == algorithm) {
        // If algorithm is not Winograd, leave out padding of length 1
        tPadding--;
        bPadding--;
        lPadding--;
        rPadding--;
        transposedCD.padding_top += 1;
        transposedCD.padding_bottom += 1;
        transposedCD.padding_left += 1;
        transposedCD.padding_right += 1;
    }

    ih = ih + (ih - 1) * (strideH - 1) + tPadding + bPadding;
    iw = iw + (iw - 1) * (strideW - 1) + lPadding + rPadding;

    TensorDesc inPaddedDesc = tensor4df(idt, idf, in, ic, ih, iw);

    EE ret = convolution_infer_forward_tmp_bytes_fp16(inPaddedDesc, filterDesc, outputDesc, transposedCD, algorithm, bytes);
    *bytes += tensorNumBytes(inPaddedDesc);  // for pre-convolution padding
    return ret;
}

EE deconvolution_fp16(TensorDesc inputDesc, F16* input,
    TensorDesc filterDesc, const F16* filter,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const F16* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F16* output,
    ActivationMode activationMode,
    Arch arch)
{
    if (nullptr == input || nullptr == filter || nullptr == output || nullptr == bias || nullptr == tmp) {
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

    if (!(idt == DT_F16 && fdt == DT_F16 && odt == DT_F16)){ 
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(idf == DF_NCHWC8 && odf == DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 strideH = convDesc.stride_h;
    U32 strideW = convDesc.stride_w;
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;

    if (fh < paddingT + 2 || fh < paddingB + 2 || fw < paddingL + 2 || fw < paddingR + 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 tPadding = fh - 1 - paddingT;
    U32 bPadding = fh - 1 - paddingB;
    U32 lPadding = fw - 1 - paddingL;
    U32 rPadding = fw - 1 - paddingR;

    ConvolutionDesc transposedCD;
    transposedCD.stride_h = 1;
    transposedCD.stride_w = 1;
    transposedCD.padding_top = 0;
    transposedCD.padding_bottom = 0;
    transposedCD.padding_left = 0;
    transposedCD.padding_right = 0;
    transposedCD.dilatedRate_h = 1;
    transposedCD.dilatedRate_w = 1;

    if (CONVOLUTION_ALGORITHM_WINOGRAD == algorithm) {
        // If algorithm is not Winograd, leave out padding of length 1
        tPadding--;
        bPadding--;
        lPadding--;
        rPadding--;
        transposedCD.padding_top += 1;
        transposedCD.padding_bottom += 1;
        transposedCD.padding_left += 1;
        transposedCD.padding_right += 1;
    }

    U32 stuffH = strideH - 1;
    U32 stuffW = strideW - 1;
    U32 ihPadded = ih + (ih - 1) * stuffH + tPadding + bPadding;
    U32 iwPadded = iw + (iw - 1) * stuffW + lPadding + rPadding;
    TensorDesc inPaddedDesc = tensor4df(idt, idf, in, ic, ihPadded, iwPadded);

    F16 *inPad = (F16*)tmp;
    F16 *inPadMov = inPad;
    F16 *inputMov = input;

    ic /= 8;

    for (U32 c = 0; c < ic; c++) {
        for (U32 h = 0; h < tPadding; h++) {
            memset(inPadMov, 0, iwPadded*8*bytesOf(idt));
            inPadMov += iwPadded*8;
        }
        for (U32 h = 0; h < ih - 1; h++) {
            memset(inPadMov, 0, lPadding*8*bytesOf(idt));
            inPadMov += lPadding*8;
            for (U32 w = 0; w < iw - 1; w++) {
                memcpy(inPadMov, inputMov, 8*bytesOf(idt));
                inPadMov += 8;
                inputMov += 8;
                memset(inPadMov, 0, stuffW*8*bytesOf(idt));
                inPadMov += stuffW * 8;
            }
            memcpy(inPadMov, inputMov, 8*bytesOf(idt));
            inPadMov += 8;
            inputMov += 8;
            memset(inPadMov, 0, rPadding*8*bytesOf(idt));
            inPadMov += rPadding*8;

            // stuffH
            memset(inPadMov, 0, iwPadded*stuffH*8*bytesOf(idt));
            inPadMov += iwPadded*stuffH*8;
        }
        memset(inPadMov, 0, lPadding*8*bytesOf(idt));
        inPadMov += lPadding*8;
        for (U32 w = 0; w < iw - 1; w++) {
            memcpy(inPadMov, inputMov, 8*bytesOf(idt));
            inPadMov += 8;
            inputMov += 8;
            memset(inPadMov, 0, stuffW*8*bytesOf(idt));
            inPadMov += stuffW * 8;
        }
        memcpy(inPadMov, inputMov, 8*bytesOf(idt));
        inPadMov += 8;
        inputMov += 8;
        memset(inPadMov, 0, rPadding*8*bytesOf(idt));
        inPadMov += rPadding*8;

        for (U32 h = ihPadded - bPadding; h < ihPadded; h++) {
            memset(inPadMov, 0, iwPadded*8*bytesOf(idt));
            inPadMov += iwPadded*8;
        }
    }

    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_GEMM:
            ret = convolution_gemm(inPaddedDesc, inPad, filterDesc, filter, transposedCD,
                                   biasDesc, bias, tmpBytes - tensorNumBytes(inPaddedDesc),
                                   inPad + tensorNumElements(inPaddedDesc), outputDesc, output, activationMode, arch);
            break;
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ret = convolution_winograd(inPaddedDesc, inPad, filterDesc, filter, transposedCD,
                                   biasDesc, bias, tmpBytes - tensorNumBytes(inPaddedDesc),
                                   inPad + tensorNumElements(inPaddedDesc), outputDesc, output, activationMode, arch);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
