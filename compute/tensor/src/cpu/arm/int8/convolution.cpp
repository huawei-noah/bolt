// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_INT8

#include "cpu/arm/int8/tensor_computing_int8.h"
#include "cpu/arm/int8/convolution_winograd.h"
#include "cpu/arm/int8/convolution_gemm.h"
#include "tensor_transpose.h"

EE convolution_int8(TensorDesc inputDesc,
    const INT8 *input,
    TensorDesc filterDesc,
    const INT8 *filter,
    F16 *scales,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const F16 *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
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

    if (idt != DT_I8 && idt != DT_F16) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (fdt != DT_I8) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (odt != DT_F16 && odt != DT_I8) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (odf != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (!(ic == fc && oc == fn)) {
        CHECK_STATUS(NOT_MATCH);
    }

    const INT8 *inputPtr = input;
    INT8 *tmpPtr = (INT8 *)tmp;
    if (idf == DF_NCHW) {
        TensorDesc prevDesc = inputDesc;
        inputDesc.df = DF_NCHWC8;
        CHECK_STATUS(transformNCHWToNCHWC8(prevDesc, input, inputDesc, tmpPtr));
        inputPtr = tmpPtr;
        tmpPtr += tensorNumBytes(inputDesc);
        tmpBytes -= tensorNumBytes(inputDesc);
    }

    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ret = convolution_winograd(inputDesc, inputPtr, scales, filterDesc, filter, scales + 2,
                convParamSpec, biasDesc, bias, tmpBytes, tmpPtr, outputDesc, output, scales + 1,
                activationDesc, arch);
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ret = convolution_gemm(inputDesc, inputPtr, scales, filterDesc, filter, scales + 2,
                convParamSpec, biasDesc, bias, tmpBytes, tmpPtr, outputDesc, output, scales + 1,
                activationDesc, arch);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
#endif
