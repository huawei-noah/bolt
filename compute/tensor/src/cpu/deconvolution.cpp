// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_OPENMP
#include <omp.h>
#endif
#include "thread_affinity.h"
#include "cpu/tensor_computing_cpu.h"
#include "cpu/cpu_functions.h"
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#include "blas_enhance.h"
#include "tensor_transpose.h"

#if defined(_USE_X86) || defined(_USE_NEON)

EE deconvolution_infer_forward_algorithm_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    Arch arch)
{
    if (nullptr == algorithm) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));

    if (1 == fn && ic != fn) {
        *algorithm = CONVOLUTION_ALGORITHM_GROUP_DECONV;
        return SUCCESS;
    }

    *algorithm = CONVOLUTION_ALGORITHM_IM2COL_GEMM;
    return SUCCESS;
}

EE deconvolution_transform_filter_bytes_cpu(TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        *bytes = tensorNumBytes(filterDesc) + 32;
        ret = SUCCESS;
    } else if (algorithm == CONVOLUTION_ALGORITHM_GROUP_DECONV) {
        ret = depthwise_convolution_transform_filter_bytes_cpu(
            filterDesc, DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT, bytes);
    }
    return ret;
}

static EE deconvolution_transform_filter_gemm_cpu(
    TensorDesc filterDesc, const void *filter, TensorDesc *ftmDesc, void *filterTransformed, Arch arch)
{
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    TensorDesc mmmDesc = tensor2df(filterDesc.dt, DF_NORMAL, fn, fc * fh * fw);
    matrix_matrix_multiply_transform_rhs(mmmDesc, filter, ftmDesc, filterTransformed, arch);
    return SUCCESS;
}

EE deconvolution_transform_filter_cpu(TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed,
    Arch arch)
{
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        return deconvolution_transform_filter_gemm_cpu(
            filterDesc, filter, ftmDesc, filterTransformed, arch);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret = deconvolution_transform_filter_arm(
            filterDesc, filter, algorithm, ftmDesc, filterTransformed);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = deconvolution_transform_filter_x86(
            filterDesc, filter, algorithm, ftmDesc, filterTransformed);
#endif
    }
    return ret;
}

EE deconvolution_infer_forward_tmp_bytes_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    Arch arch)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw, fn, fc, fh, fw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        fh = convParamSpec.kernel_h;
        fw = convParamSpec.kernel_w;
        TensorDesc matrixADesc = tensor2df(idt, DF_NKN8, ic, in * ih * iw);
        TensorDesc matrixBDesc = tensor2df(idt, DF_NORMAL, ic, oc * fh * fw);
        CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(matrixADesc, matrixBDesc, bytes, X86_AVX2));
        *bytes += in * ih * iw * oc * fh * fw * bytesOf(idt);
#ifdef _USE_NEON
        if (IS_ARM(arch) && idf == DF_NCHWC8) {
            *bytes += in * ih * iw * ic * bytesOf(idt);
        }
        *bytes += 32;
#endif
        return SUCCESS;
    }

    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;

    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    U32 tPadding = fh - 1 - paddingT;
    U32 bPadding = fh - 1 - paddingB;
    U32 lPadding = fw - 1 - paddingL;
    U32 rPadding = fw - 1 - paddingR;

    ih = ih + (ih - 1) * (strideH - 1) + tPadding + bPadding;
    iw = iw + (iw - 1) * (strideW - 1) + lPadding + rPadding;
    TensorDesc inPaddedDesc = tensor4df(idt, idf, in, ic, ih, iw);
    *bytes = tensorNumBytes(inPaddedDesc) * 2 + 32;
    return SUCCESS;
}

EE deconvolution_gemm(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc,
    Arch arch)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    CHECK_REQUIREMENT(idf == DF_NCHWC8 || idf == DF_NCHW);

    TensorDesc matrixADesc = tensor2df(idt, DF_TRANSPOSE, ic, in * ih * iw);
    if (idf == DF_NCHWC8) {
        if (IS_X86_AVX2(arch)) {
            matrixADesc = tensor2df(idt, DF_NKN8, ic, in * ih * iw);
        } else {
            TensorDesc tmpDesc = tensor4df(odt, DF_NCHW, in, ic, ih, iw);
            U8 *tmpInput = (U8 *)tmp;
            transformToNCHW(inputDesc, input, tmpDesc, tmpInput);
            input = tmpInput;
            tmp = (void *)(tmpInput + in * ic * iw * ih * bytesOf(idt));
        }
    }
    TensorDesc matrixCDesc = tensor2df(odt, DF_NORMAL, in * ih * iw, fw * fh * oc);
    U8 *tmpOutput = (U8 *)tmp;
    tmpOutput += in * ih * iw * ic * bytesOf(idt);

    memset(tmpOutput, 0, in * ih * iw * fw * fh * oc * bytesOf(idt));
    CHECK_STATUS(matrix_matrix_multiply(
        matrixADesc, input, filterDesc, filter, tmpBytes, tmp, matrixCDesc, tmpOutput, arch));

    U8 *tmpOutputPtr = (U8 *)output;
    U32 biasTileSize = bytesOf(biasDesc.dt) * 8;
    U8 *biasPtr = (U8 *)bias;
    for (U32 c = 0; c < oc / 8; c++, biasPtr += biasTileSize) {
        for (U32 n = 0; n < oh * ow; n++) {
            memcpy(tmpOutputPtr, biasPtr, biasTileSize);
            tmpOutputPtr += biasTileSize;
        }
    }

    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret =
            deconvolution_overlap_crop_arm(tmpOutput, output, inputDesc, outputDesc, convParamSpec);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret =
            deconvolution_overlap_crop_x86(tmpOutput, output, inputDesc, outputDesc, convParamSpec);
#endif
    }

    if (activationDesc.mode != ACTIVATION_NULL) {
        ArrayActivationFunction activation_func = get_array_activation_function(arch);
        CHECK_STATUS(
            activation_func(odt, output, tensorNumElements(outputDesc), activationDesc, output));
    }
    return ret;
}

EE deconvolution_cpu(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc scaleDesc,
    const void *scale,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc,
    Arch arch)
{
    UNUSED(scaleDesc);
    UNUSED(scale);

    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        return deconvolution_gemm(inputDesc, input, filterDesc, filter, convParamSpec, biasDesc,
            bias, tmpBytes, tmp, outputDesc, output, activationDesc, arch);
    }

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

    if (!(idf == DF_NCHWC8 && odf == DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;

    ConvolutionParamSpec transposedCD = convParamSpec;
    transposedCD.stride_h = 1;
    transposedCD.stride_w = 1;
    transposedCD.padding_top = 0;
    transposedCD.padding_bottom = 0;
    transposedCD.padding_left = 0;
    transposedCD.padding_right = 0;
    transposedCD.dilatedRate_h = 1;
    transposedCD.dilatedRate_w = 1;

    U32 tPadding = fh - 1 - paddingT;
    U32 bPadding = fh - 1 - paddingB;
    U32 lPadding = fw - 1 - paddingL;
    U32 rPadding = fw - 1 - paddingR;

    U32 stuffH = strideH - 1;
    U32 stuffW = strideW - 1;
    U32 ihPadded = ih + (ih - 1) * stuffH + tPadding + bPadding;
    U32 iwPadded = iw + (iw - 1) * stuffW + lPadding + rPadding;
    TensorDesc inPaddedDesc = tensor4df(idt, idf, in, ic, ihPadded, iwPadded);

    U8 *inPad = (U8 *)tmp;
    U8 *inPadMov = inPad;
    U8 *inputMov = (U8 *)input;
    U32 memUnit = 8 * bytesOf(idt);

    ic /= 8;

    for (U32 c = 0; c < ic; c++) {
        for (U32 h = 0; h < tPadding; h++) {
            memset(inPadMov, 0, iwPadded * memUnit);
            inPadMov += iwPadded * memUnit;
        }
        for (U32 h = 0; h < ih - 1; h++) {
            memset(inPadMov, 0, lPadding * memUnit);
            inPadMov += lPadding * memUnit;
            for (U32 w = 0; w < iw - 1; w++) {
                memcpy(inPadMov, inputMov, memUnit);
                inPadMov += memUnit;
                inputMov += memUnit;
                memset(inPadMov, 0, stuffW * memUnit);
                inPadMov += stuffW * memUnit;
            }
            memcpy(inPadMov, inputMov, memUnit);
            inPadMov += memUnit;
            inputMov += memUnit;
            memset(inPadMov, 0, rPadding * memUnit);
            inPadMov += rPadding * memUnit;

            // stuffH
            memset(inPadMov, 0, iwPadded * stuffH * memUnit);
            inPadMov += iwPadded * stuffH * memUnit;
        }
        memset(inPadMov, 0, lPadding * memUnit);
        inPadMov += lPadding * memUnit;
        for (U32 w = 0; w < iw - 1; w++) {
            memcpy(inPadMov, inputMov, memUnit);
            inPadMov += memUnit;
            inputMov += memUnit;
            memset(inPadMov, 0, stuffW * memUnit);
            inPadMov += stuffW * memUnit;
        }
        memcpy(inPadMov, inputMov, memUnit);
        inPadMov += memUnit;
        inputMov += memUnit;
        memset(inPadMov, 0, rPadding * memUnit);
        inPadMov += rPadding * memUnit;

        for (U32 h = ihPadded - bPadding; h < ihPadded; h++) {
            memset(inPadMov, 0, iwPadded * memUnit);
            inPadMov += iwPadded * memUnit;
        }
    }

    EE ret = NOT_SUPPORTED;
    TensorDesc blankTensorDesc;
    ActivationParamSpec blankActivationParamSpec;
    ret = depthwise_pointwise_convolution_cpu(inPaddedDesc, inPad, filterDesc, filter,
        blankTensorDesc, nullptr, transposedCD, DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT,
        biasDesc, bias, blankTensorDesc, nullptr, tmpBytes - tensorNumBytes(inPaddedDesc),
        inPad + tensorNumBytes(inPaddedDesc), outputDesc, output, activationDesc,
        blankActivationParamSpec, arch);

    return ret;
}

#endif
