// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

#ifdef _USE_X86
    if (IS_X86(arch) && idf == DF_NCHWC8) {
        *algorithm = CONVOLUTION_ALGORITHM_POINTWISE;
        return SUCCESS;
    }
#endif

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
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM ||
        algorithm == CONVOLUTION_ALGORITHM_POINTWISE) {
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
    } else if (IS_X86(arch)) {
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
    fh = convParamSpec.kernel_h;
    fw = convParamSpec.kernel_w;

    if (algorithm == CONVOLUTION_ALGORITHM_POINTWISE) {
        *bytes = (in * ih * iw + 1) * oc * fh * fw * bytesOf(idt) + 32;
        return SUCCESS;
    }
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        TensorDesc matrixADesc = tensor2df(idt, DF_NKN8, ic, in * ih * iw);
        TensorDesc matrixBDesc = tensor2df(idt, DF_NORMAL, ic, oc * fh * fw);
        CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(matrixADesc, matrixBDesc, bytes, arch));
        *bytes += in * ih * iw * oc * fh * fw * bytesOf(idt);
        if (!IS_X86(arch) || idf != DF_NCHWC8 || in > 1) {
            *bytes += in * ih * iw * ic * bytesOf(idt);
        }
        *bytes += 32;
        return SUCCESS;
    }

    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;

    U32 tPadding = fh - 1 - paddingT;
    U32 bPadding = fh - 1 - paddingB;
    U32 lPadding = fw - 1 - paddingL;
    U32 rPadding = fw - 1 - paddingR;

    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
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

    TensorDesc matrixADesc = tensor2df(idt, DF_NORMAL, in * ih * iw, ic);
    if (IS_X86(arch) && idf == DF_NCHWC8 && in == 1) {
        matrixADesc = tensor2df(idt, DF_NKN8, ic, in * ih * iw);
    } else {
        TensorDesc tmpDesc = tensor4df(odt, DF_NHWC, in, ic, ih, iw);
        U8 *tmpInput = (U8 *)tmp;
        transformFormat(inputDesc, input, tmpDesc, tmpInput);
        input = tmpInput;
        tmp = (void *)(tmpInput + in * ic * iw * ih * bytesOf(idt));
    }
    TensorDesc matrixCDesc = tensor2df(odt, DF_NORMAL, in * ih * iw, fw * fh * oc);
    U8 *tmpOutput = (U8 *)tmp;
    tmp = (void *)(tmpOutput + in * ih * iw * fw * fh * oc * bytesOf(idt));

    UNI_MEMSET(tmpOutput, 0, in * ih * iw * fw * fh * oc * bytesOf(idt));
    CHECK_STATUS(matrix_matrix_multiply(matrixADesc, input, filterDesc, filter, tmpBytes, tmp,
        matrixCDesc, tmpOutput, nullptr, arch));

    U8 *tmpOutputPtr = (U8 *)output;
    U32 biasTileSize = bytesOf(biasDesc.dt) * 8;
    for (U32 n = 0; n < on; ++n) {
        U8 *biasPtr = (U8 *)bias;
        for (U32 c = 0; c < oc / 8; c++, biasPtr += biasTileSize) {
            for (U32 hw = 0; hw < oh * ow; hw++) {
                UNI_MEMCPY(tmpOutputPtr, biasPtr, biasTileSize);
                tmpOutputPtr += biasTileSize;
            }
        }
    }

    EE ret = SUCCESS;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret =
            deconvolution_overlap_crop_arm(tmpOutput, output, inputDesc, outputDesc, convParamSpec);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
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

#ifdef _USE_X86
    if (IS_X86(arch) && algorithm == CONVOLUTION_ALGORITHM_POINTWISE) {
        return deconvolution_pointwise_x86(inputDesc, input, filterDesc, filter, convParamSpec,
            biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc, arch);
    }
#endif

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
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;

    ConvolutionParamSpec transposedCD = convParamSpec;
    transposedCD.stride_h = 1;
    transposedCD.stride_w = 1;
    transposedCD.pad_top = 0;
    transposedCD.pad_bottom = 0;
    transposedCD.pad_left = 0;
    transposedCD.pad_right = 0;
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
    TensorDesc inPaddedDesc = tensor4df(idt, idf, 1, ic, ihPadded, iwPadded);
    TensorDesc singleOutputDesc = tensor4df(idt, idf, 1, oc, oh, ow);

    U32 memUnit = 8 * bytesOf(idt);
    U32 ic8 = ic / 8;
    EE ret = NOT_SUPPORTED;
    TensorDesc blankTensorDesc;
    ActivationParamSpec blankActivationParamSpec;

    for (U32 n = 0; n < in; ++n) {
        U8 *inputMov = (U8 *)input + n * ih * iw * ic * bytesOf(idt);
        U8 *outputMov = (U8 *)output + n * oh * ow * oc * bytesOf(odt);
        U8 *inPad = (U8 *)tmp;
        U8 *inPadMov = inPad;

        for (U32 c = 0; c < ic8; c++) {
            for (U32 h = 0; h < tPadding; h++) {
                UNI_MEMSET(inPadMov, 0, iwPadded * memUnit);
                inPadMov += iwPadded * memUnit;
            }
            for (U32 h = 0; h < ih - 1; h++) {
                UNI_MEMSET(inPadMov, 0, lPadding * memUnit);
                inPadMov += lPadding * memUnit;
                for (U32 w = 0; w < iw - 1; w++) {
                    UNI_MEMCPY(inPadMov, inputMov, memUnit);
                    inPadMov += memUnit;
                    inputMov += memUnit;
                    UNI_MEMSET(inPadMov, 0, stuffW * memUnit);
                    inPadMov += stuffW * memUnit;
                }
                UNI_MEMCPY(inPadMov, inputMov, memUnit);
                inPadMov += memUnit;
                inputMov += memUnit;
                UNI_MEMSET(inPadMov, 0, rPadding * memUnit);
                inPadMov += rPadding * memUnit;

                // stuffH
                UNI_MEMSET(inPadMov, 0, iwPadded * stuffH * memUnit);
                inPadMov += iwPadded * stuffH * memUnit;
            }
            UNI_MEMSET(inPadMov, 0, lPadding * memUnit);
            inPadMov += lPadding * memUnit;
            for (U32 w = 0; w < iw - 1; w++) {
                UNI_MEMCPY(inPadMov, inputMov, memUnit);
                inPadMov += memUnit;
                inputMov += memUnit;
                UNI_MEMSET(inPadMov, 0, stuffW * memUnit);
                inPadMov += stuffW * memUnit;
            }
            UNI_MEMCPY(inPadMov, inputMov, memUnit);
            inPadMov += memUnit;
            inputMov += memUnit;
            UNI_MEMSET(inPadMov, 0, rPadding * memUnit);
            inPadMov += rPadding * memUnit;

            for (U32 h = ihPadded - bPadding; h < ihPadded; h++) {
                UNI_MEMSET(inPadMov, 0, iwPadded * memUnit);
                inPadMov += iwPadded * memUnit;
            }
        }

        ret = depthwise_pointwise_convolution_cpu(inPaddedDesc, inPad, filterDesc, filter,
            blankTensorDesc, nullptr, transposedCD,
            DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT, biasDesc, bias, blankTensorDesc,
            nullptr, tmpBytes - tensorNumBytes(inPaddedDesc), inPad + tensorNumBytes(inPaddedDesc),
            singleOutputDesc, outputMov, activationDesc, blankActivationParamSpec, arch);
    }

    return ret;
}

#endif
