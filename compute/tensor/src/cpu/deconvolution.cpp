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

    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;
    if ((strideH > 1 || strideW > 1) && fh % strideH == 0 && fw % strideW == 0) {
        *algorithm = CONVOLUTION_ALGORITHM_IM2COL_GEMM;
        return SUCCESS;
    }

    ConvolutionParamSpec transposedCD = convParamSpec;
    transposedCD.stride_h = 1;
    transposedCD.stride_w = 1;
    transposedCD.padding_top = 1;
    transposedCD.padding_bottom = 1;
    transposedCD.padding_left = 1;
    transposedCD.padding_right = 1;
    transposedCD.dilatedRate_h = 1;
    transposedCD.dilatedRate_w = 1;

    U32 tPadding = (fh - 1 - paddingT) - 1;  // Leave out padding of length 1 to activate Winograd
    U32 bPadding = (fh - 1 - paddingB) - 1;
    U32 lPadding = (fw - 1 - paddingL) - 1;
    U32 rPadding = (fw - 1 - paddingR) - 1;

    ih = ih + (ih - 1) * (strideH - 1) + tPadding + bPadding;
    iw = iw + (iw - 1) * (strideW - 1) + lPadding + rPadding;

    TensorDesc inPaddedDesc = tensor4df(idt, idf, in, ic, ih, iw);

    // Swap fn and fc
    filterDesc.dims[2] = filterDesc.dims[3];
    filterDesc.dims[3] = ic;
    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret = convolution_infer_forward_algorithm_arm(
            inPaddedDesc, filterDesc, outputDesc, transposedCD, policy, algorithm, targetDataType);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = convolution_infer_forward_algorithm_x86(
            inPaddedDesc, filterDesc, outputDesc, transposedCD, policy, algorithm, targetDataType);
#endif
    }
    return ret;
}

EE deconvolution_transform_filter_bytes_cpu(TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (algorithm == CONVOLUTION_ALGORITHM_IM2COL_GEMM) {
        *bytes = tensorNumBytes(filterDesc);
        ret = SUCCESS;
    } else if (algorithm == CONVOLUTION_ALGORITHM_GROUP_DECONV) {
        ret = depthwise_convolution_transform_filter_bytes_cpu(
            filterDesc, DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT, bytes);
    } else {
        if (IS_ARM(arch)) {
#ifdef _USE_NEON
            ret =
                convolution_transform_filter_bytes_arm(filterDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_X86
        } else if (IS_X86_AVX2(arch)) {
            ret =
                convolution_transform_filter_bytes_x86(filterDesc, convParamSpec, algorithm, bytes);
#endif
        }
    }
    return ret;
}

static EE deconvolution_transform_filter_im2col_gemm_cpu(TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *ftmDesc,
    void *filterTransformed)
{
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (convParamSpec.stride_h == convParamSpec.kernel_h &&
        convParamSpec.stride_w == convParamSpec.kernel_w) {
        U32 filterDims[5] = {fw, fh, 8, fc / 8, fn};
        U32 ftmDims[5] = {8, fw, fh, fc / 8, fn};
        U32 filterTransformDims[5] = {0, 1, 3, 4, 2};
        CHECK_STATUS(array_transpose(
            fdt, filterDims, filter, ftmDims, filterTransformed, filterTransformDims, 5));
    } else {
        U32 elementSize = bytesOf(filterDesc.dt);
        U32 fnAlignSize = fn / 8;
        U8 *ptr = (U8 *)filterTransformed;
        for (U32 i = 0; i < convParamSpec.stride_h; i++) {
            for (U32 j = 0; j < convParamSpec.stride_w; j++) {
                U32 fhStart = (fh - 1 - i - convParamSpec.padding_top) % convParamSpec.stride_h;
                U32 fwStart = (fw - 1 - j - convParamSpec.padding_left) % convParamSpec.stride_w;
                for (U32 ic = 0; ic < fnAlignSize; ic++) {
                    for (U32 h = fhStart; h < convParamSpec.kernel_h; h += convParamSpec.stride_h) {
                        for (U32 w = fwStart; w < convParamSpec.kernel_w;
                             w += convParamSpec.stride_w) {
                            for (U32 c8 = 0; c8 < 8; c8++) {
                                for (U32 oc = 0; oc < fc; oc++, ptr += elementSize) {
                                    U32 srcIndex =
                                        ((((ic * 8 + c8) * fc + oc) * fh + (fh - 1 - h)) * fw +
                                            (fw - 1 - w)) *
                                        elementSize;
                                    const U8 *src = (const U8 *)filter + srcIndex;
                                    memcpy(ptr, src, elementSize);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    *ftmDesc = tensor2df(filterDesc.dt, DF_NORMAL, fn, fc * fh * fw);
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
        return deconvolution_transform_filter_im2col_gemm_cpu(
            filterDesc, filter, convParamSpec, ftmDesc, filterTransformed);
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
        U32 ihNum = ih + convParamSpec.kernel_h * 2 - convParamSpec.padding_top -
            convParamSpec.padding_bottom;
        U32 iwNum = iw + convParamSpec.kernel_w * 2 - convParamSpec.padding_left -
            convParamSpec.padding_right;
        U32 fhNum = (U32)ceil((float)convParamSpec.kernel_h / convParamSpec.stride_h);
        U32 fwNum = (U32)ceil((float)convParamSpec.kernel_w / convParamSpec.stride_w);
        TensorDesc matrixADesc = tensor2df(idt, DF_NORMAL, in * ihNum * iwNum, ic * fhNum * fwNum);
        TensorDesc matrixBDesc = tensor2df(filterDesc.dt, DF_NORMAL, ic * fhNum * fwNum,
            oc * convParamSpec.stride_h * convParamSpec.stride_w);
        CHECK_STATUS(matrix_matrix_multiply_tmp_bytes(matrixADesc, matrixBDesc, bytes, arch));
        *bytes *= OMP_NUM_THREADS;
        *bytes += tensorNumBytes(matrixADesc) + tensorNumBytes(outputDesc);
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

    ConvolutionParamSpec transposedCD = convParamSpec;
    transposedCD.stride_h = 1;
    transposedCD.stride_w = 1;
    transposedCD.padding_top = 0;
    transposedCD.padding_bottom = 0;
    transposedCD.padding_left = 0;
    transposedCD.padding_right = 0;
    transposedCD.dilatedRate_h = 1;
    transposedCD.dilatedRate_w = 1;

    ih = ih + (ih - 1) * (strideH - 1) + tPadding + bPadding;
    iw = iw + (iw - 1) * (strideW - 1) + lPadding + rPadding;
    TensorDesc inPaddedDesc = tensor4df(idt, idf, in, ic, ih, iw);
    if (CONVOLUTION_ALGORITHM_GROUP_DECONV == algorithm) {
        *bytes = tensorNumBytes(inPaddedDesc) * 2 + 32;
        return SUCCESS;
    }
    if (DF_NCHW == filterDesc.df) {
        // Swap fn and fc
        filterDesc.dims[2] = filterDesc.dims[3];
        filterDesc.dims[3] = ic;
    }
    U32 convolution_tmp_bytes = 0;
    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret = convolution_infer_forward_tmp_bytes_arm(
            inPaddedDesc, filterDesc, outputDesc, transposedCD, algorithm, &convolution_tmp_bytes);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = convolution_infer_forward_tmp_bytes_x86(
            inPaddedDesc, filterDesc, outputDesc, transposedCD, algorithm, &convolution_tmp_bytes);
#endif
    }
    *bytes = tensorNumBytes(inPaddedDesc) + convolution_tmp_bytes;
    return ret;
}

static EE deconvolution_stride_greater_one_and_kernel_divide_stride_cpu(TensorDesc inputDesc,
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
    U8 alignSize = 8;
    U32 icAlignSize = ic / alignSize;
    U32 inputTileSize = bytesOf(idt) * alignSize;
#ifndef _USE_OPENMP
    U32 ocAlignSize = oc / alignSize;
    U32 outputTileSize = bytesOf(odt) * alignSize;
    ArrayAddFunction add_func = get_array_add_function(arch);
    ArrayActivationFunction activation_func = get_array_activation_function(arch);
#endif
    U32 iPaddingT =
        (convParamSpec.kernel_h - 1 - convParamSpec.padding_top) / convParamSpec.stride_h;
    U32 iPaddingB =
        (convParamSpec.kernel_h - 1 - convParamSpec.padding_bottom) / convParamSpec.stride_h;
    U32 iPaddingL =
        (convParamSpec.kernel_w - 1 - convParamSpec.padding_left) / convParamSpec.stride_w;
    U32 iPaddingR =
        (convParamSpec.kernel_w - 1 - convParamSpec.padding_right) / convParamSpec.stride_w;
    U32 iKernelH = convParamSpec.kernel_h / convParamSpec.stride_h;
    U32 iKernelW = convParamSpec.kernel_w / convParamSpec.stride_w;
    U8 *tmpInput = (U8 *)tmp;
    U32 iStrideT = (convParamSpec.kernel_h - 1 - convParamSpec.padding_top) % convParamSpec.stride_h;
    U32 iStrideL =
        (convParamSpec.kernel_w - 1 - convParamSpec.padding_left) % convParamSpec.stride_w;
    U32 iDumpH = 1;
    if (iStrideT == convParamSpec.stride_h - 1) {
        iDumpH = 0;
    }
    U32 iDumpW = 1;
    if (iStrideL == convParamSpec.stride_w - 1) {
        iDumpW = 0;
    }
    U32 ihNum = iPaddingT + ih + iPaddingB;
    U32 iwNum = iPaddingL + iw + iPaddingR;
    U32 mNum = 0;
    for (U32 n = 0; n < in; n++) {
        for (U32 hStart = 0; hStart <= ihNum - iKernelH; hStart++) {
            for (U32 wStart = 0; wStart <= iwNum - iKernelW; wStart++, mNum++) {
                for (U32 c = 0, k = 0; c < icAlignSize; c++) {
                    for (U32 i = 0; i < iKernelH; i++) {
                        for (U32 j = 0; j < iKernelW; j++, tmpInput += inputTileSize, k += 8) {
                            U32 h = hStart + i;
                            U32 w = wStart + j;
                            if (h < iPaddingT || h >= iPaddingT + ih || w < iPaddingL ||
                                w >= iPaddingL + iw) {
                                memset(tmpInput, 0, inputTileSize);
                            } else {
                                U32 srcIndex = (((n * icAlignSize + c) * ih + (h - iPaddingT)) * iw +
                                                   (w - iPaddingL)) *
                                    inputTileSize;
                                memcpy(tmpInput, (const U8 *)input + srcIndex, inputTileSize);
                            }
                        }
                    }
                }
            }
        }
    }
    U32 kNum = ic * iKernelH * iKernelW;
    U32 nNum = oc;
    TensorDesc tmpInputDesc = tensor2df(idt, DF_NORMAL, mNum, kNum);
    TensorDesc tmpFilterDesc = tensor2df(filterDesc.dt, DF_NORMAL, kNum, nNum);
    TensorDesc tmpOutputDesc = tensor2df(odt, DF_NORMAL, mNum, nNum);
    tmpInput = (U8 *)tmp;
    U32 bufferSize =
        (tmpBytes - tensorNumBytes(tmpInputDesc) - tensorNumBytes(tmpOutputDesc) * OMP_NUM_THREADS) /
        OMP_NUM_THREADS;
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 index = 0; index < convParamSpec.stride_h * convParamSpec.stride_w; index++) {
        U32 i = index / convParamSpec.stride_w;
        U32 j = index % convParamSpec.stride_w;
#ifdef _USE_OPENMP
        // For NDK on ARMv7, OpenMP loop cannot reference more than 14 outside variables
        ArrayAddFunction add_func = get_array_add_function(arch);
        ArrayActivationFunction activation_func = get_array_activation_function(arch);
        U32 ocAlignSize = outputDesc.dims[2] / 8;
        U32 outputTileSize = bytesOf(outputDesc.dt) * 8;
        U32 iPaddingT =
            (convParamSpec.kernel_h - 1 - convParamSpec.padding_top) / convParamSpec.stride_h;
        U32 iPaddingB =
            (convParamSpec.kernel_h - 1 - convParamSpec.padding_bottom) / convParamSpec.stride_h;
        U32 iPaddingL =
            (convParamSpec.kernel_w - 1 - convParamSpec.padding_left) / convParamSpec.stride_w;
        U32 iPaddingR =
            (convParamSpec.kernel_w - 1 - convParamSpec.padding_right) / convParamSpec.stride_w;
        U32 iKernelH = convParamSpec.kernel_h / convParamSpec.stride_h;
        U32 iKernelW = convParamSpec.kernel_w / convParamSpec.stride_w;
        U32 iStrideT =
            (convParamSpec.kernel_h - 1 - convParamSpec.padding_top) % convParamSpec.stride_h;
        U32 iStrideL =
            (convParamSpec.kernel_w - 1 - convParamSpec.padding_left) % convParamSpec.stride_w;
        U32 ihNum = iPaddingT + inputDesc.dims[1] + iPaddingB;
        U32 iwNum = iPaddingL + inputDesc.dims[0] + iPaddingR;
        U32 iDumpH = (iStrideT == convParamSpec.stride_h - 1) ? 0 : 1;
        U32 iDumpW = (iStrideL == convParamSpec.stride_w - 1) ? 0 : 1;
        U32 threadId = omp_get_thread_num();
#else
        U32 threadId = 0;
#endif
        U8 *tmpOutput = (U8 *)tmpInput + tensorNumBytes(tmpInputDesc) +
            (tensorNumBytes(tmpOutputDesc) + bufferSize) * threadId;
        U8 *buffer = (U8 *)tmpOutput + tensorNumBytes(tmpOutputDesc);
        memset(tmpOutput, 0, tensorNumBytes(tmpOutputDesc));
        const U8 *tmpFilter = (const U8 *)filter + tensorNumBytes(tmpFilterDesc) * index;
        CHECK_STATUS(matrix_matrix_multiply(tmpInputDesc, tmpInput, tmpFilterDesc, tmpFilter,
            bufferSize, buffer, tmpOutputDesc, tmpOutput, arch));
        U32 ihStart = 0;
        U32 ihEnd = iPaddingT + inputDesc.dims[1] + iPaddingB - iKernelH - iDumpH;
        U32 iwStart = 0;
        U32 iwEnd = iPaddingL + inputDesc.dims[0] + iPaddingR - iKernelW - iDumpW;
        if (i > iStrideT) {
            ihStart += iDumpH;
            ihEnd += iDumpH;
        }
        if (j > iStrideL) {
            iwStart += iDumpW;
            iwEnd += iDumpW;
        }
        for (U32 n = 0; n < in; n++) {
            for (U32 hStart = ihStart, h = 0; hStart <= ihEnd; hStart++, h++) {
                for (U32 wStart = iwStart, w = 0; wStart <= iwEnd; wStart++, w++) {
                    U32 srcIndex =
                        (((n * (ihNum - iKernelH + 1) + hStart) * (iwNum - iKernelW + 1) + wStart) *
                            ocAlignSize) *
                        outputTileSize;
                    add_func(outputDesc.dt, (U8 *)tmpOutput + srcIndex, bias,
                        (U8 *)tmpOutput + srcIndex, outputDesc.dims[2]);
                    CHECK_STATUS(activation_func(outputDesc.dt, (U8 *)tmpOutput + srcIndex,
                        outputDesc.dims[2], activationDesc, (U8 *)tmpOutput + srcIndex));
                    for (U32 c = 0; c < ocAlignSize; c++) {
                        U32 srcIndex =
                            (((n * (ihNum - iKernelH + 1) + hStart) * (iwNum - iKernelW + 1) +
                                 wStart) *
                                    ocAlignSize +
                                c) *
                            outputTileSize;
                        U32 dstIndex = (((n * ocAlignSize + c) * outputDesc.dims[1] +
                                            h * convParamSpec.stride_h + i) *
                                               outputDesc.dims[0] +
                                           w * convParamSpec.stride_w + j) *
                            outputTileSize;
                        memcpy((U8 *)output + dstIndex, (U8 *)tmpOutput + srcIndex, outputTileSize);
                    }
                }
            }
        }
    }
    return SUCCESS;
}

static EE deconvolution_stride_greater_one_and_kernel_equal_stride_cpu(TensorDesc inputDesc,
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
    ArrayActivationFunction activation_func = get_array_activation_function(arch);
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw, on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 ihNum = ih + convParamSpec.padding_top + convParamSpec.padding_bottom;
    U32 iwNum = iw + convParamSpec.padding_left + convParamSpec.padding_right;
    U32 fh = convParamSpec.kernel_h;
    U32 fw = convParamSpec.kernel_w;
    U32 fhNum = fh / convParamSpec.stride_h;
    U32 fwNum = fw / convParamSpec.stride_w;

    U32 inputBytes = tensorNumBytes(inputDesc);
    U32 tmpInputDescDims[5] = {8, iw, ih, ic / 8, in};
    U32 finalInputDescDims[5] = {8, ic / 8, iw, ih, in};
    U32 inputTransformDims[5] = {0, 2, 3, 1, 4};
    void *tmpInput = tmp;
    tmp = (U8 *)tmp + inputBytes;
    tmpBytes -= inputBytes;
    CHECK_STATUS(array_transpose(
        idt, tmpInputDescDims, input, finalInputDescDims, tmpInput, inputTransformDims, 5));

    TensorDesc matrixADesc = tensor2df(idt, DF_NORMAL, in * ihNum * iwNum, ic * fhNum * fwNum);
    TensorDesc matrixCDesc = tensor2df(odt, DF_NORMAL, in * ihNum * iwNum, oc * fh * fw);
    void *tmpOutput = tmp;
    tmp = (U8 *)tmp + tensorNumBytes(matrixCDesc);
    tmpBytes -= tensorNumBytes(matrixCDesc);
    U32 biasTileSize = bytesOf(biasDesc.dt) * 8;
    U8 *tmpOutputPtr = (U8 *)tmpOutput;
    for (U32 n = 0; n < on * ih * iw; n++) {
        U8 *biasPtr = (U8 *)bias;
        for (U32 c = 0; c < oc / 8; c++, biasPtr += biasTileSize) {
            for (U32 i = 0; i < oh * ow / (ih * iw); i++, tmpOutputPtr += biasTileSize) {
                memcpy(tmpOutputPtr, biasPtr, biasTileSize);
            }
        }
    }
    CHECK_STATUS(matrix_matrix_multiply(
        matrixADesc, tmpInput, filterDesc, filter, tmpBytes, tmp, matrixCDesc, tmpOutput, arch));

    U32 tmpOutputDims[7] = {8, ow / iw, oh / ih, oc / 8, iw, ih, on};
    U32 finalOutputDims[7] = {8, ow / iw, iw, oh / ih, ih, oc / 8, on};
    U32 outputTransformDims[7] = {0, 3, 1, 4, 2, 5, 6};
    CHECK_STATUS(array_transpose(
        odt, tmpOutputDims, tmpOutput, finalOutputDims, output, outputTransformDims, 7));
    CHECK_STATUS(
        activation_func(odt, output, tensorNumElements(outputDesc), activationDesc, output));
    return SUCCESS;
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
        if (convParamSpec.stride_h == convParamSpec.kernel_h &&
            convParamSpec.stride_w == convParamSpec.kernel_w) {
            return deconvolution_stride_greater_one_and_kernel_equal_stride_cpu(inputDesc, input,
                filterDesc, filter, convParamSpec, biasDesc, bias, tmpBytes, tmp, outputDesc,
                output, activationDesc, arch);
        } else {
            return deconvolution_stride_greater_one_and_kernel_divide_stride_cpu(inputDesc, input,
                filterDesc, filter, convParamSpec, biasDesc, bias, tmpBytes, tmp, outputDesc,
                output, activationDesc, arch);
        }
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

    if (CONVOLUTION_ALGORITHM_WINOGRAD == algorithm) {
        fh = 3;
        fw = 3;
    }

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
    if (algorithm == CONVOLUTION_ALGORITHM_GROUP_DECONV) {
        TensorDesc blankTensorDesc;
        ActivationParamSpec blankActivationParamSpec;
        ret = depthwise_pointwise_convolution_cpu(inPaddedDesc, inPad, filterDesc, filter,
            blankTensorDesc, nullptr, transposedCD,
            DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT, biasDesc, bias, blankTensorDesc,
            nullptr, tmpBytes - tensorNumBytes(inPaddedDesc), inPad + tensorNumBytes(inPaddedDesc),
            outputDesc, output, activationDesc, blankActivationParamSpec, arch);
    } else {
        ret = convolution_cpu(inPaddedDesc, inPad, filterDesc, filter, transposedCD, algorithm,
            scaleDesc, scale, biasDesc, bias, tmpBytes - tensorNumBytes(inPaddedDesc),
            inPad + tensorNumBytes(inPaddedDesc), outputDesc, output, activationDesc, arch);
    }

    return ret;
}

#endif
