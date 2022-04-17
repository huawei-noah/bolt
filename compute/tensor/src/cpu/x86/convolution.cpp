// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_INT8
#include "cpu/x86/int8/tensor_computing_int8.h"
#endif
#include "tensor_transpose.h"

EE convolution_infer_forward_algorithm_x86(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType)
{
    UNUSED(filterDesc);
    UNUSED(outputDesc);
    UNUSED(convParamSpec);
    UNUSED(policy);
    UNUSED(targetDataType);
    if (nullptr == algorithm) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (*algorithm != CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 group = convParamSpec.group;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;

    if ((targetDataType != DT_I8) && (targetDataType != DT_U8_Q) &&
        ((idf != DF_NCHWC8) || (ic / group % 8 != 0))) {
        *algorithm = CONVOLUTION_ALGORITHM_GEMM_ICNCHW;
        return SUCCESS;
    }

    if ((targetDataType == DT_F32) && (idf == DF_NCHWC8) && (group == 1) && (fh == 3) &&
        (fw == 3) && (dilateH == 1) && (dilateW == 1) && (oh > 8) && (ow > 8) && (strideH == 1) &&
        (strideW == 1)) {
        *algorithm = CONVOLUTION_ALGORITHM_WINOGRAD;
        return SUCCESS;
    }

    if ((fh == 1) && (fw == 1)) {
        *algorithm = CONVOLUTION_ALGORITHM_POINTWISE;
        return SUCCESS;
    }

    *algorithm = CONVOLUTION_ALGORITHM_DIRECT;
    return SUCCESS;
}

EE convolution_transform_filter_bytes_x86(TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;

    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    U32 fnAlignSize = 8;
    U32 fnGroupSize = fn / convParamSpec.group;
    U32 fnPadding = (fnGroupSize / fnAlignSize + ((fnGroupSize % fnAlignSize) == 0 ? 0 : 1)) *
        fnAlignSize * convParamSpec.group;
    U32 fcPadding = (fc / fnAlignSize + ((fc % fnAlignSize) == 0 ? 0 : 1)) * fnAlignSize;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = fnPadding * fcPadding * fh * fw;
            break;
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            *bytes = fnPadding * fc * fh * fw;
            break;
        case CONVOLUTION_ALGORITHM_POINTWISE:
            *bytes = fnPadding * fcPadding;
            break;
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            *bytes =
                fnPadding * fcPadding * 36 + 16 * 32 * 18;  // bolckIc:16, blockOc:32, weight:3*6=18
            break;
        default:
            return NOT_SUPPORTED;
    }
    *bytes *= bytesOf(fdt);
    if (fdt == DT_I8) {
        fnAlignSize = 4;
        *bytes /= fcPadding;
        fcPadding = (fc + fnAlignSize - 1) / fnAlignSize * fnAlignSize;
        *bytes = *bytes * fcPadding + fn * bytesOf(DT_I32);  //offsetC
    }
    *bytes += 32;
    return ret;
}

EE convolution_transform_filter_x86(TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = convolution_transform_filter_fp32(filterDesc, (F32 *)filter, convParamSpec,
                algorithm, ftmDesc, (F32 *)filterTransformed);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_transform_filter_int8(filterDesc, (INT8 *)filter, convParamSpec,
                algorithm, ftmDesc, (INT8 *)filterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_infer_forward_tmp_bytes_x86(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = convolution_infer_forward_tmp_bytes_fp32(
                inputDesc, filterDesc, outputDesc, convParamSpec, algorithm, bytes);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_infer_forward_tmp_bytes_int8(
                inputDesc, filterDesc, outputDesc, convParamSpec, algorithm, bytes);
            break;
        }
#endif
        default:
            CHECK_STATUS(NOT_SUPPORTED);
            break;
    }

    return ret;
}

EE convolution_x86(TensorDesc inputDesc,
    void *input,
    void *eltwiseInput,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc scaleDesc,
    void *scale,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc,
    Arch arch)
{
    U32 group = convParamSpec.group;
    U32 batchAxis = inputDesc.nDims - 1;
    U32 dataChannelAxis = inputDesc.nDims - 2;
    U32 filterChannelAxis = filterDesc.nDims - 1;
    U32 biasChannelAxis = 0;
    if (group > 1) {
        CHECK_REQUIREMENT(inputDesc.dims[batchAxis] == 1);
    }
    U32 icGroupSize = inputDesc.dims[dataChannelAxis] / group;

    void *inputTransform;
    if ((inputDesc.df == DF_NCHWC8 && icGroupSize % 8 != 0)) {
        TensorDesc tmpInputDesc = inputDesc;
        tmpInputDesc.df = DF_NCHW;
        transformToNCHW(inputDesc, input, tmpInputDesc, tmp);
        inputTransform = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(tmpInputDesc);
        tmpBytes -= tensorNumBytes(tmpInputDesc);
        inputDesc.df = DF_NCHW;
    } else {
        inputTransform = input;
    }

    TensorDesc tmpInputDesc = inputDesc;
    tmpInputDesc.dims[dataChannelAxis] /= group;
    TensorDesc tmpOutputDesc = outputDesc;
    tmpOutputDesc.dims[dataChannelAxis] /= group;
    TensorDesc tmpFilterDesc = filterDesc;
    tmpFilterDesc.dims[filterChannelAxis] /= group;
    TensorDesc tmpBiasDesc = biasDesc;
    tmpBiasDesc.dims[biasChannelAxis] /= group;

    TensorDesc paddingFilterDesc = tmpFilterDesc;
    paddingFilterDesc.dims[filterChannelAxis] = (tmpFilterDesc.dims[filterChannelAxis] + 7) / 8 * 8;

    EE ret = SUCCESS;
    for (U32 g = 0; g < group; g++) {
        void *tmpInput = (U8 *)inputTransform + g * tensorNumBytes(tmpInputDesc);
        const void *tmpFilter = (U8 *)filter + g * tensorNumBytes(paddingFilterDesc);
        const void *tmpBias = (U8 *)bias + g * tensorNumBytes(tmpBiasDesc);
        void *tmpOutput = (U8 *)output + g * tensorNumBytes(tmpOutputDesc);
        switch (filterDesc.dt) {
#ifdef _USE_FP32
            case DT_F32: {
                ret = convolution_fp32(tmpInputDesc, (F32 *)tmpInput, (F32 *)eltwiseInput,
                    tmpFilterDesc, (F32 *)tmpFilter, convParamSpec, algorithm, tmpBiasDesc,
                    (F32 *)tmpBias, tmpBytes, tmp, tmpOutputDesc, (F32 *)tmpOutput, activationDesc,
                    arch);
                break;
            }
#endif
#ifdef _USE_INT8
            case DT_I8: {
                ret = convolution_int8(tmpInputDesc, (UINT8 *)tmpInput, (F32 *)eltwiseInput,
                    tmpFilterDesc, (INT8 *)tmpFilter, convParamSpec, algorithm, tmpBiasDesc,
                    (F32 *)tmpBias, tmpBytes, tmp, tmpOutputDesc, tmpOutput, (F32 *)scale,
                    activationDesc, arch);
                break;
            }
#endif
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    }
    return ret;
}
