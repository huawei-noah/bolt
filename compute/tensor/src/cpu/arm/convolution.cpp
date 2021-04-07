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
#ifdef _USE_FP16
#include "cpu/arm/bnn/tensor_computing_bnn.h"
#endif
#include "tensor_transpose.h"
#include "ut_util.h"

EE convolution_infer_forward_algorithm_arm(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec p,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType)
{
    UNUSED(outputDesc);
    if (nullptr == algorithm) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (*algorithm != CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }

    EE ret = SUCCESS;
    if (policy == CONVOLUTION_FASTEST) {
        DataType idt, fdt;
        DataFormat idf, fdf;
        U32 in, ic, it, ih, iw;
        U32 fn, fc, ft, fh, fw;
        if (tensorIs4d(inputDesc)) {
            CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
            CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
            it = ft = 1;
            p.dilatedRate_t = p.stride_t = 1;
            p.padding_before = p.padding_after = 0;
        } else if (tensorIs5d(inputDesc)) {
            CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
            CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        } else {
            return NOT_SUPPORTED;
        }
        if (p.dilatedRate_t > 1 || p.dilatedRate_h > 1 || p.dilatedRate_w > 1) {
            *algorithm = CONVOLUTION_ALGORITHM_GEMM;
            return SUCCESS;
        }

        if ((idf != DF_NCHWC8 || ic / p.group % 8 != 0) && DT_I8 != idt) {
            *algorithm = CONVOLUTION_ALGORITHM_GEMM_ICNCHW;
        } else if (ft == 1 && fh == 3 && fw == 3 && p.stride_t == 1 && p.stride_h == 1 &&
            p.stride_w == 1 && p.padding_before == 0 && p.padding_after == 0 && p.padding_top == 1 &&
            p.padding_bottom == 1 && p.padding_left == 1 && p.padding_right == 1) {
            *algorithm = CONVOLUTION_ALGORITHM_WINOGRAD;
        } else {
            *algorithm = CONVOLUTION_ALGORITHM_GEMM;
        }

        switch (targetDataType) {
            case DT_BIN01: {
                *algorithm = CONVOLUTION_ALGORITHM_BNN;
                break;
            }
            case DT_BIN11: {
                *algorithm = CONVOLUTION_ALGORITHM_BNN;
                break;
            }
            case DT_I8: {
                if (*algorithm == CONVOLUTION_ALGORITHM_WINOGRAD) {
                    *algorithm = CONVOLUTION_ALGORITHM_GEMM;
                }
                break;
            }
            default:
                break;
        }

#ifndef __aarch64__
        if (CONVOLUTION_ALGORITHM_GEMM_ICNCHW != *algorithm) {
            *algorithm = CONVOLUTION_ALGORITHM_GEMM;
        }
        return SUCCESS;
#endif
    } else if (policy == CONVOLUTION_TUNNING) {
        std::vector<ConvolutionForwardAlgorithm> convolutionAlgorithms;
        U32 filterBytes = 0;
        U32 tmpBytes = 0;
        for (U32 i = 0; i < convolutionAlgorithms.size(); i++) {
            U32 bytes = 0;
            CHECK_STATUS(convolution_transform_filter_bytes_arm(
                filterDesc, p, convolutionAlgorithms[i], &bytes));
            filterBytes = (bytes > filterBytes) ? bytes : filterBytes;
            CHECK_STATUS(convolution_infer_forward_tmp_bytes_arm(
                inputDesc, filterDesc, outputDesc, p, convolutionAlgorithms[i], &bytes));
            tmpBytes = (bytes > tmpBytes) ? bytes : tmpBytes;
        }
        TensorDesc biasDesc = tensor1d(filterDesc.dt, outputDesc.dims[3]);
        TensorDesc scaleDesc = tensor1d(DT_F32, outputDesc.dims[2]);
        U8 *input = ut_input_v(tensorNumElements(inputDesc), inputDesc.dt, UT_INIT_RANDOM);
        U8 *filter = ut_input_v(tensorNumElements(filterDesc), filterDesc.dt, UT_INIT_RANDOM);
        U8 *filterTransformed =
            ut_input_v(filterBytes / bytesOf(filterDesc.dt), filterDesc.dt, UT_INIT_RANDOM);
        U8 *bias = ut_input_v(tensorNumElements(biasDesc), biasDesc.dt, UT_INIT_RANDOM);
        U8 *scale = ut_input_v(tensorNumElements(scaleDesc), scaleDesc.dt, UT_INIT_RANDOM);
        U8 *tmp = ut_input_v(tmpBytes / bytesOf(inputDesc.dt), inputDesc.dt, UT_INIT_ZERO);
        U8 *output = ut_input_v(tensorNumElements(outputDesc), outputDesc.dt, UT_INIT_ZERO);
        U32 algorithmIndex = 0;
        ActivationParamSpec activationDesc;
        activationDesc.mode = ACTIVATION_RELU;
        activationDesc.value[0] = 0;
        for (U32 i = 0; i < convolutionAlgorithms.size(); i++) {
            TensorDesc ftmDesc;
            CHECK_STATUS(convolution_transform_filter_arm(
                filterDesc, filter, p, convolutionAlgorithms[i], &ftmDesc, filterTransformed));

            memset(tmp, 0, tmpBytes);
            double timeStart = ut_time_ms();
            CHECK_STATUS(convolution_arm(inputDesc, input, ftmDesc, filterTransformed, p,
                convolutionAlgorithms[i], scaleDesc, scale, biasDesc, bias, tmpBytes, tmp,
                outputDesc, output, activationDesc, ARM_A76));
            double timeEnd = ut_time_ms();
            double timeMin = INT_MAX;
            if (timeMin > timeEnd - timeStart) {
                timeMin = timeEnd - timeStart;
                algorithmIndex = i;
            }
        }
        free(input);
        free(filter);
        free(filterTransformed);
        free(bias);
        free(scale);
        free(tmp);
        free(output);
        *algorithm = convolutionAlgorithms[algorithmIndex];
        ret = SUCCESS;
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

EE convolution_transform_filter_bytes_arm(TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, ft, fh, fw;
    if (tensorIs4d(filterDesc)) {
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        ft = 1;
    } else if (tensorIs5d(filterDesc)) {
        CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
    } else {
        return NOT_SUPPORTED;
    }
    U32 fnAlignSize = 8;
    if (filterDesc.dt == DT_F16) {
        fnAlignSize = 16;
    }
    U32 fnGroupSize = fn / convParamSpec.group;
    U32 fnPadding = (fnGroupSize / fnAlignSize + ((fnGroupSize % fnAlignSize) == 0 ? 0 : 1)) *
        fnAlignSize * convParamSpec.group;
    EE ret = SUCCESS;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            *bytes = fnPadding * fc * 6 * 6;
            break;
        case CONVOLUTION_ALGORITHM_DIRECT:
            *bytes = fnPadding * fc * ft * fh * fw;
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            *bytes = fnPadding * fc * ft * fh * fw;
            break;
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            *bytes = fnPadding * fc * ft * fh * fw;
            break;
        case CONVOLUTION_ALGORITHM_BNN:
            *bytes = fnPadding * fc * ft * fh * fw;
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    *bytes *= bytesOf(fdt);

    switch (filterDesc.dt) {
        case DT_BIN01: {
            *bytes /= 8;
            break;
        }
        case DT_BIN11: {
            *bytes /= 8;
            break;
        }
        default:
            break;
    }
    *bytes += 32;
    return ret;
}

EE convolution_transform_filter_arm(TensorDesc filterDesc,
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
#ifdef _USE_FP16
        case DT_F16: {
            ret = convolution_transform_filter_fp16(filterDesc, (F16 *)filter, convParamSpec,
                algorithm, ftmDesc, (F16 *)filterTransformed);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_transform_filter_int8(
                filterDesc, filter, convParamSpec, algorithm, ftmDesc, filterTransformed);
            break;
        }
        case DT_F16_8Q: {
            ret = convolution_transform_filter_int8(
                filterDesc, filter, convParamSpec, algorithm, ftmDesc, filterTransformed);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_BIN01: {
            ret = convolution_transform_filter_bnn(
                filterDesc, (BIN8 *)filter, ftmDesc, (BIN8 *)filterTransformed);
            break;
        }
        case DT_BIN11: {
            ret = convolution_transform_filter_bnn(
                filterDesc, (BIN8 *)filter, ftmDesc, (BIN8 *)filterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec p,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, it, ih, iw;
    U32 fn, fc, ft, fh, fw;
    U32 on, oc, ot, oh, ow;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
        it = ft = ot = 1;
        p.dilatedRate_t = p.stride_t = 1;
        p.padding_before = p.padding_after = 0;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        return NOT_SUPPORTED;
    }
    U32 it_pad = it + p.padding_before + p.padding_after;
    U32 ih_pad = ih + p.padding_top + p.padding_bottom;
    U32 iw_pad = iw + p.padding_left + p.padding_right;
    U32 tile_size = 0;
    switch (fdt) {
        case DT_F32:
#ifdef __aarch64__
            tile_size = 12;
#else
            tile_size = 6;
#endif
            break;
        case DT_F16:
            tile_size = 8;
            break;
        case DT_I8:
            tile_size = 12;
            break;
        case DT_BIN01:
            tile_size = 0;
            break;
        case DT_BIN11:
            tile_size = 0;
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    U32 element_size = bytesOf(idt);
    *bytes = (ic * it_pad * ih_pad * iw_pad) * element_size;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT:
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            *bytes += tile_size * ft * fh * fw * ic * OMP_NUM_THREADS * element_size;
            if (fdt == DT_I8) {
                *bytes += ic * it * ih * iw;
            }
            if (odt == DT_I8) {
                // scaled bias + results before quantization
                *bytes += (oc + on * oc * ot * oh * ow) * bytesOf(DT_I32);
            }
            break;
        case CONVOLUTION_ALGORITHM_WINOGRAD: {
            U32 tile_h = (oh + 3) / 4;
            U32 tile_w = (ow + 3) / 4;
            U32 pad_left = p.padding_left;
            U32 pad_right = p.padding_right + (tile_w * 4 - ow);
            U32 pad_top = p.padding_top;
            U32 pad_bottom = p.padding_bottom + (tile_h * 4 - oh);
            ih_pad = ih + pad_top + pad_bottom;
            iw_pad = iw + pad_left + pad_right;
            *bytes = ic * ih_pad * iw_pad * element_size;
            if (fdt == DT_F32) {
                *bytes += (ic + 8) * 6 * 6 * 12 * element_size * OMP_NUM_THREADS;
            } else if (fdt == DT_F16) {
                *bytes += (ic + oc) * 6 * 6 * 8 * element_size * OMP_NUM_THREADS;
            } else if (fdt == DT_I8) {
                // itm (int16 for int8 inputs) and otm (otm just contains o8 each time)
                *bytes += (ic + 8) * 6 * 6 * 12 * bytesOf(DT_F16);
                // quantized transformed input
                *bytes += ic * 6 * 6 * 12;
                if (odt == DT_I8) {
                    // Output before quantization
                    *bytes += on * oc * oh * ow * bytesOf(DT_F16);
                }
            } else {
                ret = NOT_SUPPORTED;
            }
            break;
        }
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            *bytes += tile_size * ft * fh * fw * ic * OMP_NUM_THREADS * element_size;
            break;
        case CONVOLUTION_ALGORITHM_BNN:
            *bytes += (8 * ft * fh * fw * ic + ic * it * ih * iw) * element_size;
            *bytes /= 8;
            break;
        default:
            ret = NOT_MATCH;
            break;
    }
    if (DT_I8 == fdt && DF_NCHW == idf) {
        CHECK_REQUIREMENT(ic % 8 == 0);
        *bytes += tensorNumBytes(inputDesc);
    }
    *bytes += 32;

    // pre data processing space for not complete NCHWC8 group convolution input
    U32 icGroupSize = ic / p.group;
    if (idf == DF_NCHWC8 && icGroupSize % 8 != 0) {
        *bytes += tensorNumBytes(inputDesc);
    }
    return ret;
}

EE convolution_arm(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec p,
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
    U32 group = p.group;
    U32 batchAxis = inputDesc.nDims - 1;
    U32 dataChannelAxis = inputDesc.nDims - 2;
    U32 filterChannelAxis = filterDesc.nDims - 1;
    U32 biasChannelAxis = 0;
    if (group > 1) {
        CHECK_REQUIREMENT(inputDesc.dims[batchAxis] == 1);
    }
    U32 icGroupSize = inputDesc.dims[dataChannelAxis] / group;
    // pre data processing space for not complete NCHWC8 group convolution input
    void *inputTransform;
    if (inputDesc.df == DF_NCHWC8 && icGroupSize % 8 != 0) {
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
    EE ret = SUCCESS;
    for (U32 g = 0; g < group; g++) {
        void *tmpInput = (U8 *)inputTransform + g * tensorNumBytes(tmpInputDesc);
        const void *tmpFilter = (U8 *)filter + g * tensorNumBytes(tmpFilterDesc);
        const void *tmpBias = (U8 *)bias + g * tensorNumBytes(tmpBiasDesc);
        void *tmpOutput = (U8 *)output + g * tensorNumBytes(tmpOutputDesc);
        switch (filterDesc.dt) {
#ifdef _USE_FP32
            case DT_F32: {
                ret = convolution_fp32(tmpInputDesc, (F32 *)tmpInput, tmpFilterDesc,
                    (F32 *)tmpFilter, p, algorithm, tmpBiasDesc, (F32 *)tmpBias, tmpBytes, tmp,
                    tmpOutputDesc, (F32 *)tmpOutput, activationDesc, arch);
                break;
            }
#endif
#ifdef _USE_FP16
            case DT_F16: {
                ret = convolution_fp16(tmpInputDesc, (F16 *)tmpInput, tmpFilterDesc,
                    (F16 *)tmpFilter, p, algorithm, tmpBiasDesc, (F16 *)tmpBias, tmpBytes, tmp,
                    tmpOutputDesc, (F16 *)tmpOutput, activationDesc, arch);
                break;
            }
#endif
#ifdef _USE_INT8
            case DT_I8: {
                ret = convolution_int8(tmpInputDesc, (INT8 *)tmpInput, tmpFilterDesc,
                    (INT8 *)tmpFilter, (F16 *)scale, p, algorithm, tmpBiasDesc, (F16 *)tmpBias,
                    tmpBytes, tmp, tmpOutputDesc, tmpOutput, activationDesc, arch);
                break;
            }
#endif
#ifdef _USE_FP16
            case DT_BIN01: {
                ret = convolution_bnn(tmpInputDesc, (F16 *)tmpInput, tmpFilterDesc,
                    (BIN8 *)tmpFilter, p, scaleDesc, (F16 *)scale, tmpBiasDesc, (F16 *)tmpBias,
                    tmpBytes, tmp, tmpOutputDesc, (F16 *)tmpOutput, activationDesc, arch);
                break;
            }
            case DT_BIN11: {
                ret = convolution_bnn(tmpInputDesc, (F16 *)tmpInput, tmpFilterDesc,
                    (BIN8 *)tmpFilter, p, scaleDesc, (F16 *)scale, tmpBiasDesc, (F16 *)tmpBias,
                    tmpBytes, tmp, tmpOutputDesc, (F16 *)tmpOutput, activationDesc, arch);
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
