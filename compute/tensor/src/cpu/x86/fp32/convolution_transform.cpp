// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstring>
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/fp32/transform_functions_fp32.h"

// N is 32/24
template <U32 N>
inline EE transformNCHWToNCHWCxNxWrapper(
    TensorDesc filterDesc, const F32 *filterArray, TensorDesc ftmDesc, F32 *ftmArray, U32 cx)
{
    EE ret = NOT_SUPPORTED;
    switch (cx) {
        case 128:
            ret = transformNCHWToNCHWCxNx<128, N>(filterDesc, filterArray, ftmDesc, ftmArray);
            break;
        case 8:
            ret = transformNCHWToNCHWCxNx<8, N>(filterDesc, filterArray, ftmDesc, ftmArray);
            break;
        case 1:
            ret = transformNCHWToNCHWCxNx<1, N>(filterDesc, filterArray, ftmDesc, ftmArray);
            break;
        default:
            break;
    }
    return ret;
}

inline EE convolution_transform_filter_kernel_fp32(TensorDesc filterDesc,
    const F32 *filterArray,
    TensorDesc *ftmDesc,
    F32 *ftmArray,
    DataFormat ftmDataFormat,
    U32 cx)
{
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fdf == ftmDataFormat) {
        *ftmDesc = filterDesc;
        memcpy(ftmArray, filterArray, fn * fc * fh * fw * bytesOf(fdt));
        return SUCCESS;
    }
    if (fdf != DF_NCHW) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    EE ret = SUCCESS;
    switch (ftmDataFormat) {
        case DF_NCHWCxN32: {
            /*
         *  NCHW => NCHWCxN32
         */
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, fc, fh, fw);
            transformNCHWToNCHWCxNxWrapper<32>(filterDesc, filterArray, *ftmDesc, ftmArray, cx);
            break;
        }
        case DF_NCHWCxN24: {
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, fc, fh, fw);
            transformNCHWToNCHWCxNxWrapper<24>(filterDesc, filterArray, *ftmDesc, ftmArray, cx);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_transform_filter_fp32(TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F32 *filterTransformed)
{
    U32 cx = 0;
    DataFormat ftmDataFormat;
    DataType fdt;
    DataFormat fdf;
    U32 fn, fc, fh, fw;
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    fn = (fn + 7) / 8 * 8 / convParamSpec.group;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_DIRECT: {
            if ((fn % 24 == 0) && (fn % 32 != 0)) {
                ftmDataFormat = DF_NCHWCxN24;
            } else {
                ftmDataFormat = DF_NCHWCxN32;
            }
            cx = 8;
            break;
        }
        case CONVOLUTION_ALGORITHM_POINTWISE: {
            if ((fn % 24 != 0) && (fn % 32 == 0)) {
                ftmDataFormat = DF_NCHWCxN32;
            } else {
                ftmDataFormat = DF_NCHWCxN24;
            }
            cx = 128;
            break;
        }
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW: {
            if ((fn % 24 != 0) && (fn % 32 == 0)) {
                ftmDataFormat = DF_NCHWCxN32;
            } else {
                ftmDataFormat = DF_NCHWCxN24;
            }
            cx = 1;
            break;
        }
        default:
            return NOT_MATCH;
    }

    U32 channelAxis = filterDesc.nDims - 1;
    TensorDesc tmpFilterDesc = filterDesc;
    tmpFilterDesc.dims[channelAxis] /= convParamSpec.group;
    U32 fnPadding = tmpFilterDesc.dims[channelAxis];
    if (fnPadding % 8 != 0) {
        fnPadding = (fnPadding / 8 + 1) * 8;
    }
    U32 originalTileSize = tensorNumElements(tmpFilterDesc);
    for (U32 g = 0; g < convParamSpec.group; g++) {
        CHECK_STATUS(convolution_transform_filter_kernel_fp32(
            tmpFilterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat, cx));
        U32 newTileSize = tensorNumElements(*ftmDesc) / tmpFilterDesc.dims[channelAxis] * fnPadding;
        filter += originalTileSize;
        filterTransformed += newTileSize;
    }
    ftmDesc->dims[channelAxis] = filterDesc.dims[channelAxis];
    return SUCCESS;
}
