// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/fp32/transform_functions_fp32.h"

inline EE depthwise_convolution_transform_filter_kernel_fp32(TensorDesc filterDesc,
    const F32 *filterArray,
    TensorDesc *ftmDesc,
    F32 *ftmArray,
    DataFormat ftmDataFormat)
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
    switch (ftmDataFormat) {
        case DF_NCHWC24: {
            filterDesc = tensor4df(fdt, fdf, fc, 1, fh, fw);
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fc, 1, fh, fw);
            transformNCHWToNCHWCxNx<1, 24>(filterDesc, filterArray, *ftmDesc, ftmArray);
            *ftmDesc = tensor4df(fdt, ftmDataFormat, fn, fc, fh, fw);
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE depthwise_convolution_transform_filter_fp32(TensorDesc filterDesc,
    const F32 *filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F32 *filterTransformed)
{
    DataFormat ftmDataFormat;
    switch (algorithm) {
        case DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT:
            ftmDataFormat = DF_NCHWC24;
            break;
        default:
            return NOT_MATCH;
    }
    EE ret = depthwise_convolution_transform_filter_kernel_fp32(
        filterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat);
    CHECK_STATUS(ret);
    return ret;
}
