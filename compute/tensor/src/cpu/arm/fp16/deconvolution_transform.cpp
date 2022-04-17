// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/transform_functions.h"
#include "cpu/arm/fp16/tensor_computing_fp16.h"

inline EE deconvolution_transform_filter_kernel_fp16(TensorDesc filterDesc,
    const F16 *filterArray,
    TensorDesc *ftmDesc,
    F16 *ftmArray,
    DataFormat ftmDataFormat)
{
    if (nullptr == filterArray || nullptr == ftmDesc || nullptr == ftmArray) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (filterDesc.df == ftmDataFormat) {
        *ftmDesc = filterDesc;
        UNI_MEMCPY(ftmArray, filterArray, tensorNumBytes(filterDesc));
        return SUCCESS;
    }
    if (filterDesc.df != DF_NCHW) {
        return NOT_SUPPORTED;
    }
    EE ret = SUCCESS;
    switch (ftmDataFormat) {
        case DF_NHWCN16: {
            /*
         *  CNHW => NHWCN16
         *  if there is remainder, it should be NHWCN8
         */
            ret = transformCNHWToNHWCNx<F16, 16>(
                filterDesc, filterArray, ftmDataFormat, ftmDesc, ftmArray);
            break;
        }
        case DF_HWNCN16: {
            /*
         *  CNHW => NHWCN16 + NHWCN8 if there is remainder divided by 16
         */
            ret = transformCNHWToHWNCNx<F16, 16>(
                filterDesc, filterArray, ftmDataFormat, ftmDesc, ftmArray);
            break;
        }
        case DF_NCHWC8: {
            ret = transformCNHWToNCHWC8<F16>(
                filterDesc, filterArray, ftmDataFormat, ftmDesc, ftmArray);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE deconvolution_transform_filter_fp16(TensorDesc filterDesc,
    const F16 *filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F16 *filterTransformed)
{
    DataFormat ftmDataFormat;
    switch (algorithm) {
        case CONVOLUTION_ALGORITHM_WINOGRAD:
            ftmDataFormat = DF_HWNCN16;
            break;
        case CONVOLUTION_ALGORITHM_GEMM_ICNCHW:
            ftmDataFormat = DF_NHWCN16;
            break;
        case CONVOLUTION_ALGORITHM_GEMM:
            ftmDataFormat = DF_NHWCN16;
            break;
        case CONVOLUTION_ALGORITHM_GROUP_DECONV:
            ftmDataFormat = DF_NCHWC8;
            break;
        default:
            return NOT_MATCH;
    }
    EE ret = deconvolution_transform_filter_kernel_fp16(
        filterDesc, filter, ftmDesc, filterTransformed, ftmDataFormat);
    CHECK_STATUS(ret);
    return ret;
}
