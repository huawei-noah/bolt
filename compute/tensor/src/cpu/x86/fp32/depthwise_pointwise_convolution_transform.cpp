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

EE depthwise_pointwise_convolution_transform_filter_fp32(TensorDesc dwFilterDesc,
    const F32 *dwFilter,
    TensorDesc pwFilterDesc,
    const F32 *pwFilter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *dwFtmDesc,
    F32 *dwFilterTransformed,
    TensorDesc *pwFtmDesc,
    F32 *pwFilterTransformed)
{
    EE ret = depthwise_convolution_transform_filter_fp32(dwFilterDesc, dwFilter,
        DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT, dwFtmDesc, dwFilterTransformed);
    CHECK_STATUS(ret);
    if (pwFilter == nullptr) {
        return ret;
    }

    ConvolutionParamSpec p = createConvolutionParamSpec(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
        1, pwFilterDesc.dims[pwFilterDesc.nDims - 1], Convolution_Pointwise);
    ret = convolution_transform_filter_fp32(
        pwFilterDesc, pwFilter, p, CONVOLUTION_ALGORITHM_POINTWISE, pwFtmDesc, pwFilterTransformed);
    CHECK_STATUS(ret);
    return ret;
}
