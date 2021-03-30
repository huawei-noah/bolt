// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_INT8
#define _H_TENSOR_COMPUTING_INT8
#ifdef _USE_INT8
#include <vector>
#include "sys.h"

#include "error.h"
#include "cpu/arm/int8/arm_functions_int8.h"

EE convolution_infer_forward_algorithm_int8(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm);

EE convolution_transform_filter_bytes_int8(
    TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE convolution_transform_filter_int8(TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed);

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
    Arch arch);

EE depthwise_pointwise_convolution_int8(TensorDesc inputDesc,
    INT8 *input,
    TensorDesc dwFilterDesc,
    const INT8 *dwFilter,
    TensorDesc pwFilterDesc,
    const INT8 *pwFilter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc dwBiasDesc,
    const I32 *dwBias,
    TensorDesc pwBiasDesc,
    const I32 *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    I32 *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    Arch arch);

EE pooling_c8_int8(I32 tstart,
    I32 tend,
    I32 hstart,
    I32 hend,
    I32 wstart,
    I32 wend,
    I32 poolSize,
    const INT8 *input,
    I32 it,
    I32 ih,
    I32 iw,
    PoolingParamSpec p,
    INT8 *output,
    void *scale);

EE concat_int8(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    F32 *inputScale,
    int axis,
    TensorDesc outputDesc,
    void *output,
    F32 *outputScale);

EE quantize_tensor_int32(
    TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, F32 *scale);
#endif
#endif
