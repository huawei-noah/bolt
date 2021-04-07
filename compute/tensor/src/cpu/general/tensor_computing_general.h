// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_GENERAL
#define _H_TENSOR_COMPUTING_GENERAL

#include <vector>

#include "error.h"
#include "sys.h"
#include "uni.h"
#include "tensor_desc.h"
#include "parameter_spec.h"

EE convolution_general(TensorDesc inputDesc,
    void *input,
    void *eltwiseInput,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc scaleDesc,
    const void *scale,
    TensorDesc biasDesc,
    const void *bias,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc);

EE deconvolution_general(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc scaleDesc,
    const void *scale,
    TensorDesc biasDesc,
    const void *bias,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec activationDesc);

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes_general(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes);

EE depthwise_pointwise_convolution_general(TensorDesc inputDesc,
    void *input,
    TensorDesc dwFilterDesc,
    const void *dwFilter,
    TensorDesc pwFilterDesc,
    const void *pwFilter,
    ConvolutionParamSpec convParamSpec,
    TensorDesc dwBiasDesc,
    const void *dwBias,
    TensorDesc pwBiasDesc,
    const void *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec);

EE depthwise_convolution_general(TensorDesc inputDesc,
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
    ActivationParamSpec depthwiseActivationParamSpec);

EE pooling_general(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    void *output);

EE pooling_bp_general(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    void *output);

EE attention_general(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output);

EE clip_general(
    TensorDesc inputDesc, void *input, ClipParamSpec p, TensorDesc outputDesc, void *output);

EE eltwise_general(DataType dataType,
    std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode);

EE rnncell_general(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    void *state,
    U32 tmpBytes,
    void *tmp,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    void *currentH);

EE transpose_general(
    TensorDesc inputDesc, const void *input, U32 *dim, TensorDesc outputDesc, void *output);

EE scale_general(TensorDesc inputDesc,
    void *input,
    void *alpha,
    void *beta,
    ScaleParamSpec p,
    TensorDesc outputDesc,
    void *output);

EE softmax_general(
    TensorDesc inputDesc, const void *input, SoftmaxParamSpec p, TensorDesc outputDesc, void *output);

EE check_general(TensorDesc inputDescA,
    const void *inputA,
    TensorDesc inputDescB,
    const void *inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    void *output);

EE layer_normalization_general(
    TensorDesc inputDesc, void *input, void *alpha, void *beta, TensorDesc outputDesc, void *output);

EE attention_mask_general(TensorDesc inputDesc,
    const void *input,
    AttentionMaskParamSpec p,
    TensorDesc outputDesc,
    void *output);

EE prelu_general(TensorDesc inputDesc,
    void *input,
    void *weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    void *output);
#endif
