// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CHEETAH_TENSOR_COMPUTING_X86_H
#define CHEETAH_TENSOR_COMPUTING_X86_H

#include <vector>
#include "error.h"
#include "sys.h"
#include "tensor_desc.h"
#include "parameter_spec.h"

EE attention_x86(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output);

EE attention_mask_x86(TensorDesc inputDesc,
    const void *input,
    AttentionMaskParamSpec p,
    TensorDesc outputDesc,
    void *output);

EE check_x86(TensorDesc inputDescA,
    const void *inputA,
    TensorDesc inputDescB,
    const void *inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    void *output);

EE clip_x86(TensorDesc inputDesc, void *input, ClipParamSpec p, TensorDesc outputDesc, void *output);

EE convolution_infer_forward_algorithm_x86(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType);

EE convolution_transform_filter_bytes_x86(TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes);

EE convolution_transform_filter_x86(TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed);

EE convolution_infer_forward_tmp_bytes_x86(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes);

EE convolution_x86(TensorDesc inputDesc,
    void *input,
    void *eltwiseInput,
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
    Arch arch);

EE depthwise_pointwise_convolution_transform_filter_x86(TensorDesc dwFilterDesc,
    const void *dwFilter,
    TensorDesc pwFilterDesc,
    const void *pwFilter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *dwFtmDesc,
    void *dwFilterTransformed,
    TensorDesc *pwFtmDesc,
    void *pwFilterTransformed);

EE depthwise_pointwise_convolution_x86(TensorDesc inputDesc,
    void *input,
    TensorDesc dwFilterDesc,
    const void *dwFilter,
    TensorDesc pwFilterDesc,
    const void *pwFilter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc dwBiasDesc,
    const void *dwBias,
    TensorDesc pwBiasDesc,
    const void *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    Arch arch);

EE depthwise_convolution_transform_filter_bytes_x86(
    TensorDesc filterDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE depthwise_convolution_transform_filter_x86(TensorDesc filterDesc,
    const void *filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed);

EE depthwise_convolution_infer_forward_tmp_bytes_x86(TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes);

EE depthwise_convolution_x86(TensorDesc inputDesc,
    void *input,
    TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const void *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    Arch arch);

EE eltwise_x86(DataType dataType,
    std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode);

EE layer_normalization_x86(
    TensorDesc inputDesc, void *input, void *alpha, void *beta, TensorDesc outputDesc, void *output);

EE rnncell_x86(TensorDesc xDesc,
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
    void *currentH,
    Arch arch);

EE scale_x86(TensorDesc inputDesc,
    void *input,
    void *alpha,
    void *beta,
    ScaleParamSpec p,
    TensorDesc outputDesc,
    void *output);

EE pooling_x86(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec poolingParamSpec,
    const void *scale,
    TensorDesc outputDesc,
    void *output);

EE pooling_bp_x86(TensorDesc inputDesc,
    const void *input,
    PoolingParamSpec poolingParamSpec,
    TensorDesc outputDesc,
    void *output);

EE reshape_x86(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output);

EE softmax_x86(
    TensorDesc inputDesc, const void *input, SoftmaxParamSpec p, TensorDesc outputDesc, void *output);

EE deconvolution_transform_filter_x86(TensorDesc filterDesc,
    const void *filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed);

EE deconvolution_overlap_crop_x86(void *input,
    void *output,
    TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec);

EE prelu_x86(TensorDesc inputDesc,
    void *input,
    void *weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    void *output);

#endif  //CHEETAH_TENSOR_COMPUTING_X86_H
