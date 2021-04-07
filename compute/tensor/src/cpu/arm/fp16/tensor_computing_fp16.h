// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_FP16
#define _H_TENSOR_COMPUTING_FP16
#include <vector>

#include "sys.h"
#include "tensor_desc.h"
#include "parameter_spec.h"
#include "cpu/arm/fp16/arm_functions_fp16.h"

EE convolution_transform_filter_fp16(TensorDesc filterDesc,
    const F16 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F16 *filterTransformed);

EE convolution_fp16(TensorDesc inputDesc,
    F16 *input,
    TensorDesc filterDesc,
    const F16 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const F16 *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F16 *output,
    ActivationParamSpec activationDesc,
    Arch arch);

EE deconvolution_transform_filter_fp16(TensorDesc filterDesc,
    const F16 *filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F16 *filterTransformed);

EE pooling_c8_fp16(I32 tstart,
    I32 tend,
    I32 hstart,
    I32 hend,
    I32 wstart,
    I32 wend,
    I32 poolSize,
    const F16 *input,
    I32 it,
    I32 ih,
    I32 iw,
    PoolingParamSpec p,
    F16 *output);

EE softmax_fp16(
    TensorDesc inputDesc, const F16 *input, int axis, TensorDesc outputDesc, F16 *output);

EE attention_fp16(U32 batch,
    U32 numHeads,
    I32 fromSequenceLength,
    I32 toSequenceLength,
    const F16 *input,
    F16 *output);

EE clip_fp16(F16 *input, F16 *output, I32 len, F32 minValue, F32 maxValue);

EE concat_fp16(std::vector<TensorDesc> inputDesc,
    std::vector<F16 *> input,
    F16 *inputScale,
    TensorDesc outputDesc,
    F16 *output,
    F16 *outputScale,
    U32 concatDim);

EE depthwise_pointwise_convolution_fp16(TensorDesc inputDesc,
    F16 *input,
    TensorDesc dwFilterDesc,
    const F16 *dwFilter,
    TensorDesc pwFilterDesc,
    const F16 *pwFilter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc dwBiasDesc,
    const F16 *dwBias,
    TensorDesc pwBiasDesc,
    const F16 *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F16 *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    Arch arch);

EE eltwise_fp16(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode);

EE rnncell_fp16(TensorDesc xDesc,
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
    void *output,
    Arch arch);

EE lstmcell_fp16(TensorDesc xDesc,
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
    void *output,
    Arch arch);

EE grucell_fp16(TensorDesc xDesc,
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
    void *output,
    Arch arch);

EE power_fp16(TensorDesc inputDesc,
    F16 *input,
    F32 scale,
    F32 shift,
    F32 power,
    TensorDesc outputDesc,
    F16 *output);

EE layer_normalization_fp16(
    TensorDesc inputDesc, F16 *input, F16 *alpha, F16 *beta, TensorDesc outputDesc, F16 *output);

EE scale_fp16(F16 *input,
    I32 axis,
    I32 nDims,
    F16 *alpha,
    F16 *beta,
    I32 in,
    I32 ic,
    I32 elements_per_channel,
    F16 *output);

EE softmax_fp16(TensorDesc inputDesc, const F16 *input, TensorDesc outputDesc, F16 *output);

EE check_fp16(TensorDesc inputDescA,
    const F16 *inputA,
    TensorDesc inputDescB,
    const F16 *inputB,
    CheckMode checkMode,
    TensorDesc outputDesc,
    I32 *output);

EE quantize_tensor_fp16(
    TensorDesc dDesc, const void *data, TensorDesc *qDesc, void *qData, F16 *scale);

EE attention_mask_fp16(TensorDesc inputDesc,
    const F16 *input,
    AttentionMaskParamSpec p,
    TensorDesc outputDesc,
    F16 *output);

EE prelu_fp16(TensorDesc inputDesc,
    F16 *input,
    F16 *weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    F16 *output);
#endif
