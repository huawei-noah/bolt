// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_FP32
#define _H_TENSOR_COMPUTING_FP32
#include <vector>
#include "sys.h"
#include "error.h"
#include "thread_affinity.h"

#include "cpu/arm/fp32/arm_functions_fp32.h"

EE convolution_transform_filter_fp32(TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F32 *filterTransformed);

EE convolution_fp32(TensorDesc inputDesc,
    F32 *input,
    TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc,
    const F32 *bias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *output,
    ActivationParamSpec activationDesc,
    Arch arch);

#ifdef __aarch64__
EE convolution_gemm_V8(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc);
#else
EE convolution_gemm_V7(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc);
#endif

#ifdef __aarch64__
EE convolution_gemm_icnchw_V8(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc);
#else
EE convolution_gemm_icnchw_V7(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc);
#endif

EE convolution_winograd_V8(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc);

EE deconvolution_transform_filter_fp32(TensorDesc filterDesc,
    const F32 *filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    F32 *filterTransformed);

EE pooling_c8_fp32(I32 tstart,
    I32 tend,
    I32 hstart,
    I32 hend,
    I32 wstart,
    I32 wend,
    I32 poolSize,
    const F32 *input,
    I32 it,
    I32 ih,
    I32 iw,
    PoolingParamSpec p,
    F32 *output);

EE pooling_bp_c8_fp32(const F32 *input,
    int hstart,
    int hend,
    int wstart,
    int wend,
    F32 *output,
    U32 stride,
    PoolingParamSpec poolingParamSpec);

EE softmax_fp32(
    TensorDesc inputDesc, const F32 *input, int axis, TensorDesc outputDesc, F32 *output);

EE concat_fp32(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    TensorDesc outputDesc,
    void *output,
    U32 concatDim);

EE attention_fp32(U32 batch,
    U32 numHeads,
    I32 fromSequenceLength,
    I32 toSequenceLength,
    const F32 *input,
    F32 *output);

EE clip_fp32(F32 *input, F32 *output, I32 len, F32 minValue, F32 maxValue);

EE depthwise_pointwise_convolution_fp32(TensorDesc inputDesc,
    F32 *input,
    TensorDesc dwFilterDesc,
    const F32 *dwFilter,
    TensorDesc pwFilterDesc,
    const F32 *pwFilter,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc dwBiasDesc,
    const F32 *dwBias,
    TensorDesc pwBiasDesc,
    const F32 *pwBias,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    Arch arch);

EE eltwise_fp32(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode);

EE rnncell_fp32(TensorDesc xDesc,
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

EE lstmcell_fp32(TensorDesc xDesc,
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

EE grucell_fp32(TensorDesc xDesc,
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

EE power_fp32(TensorDesc inputDesc,
    F32 *input,
    F32 scale,
    F32 shift,
    F32 power,
    TensorDesc outputDesc,
    F32 *output);

EE layer_normalization_fp32(
    TensorDesc inputDesc, F32 *input, F32 *alpha, F32 *beta, TensorDesc outputDesc, F32 *output);

EE scale_fp32(F32 *input,
    I32 axis,
    I32 nDims,
    F32 *alpha,
    F32 *beta,
    I32 in,
    I32 ic,
    I32 elements_per_channel,
    F32 *output);

EE softmax_fp32(TensorDesc inputDesc, const F32 *input, TensorDesc outputDesc, F32 *output);

EE check_fp32(TensorDesc inputDescA,
    const F32 *inputA,
    TensorDesc inputDescB,
    const F32 *inputB,
    CheckMode checkMode,
    TensorDesc outputDesc,
    I32 *output);

EE attention_mask_fp32(TensorDesc inputDesc,
    const F32 *input,
    AttentionMaskParamSpec p,
    TensorDesc outputDesc,
    F32 *output);

EE prelu_fp32(TensorDesc inputDesc,
    F32 *input,
    F32 *weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    F32 *output);
#endif
