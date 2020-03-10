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
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing_type.h"
#include "cpu/arm/fp32/arm_neon_expand_fp32.h"

EE convolution_transform_filter_fp32(TensorDesc filterDesc, const F32* filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, F32* filterTransformed);

EE convolution_infer_forward_tmp_bytes_fp32(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE convolution_fp32(TensorDesc inputDesc, F32* input,
    TensorDesc filterDesc, const F32* filter,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const F32* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* output,
    ActivationMode activationMode,
    Arch arch);

EE convolution_gemm_V8(TensorDesc inputDesc, F32* inArray,
    TensorDesc filterDesc, const F32* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* outArray,
    ActivationMode activationMode);

EE convolution_gemm_icnchw_V8(TensorDesc inputDesc, F32* inArray,
    TensorDesc filterDesc, const F32* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* outArray,
    ActivationMode activationMode);

EE convolution_winograd_V8(TensorDesc inputDesc, F32* inArray,
    TensorDesc filterDesc, const F32* filterArray,
    ConvolutionDesc convDesc,
    TensorDesc biasDesc, const F32* biasArray,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* outArray,
    ActivationMode activationMode);

EE deconvolution_infer_forward_algorithm_fp32(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm);

EE deconvolution_infer_forward_tmp_bytes_fp32(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE deconvolution_transform_filter_bytes_fp32(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes);

EE deconvolution_transform_filter_fp32(TensorDesc filterDesc, const F32* filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, F32* filterTransformed);

EE deconvolution_fp32(TensorDesc inputDesc, F32* input,
    TensorDesc filterDesc, const F32* filter,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const F32* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* output,
    ActivationMode activationMode,
    Arch arch);

EE pooling_fp32(TensorDesc inputDesc, const F32* input, PoolingDesc poolingDesc, TensorDesc outputDesc, F32* output);

EE softmax_fp32(TensorDesc inputDesc, const F32* input, TensorDesc outputDesc, F32* output);

EE concat_fp32(std::vector<TensorDesc> inputDesc, std::vector<void*> input, TensorDesc outputDesc, void* output, U32 concatDim);

EE attention_fp32(U32 batch, U32 numHeads, I32 fromSequenceLength, I32 toSequenceLength, const F32 *input, F32 *output);

EE clip_fp32(F32 *input, F32 *output, I32 len, F32 minValue, F32 maxValue);

EE depthwise_convolution_infer_forward_algorithm_fp32(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm, DataType targetDataType);

EE depthwise_convolution_transform_filter_bytes_fp32(TensorDesc filterDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32* bytes);

EE depthwise_convolution_transform_filter_fp32(TensorDesc filterDesc, const F32* filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, F32* filterTransformed);

EE depthwise_convolution_infer_forward_tmp_bytes_fp32(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE depthwise_convolution_transform_filter_fp32(TensorDesc filterDesc, const F32* filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, F32* filterTransformed);

EE depthwise_convolution_fp32(TensorDesc inputDesc, F32* input,
    TensorDesc filterDesc, const F32* filter,
    ConvolutionDesc convDesc,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const F32* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, F32* output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch);

EE eltwise_fp32(std::vector<void*>input, U32 num, U32 len, void *output, EltwiseMode eltwiseMode);

EE lstmcell_fp32(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* output);

EE multiply_fp32(F32 *alpha, F32 *beta, TensorDesc inputDesc, F32* input, TensorDesc outputDesc, F32 *output);

EE layer_normalization_fp32(F32 *alpha, F32 *beta,
    TensorDesc inputDesc, F32* input,
    TensorDesc outputDesc, F32* output);

EE pooling_fp32(TensorDesc inputDesc, const F32* input, PoolingDesc poolingDesc, const F32* scale, TensorDesc outputDesc, F32* output);

EE scale_nchwc8_fp32(F32* alpha, F32* beta, F32* data, U32 in, U32 ic, U32 elements_per_channel);

EE softmax_fp32(TensorDesc inputDesc, const F32* input,
    TensorDesc outputDesc, F32* output);

EE check_fp32(TensorDesc inputDescA, const F32* inputA,
    TensorDesc inputDescB, const F32* inputB,
    CheckMode checkMode,
    TensorDesc outputDesc, I32* output);
#endif
