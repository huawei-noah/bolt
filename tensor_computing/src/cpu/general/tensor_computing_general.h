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
#include "tensor_desc.h"
#include "tensor_computing_type.h"

EE convolution_general(TensorDesc inputDesc, void* input,
        TensorDesc filterDesc, const void* filter,
        ConvolutionDesc convDesc,
        TensorDesc scaleDesc, const void* scale,
        TensorDesc biasDesc, const void* bias,
        TensorDesc outputDesc, void* output,
        ActivationMode activationMode);

EE deconvolution_general(TensorDesc inputDesc, void* input,
        TensorDesc filterDesc, const void* filter,
        ConvolutionDesc convDesc,
        TensorDesc scaleDesc, const void* scale,
        TensorDesc biasDesc, const void* bias,
        TensorDesc outputDesc, void* output,
        ActivationMode activationMode);

EE depthwise_convolution_general(TensorDesc inputDesc, void* input,
        TensorDesc filterDesc, const void* filter,
        ConvolutionDesc convDesc,
        TensorDesc biasDesc, const void* bias,
        TensorDesc outputDesc, void* output,
        ActivationMode depthwiseActivationMode,
        ActivationMode pointwiseActivationMode);

EE pooling_general(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, TensorDesc outputDesc, void* output);

EE activation_general(TensorDesc inputDesc, void* data, ActivationMode activationMode);

EE attention_general(TensorDesc inputDesc, const void *input,
       TensorDesc outputDesc, void *output);

EE clip_general(void *minValue, void *maxValue, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output);

EE eltwise_general(std::vector<TensorDesc> inputDesc, std::vector<void*> input,
    TensorDesc outputDesc, void* output, EltwiseMode eltwiseMode);

EE lstmcell_general(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* currentH);

EE lstm_general(TensorDesc inputDesc, const void* input,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    LSTMDesc lstmDesc,
    TensorDesc outputDesc, void* output);

EE transpose_general(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output, U32 *dim);

EE slice_general(TensorDesc inputDesc, void* input,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output);

EE split_general(TensorDesc inputDesc, void* input,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output);

EE multiply_general(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output);

EE scale_general(void *alpha, void *beta, TensorDesc inputDesc, void* data);

EE softmax_general(TensorDesc inputDesc, const void* input,
    TensorDesc outputDesc, void* output);

EE reshape_general(TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output);

EE argmax_general(TensorDesc inputDesc, const void* input,
    I32 axis,
    TensorDesc outputDesc, void* output);

EE axis_mean_general(TensorDesc inputDesc, const void* input,
    I32 axis,
    TensorDesc outputDesc, void* output);

EE check_general(TensorDesc inputDescA, const void* inputA,
    TensorDesc inputDescB, const void* inputB,
    CheckMode checkMode,
    TensorDesc outputDesc, void* output);

EE layer_normalization_general(void *alpha, void *beta,
    TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output);
#endif
