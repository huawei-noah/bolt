// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_TENSOR_COMPUTING_ARM
#define _H_TENSOR_COMPUTING_ARM

#include <vector>

#include "sys.h"
#include "tensor_desc.h"
#include "tensor_computing_type.h"

EE activation_arm(TensorDesc inputDesc, void* data, ActivationMode activationMode);


EE attention_arm(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output);


EE clip_arm(void *minValue, void *maxValue, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output);


EE concat_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input, std::vector<F16> inputScale,
    TensorDesc outputDesc, void* output, F16* outputScale, U32 concatDim);


EE convolution_infer_forward_algorithm_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType);

EE convolution_transform_filter_bytes_arm(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes);

EE convolution_transform_filter_arm(TensorDesc filterDesc, const void* filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed);

EE convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE convolution_arm(TensorDesc inputDesc, void* input,
    TensorDesc filterDesc, const void* filter,
    ConvolutionDesc convDesc,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc scaleDesc, const void* scale,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode activationMode,
    Arch arch);


EE depthwise_convolution_infer_forward_algorithm_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm, DataType targetDataType);

EE depthwise_convolution_transform_filter_bytes_arm(TensorDesc filterDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32* bytes);

EE depthwise_convolution_transform_filter_arm(TensorDesc filterDesc, const void* filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed);

EE depthwise_convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE depthwise_convolution_arm(TensorDesc inputDesc, void* input,
    TensorDesc filterDesc, const void* filter,
    ConvolutionDesc convDesc,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch);


EE eltwise_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input,
    TensorDesc outputDesc, void* output, EltwiseMode eltwiseMode);


EE lstm_transform_filter_arm(TensorDesc filterDesc, const void* filter, TensorDesc *ftmDesc, void* filterTransformed, U32 x_dim, U32 h_dim);

EE lstm_transform_filter_bytes_arm(TensorDesc filterDesc, U32* bytes);

EE lstm_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes);

EE lstm_arm(TensorDesc inputDesc, const void* input, TensorDesc filterDesc, const void* filter,
    LSTMDesc lstmDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output);


EE matmul_infer_forward_tmp_bytes_arm(TensorDesc matrixADesc, TensorDesc matrixBDesc, U32 *bytes, Arch arch);

EE matmul_arm(TensorDesc matrixADesc, const void* matrixA,
    TensorDesc matrixBDesc, const void* matrixB,
    void* tmp, U32 bytes,
    TensorDesc matrixCDesc, void* matrixC,
    Arch arch);


EE multiply_arm(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output);


EE layer_normalization_arm(void *alpha, void *beta,
    TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output);


EE pooling_arm(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, const void* scale, TensorDesc outputDesc, void* output);


EE reshape_arm(TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output);


EE scale_arm(void *alpha, void *beta, TensorDesc inputDesc, void* data);


EE slice_arm(TensorDesc inputDesc, void* input,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output);


EE softmax_arm(TensorDesc inputDesc, const void* input,
    TensorDesc outputDesc, void* output);


EE split_arm(TensorDesc inputDesc, void* input,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output);


EE transpose_arm(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output, U32 *dim);

EE quantize_tensor_arm(TensorDesc dDesc, const void* data, TensorDesc* qDesc, void* qData, F16 *scale);
#endif
