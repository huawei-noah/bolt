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

EE activation_arm(TensorDesc inputDesc, void* input, ActivationDesc activationDesc, TensorDesc outputDesc, void* output);

EE attention_arm(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output);

EE clip_arm(void *minValue, void *maxValue, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output);

EE concat_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input, void* inputScale,
    TensorDesc outputDesc, void* output, void* outputScale, int axis);

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
    ActivationDesc activationDesc,
    Arch arch);

EE deconvolution_infer_forward_algorithm_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType);

EE deconvolution_transform_filter_bytes_arm(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes);

EE deconvolution_transform_filter_arm(TensorDesc filterDesc, const void* filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed);

EE deconvolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE deconvolution_arm(TensorDesc inputDesc, void* input,
    TensorDesc filterDesc, const void* filter,
    ConvolutionDesc convDesc,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc scaleDesc, const void* scale,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationDesc activationDesc,
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
    ActivationDesc depthwiseActivationDesc,
    ActivationDesc pointwiseActivationDesc,
    Arch arch);

EE detectionoutput_qsort_descent_arm(std::vector<BoxRect>& boxes, std::vector<F32>& scores, int left, int right);

F32 detectionoutput_intersectionarea_arm(BoxRect a, BoxRect b);

EE detectionoutput_nms_pickedboxes_arm(std::vector<BoxRect> boxes, std::vector<I64>& picked, F32 nms_threshold);

EE detectionoutput_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input, DetectionOutputDesc detectionoutputDesc, TensorDesc outputDesc, void* output);

EE eltwise_arm(std::vector<TensorDesc> inputDesc, std::vector<void*> input,
    TensorDesc outputDesc, void* output, EltwiseMode eltwiseMode);

EE lstm_transform_filter_arm(TensorDesc filterDesc, const void* filter, LSTMDesc lstmDesc, TensorDesc *ftmDesc, void* ftm);

EE lstm_transform_filter_bytes_arm(TensorDesc filterDesc, LSTMDesc lstmDesc, U32* bytes);

EE lstm_arm(TensorDesc inputDesc, const void* input,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    LSTMDesc lstmDesc,
    TensorDesc outputDesc, void* output,
    Arch arch);

EE lstmcell_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes, Arch arch);

EE lstmcell_arm(TensorDesc xDesc, const void* currentX,
    TensorDesc filterDesc, const void* filter,
    TensorDesc biasDesc, const void* bias,
    void *state,
    U32 tmpBytes, void *tmp,
    LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
    TensorDesc hDesc, void* currentH,
    Arch arch);

EE lstm_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc, LSTMDesc lstmDesc, U32 *bytes, Arch arch);

EE multiply_arm(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output);

EE layer_normalization_arm(void *alpha, void *beta,
    TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output);

EE pooling_arm(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, const void* scale, TensorDesc outputDesc, void* output);

EE priorbox_arm(std::vector<TensorDesc> inputDesc, PriorBoxDesc priorboxDesc, TensorDesc outputDesc, void* output);

EE reshape_arm(TensorDesc inputDesc, void* input,
    TensorDesc outputDesc, void* output);

EE scale_arm(TensorDesc inputDesc, void* input, I32 axis, void *alpha, void *beta, TensorDesc outputDesc, void* output);

EE slice_arm(TensorDesc inputDesc, void* input, int axis,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output);

EE softmax_arm(TensorDesc inputDesc, const void* input,
    int axis,
    TensorDesc outputDesc, void* output);

EE split_arm(TensorDesc inputDesc, void* input,
    std::vector<TensorDesc> outputDesc, std::vector<void*>* output);

EE transpose_arm(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output, U32 *dim);

EE quantize_tensor_arm(TensorDesc dDesc, const void* data, TensorDesc* qDesc, void* qData, void *scale);

EE argmax_arm(TensorDesc inputDesc, const void* input,
    I32 axis,
    TensorDesc outputDesc, void* output);

EE reduction_arm(TensorDesc inputDesc, const void* input,
    TensorDesc maskDesc, const void* mask,
    I32 axis,
    ReductionMode reductionMode,
    float coeff,
    TensorDesc outputDesc, void* output);

EE check_arm(TensorDesc inputDescA, const void* inputA,
    TensorDesc inputDescB, const void* inputB,
    CheckMode checkMode,
    TensorDesc outputDesc, void* output);

EE attention_mask_arm(TensorDesc inputDesc, const void* input,
    I32 attentionLength, bool sameLength, float mask,
    TensorDesc outputDesc, void* output);

EE padding_arm(TensorDesc inputDesc, const void* input, PadDesc padDesc, TensorDesc outputDesc, void* output);
#endif
