// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_CPU
#define _H_TENSOR_COMPUTING_CPU

#include "uni.h"
#include "sys.h"
#include "tensor_desc.h"
#include "parameter_spec.h"

EE rnn_transform_filter_cpu(const TensorDesc *filterDescs,
    const void **filterArray,
    RNNParamSpec rnnParamSpec,
    TensorDesc *ftmDesc,
    void **ftmArray,
    Arch arch);

EE rnn_transform_filter_bytes_cpu(
    const TensorDesc *filterDesc, RNNParamSpec rnnParamSpec, U32 *bytes);

EE rnncell_infer_forward_tmp_bytes_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    Arch arch);

EE rnn_infer_forward_tmp_bytes_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    Arch arch);

EE rnncell_cpu(TensorDesc xDesc,
    const void *currentX,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    void *state,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    U32 tmpBytes,
    void *tmp,
    TensorDesc hDesc,
    void *currentH,
    Arch arch);

EE rnn_cpu(TensorDesc inputDesc,
    const void *input,
    const TensorDesc *filterDesc,
    const void **filter,
    const TensorDesc *biasDesc,
    const void **bias,
    RNNParamSpec rnnParamSpec,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch);

EE embedding_cpu(TensorDesc inputDesc,
    void *input,
    void *weight,
    EmbedParamSpec p,
    TensorDesc outputDesc,
    void *output);

EE tfslice_infer_output_size_cpu(TensorDesc inputDesc, TfSliceParamSpec p, TensorDesc *outputDesc);

EE tfslice_cpu(
    TensorDesc inputDesc, void *input, TfSliceParamSpec p, TensorDesc outputDesc, void *output);

EE padding_infer_output_size_cpu(
    TensorDesc inputDesc, PadParamSpec padParamSpec, TensorDesc *outputDesc);

EE padding_cpu(TensorDesc inputDesc,
    const void *input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    void *output);

EE reshape_infer_output_size_cpu(TensorDesc inputDesc, ReshapeParamSpec p, TensorDesc *outputDesc);

EE reshape_cpu(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output);

EE depthwise_convolution_transform_filter_bytes_cpu(
    TensorDesc filterDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes);

EE eltwise_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec eltwiseDesc,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch);

EE roialign_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    RoiAlignParamSpec roiAlignParamSpec,
    TensorDesc outputDesc,
    void *output);

EE split_cpu(TensorDesc inputDesc,
    void *input,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output);

EE transpose_cpu(
    TensorDesc inputDesc, const void *input, U32 *dim, TensorDesc outputDesc, void *output);

EE reduction_cpu(TensorDesc inputDesc,
    const void *input,
    TensorDesc maskDesc,
    const void *mask,
    ReductionParamSpec p,
    int tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch);

EE non_max_suppression_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    NonMaxSuppressionParamSpec nonMaxSuppressionParamSpec,
    TensorDesc outputDesc,
    void *output);

EE concat_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    void *inputScale,
    ConcatParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    void *outputScale);

EE l2normalization_cpu(
    TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output, Arch arch);

EE power_cpu(TensorDesc inputDesc,
    void *input,
    PowerParamSpec p,
    TensorDesc outputDesc,
    void *output,
    Arch arch);

EE slice_cpu(TensorDesc inputDesc,
    void *input,
    SliceParamSpec p,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output);

EE priorbox_cpu(std::vector<TensorDesc> inputDesc,
    PriorBoxParamSpec priorBoxParamSpec,
    TensorDesc outputDesc,
    void *output,
    Arch arch);

EE clip_cpu(TensorDesc inputDesc,
    void *input,
    ClipParamSpec p,
    TensorDesc outputDesc,
    void *output,
    Arch arch);

EE detectionoutput_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    DetectionOutputParamSpec detectionOutputParamSpec,
    TensorDesc outputDesc,
    void *output);

EE deconvolution_infer_forward_algorithm_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    Arch arch);

EE deconvolution_transform_filter_bytes_cpu(TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    Arch arch);

EE deconvolution_transform_filter_cpu(TensorDesc filterDesc,
    const void *filter,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc,
    void *filterTransformed,
    Arch arch);

EE deconvolution_infer_forward_tmp_bytes_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    Arch arch);

EE deconvolution_cpu(TensorDesc inputDesc,
    void *input,
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

EE convolution_cpu(TensorDesc inputDesc,
    void *input,
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

EE depthwise_pointwise_convolution_cpu(TensorDesc inputDesc,
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

EE activation_cpu(TensorDesc inputDesc,
    void *input,
    ActivationParamSpec activationDesc,
    TensorDesc outputDesc,
    void *output,
    Arch arch);

EE yolov3detectionoutput_cpu(std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    Yolov3DetectionOutputParamSpec yolov3DetectionOutputParamSpec,
    TensorDesc outputDesc,
    void *output,
    Arch arch);

EE argmax_cpu(
    TensorDesc inputDesc, const void *input, ArgMaxParamSpec p, TensorDesc outputDesc, void *output);

#endif
