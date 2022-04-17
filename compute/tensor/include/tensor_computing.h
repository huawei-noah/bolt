// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING
#define _H_TENSOR_COMPUTING

#include <vector>
#include "sys.h"
#include "parameter_spec.h"
#include "tensor_auxiliary.h"
#ifdef _USE_GPU
#include "gcl.h"
#include "ocl_desc_trans.h"
#define ALIGN(len, align_num) ((len + align_num - 1) / align_num * align_num)
#endif

EE convolution_infer_output_size(Tensor *inputTensor,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    Tensor *outputTensor,
    DataType targetDataType,
    ArchInfo_t archInfo);

EE convolution_infer_forward_algorithm(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    ActivationParamSpec activationDesc,
    ArchInfo_t archInfo);

EE convolution_transform_filter_bytes(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    void *bytes,
    ArchInfo_t archInfo);

EE convolution_transform_filter(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    Tensor tmpTensor,
    Tensor *ftmTensor,
    ArchInfo_t archInfo);

EE convolution_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    ArchInfo_t archInfo);

EE convolution(std::vector<Tensor> inputTensors,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    void *scale,
    Tensor biasTensor,
    std::vector<Tensor> tmpTensors,
    Tensor outputTensor,
    ActivationParamSpec activationDesc,
    ArchInfo_t archInfo);

EE deconvolution_infer_output_size(Tensor *inputTensor,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    Tensor *outputTensor,
    DataType targetDataType,
    ArchInfo_t archInfo);

EE deconvolution_transform_filter_bytes(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    void *bytes,
    ArchInfo_t archInfo);

EE deconvolution_transform_filter(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    Tensor tmpTensor,
    Tensor *ftmTensor,
    ArchInfo_t archInfo);

EE deconvolution_infer_forward_algorithm(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    ActivationParamSpec activationDesc,
    ArchInfo_t archInfo);

EE deconvolution_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    ArchInfo_t archInfo);

EE deconvolution(Tensor inputTensor,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    void *scale,
    Tensor biasTensor,
    Tensor tmpTensor,
    Tensor outputTensor,
    ActivationParamSpec activationDesc,
    ArchInfo_t archInfo);

EE depthwise_pointwise_convolution_infer_output_size(Tensor *inputTensor,
    Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    ConvolutionParamSpec convParamSpec,
    Tensor *outputTensor,
    DataType targetDataType,
    ArchInfo_t archInfo);

EE depthwise_pointwise_convolution_infer_forward_algorithm(Tensor inputTensor,
    Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    DepthwiseConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    ArchInfo_t archInfo);

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    ArchInfo_t archInfo);

EE depthwise_pointwise_convolution_transform_filter_bytes(Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    void *dwBytes,
    void *pwBytes,
    ArchInfo_t archInfo);

EE depthwise_pointwise_convolution_transform_filter(Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    Tensor *dwFtm,
    Tensor *pwFtm,
    ArchInfo_t archInfo);

EE depthwise_pointwise_convolution(std::vector<Tensor> inputTensors,
    Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    void *scale,
    Tensor dwBiasTensor,
    Tensor pwBiasTensor,
    std::vector<Tensor> tmpTensors,
    Tensor outputTensor,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    ArchInfo_t archInfo);

EE depthwise_convolution_infer_output_size(Tensor *inputTensor,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    Tensor *outputTensor,
    DataType targetDataType,
    ArchInfo_t archInfo);

EE depthwise_convolution_infer_forward_algorithm(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    DepthwiseConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    ActivationParamSpec depthwiseActivationParamSpec,
    ArchInfo_t archInfo);

EE depthwise_convolution_transform_filter_bytes(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    void *bytes,
    ArchInfo_t archInfo);

EE depthwise_convolution_transform_filter(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    Tensor *ftmTensor,
    ArchInfo_t archInfo);

EE depthwise_convolution_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    ArchInfo_t archInfo);

EE depthwise_convolution(Tensor inputTensor,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    void *scale,
    Tensor biasTensor,
    Tensor tmpTensor,
    Tensor outputTensor,
    ActivationParamSpec depthwiseActivationParamSpec,
    ArchInfo_t archInfo);

EE detectionoutput_infer_output_size(std::vector<Tensor *> inputTensor,
    DetectionOutputParamSpec detectionOutputParamSpec,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE detectionoutput(std::vector<Tensor> inputTensor,
    DetectionOutputParamSpec detectionOutputParamSpec,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE pooling_infer_output_size(Tensor *inputTensor,
    PoolingParamSpec poolingParamSpec,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE pooling_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE pooling(Tensor inputTensor,
    PoolingParamSpec poolingParamSpec,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE pooling_bp(
    Tensor inputTensor, PoolingParamSpec poolingParamSpec, Tensor outputTensor, ArchInfo_t archInfo);

EE priorbox_infer_output_size(std::vector<Tensor *> inputTensor,
    PriorBoxParamSpec priorBoxParamSpec,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE priorbox(std::vector<Tensor> inputTensor,
    PriorBoxParamSpec priorBoxParamSpec,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE activation_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE activation(
    Tensor inputTensor, ActivationParamSpec activationDesc, Tensor outputTensor, ArchInfo_t archInfo);

EE concat_infer_output_size(
    std::vector<Tensor *> inputTensor, ConcatParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE concat_infer_forward_tmp_bytes(
    std::vector<Tensor> inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE concat(std::vector<Tensor> inputTensor,
    ConcatParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE eltwise_infer_output_size(
    std::vector<Tensor *> inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE eltwise_infer_forward_tmp_bytes(
    std::vector<Tensor> inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE eltwise(std::vector<Tensor> inputTensor,
    EltwiseParamSpec eltwiseDesc,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE split_infer_output_size(Tensor *inputTensor, std::vector<Tensor *> output);

EE split(Tensor inputTensor, std::vector<Tensor> outputTensor, ArchInfo_t archInfo);

EE fully_connected_infer_output_size(
    Tensor *inputTensor, Tensor filterTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE fully_connected_infer_forward_algorithm(
    Tensor inputTensor, Tensor filterTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE fully_connected_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor filterTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE fully_connected_transform_filter_bytes(Tensor filterTensor, void *bytes, ArchInfo_t archInfo);

EE fully_connected_transform_filter(
    Tensor inputTensor, Tensor filterTensor, Tensor *ftmTensor, ArchInfo_t archInfo);

EE fully_connected(Tensor inputTensor,
    Tensor filterTensor,
    Tensor biasTensor,
    std::vector<Tensor> tmpTensors,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE softmax_infer_output_size(
    Tensor *inputTensor, SoftmaxParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE softmax_infer_forward_tmp_bytes(
    Tensor inputTensor, SoftmaxParamSpec p, U32 *bytes, ArchInfo_t archInfo);

EE softmax(Tensor inputTensor,
    SoftmaxParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE logsoftmax(Tensor inputTensor,
    SoftmaxParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE rnn_infer_output_size(std::vector<Tensor *> inputTensor,
    RNNParamSpec rnnParamSpec,
    std::vector<Tensor *> outputTensor,
    ArchInfo_t archInfo);

EE rnn_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    ArchInfo_t archInfo);

EE rnn_transform_filter_bytes(
    std::vector<Tensor> filterTensor, RNNParamSpec rnnParamSpec, void *bytes, ArchInfo_t archInfo);

EE rnn_transform_filter(std::vector<Tensor> filterTensor,
    RNNParamSpec rnnParamSpec,
    Tensor tmpTensor,
    std::vector<Tensor *> ftmTensor,
    ArchInfo_t archInfo);

EE rnn_infer_forward_algorithm(Tensor inputTensor,
    std::vector<Tensor> filterTensors,
    std::vector<Tensor> biasTensors,
    RNNParamSpec rnnParamSpec,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE rnn(std::vector<Tensor> inputTensor,
    std::vector<Tensor> filterTensors,
    std::vector<Tensor> biasTensors,
    RNNParamSpec rnnParamSpec,
    std::vector<Tensor> tmpTensors,
    std::vector<Tensor> outputTensor,
    ArchInfo_t archInfo);

EE rnncell_infer_output_size(std::vector<Tensor *> inputTensor,
    RNNParamSpec rnnParamSpec,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE rnncell_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    ArchInfo_t archInfo);

EE rnncell_transform_filter_bytes(
    std::vector<Tensor> filterTensor, RNNParamSpec rnnParamSpec, void *bytes, ArchInfo_t archInfo);

EE rnncell_transform_filter(std::vector<Tensor> filterTensor,
    RNNParamSpec rnnParamSpec,
    std::vector<Tensor *> ftmTensor,
    ArchInfo_t archInfo);

EE rnncell_infer_forward_algorithm(Tensor xTensor,
    Tensor filterTensor,
    Tensor biasTensor,
    Tensor stateTensor,
    RNNParamSpec rnncellDesc,
    U32 batchStrideX,
    U32 batchStrideH,
    Tensor hTensor,
    ArchInfo_t archInfo);

EE rnncell(Tensor xTensor,
    std::vector<Tensor> filterTensors,
    std::vector<Tensor> biasTensors,
    Tensor stateTensor,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    U32 tmpOffset,
    Tensor tmpTensor,
    Tensor hTensor,
    ArchInfo_t archInfo);

EE scale_infer_output_size(
    Tensor *inputTensor, ScaleParamSpec p, U32 axisLen, Tensor *outputTensor, ArchInfo_t archInfo);

EE scale(Tensor inputTensor,
    void *alpha,
    void *beta,
    ScaleParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE batch_norm_infer_output_size(
    Tensor *inputTensor, BatchNormParamSpec bnParamSpec, Tensor *outputTensor, ArchInfo_t archInfo);

EE batch_norm_transform_filter_bytes(Tensor varianceTensor,
    Tensor meanTensor,
    BatchNormParamSpec bnParamSpec,
    U32 *bytes,
    ArchInfo_t archInfo);

// transform batch norm weight to scale weight
EE batch_norm_transform_filter(Tensor varianceTensor,
    Tensor meanTensor,
    BatchNormParamSpec bnParamSpec,
    Tensor alphaTensor,
    Tensor betaTensor,
    ArchInfo_t archInfo);

EE batch_norm(Tensor inputTensor,
    Tensor alphaTensor,
    Tensor betaTensor,
    BatchNormParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE instance_norm(Tensor inputTensor,
    Tensor tmpTensor,
    Tensor scaleTensor,
    Tensor biasTensor,
    InstanceNormParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE instance_norm_infer_forward_tmp_bytes(
    TensorDesc inputDesc, InstanceNormParamSpec p, U32 *bytes, ArchInfo_t archInfo);

EE prelu_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE prelu(Tensor inputTensor,
    Tensor weightTensor,
    PReLUParamSpec preluDesc,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE normalization_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE normalization_infer_forward_tmp_bytes(Tensor inputTensor, U32 *bytes, ArchInfo_t archInfo);

EE layer_normalization(Tensor inputTensor,
    LayerNormParamSpec p,
    Tensor alphaTensor,
    Tensor betaTensor,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE slice_infer_output_size(
    Tensor *inputTensor, SliceParamSpec p, std::vector<Tensor *> outputTensor, ArchInfo_t archInfo);

EE slice_infer_forward_tmp_bytes(Tensor inputTensor,
    SliceParamSpec p,
    std::vector<Tensor> outputTensor,
    U32 *bytes,
    ArchInfo_t archInfo);

EE slice(Tensor inputTensor,
    SliceParamSpec p,
    Tensor tmpTensor,
    std::vector<Tensor> outputTensor,
    ArchInfo_t archInfo);

EE tfslice_infer_output_size(
    Tensor *inputTensor, TfSliceParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE tfslice_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE tfslice(Tensor inputTensor,
    TfSliceParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);
EE tfslice(Tensor inputTensor, TfSliceParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE transpose_infer_output_size(
    Tensor *inputTensor, TransposeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE transpose_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE transpose(Tensor inputTensor,
    TransposeParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE matmul_infer_output_size(Tensor *matrixATensor,
    bool transposeA,
    Tensor *matrixBTensor,
    bool transposeB,
    Tensor *matrixCTensor,
    ArchInfo_t archInfo);

EE matmul_infer_forward_algorithm(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    Tensor matrixCTensor,
    ArchInfo_t archInfo);

EE matmul_infer_forward_tmp_bytes(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    Tensor matrixCTensor,
    U32 *bytes,
    ArchInfo_t archInfo);

EE matmul(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    Tensor biasTensor,
    std::vector<Tensor> tmpTensors,
    Tensor matirxCTensor,
    ArchInfo_t archInfo);

EE reshape_infer_output_size(
    Tensor *inputTensor, ReshapeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE reshape_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE reshape(Tensor inputTensor, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE attention_infer_output_size(Tensor *inputTensor, AttentionParamSpec p, Tensor *outputTensor);

EE attention(Tensor inputTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE power_infer_output_size(
    Tensor *inputTensor, PowerParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE power(Tensor inputTensor, PowerParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE clip_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE clip(Tensor inputTensor, ClipParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE bilateral_slice_apply_infer_output_size(Tensor *inputTensor,
    Tensor *guideTensor,
    Tensor *gridTensor,
    BilateralSliceApplyParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE bilateral_slice_apply_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor guideTensor,
    Tensor gridTensor,
    BilateralSliceApplyParamSpec p,
    U32 *bytes,
    ArchInfo_t archInfo);

EE bilateral_slice_apply(Tensor inputTensor,
    Tensor guideTensor,
    Tensor gridTensor,
    BilateralSliceApplyParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE argmax_infer_output_size(
    Tensor *inputTensor, ArgMaxParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE argmax_infer_forward_tmp_bytes(
    Tensor inputTensor, ArgMaxParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE argmax(
    Tensor inputTensor, ArgMaxParamSpec p, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE reduction_infer_output_size(Tensor *inputTensor,
    Tensor maskTensor,
    ReductionParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE reduction_infer_forward_tmp_bytes(
    Tensor inputTensor, ReductionParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE reduction(Tensor inputTensor,
    Tensor maskTensor,
    ReductionParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE check_infer_output_size(
    std::vector<Tensor *> inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE check(Tensor inputTensorA,
    Tensor inputTensorB,
    CheckParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE squeeze_infer_output_size(
    Tensor *inputTensor, SqueezeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE squeeze_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE squeeze(Tensor inputTensor, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE unsqueeze_infer_output_size(
    Tensor *inputTensor, UnsqueezeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE unsqueeze_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE unsqueeze(Tensor inputTensor, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE space2depth_infer_output_size(Tensor *inputTensor,
    Space2DepthParamSpec space2DepthPara,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE space2depth(Tensor inputTensor,
    Space2DepthParamSpec space2DepthPara,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE depth2space_infer_output_size(
    Tensor *inputTensor, Depth2SpaceParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE depth2space_infer_forward_tmp_bytes(
    Tensor inputTensor, Depth2SpaceParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE depth2space(Tensor inputTensor,
    Depth2SpaceParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE attention_mask(
    Tensor inputTensor, AttentionMaskParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE attention_mask_infer_output_size(Tensor *inputTensor, Tensor *outputTensor);

EE padding_infer_output_size(
    Tensor *inputTensor, PadParamSpec padParamSpec, Tensor *outputTensor, ArchInfo_t archInfo);

EE padding(Tensor inputTensor, PadParamSpec padParamSpec, Tensor outputTensor, ArchInfo_t archInfo);

EE embedding_infer_output_size(Tensor *inputTensor,
    EmbedParamSpec p,
    DataType outputDt,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE embedding(Tensor inputTensor,
    Tensor weightTensor,
    EmbedParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE yolov3detectionoutput_infer_output_size(std::vector<Tensor *> inputTensor,
    Yolov3DetectionOutputParamSpec yolov3DetectionOutputParamSpec,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE yolov3detectionoutput(std::vector<Tensor> inputTensor,
    Yolov3DetectionOutputParamSpec yolov3DetectionOutputParamSpec,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE preallocated_memory_infer_output_size(std::vector<Tensor *> inputTensors,
    PreAllocatedMemoryParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE preallocated_memory(PreAllocatedMemoryParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE copy_infer_output_size(std::vector<Tensor *> inputTensor, ArchInfo_t archInfo);

EE copy(std::vector<Tensor> inputTensor,
    U32 srcOffset,
    U32 dstOffset,
    U32 srcStride,
    U32 dstStride,
    U32 length,
    ArchInfo_t archInfo);

EE non_max_suppression_infer_output_size(std::vector<Tensor *> inputTensor,
    NonMaxSuppressionParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE non_max_suppression(std::vector<Tensor> inputTensor,
    NonMaxSuppressionParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE roialign_infer_output_size(std::vector<Tensor *> inputTensor,
    RoIAlignParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE roialign_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE roialign(std::vector<Tensor> inputTensor,
    RoIAlignParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);
/*
EE multihead_attention_infer_output_size(Tensor *inputTensor,
    std::vector<Tensor> filterTensor,
    Tensor *outputTensor,
    U32 *firstFCSliceNum,
    ArchInfo_t archInfo);

EE multihead_attention_infer_forward_algorithm(Tensor inputTensor,
    std::vector<Tensor> filterTensor,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE multihead_attention_infer_forward_tmp_bytes(Tensor inputTensor,
    std::vector<Tensor> filterTensor,
    std::vector<bool> eltwiseWithLayerNormIn,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    U32 *bytes,
    ArchInfo_t archInfo);

EE multihead_attention_transform_filter_bytes(
    std::vector<Tensor> filterTensor, U32 *bytes, ArchInfo_t archInfo);

EE multihead_attention_transform_filter(
    std::vector<Tensor> filterTensor, std::vector<Tensor *> ftmTensor, ArchInfo_t archInfo);

EE multihead_attention(Tensor inputTensor,
    std::vector<Tensor> filterTensor,
    std::vector<Tensor> biasTensor,
    std::vector<Tensor> layerNormAlphaTensor,
    std::vector<Tensor> layerNormBetaTensor,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);*/

EE channel_resize_infer_output_size(
    Tensor *inputTensor, ChannelResizeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE channel_resize(
    Tensor inputTensor, ChannelResizeParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE l2normalization_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE l2normalization(Tensor inputTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE tile_infer_output_size(
    Tensor *inputTensor, TileParamSpec tileParamSpec, Tensor *outputTensor, ArchInfo_t archInfo);

EE tile_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE tile(Tensor inputTensor,
    TileParamSpec tileParamSpec,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE where_infer_output_size(
    Tensor *xTensor, Tensor *yTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE where(
    Tensor conditionTensor, Tensor xTensor, Tensor yTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE cast_infer_output_size(
    Tensor *inputTensor, CastParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE cast(Tensor inputTensor, CastParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE quantize(Tensor inputTensor, Tensor *outputTensor, F32 *scale, ArchInfo_t archInfo);

EE expand_infer_output_size(
    Tensor *inputTensor, ExpandParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE expand_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE expand(
    Tensor inputTensor, ExpandParamSpec p, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo);

EE topk_infer_output_size(Tensor *inputTensor,
    TopKParamSpec p,
    Tensor *outputTensor,
    Tensor *outputIndicesTensor,
    ArchInfo_t archInfo);

EE topk_infer_forward_tmp_bytes(
    Tensor inputTensor, TopKParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo);

EE topk(Tensor inputTensor,
    TopKParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    Tensor outputIndicesTensor,
    ArchInfo_t archInfo);

EE cast_infer_output_size(
    Tensor *inputTensor, CastParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE cast(Tensor inputTensor, CastParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE dequantize(Tensor input, const F32 *scale, Tensor bias, Tensor output, ArchInfo_t archInfo);

std::vector<F32> compress_histogram(std::vector<F32> &histogram, F32 numPerBin, F32 last_max);

std::vector<F32> compute_scale_with_KL(std::vector<F32> &histogram, F32 interval);

EE scatter_infer_output_size(Tensor *dataTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE scatter_infer_forward_tmp_bytes(
    Tensor dataTensor, Tensor updateTensor, U32 *bytes, ArchInfo_t archInfo);

EE scatter(Tensor dataTensor,
    Tensor indexTensor,
    Tensor updateTensor,
    ScatterParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE gather_infer_output_size(Tensor *dataTensor,
    Tensor *indexTensor,
    GatherParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE gather_infer_forward_tmp_bytes(Tensor dataTensor,
    Tensor indexTensor,
    GatherParamSpec p,
    Tensor outputTensor,
    U32 *bytes,
    ArchInfo_t archInfo);

EE gather(Tensor dataTensor,
    Tensor indexTensor,
    GatherParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE select_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE select(Tensor boolChoice, Tensor a, Tensor b, Tensor outputTensor, ArchInfo_t archInfo);

EE gat_infer_output_size(
    Tensor *nodeFeatureTensor, GATParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo);

EE gat_infer_forward_tmp_bytes(Tensor nodeFeatureTensor,
    Tensor edgeFeatureTensor,
    GATParamSpec p,
    U32 *bytes,
    ArchInfo_t archInfo);

EE gat(Tensor nodeFeature0,
    Tensor node0,
    Tensor nodeFeature1,
    Tensor node1,
    Tensor edgeFeature,
    GATParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE generate_proposals_infer_output_size(Tensor *deltaTensor,
    Tensor *logitTensor,
    GenerateProposalsParamSpec generateProposalsParam,
    Tensor *outputTensor,
    ArchInfo_t archInfo);

EE generate_proposals_infer_forward_tmp_bytes(Tensor deltaTensor,
    Tensor logitTensor,
    GenerateProposalsParamSpec generateProposalsParam,
    U32 *bytes,
    ArchInfo_t archInfo);

EE generate_proposals(Tensor deltaTensor,
    Tensor logitTensor,
    Tensor imgInfoTensor,
    Tensor anchorTensor,
    GenerateProposalsParamSpec generateProposalsParam,
    std::vector<Tensor> tmpTensors,
    Tensor outputTensor,
    ArchInfo_t archInfo);

EE onehot_infer_output_size(
    Tensor *inputTensor, OneHotParamSpec p, DataType type, Tensor *outputTensor, ArchInfo_t archInfo);

EE onehot(Tensor inputTensor, OneHotParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE cumsum_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo);

EE cumsum(Tensor inputTensor, CumSumParamSpec p, Tensor outputTensor, ArchInfo_t archInfo);

EE non_zero(Tensor inputTensor, Tensor outputTensor, ArchInfo_t archInfo);
#endif
