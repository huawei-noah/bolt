// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_TENSOR_COMPUTING_MALI
#define _H_TENSOR_COMPUTING_MALI

#include "tensor_desc.h"
#include "parameter_spec.h"
#include "gcl.h"
#include "ocl_desc_trans.h"
#define ALIGN(len, align_num) ((len + align_num - 1) / align_num * align_num)

EE pooling_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo);

EE pooling_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PoolingParamSpec poolingParamSpec,
    const void *scale,
    GCLMem_t temp,
    TensorDesc outputDesc,
    GCLMem_t output);

EE pooling_infer_output_size_mali(TensorDesc inputDesc,
    PoolingParamSpec poolingParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE padding_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    PadParamSpec padParamSpec,
    TensorDesc outputDesc,
    GCLMem_t output);

EE padding_infer_output_size_mali(TensorDesc inputDesc,
    PadParamSpec padParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE convolution_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE convolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc outputDesc,
    GCLMemDesc inputMemDesc,
    GCLMemDesc outputMemDesc,
    ConvolutionPolicy policy,
    ActivationMode activationMode,
    ForwardRunInfoMali_t forwardRunInfo);

EE convolution_transform_filter_bytes_mali(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes);

EE convolution_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMem_t tmp,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem);

EE convolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes);

EE convolution_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc scaleDesc,
    const GCLMem_t scale,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode);

EE depthwise_pointwise_convolution_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE depthwise_pointwise_convolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    ForwardRunInfoMali_t forwardRunInfo);

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes);

EE depthwise_pointwise_convolution_transform_filter_bytes_mali(TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemDwFilterDesc,
    GCLMemDesc_t gclmemPwFilterDesc,
    U32 *bytes);

EE depthwise_pointwise_convolution_transform_filter_mali(GCLHandle_t handle,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    GCLMem_t dwFilter,
    GCLMem_t pwFilter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *dwfltmemDesc,
    TensorDesc *pwfltmemDesc,
    GCLMem_t dwfltmem,
    GCLMem_t pwfltmem);

EE depthwise_pointwise_convolution_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    const GCLMem_t dwFilter,
    const GCLMem_t pwFilter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc dwBiasDesc,
    TensorDesc pwBiasDesc,
    const GCLMem_t dwBias,
    const GCLMem_t pwBias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode);

EE depthwise_convolution_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE depthwise_convolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ActivationMode depthwiseActivationMode,
    ForwardRunInfoMali_t forwardRunInfo);

EE depthwise_convolution_transform_filter_bytes_mali(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes);

EE depthwise_convolution_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem);

EE depthwise_convolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes);

EE depthwise_convolution_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode);

EE deconvolution_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE deconvolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc outputDesc,
    ConvolutionPolicy policy,
    ActivationMode activationMode,
    ForwardRunInfoMali_t forwardRunInfo);

EE deconvolution_transform_filter_bytes_mali(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes);

EE deconvolution_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMem_t tmp,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem);

EE deconvolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes);

EE deconvolution_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc scaleDesc,
    const GCLMem_t scale,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode);

EE bilateral_slice_apply_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc guideDesc,
    TensorDesc gridDesc,
    BilateralSliceApplyParamSpec bilateralSliceApplyParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemGuideDesc,
    GCLMemDesc_t gclmemGridDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE bilateral_slice_apply_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc guideDesc,
    TensorDesc gridDesc,
    BilateralSliceApplyParamSpec bilateralSliceApplyParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes);

EE bilateral_slice_apply_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc guideDesc,
    const GCLMem_t guide,
    TensorDesc gridDesc,
    const GCLMem_t grid,
    BilateralSliceApplyParamSpec bilateralSliceApplyParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE eltwise_infer_output_size_mali(std::vector<TensorDesc> inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE eltwise_infer_forward_tmp_bytes_mali(
    std::vector<TensorDesc> inputDesc, std::vector<GCLMemDesc> gclmemInputDesc, U32 *bytes);

EE eltwise_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    EltwiseParamSpec eltwiseDesc,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE softmax_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE softmax_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SoftmaxParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output);

EE softmax_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo);

EE activation_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE activation_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode activationMode);

EE fully_connected_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE fully_connected_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    GCLMemDesc inputMemDesc,
    GCLMemDesc outputMemDesc,
    ForwardRunInfoMali_t forwardRunInfo);

EE fully_connected_transform_filter_bytes_mali(TensorDesc filterDesc,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE fully_connected_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo);

EE fully_connected_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, TensorDesc filterDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo);

EE fully_connected_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo);

EE scale_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE scale_mali(GCLHandle_t handle,
    GCLMem_t alpha,
    GCLMem_t beta,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc outputDesc,
    GCLMem_t output);

EE prelu_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE prelu_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t weight,
    PReLUParamSpec preluDesc,
    TensorDesc outputDesc,
    GCLMem_t output);

EE concat_infer_output_size_mali(std::vector<TensorDesc> inputDesc,
    ConcatParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE concat_infer_forward_tmp_bytes_mali(std::vector<TensorDesc> inputDesc, 
    std::vector<GCLMemDesc> gclmemInputDesc,
    U32 *bytes);

EE concat_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    GCLMem_t inputScale,
    ConcatParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    GCLMem_t outputScale);

EE clip_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE clip_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ClipParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output);

EE squeeze_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE squeeze_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes);

EE squeeze_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE unsqueeze_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE unsqueeze_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes);

EE unsqueeze_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE reshape_infer_output_size_mali(TensorDesc inputDesc,
    ReshapeParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE reshape_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    U32 *bytes);

EE reshape_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE space2depth_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE space2depth_mali(
    GCLHandle_t handle, TensorDesc inputDesc, GCLMem_t input, TensorDesc outputDesc, GCLMem_t output);

EE depth2space_infer_output_size_mali(TensorDesc inputDesc,
    Depth2SpaceParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE depth2space_infer_tmpBuf_size_mali(
    TensorDesc inputDesc, Depth2SpaceParamSpec p, TensorDesc outputDesc, U32 *bytes);

EE depth2space_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    Depth2SpaceParamSpec p,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE embedding_infer_output_size_mali(TensorDesc inputDesc,
    EmbedParamSpec p,
    DataType odt,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE embedding_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc weightDesc,
    GCLMem_t weight,
    EmbedParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output);

EE normalization_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE normalization_infer_forward_tmp_bytes_mali(GCLMemDesc gclmemInputDesc, U32 *bytes);

EE layer_normalization_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    GCLMem_t alpha,
    GCLMem_t beta,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE matmul_infer_output_size_mali(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc *matrixCDesc,
    GCLMemDesc_t gclmemMatrixADesc,
    GCLMemDesc_t gclmemMatrixBDesc,
    GCLMemDesc_t gclmemMatrixCDesc);

EE matmul_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool TransposeA,
    TensorDesc matrixBDesc,
    bool TransposeB,
    TensorDesc matrixCDesc,
    GCLMemDesc gclmemMatrixADesc,
    GCLMemDesc gclmemMatrixBDesc,
    GCLMemDesc gclmemMatrixCDesc,
    ForwardRunInfoMali_t forwardRunInfo);

EE matmul_infer_forward_tmp_bytes_mali(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE matmul_mali(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    const GCLMem_t matrixA,
    TensorDesc matrixBDesc,
    bool transposeB,
    const GCLMem_t matrixB,
    GCLMem_t tmp,
    TensorDesc matrixCDesc,
    GCLMem_t matrixC,
    ForwardRunInfoMali_t forwardRunInfo);

EE power_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE power_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    PowerParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output);

EE transpose_infer_output_size_mali(TensorDesc inputDesc,
    TransposeParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE transpose_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    U32 *bytes);

EE transpose_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TransposeParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE slice_infer_output_size_mali(TensorDesc inputDesc,
    SliceParamSpec p,
    std::vector<TensorDesc> *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE slice_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
    SliceParamSpec p,
    std::vector<TensorDesc> outputDesc,
    U32 *bytes);

EE slice_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    SliceParamSpec p,
    GCLMem_t tmpbuf,
    std::vector<TensorDesc> outputDesc,
    std::vector<void *> *output);

EE rnncell_infer_output_size_mali(TensorDesc inputDesc,
    RNNParamSpec rnnParamSpec,
    TensorDesc outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemStateDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE rnncell_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc xDesc,
    TensorDesc filterDesc,
    TensorDesc biasDesc,
    RNNParamSpec rnnPara,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    GCLMemDesc inputMemDesc,
    GCLMemDesc stateMemDesc,
    GCLMemDesc outputMemDesc,
    ForwardRunInfoMali_t forwardRunInfo);

EE rnn_transform_filter_bytes_mali(TensorDesc filterDesc,
    RNNParamSpec rnnParamSpec,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE rnn_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    RNNParamSpec rnnParamSpec,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo);

EE rnncell_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnncellDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE rnncell_mali(GCLHandle_t handle,
    TensorDesc xDesc,
    const GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    GCLMem_t state,
    RNNParamSpec rnncellDesc,
    U32 batchStrideX,
    U32 batchStrideH,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc hDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo);

EE argmax_infer_output_size_mali(TensorDesc inputDesc,
    ArgMaxParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE argmax_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, ArgMaxParamSpec p, TensorDesc outputDesc, U32 *bytes);

EE argmax_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ArgMaxParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE preallocated_memory_infer_output_size_mali(TensorDesc *outputDesc, GCLMemDesc_t gclmemOutputDesc);

EE preallocated_memory_mali(GCLHandle_t handle, TensorDesc outputDesc, GCLMem_t output);

EE copy_infer_output_size_mali(std::vector<TensorDesc> inputDesc, GCLMemDesc_t gclmemInputDesc);

EE copy_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDesc,
    std::vector<void *> input,
    U32 srcOffset,
    U32 dstOffset,
    U32 srcStride,
    U32 dstStride,
    U32 length);

EE check_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputADesc,
    GCLMemDesc_t gclmemInputBDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE check_mali(GCLHandle_t handle,
    TensorDesc inputDescA,
    GCLMem_t inputA,
    TensorDesc inputDescB,
    GCLMem_t inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output);

EE multihead_attention_infer_output_size_mali(TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    TensorDesc *outputDesc,
    U32 *firstFCSliceNum,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    ForwardRunInfoMali_t forwardRunInfo);

EE multihead_attention_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    TensorDesc outputDesc,
    ForwardRunInfoMali_t forwardRunInfo);

EE multihead_attention_transform_filter_bytes_mali(std::vector<TensorDesc> filterDesc,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE multihead_attention_transform_filter_mali(GCLHandle_t handle,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> *fltmemDesc,
    std::vector<void *> fltmem,
    ForwardRunInfoMali_t forwardRunInfo);

EE multihead_attention_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    std::vector<bool> eltwiseWithLayerNormIn,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo);

EE multihead_attention_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> biasDesc,
    std::vector<void *> bias,
    std::vector<void *> layerNormAlpha,
    std::vector<void *> layerNormBeta,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo);

EE channel_resize_infer_output_size_mali(TensorDesc inputDesc,
    ChannelResizeParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE channel_resize_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    ChannelResizeParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output);

EE reduction_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc maskDesc,
    ReductionParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE reduction_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    ReductionParamSpec p,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes);

EE reduction_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TensorDesc maskDesc,
    GCLMem_t mask,
    ReductionParamSpec p,
    GCLMem_t tmp,
    TensorDesc outputDesc,
    GCLMem_t output);

EE topk_infer_output_size_mali(TensorDesc inputDesc,
    TopKParamSpec p,
    TensorDesc *outputDesc,
    TensorDesc *outputDescIndices,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    GCLMemDesc_t gclmemOutputIndicesDesc);

EE topk_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, TopKParamSpec p, TensorDesc outputDesc, U32 *bytes);

EE topk_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TopKParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    TensorDesc outputIndicesDesc,
    GCLMem_t outputIndices);

EE tfslice_infer_output_size_mali(TensorDesc inputDesc,
    TfSliceParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE tfslice_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    GCLMemDesc gclmemOutputDesc,
    U32 *bytes);

EE tfslice_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    TfSliceParamSpec p,
    GCLMem_t tmpbuf,
    TensorDesc outputDesc,
    GCLMem_t output);

EE cast_infer_output_size_mali(TensorDesc inputDesc,
    CastParamSpec p,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc);

EE cast_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    GCLMem_t input,
    CastParamSpec p,
    TensorDesc outputDesc,
    GCLMem_t output);
#endif
