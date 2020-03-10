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

#include <tensor_desc.h>
#include <vector>
#include "sys.h"
#include "tensor_computing_type.h"

#ifdef __cplusplus
extern "C" {
#endif
    EE convolution_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
        TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes, Arch arch, ExtInfo_t extInfo = NULL);

    EE convolution_infer_forward_algorithm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
        ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType,
        ActivationMode activationMode, Arch arch, ExtInfo_t extInfo = NULL);

    EE convolution_transform_filter_bytes(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes, Arch arch, ExtInfo_t extInfo = NULL);

    EE convolution_transform_filter(TensorDesc filterDesc, const void* filter, ConvolutionForwardAlgorithm algorithm,
        TensorDesc *ftmDesc, void* filterTransformed, Arch arch, ExtInfo_t extInfo = NULL);

    EE convolution_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
        ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes, Arch arch, ExtInfo_t extInfo = NULL);

    EE convolution(TensorDesc inputDesc, void* input,
            TensorDesc filterDesc, const void* filter,
            ConvolutionDesc convDesc,
            ConvolutionForwardAlgorithm algorithm,
            TensorDesc scaleDesc, const void* scale,
            TensorDesc biasDesc, const void* bias,
            U32 tmpBytes, void* tmp,
            TensorDesc outputDesc, void* output,
            ActivationMode activationMode,
            Arch arch, ExtInfo_t extInfo = NULL);

    EE deconvolution_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
        TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes);

    EE deconvolution_infer_forward_algorithm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
        ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType, Arch arch);

    EE deconvolution_transform_filter_bytes(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes, Arch arch);

    EE deconvolution_transform_filter(TensorDesc filterDesc, const void* filter, ConvolutionForwardAlgorithm algorithm,
        TensorDesc *ftmDesc, void* filterTransformed, Arch arch);

    EE deconvolution_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
        ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes, Arch arch);

    EE deconvolution(TensorDesc inputDesc, void* input,
            TensorDesc filterDesc, const void* filter,
            ConvolutionDesc convDesc,
            ConvolutionForwardAlgorithm algorithm,
            TensorDesc scaleDesc, const void* scale,
            TensorDesc biasDesc, const void* bias,
            U32 tmpBytes, void* tmp,
            TensorDesc outputDesc, void* output,
            ActivationMode activationMode,
            Arch arch);

    EE deconvolution_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
        TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes);

    EE deconvolution_infer_forward_algorithm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
        ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType, Arch arch);

    EE deconvolution_transform_filter_bytes(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes, Arch arch);

    EE deconvolution_transform_filter(TensorDesc filterDesc, const void* filter, ConvolutionForwardAlgorithm algorithm,
        TensorDesc *ftmDesc, void* filterTransformed, Arch arch);

    EE deconvolution_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
        ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes, Arch arch);

    EE deconvolution(TensorDesc inputDesc, void* input,
            TensorDesc filterDesc, const void* filter,
            ConvolutionDesc convDesc,
            ConvolutionForwardAlgorithm algorithm,
            TensorDesc scaleDesc, const void* scale,
            TensorDesc biasDesc, const void* bias,
            U32 tmpBytes, void* tmp,
            TensorDesc outputDesc, void* output,
            ActivationMode activationMode,
            Arch arch);

    EE depthwise_convolution_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
        TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes, Arch arch, ExtInfo_t extInfo = NULL);
    
    EE depthwise_convolution_infer_forward_algorithm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
        ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm, DataType targetDataType, 
        ActivationMode depthwiseActivationMode, ActivationMode pointwiseActivationMode, Arch arch, ExtInfo_t extInfo = NULL);
    
    EE depthwise_convolution_transform_filter_bytes(TensorDesc filterDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32* bytes, Arch arch, ExtInfo_t extInfo = NULL);
    
    EE depthwise_convolution_transform_filter(TensorDesc filterDesc, const void* filter, DepthwiseConvolutionForwardAlgorithm algorithm,
        TensorDesc *ftmDesc, void* filterTransformed, Arch arch, ExtInfo_t extInfo = NULL);
    
    EE depthwise_convolution_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
        ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes, Arch arch, ExtInfo_t extInfo = NULL);
    
    EE depthwise_convolution(TensorDesc inputDesc, void* input,
            TensorDesc filterDesc, const void* filter,
            ConvolutionDesc convDesc,
            DepthwiseConvolutionForwardAlgorithm algorithm,
            TensorDesc biasDesc, const void* bias,
            U32 tmpBytes, void* tmp,
            TensorDesc outputDesc, void* output,
            ActivationMode depthwiseActivationMode,
            ActivationMode pointwiseActivationMode,
            Arch arch, ExtInfo_t extInfo = NULL);

    EE pooling_infer_output_size(TensorDesc inputDesc, PoolingDesc poolingDesc, TensorDesc *outputDesc, Arch arch, ExtInfo_t extInfo = NULL);

    EE pooling(TensorDesc inputDesc, const void* input, PoolingDesc poolingDesc, const void* scale, TensorDesc outputDesc, void* output, Arch arch, ExtInfo_t extInfo = NULL);

    EE activation_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc, Arch arch, ExtInfo_t extInfo = NULL);

    EE activation(TensorDesc inputDesc, void* data, ActivationMode mode, Arch arch, ExtInfo_t extInfo = NULL);

    EE concat_infer_output_size(std::vector<TensorDesc> inputDesc, TensorDesc* outputDesc, U32 concatDim, Arch arch, ExtInfo_t extInfo = NULL);

    EE concat(std::vector<TensorDesc> inputDesc, std::vector<void*> input, void* inputScale,
            TensorDesc outputDesc, void* output, void* outputScale, U32 concatDim, Arch arch, ExtInfo_t extInfo = NULL);

    EE eltwise(std::vector<TensorDesc> inputDesc, std::vector<void*> input, TensorDesc outputDesc, void* output, EltwiseMode eltwiseMode, Arch arch, ExtInfo_t extInfo = NULL);

    EE eltwise_infer_output_size(std::vector<TensorDesc> inputDesc, TensorDesc* outputDesc, Arch arch, ExtInfo_t extInfo = NULL);

    EE split(TensorDesc inputDesc, void* input, std::vector<TensorDesc> outputDesc, std::vector<void*>* output, Arch arch);

    EE split_infer_output_size(TensorDesc inputDesc, std::vector<TensorDesc>* outputDesc);

    EE fully_connected_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc *outputDesc, Arch arch, ExtInfo_t extInfo = NULL);

    EE fully_connected_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, U32 *bytes, Arch arch);

    EE fully_connected_transform_filter_bytes(TensorDesc filterDesc, U32* bytes, Arch arch, ExtInfo_t extInfo = NULL);

    EE fully_connected_transform_filter(TensorDesc inputDesc, TensorDesc filterDesc, const void* filter,
            TensorDesc *ftmDesc, void* filterTransformed, Arch arch, ExtInfo_t extInfo = NULL);

    EE fully_connected(TensorDesc inputDesc, const void* input, TensorDesc weightDesc, const void* weight, void* tmp, U32 bytes,
            TensorDesc outputDesc, void* output, TensorDesc biasDesc, const void* bias, Arch arch, ExtInfo_t extInfo = NULL);

    EE softmax_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc, Arch arch, ExtInfo_t extInfo = NULL);

    EE softmax(TensorDesc inputDesc, const void* input, TensorDesc outputDesc, void* output, Arch arch, ExtInfo_t extInfo = NULL);

    EE lstm_transform_filter(TensorDesc filterDesc,
            const void* filter,
            TensorDesc *ftmDesc,
            void* filterTransformed,
            U32 x_dim, U32 h_dim, Arch arch);

    EE lstm_transform_filter_bytes(TensorDesc filterDesc, U32* bytes, Arch arch);


    EE lstm_infer_output_size(TensorDesc inputDesc,
            TensorDesc filterDesc,
            LSTMDesc lstmDesc,
            TensorDesc* outputDesc,
            U32* outputBytes);

    EE lstm_infer_forward_tmp_bytes(TensorDesc inputDesc,
            TensorDesc filterDesc,
            TensorDesc outputDesc,
            LSTMDesc lstmDesc,
            U32 *bytes, Arch arch);

    EE lstm(TensorDesc inputDesc, const void* input,
        TensorDesc filterDesc, const void* filter,
        TensorDesc biasDesc, const void* bias,
        U32 tmpBytes, void* tmp,
        LSTMDesc lstmDesc,
        TensorDesc outputDesc, void* output,
        Arch arch);

    EE lstmcell_infer_output_size(TensorDesc inputDesc,
            TensorDesc filterDesc,
            LSTMDesc lstmDesc,
            TensorDesc* outputDesc,
            U32* outputBytes);

    EE lstmcell_infer_forward_tmp_bytes(TensorDesc inputDesc,
            TensorDesc filterDesc,
            TensorDesc outputDesc,
            LSTMDesc lstmDesc,
            U32 *bytes, Arch arch);

    EE lstmcell(TensorDesc xDesc, const void* currentX,
        TensorDesc filterDesc, const void* filter,
        TensorDesc biasDesc, const void* bias,
        void *state,
        U32 tmpBytes, void *tmp,
        LSTMDesc lstmDesc, U32 batchStrideX, U32 batchStrideH,
        TensorDesc hDesc, void* currentH,
        Arch arch);

    EE scale(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void* output, Arch arch, ExtInfo_t extInfo = NULL);
    
    EE scale_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc, Arch arch, ExtInfo_t extInfo = NULL);

    EE normalization_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc);

    EE layer_normalization(void *alpha, void *beta,
                                TensorDesc inputDesc, void* input,
                                TensorDesc outputDesc, void* output, Arch arch);

    EE slice_infer_output_size(TensorDesc inputDesc, std::vector<TensorDesc>* outputDesc, U32 axis, U32 *slice_point);

    EE slice(TensorDesc inputDesc, void* input, std::vector<TensorDesc> outputDesc, std::vector<void*>* output, Arch arch);

    EE transpose(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output, U32 *dim, Arch arch);

    EE transpose_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc, U32 *dim);

    EE matmul_infer_output_size(TensorDesc matrixADesc, bool transposeA, TensorDesc matrixBDesc, bool transposeB, TensorDesc *matrixCDesc);

    EE matmul_infer_forward_tmp_bytes(TensorDesc matrixADesc, bool transposeA, TensorDesc matrixBDesc, bool transposeB, U32 *bytes, Arch arch);

    EE matmul(TensorDesc matrixADesc, bool transposeA, const void* matrixA,
           TensorDesc matrixBDesc, bool transposeB, const void* matrixB,
           void* tmp, U32 bytes,
           TensorDesc matirxCDesc, void* matrixC, Arch arch);

    EE reshape_infer_output_size(TensorDesc inputDesc, TensorDesc* outputDesc, I32 *shape, I32 shape_size);

    EE reshape(TensorDesc inputDesc, void* input,
           TensorDesc outputDesc, void* output, Arch arch);

    EE attention(TensorDesc inputDesc, const void *input,
           TensorDesc outputDesc, void *output,
           Arch arch);

    EE attention_infer_output_size(TensorDesc inputDesc,
           U32 numHeads, U32 fromSequenceLength, U32 toSequenceLength,
           TensorDesc *outputDesc);

    EE multiply(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output, Arch arch);

    EE multiply_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc);

    EE clip(void *min_value, void *max_value, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output, Arch arch);

    EE clip_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc);

    EE quantize_tensor(TensorDesc dDesc, const void* data, TensorDesc* qDesc, void* qData, void *scale);

    EE tensor_computing_set_input_infer_tmpBuf_size(void* input, TensorDesc hostDesc, U32* tmpBufSize, Arch arch);

    EE tensor_computing_set_input(void* input, TensorDesc hostDesc, const void* hostPtr, void* tmpBuf, bool blocking, Arch arch, ExtInfo_t extInfo = NULL);

    EE tensor_computing_get_output_infer_tmpBuf_size(const void* input, TensorDesc hostDesc, U32* tmpBufSize, Arch arch);

    EE tensor_computing_get_output(const void* input, TensorDesc hostDesc, void** hostPtr, void* tmpBuf, bool blocking, Arch arch, ExtInfo_t extInfo = NULL);

    EE bilateral_slice_apply_infer_output_size(TensorDesc inputDesc, TensorDesc guideDesc, TensorDesc gridDesc, BilateralSliceApplyDesc bilateralSliceApplyDesc,
        TensorDesc* outputDesc, Arch arch, ExtInfo_t extInfo = NULL);

    EE bilateral_slice_apply_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc guideDesc, TensorDesc gridDesc, BilateralSliceApplyDesc bilateralSliceApplyDesc,
        U32* bytes, Arch arch, ExtInfo_t extInfo = NULL);

    EE bilateral_slice_apply(TensorDesc inputDesc, const void* input, TensorDesc guideDesc, const void* guide, TensorDesc gridDesc, const void* grid, 
        BilateralSliceApplyDesc bilateralSliceApplyDesc, U32 tmpBytes, const void* tmpBuf, TensorDesc outputDesc, const void* output, Arch arch, ExtInfo_t extInfo = NULL);

    EE argmax(TensorDesc inputDesc, const void* input,
        I32 axis,
        TensorDesc outputDesc, void* output, Arch arch);

    EE argmax_infer_output_size(TensorDesc inputDesc, int axis, TensorDesc *outputDesc);

    EE axis_mean(TensorDesc inputDesc, const void* input,
        I32 axis,
        TensorDesc outputDesc, void* output, Arch arch);

    EE axis_mean_infer_output_size(TensorDesc inputDesc, int axis, TensorDesc *outputDesc);

    EE check(TensorDesc inputDescA, const void* inputA,
        TensorDesc inputDescB, const void* inputB,
        CheckMode checkMode,
        TensorDesc outputDesc, void* output, Arch arch);

    EE check_infer_output_size(TensorDesc inputDesc, TensorDesc *outputDesc);
#ifdef __cplusplus
}
#endif
#endif

