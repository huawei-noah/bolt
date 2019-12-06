// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "cpu/arm/tensor_computing_arm.h"
#include "cpu/arm/fp16/depthwise_convolution_fp16.h"
#ifdef _USE_INT8
#include "cpu/arm/int8/depthwise_convolution_int8.h"
#endif

EE depthwise_convolution_infer_forward_algorithm_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm, DataType targetDataType)
{
    if (nullptr == algorithm)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    EE ret = SUCCESS;
    switch (targetDataType) {
        case DT_F16: {
            ret = depthwise_convolution_infer_forward_algorithm_fp16(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = depthwise_convolution_infer_forward_algorithm_int8(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_transform_filter_bytes_arm(TensorDesc filterDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32* bytes) {
    if (nullptr == bytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = depthwise_convolution_transform_filter_bytes_fp16(filterDesc, algorithm, bytes);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = depthwise_convolution_transform_filter_bytes_int8(filterDesc, algorithm, bytes);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_transform_filter_arm(TensorDesc filterDesc, const void* filter,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = depthwise_convolution_transform_filter_fp16(filterDesc, (F16*)filter, algorithm, ftmDesc, (F16*)filterTransformed);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = depthwise_convolution_transform_filter_int8(filterDesc, (INT8*)filter, algorithm, ftmDesc, (INT8*)filterTransformed);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = depthwise_convolution_infer_forward_tmp_bytes_fp16(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = depthwise_convolution_infer_forward_tmp_bytes_int8(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;

}

EE depthwise_convolution_arm(TensorDesc inputDesc, void* input,
    TensorDesc filterDesc, const void* filter,
    ConvolutionDesc convDesc,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
    Arch arch)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = depthwise_convolution_fp16(inputDesc, (F16*)input,
                                   filterDesc, (F16*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (F16*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (F16*)output,
                                   depthwiseActivationMode,
                                   pointwiseActivationMode,
                                   arch);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = depthwise_convolution_int8(inputDesc, (INT8*)input,
                                   filterDesc, (INT8*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (I32*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (I32*)output,
                                   depthwiseActivationMode,
                                   pointwiseActivationMode,
                                   arch);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
