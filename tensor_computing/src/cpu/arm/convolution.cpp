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
#include "cpu/arm/fp16/convolution_fp16.h"
#ifdef _USE_INT8
#include "cpu/arm/int8/convolution_int8.h"
#endif
#include "cpu/arm/bnn/convolution_bnn.h"

EE convolution_infer_forward_algorithm_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm, DataType targetDataType)
{
    if (nullptr == algorithm)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    EE ret = SUCCESS;
    switch (targetDataType) {
        case DT_F16: {
            ret = convolution_infer_forward_algorithm_fp16(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_infer_forward_algorithm_fp16(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm);
            break;
        }
#endif
        case DT_DOREFA: {
            ret = convolution_infer_forward_algorithm_bnn(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm);
            break;
        }
        case DT_XNOR: {
            ret = convolution_infer_forward_algorithm_bnn(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_transform_filter_bytes_arm(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes) {
    if (nullptr == bytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = convolution_transform_filter_bytes_fp16(filterDesc, algorithm, bytes);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_transform_filter_bytes_int8(filterDesc, algorithm, bytes);
            break;
        }
#endif
        case DT_DOREFA: {
            ret = convolution_transform_filter_bytes_bnn(filterDesc, algorithm, bytes);
            break;
        }
        case DT_XNOR: {
            ret = convolution_transform_filter_bytes_bnn(filterDesc, algorithm, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_transform_filter_arm(TensorDesc filterDesc, const void* filter,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = convolution_transform_filter_fp16(filterDesc, (F16*)filter, algorithm, ftmDesc, (F16*)filterTransformed);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_transform_filter_int8(filterDesc, filter, algorithm, ftmDesc, filterTransformed);
            break;
        }
        case DT_F16_8Q: {
            ret = convolution_transform_filter_int8(filterDesc, filter, algorithm, ftmDesc, filterTransformed);
            break;
        }
#endif
        case DT_DOREFA: {
            ret = convolution_transform_filter_bnn(filterDesc, (BIN8*)filter, ftmDesc, (BIN8*)filterTransformed);
            break;
        }
        case DT_XNOR: {
            ret = convolution_transform_filter_bnn(filterDesc, (BIN8*)filter, ftmDesc, (BIN8*)filterTransformed);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE convolution_infer_forward_tmp_bytes_arm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = convolution_infer_forward_tmp_bytes_fp16(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_infer_forward_tmp_bytes_int8(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
#endif
        case DT_DOREFA: {
            ret = convolution_infer_forward_tmp_bytes_bnn(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
        case DT_XNOR: {
            ret = convolution_infer_forward_tmp_bytes_bnn(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;

}

EE convolution_arm(TensorDesc inputDesc, void* input,
    TensorDesc filterDesc, const void* filter,
    ConvolutionDesc convDesc,
    ConvolutionForwardAlgorithm algorithm,
    TensorDesc scaleDesc, const void* scale,
    TensorDesc biasDesc, const void* bias,
    U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output,
    ActivationMode activationMode,
    Arch arch)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = convolution_fp16(inputDesc, (F16*)input,
                                   filterDesc, (F16*)filter,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (F16*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, (F16*)output,
                                   activationMode,
                                   arch);
            break;
        }
#ifdef _USE_INT8
        case DT_I8: {
            ret = convolution_int8(inputDesc, (INT8*)input,
                                   filterDesc, (INT8*)filter,
                                   (F16*)scale,
                                   convDesc,
                                   algorithm,
                                   biasDesc, (F16*)bias,
                                   tmpBytes, tmp,
                                   outputDesc, output,
                                   activationMode,
                                   arch);
            break;
        }
#endif
        case DT_DOREFA: {
            ret = convolution_bnn(inputDesc, (F16*)input,
                                         filterDesc, (BIN8*)filter,
                                         convDesc,
                                         scaleDesc, (F16*)scale,
                                         biasDesc, (F16*)bias,
                                         tmpBytes, tmp,
                                         outputDesc, (F16*)output,
                                         activationMode,
                                         arch);
            break;
        }
        case DT_XNOR: {
            ret = convolution_bnn(inputDesc, (F16*)input,
                                       filterDesc, (BIN8*)filter,
                                       convDesc,
                                       scaleDesc, (F16*)scale,
                                       biasDesc, (F16*)bias,
                                       tmpBytes, tmp,
                                       outputDesc, (F16*)output,
                                       activationMode,
                                       arch);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
