// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include "sys.h"
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "tensor_computing.h"
#include "cpu/general/tensor_computing_general.h"
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

inline EE depthwise_convolution_infer_output_size_cpu(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
    TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes)
{
    if (nullptr == outputDesc || nullptr == outputBytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fh < 1 || fw < 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    
    U32 strideH = convDesc.stride_h;
    U32 strideW = convDesc.stride_w;
    U32 paddingT = convDesc.padding_top;
    U32 paddingB = convDesc.padding_bottom;
    U32 paddingL = convDesc.padding_left;
    U32 paddingR = convDesc.padding_right;
    if (fdf == DF_NCHW || fdf == DF_NCHWC8) {
        oc = ic;
    } else {
        oc = fn;
    }
    oh = (ih + paddingT + paddingB - fh) / strideH + 1;
    ow = (iw + paddingL + paddingR - fw) / strideW + 1;

    if (fn % 8 != 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    *outputDesc = tensor4df(targetDataType, DF_NCHWC8, in, oc, oh, ow);
    *outputBytes = tensorNumBytes(*outputDesc);
    return SUCCESS;
}

EE depthwise_convolution_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
    TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes, Arch arch, ExtInfo_t extInfo)
{
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8) {
        UNUSED(extInfo);
        ret = depthwise_convolution_infer_output_size_cpu(inputDesc, filterDesc, convDesc, outputDesc, targetDataType, outputBytes);
#ifdef _USE_MALI        
    } else if (arch == MALI) {
        ret = depthwise_convolution_infer_output_size_mali(inputDesc, filterDesc, convDesc, outputDesc, 
                extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc, extInfo->maliInfo.forwardRunInfo);
#endif                                                           

    } else {
        ret = NOT_SUPPORTED;
    }        
    return ret;
}

EE depthwise_convolution_infer_forward_algorithm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, DepthwiseConvolutionForwardAlgorithm *algorithm, DataType targetDataType, 
    ActivationMode depthwiseActivationMode, ActivationMode pointwiseActivationMode, Arch arch, ExtInfo_t extInfo)
{
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8) { 
        UNUSED(depthwiseActivationMode);
        UNUSED(pointwiseActivationMode);
        UNUSED(extInfo);
        ret = depthwise_convolution_infer_forward_algorithm_arm(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm, targetDataType);
#ifdef _USE_MALI        
    } else if (arch == MALI) {
        ret = depthwise_convolution_infer_forward_algorithm_mali(extInfo->maliInfo.handle, inputDesc, filterDesc, outputDesc, convDesc, policy, 
                depthwiseActivationMode, pointwiseActivationMode, extInfo->maliInfo.forwardRunInfo);
#endif        
    } else {
        ret = NOT_SUPPORTED;
    }        
    return ret;
}

EE depthwise_convolution_transform_filter_bytes(TensorDesc filterDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32* bytes, Arch arch, ExtInfo_t extInfo) {
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8) { 
        UNUSED(extInfo);
        ret = depthwise_convolution_transform_filter_bytes_arm(filterDesc, algorithm, bytes);
#ifdef _USE_MALI        
    } else if (arch == MALI) {
        ret = depthwise_convolution_transform_filter_bytes_mali(filterDesc, extInfo->maliInfo.forwardRunInfo, extInfo->maliInfo.gclmemFilterDesc, bytes);
#endif        
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

EE depthwise_convolution_transform_filter(TensorDesc filterDesc, const void* filter, DepthwiseConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed, Arch arch, ExtInfo_t extInfo)
{
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8) { 
        UNUSED(extInfo);
        ret = depthwise_convolution_transform_filter_arm(filterDesc, filter, algorithm, ftmDesc, filterTransformed);
#ifdef _USE_MALI        
    } else if (arch == MALI) {
        ret = depthwise_convolution_transform_filter_mali(extInfo->maliInfo.handle, filterDesc, (GCLMem_t)filter, extInfo->maliInfo.forwardRunInfo, ftmDesc, 
                                                         (GCLMem_t)filterTransformed);
#endif                                                         
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

EE depthwise_convolution_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, DepthwiseConvolutionForwardAlgorithm algorithm, U32 *bytes, Arch arch, ExtInfo_t extInfo) {
    EE ret = SUCCESS;
    if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8) {
        UNUSED(extInfo);
        ret = depthwise_convolution_infer_forward_tmp_bytes_arm(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
#ifdef _USE_MALI        
    } else if (arch == MALI) {
        ret = depthwise_convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc, convDesc, extInfo->maliInfo.forwardRunInfo, bytes);
#endif        
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

EE depthwise_convolution(TensorDesc inputDesc, void* input,
        TensorDesc filterDesc, const void* filter,
        ConvolutionDesc convDesc,
        DepthwiseConvolutionForwardAlgorithm algorithm,
        TensorDesc biasDesc, const void* bias,
        U32 tmpBytes, void* tmp,
        TensorDesc outputDesc, void* output,
        ActivationMode depthwiseActivationMode,
        ActivationMode pointwiseActivationMode,
        Arch arch, ExtInfo_t extInfo)
{
#ifndef _USE_MALI
    UNUSED(extInfo);
#endif  
    EE ret = SUCCESS;
    switch (arch) {
        case CPU_GENERAL:
            ret = depthwise_convolution_general(inputDesc, input,
                                                filterDesc, filter,
                                                convDesc,
                                                biasDesc, bias,
                                                outputDesc, output,
                                                depthwiseActivationMode,
                                                pointwiseActivationMode);
            break;
        case ARM_A55:
            ret = depthwise_convolution_arm(inputDesc, input,
                                            filterDesc, filter,
                                            convDesc,
                                            algorithm,
                                            biasDesc, bias,
                                            tmpBytes, tmp,
                                            outputDesc, output,
                                            depthwiseActivationMode,
                                            pointwiseActivationMode,
                                            arch);
            break;
        case ARM_A76:
            ret = depthwise_convolution_arm(inputDesc, input,
                                            filterDesc, filter,
                                            convDesc,
                                            algorithm,
                                            biasDesc, bias,
                                            tmpBytes, tmp,
                                            outputDesc, output,
                                            depthwiseActivationMode,
                                            pointwiseActivationMode,
                                            arch);
            break;
#ifdef _USE_MALI                                            
        case MALI:
            ret = depthwise_convolution_mali(extInfo->maliInfo.handle, inputDesc, (GCLMem_t)input,
                                             filterDesc, (GCLMem_t)filter,
                                             convDesc,
                                             extInfo->maliInfo.forwardRunInfo,
                                             biasDesc,   (GCLMem_t)bias,
                                             tmpBytes,   (GCLMem_t)tmp,
                                             outputDesc, (GCLMem_t)output,
                                             depthwiseActivationMode,
                                             pointwiseActivationMode);
            break;
#endif            
        case ARM_V8:
            ret = depthwise_convolution_arm(inputDesc, input,
                                            filterDesc, filter,
                                            convDesc,
                                            algorithm,
                                            biasDesc, bias,
                                            tmpBytes, tmp,
                                            outputDesc, output,
                                            depthwiseActivationMode,
                                            pointwiseActivationMode,
                                            arch);
            break;
        default:
            ret = NOT_SUPPORTED;
    }
    return ret;
}
