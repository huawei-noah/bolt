// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "tensor_computing.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

inline EE convolution_infer_output_size_cpu(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
    TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes)
{
    if (nullptr == outputDesc || nullptr == outputBytes)
        CHECK_STATUS(NULL_POINTER);
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    I32 oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fh < 1 || fw < 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    
    I32 strideH = convDesc.stride_h;
    I32 strideW = convDesc.stride_w;
    I32 paddingT = convDesc.padding_top;
    I32 paddingB = convDesc.padding_bottom;
    I32 paddingL = convDesc.padding_left;
    I32 paddingR = convDesc.padding_right;
    I32 dilateH = convDesc.dilatedRate_h;
    I32 dilateW = convDesc.dilatedRate_w;

    U32 fhDilated = (fh - 1) * dilateH + 1;
    U32 fwDilated = (fw - 1) * dilateW + 1;
    oh = (ih + paddingT + paddingB - fhDilated) / strideH + 1;
    ow = (iw + paddingL + paddingR - fwDilated) / strideW + 1;

    if (fn % 8 != 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    *outputDesc = tensor4df(targetDataType, DF_NCHWC8, in, fn, oh, ow);
    *outputBytes = tensorNumBytes(*outputDesc);
    return SUCCESS;
}

EE convolution_infer_output_size(TensorDesc inputDesc, TensorDesc filterDesc, ConvolutionDesc convDesc,
    TensorDesc* outputDesc, DataType targetDataType, U32* outputBytes, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if (arch == MALI) {
#ifdef _USE_MALI
        ret = convolution_infer_output_size_mali(inputDesc, filterDesc, convDesc, outputDesc, 
                  extInfo->maliInfo.gclmemInputDesc, extInfo->maliInfo.gclmemOutputDesc, extInfo->maliInfo.forwardRunInfo);
#endif
    } else {
        ret = convolution_infer_output_size_cpu(inputDesc, filterDesc, convDesc, outputDesc, targetDataType, outputBytes);
    }
    return ret;
}

EE convolution_infer_forward_algorithm(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionPolicy policy, ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType, ActivationDesc activationDesc, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = convolution_infer_forward_algorithm_arm(inputDesc, filterDesc, outputDesc, convDesc, policy, algorithm, targetDataType);
#endif
#ifdef _USE_MALI
    } else if (arch == MALI) {
        ret = convolution_infer_forward_algorithm_mali(extInfo->maliInfo.handle, inputDesc,
                  filterDesc, convDesc, outputDesc, policy, activationDesc.mode, extInfo->maliInfo.forwardRunInfo);
#endif
    }
    return ret;
}

EE convolution_transform_filter_bytes(TensorDesc filterDesc, ConvolutionForwardAlgorithm algorithm, U32* bytes, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = convolution_transform_filter_bytes_arm(filterDesc, algorithm, bytes);
#endif
#ifdef _USE_MALI
    } else if (arch == MALI) {
        ret = convolution_transform_filter_bytes_mali(filterDesc, extInfo->maliInfo.forwardRunInfo, extInfo->maliInfo.gclmemFilterDesc, bytes);
#endif
    }
    return ret;
}

EE convolution_transform_filter(TensorDesc filterDesc, const void* filter, ConvolutionForwardAlgorithm algorithm,
    TensorDesc *ftmDesc, void* filterTransformed, void* tmp, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = convolution_transform_filter_arm(filterDesc, filter, algorithm, ftmDesc, filterTransformed);
#endif
#ifdef _USE_MALI
    } else if (arch == MALI) {
        ret = convolution_transform_filter_mali(extInfo->maliInfo.handle, filterDesc, (GCLMem_t)filter, extInfo->maliInfo.forwardRunInfo, 
                                                ftmDesc, (GCLMem_t)filterTransformed, (GCLMem_t)tmp);
#endif
    }
    return ret;
}

EE convolution_infer_forward_tmp_bytes(TensorDesc inputDesc, TensorDesc filterDesc, TensorDesc outputDesc,
    ConvolutionDesc convDesc, ConvolutionForwardAlgorithm algorithm, U32 *bytes, Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = convolution_infer_forward_tmp_bytes_arm(inputDesc, filterDesc, outputDesc, convDesc, algorithm, bytes);
#endif
#ifdef _USE_MALI
    } else if (arch == MALI) {
        ret = convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc, convDesc, extInfo->maliInfo.forwardRunInfo, bytes);
#endif
    }
    return ret;
}

EE convolution(TensorDesc inputDesc, void* input,
        TensorDesc filterDesc, const void* filter,
        ConvolutionDesc convDesc,
        ConvolutionForwardAlgorithm algorithm,
        TensorDesc scaleDesc, const void* scale,
        TensorDesc biasDesc, const void* bias,
        U32 tmpBytes, void* tmp,
        TensorDesc outputDesc, void* output,
        ActivationDesc activationDesc,
        Arch arch, ExtInfo_t extInfo)
{
    EE ret = NOT_SUPPORTED;
    if (arch == CPU_GENERAL) {
#ifdef _USE_GENERAL
        ret = convolution_general(inputDesc, input,
                                  filterDesc, filter,
                                  convDesc,
                                  scaleDesc, scale,
                                  biasDesc, bias,
                                  outputDesc, output,
                                  activationDesc);
#endif
#ifdef _USE_NEON
    } else if (arch == ARM_A55 || arch == ARM_A76 || arch == ARM_V8 || arch == ARM_V7) {
        ret = convolution_arm(inputDesc, input,
                              filterDesc, filter,
                              convDesc,
                              algorithm,
                              scaleDesc, scale,
                              biasDesc, bias,
                              tmpBytes, tmp,
                              outputDesc, output,
                              activationDesc,
                              arch);
#endif
#ifdef _USE_MALI
    } else if (arch == MALI) {
        ret = convolution_mali(extInfo->maliInfo.handle, inputDesc, (GCLMem_t)input,
                               filterDesc, (GCLMem_t)filter,
                               convDesc,
                               extInfo->maliInfo.forwardRunInfo,
                               scaleDesc,  (GCLMem_t)scale,
                               biasDesc,   (GCLMem_t)bias,
                               tmpBytes,   (GCLMem_t)tmp,
                               outputDesc, (GCLMem_t)output,
                               activationDesc.mode);
#endif
    }                            
    return ret;
}
