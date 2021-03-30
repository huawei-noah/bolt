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
#include "cpu/tensor_computing_cpu.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

inline EE depthwise_convolution_infer_output_size_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    DataType targetDataType)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    if (fh < 1 || fw < 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;

    U32 fhDilated = (fh - 1) * dilateH + 1;
    U32 fwDilated = (fw - 1) * dilateW + 1;

    CHECK_REQUIREMENT(fdf == DF_NCHW || fdf == DF_NCHWC8);
    oh = (ih + paddingT + paddingB - fhDilated) / strideH + 1;
    ow = (iw + paddingL + paddingR - fwDilated) / strideW + 1;

    if (ic % 8 != 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    *outputDesc = tensor4df(targetDataType, DF_NCHWC8, in, ic, oh, ow);
    return SUCCESS;
}

EE depthwise_convolution_infer_output_size(Tensor *inputTensor,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    Tensor *outputTensor,
    DataType targetDataType,
    ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = depthwise_convolution_infer_output_size_mali(
            inputDesc, filterDesc, convParamSpec, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        ret = depthwise_convolution_infer_output_size_cpu(
            inputDesc, filterDesc, convParamSpec, &outputDesc, targetDataType);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE depthwise_convolution_infer_forward_algorithm(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    DepthwiseConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    ActivationParamSpec depthwiseActivationParamSpec,
    ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        *algorithm = DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT;
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        *algorithm = DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT;
        ret = SUCCESS;
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc filterDesc = filterTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = depthwise_convolution_infer_forward_algorithm_mali(
            ((MaliPara_t)(archInfo->archPara))->handle, inputDesc, filterDesc, outputDesc,
            convParamSpec, policy, depthwiseActivationParamSpec.mode,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}

EE depthwise_convolution_transform_filter_bytes(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    TensorDesc filterDesc = filterTensor.get_desc();

    UNUSED(convParamSpec);
    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        *bytes = tensorNumBytes(filterDesc);
        ret = SUCCESS;
#endif
#if defined(_USE_X86) || defined(_USE_NEON)
    } else if (IS_CPU(arch)) {
        ret = depthwise_convolution_transform_filter_bytes_cpu(filterDesc, algorithm, bytes);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = depthwise_convolution_transform_filter_bytes_mali(filterDesc,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo,
            ((MaliPara_t)(archInfo->archPara))->gclmemFilterDesc, bytes);
#endif
    }
    return ret;
}

EE depthwise_convolution_transform_filter(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    Tensor *ftmTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc filterDesc = filterTensor.get_desc();
    void *filter = get_ptr_from_tensor(filterTensor, arch);
    TensorDesc ftmDesc = ftmTensor->get_desc();
    void *filterTransformed = get_ptr_from_tensor(*ftmTensor, arch);

    UNUSED(convParamSpec);
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        UNI_MEMCPY(filterTransformed, filter, tensorNumBytes(filterDesc));
        ftmDesc = filterDesc;
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = depthwise_convolution_transform_filter_x86(
            filterDesc, filter, algorithm, &ftmDesc, filterTransformed);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_convolution_transform_filter_arm(
            filterDesc, filter, algorithm, &ftmDesc, filterTransformed);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = depthwise_convolution_transform_filter_mali(((MaliPara_t)(archInfo->archPara))->handle,
            filterDesc, (GCLMem_t)filter, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo,
            &ftmDesc, (GCLMem_t)filterTransformed);
#endif
    }
    ftmTensor->resize(ftmDesc);
    return ret;
}

EE depthwise_convolution_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();

    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        *bytes = 0;
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = depthwise_convolution_infer_forward_tmp_bytes_x86(
            inputDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_convolution_infer_forward_tmp_bytes_arm(
            inputDesc, filterDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = depthwise_convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc,
            convParamSpec, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, bytes);
        ret = SUCCESS;
#endif
    }
    return ret;
}

EE depthwise_convolution(Tensor inputTensor,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    Tensor biasTensor,
    Tensor tmpTensor,
    Tensor outputTensor,
    ActivationParamSpec depthwiseActivationParamSpec,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc filterDesc = filterTensor.get_desc();
    void *filter = get_ptr_from_tensor(filterTensor, arch);
    TensorDesc biasDesc = biasTensor.get_desc();
    void *bias = get_ptr_from_tensor(biasTensor, arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = depthwise_convolution_general(inputDesc, input, filterDesc, filter, convParamSpec,
            biasDesc, bias, tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = depthwise_convolution_x86(inputDesc, input, filterDesc, filter, convParamSpec,
            algorithm, biasDesc, bias, tmpBytes, tmp, outputDesc, output,
            depthwiseActivationParamSpec, archInfo->arch);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_convolution_arm(inputDesc, input, filterDesc, filter, convParamSpec,
            algorithm, biasDesc, bias, tmpBytes, tmp, outputDesc, output,
            depthwiseActivationParamSpec, archInfo->arch);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = depthwise_convolution_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, filterDesc, (GCLMem_t)filter, convParamSpec,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, biasDesc, (GCLMem_t)bias, tmpBytes,
            (GCLMem_t)tmp, outputDesc, (GCLMem_t)output, depthwiseActivationParamSpec.mode);
#endif
    }
    return ret;
}
