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
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

inline EE depthwise_pointwise_convolution_infer_output_size_cpu(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    DataType targetDataType)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt, fdt2;
    DataFormat idf, fdf, fdf2;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 fn2, fc2, fh2, fw2;
    U32 oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(pwFilterDesc, &fdt2, &fdf2, &fn2, &fc2, &fh2, &fw2));
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

    oh = (ih + paddingT + paddingB - fhDilated) / strideH + 1;
    ow = (iw + paddingL + paddingR - fwDilated) / strideW + 1;

    if (fn2 % 8 != 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    *outputDesc = tensor4df(targetDataType, DF_NCHWC8, in, fn2, oh, ow);
    return SUCCESS;
}

EE depthwise_pointwise_convolution_infer_output_size(Tensor *inputTensor,
    Tensor dwFilterTensor,
    Tensor pwFilterTensor,
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
    TensorDesc outputDesc = outputTensor->get_desc();
    TensorDesc dwFilterDesc = dwFilterTensor.get_desc();
    TensorDesc pwFilterDesc = pwFilterTensor.get_desc();

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = depthwise_pointwise_convolution_infer_output_size_mali(inputDesc, dwFilterDesc,
            pwFilterDesc, convParamSpec, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        ret = depthwise_pointwise_convolution_infer_output_size_cpu(
            inputDesc, dwFilterDesc, pwFilterDesc, convParamSpec, &outputDesc, targetDataType);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

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
    ArchInfo_t archInfo)
{
#if defined(_USE_NEON) || defined(_USE_MALI)
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();
    TensorDesc dwFilterDesc = dwFilterTensor.get_desc();
    TensorDesc pwFilterDesc = pwFilterTensor.get_desc();
#endif
    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT;
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_pointwise_convolution_infer_forward_algorithm_arm(inputDesc, dwFilterDesc,
            pwFilterDesc, outputDesc, convParamSpec, policy, algorithm, targetDataType);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = depthwise_pointwise_convolution_infer_forward_algorithm_mali(
            ((MaliPara_t)(archInfo->archPara))->handle, inputDesc, dwFilterDesc, pwFilterDesc,
            outputDesc, convParamSpec, policy, depthwiseActivationParamSpec.mode,
            pointwiseActivationParamSpec.mode, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}

EE depthwise_pointwise_convolution_transform_filter_bytes(Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *dwBytes,
    U32 *pwBytes,
    ArchInfo_t archInfo)
{
    TensorDesc dwFilterDesc = dwFilterTensor.get_desc();
    TensorDesc pwFilterDesc = pwFilterTensor.get_desc();
    UNUSED(convParamSpec);
    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        *dwBytes = tensorNumBytes(dwFilterDesc);
        *pwBytes = tensorNumBytes(pwFilterDesc);
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        *dwBytes = tensorNumBytes(dwFilterDesc) + 32;
        *pwBytes = tensorNumBytes(pwFilterDesc) + 32;
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        *dwBytes = tensorNumBytes(dwFilterDesc) + 32;
        *pwBytes = tensorNumBytes(pwFilterDesc) + 32;
        ret = SUCCESS;
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        GCLMemDesc_t gclmemFilterDesc = ((MaliPara_t)(archInfo->archPara))->gclmemFilterDesc;
        GCLMemDesc_t gclmemDwFilterDesc = &gclmemFilterDesc[0];
        GCLMemDesc_t gclmemPwFilterDesc = &gclmemFilterDesc[1];
        ret = depthwise_pointwise_convolution_transform_filter_bytes_mali(dwFilterDesc,
            pwFilterDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, gclmemDwFilterDesc,
            gclmemPwFilterDesc, dwBytes);
        *pwBytes = 0;
#endif
    }
    return ret;
}

EE depthwise_pointwise_convolution_transform_filter(Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    Tensor *dwFtm,
    Tensor *pwFtm,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc dwFilterDesc = dwFilterTensor.get_desc();
    void *dwFilter = get_ptr_from_tensor(dwFilterTensor, arch);
    TensorDesc pwFilterDesc = pwFilterTensor.get_desc();
    void *pwFilter = get_ptr_from_tensor(pwFilterTensor, arch);
    TensorDesc dwFtmDesc = dwFtm->get_desc();
    void *dwFilterTransformed = get_ptr_from_tensor(*dwFtm, arch);
    TensorDesc pwFtmDesc = pwFtm->get_desc();
    void *pwFilterTransformed = get_ptr_from_tensor(*pwFtm, arch);
    UNUSED(convParamSpec);
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        UNI_MEMCPY(dwFilterTransformed, dwFilter, tensorNumBytes(dwFilterDesc));
        dwFtmDesc = dwFilterDesc;
        UNI_MEMCPY(pwFilterTransformed, pwFilter, tensorNumBytes(pwFilterDesc));
        pwFtmDesc = pwFilterDesc;
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = depthwise_pointwise_convolution_transform_filter_x86(dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, algorithm, &dwFtmDesc, dwFilterTransformed, &pwFtmDesc,
            pwFilterTransformed);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_pointwise_convolution_transform_filter_arm(dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, convParamSpec, algorithm, &dwFtmDesc, dwFilterTransformed,
            &pwFtmDesc, pwFilterTransformed);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = depthwise_pointwise_convolution_transform_filter_mali(
            ((MaliPara_t)(archInfo->archPara))->handle, dwFilterDesc, pwFilterDesc,
            (GCLMem_t)dwFilter, (GCLMem_t)pwFilter,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, &dwFtmDesc, &pwFtmDesc,
            (GCLMem_t)dwFilterTransformed, (GCLMem_t)pwFilterTransformed);
#endif
    }
    dwFtm->resize(dwFtmDesc);
    pwFtm->resize(pwFtmDesc);
    return ret;
}

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();
    TensorDesc dwFilterDesc = dwFilterTensor.get_desc();
    TensorDesc pwFilterDesc = pwFilterTensor.get_desc();

    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = depthwise_pointwise_convolution_infer_forward_tmp_bytes_general(
            inputDesc, dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = depthwise_convolution_infer_forward_tmp_bytes_x86(
            inputDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_pointwise_convolution_infer_forward_tmp_bytes_arm(
            inputDesc, dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali(inputDesc, dwFilterDesc,
            pwFilterDesc, outputDesc, convParamSpec,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, bytes);
#endif
    }
    return ret;
}

EE depthwise_pointwise_convolution(Tensor inputTensor,
    Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    Tensor dwBiasTensor,
    Tensor pwBiasTensor,
    Tensor tmpTensor,
    Tensor outputTensor,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    TensorDesc dwFilterDesc = dwFilterTensor.get_desc();
    void *dwFilter = get_ptr_from_tensor(dwFilterTensor, arch);
    TensorDesc pwFilterDesc = pwFilterTensor.get_desc();
    void *pwFilter = get_ptr_from_tensor(pwFilterTensor, arch);
    TensorDesc dwBiasDesc = dwBiasTensor.get_desc();
    void *dwBias = get_ptr_from_tensor(dwBiasTensor, arch);
    TensorDesc pwBiasDesc = pwBiasTensor.get_desc();
    void *pwBias = get_ptr_from_tensor(pwBiasTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = depthwise_pointwise_convolution_general(inputDesc, input, dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, convParamSpec, dwBiasDesc, dwBias, pwBiasDesc, pwBias, tmpBytes,
            tmp, outputDesc, output, depthwiseActivationParamSpec, pointwiseActivationParamSpec);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = depthwise_pointwise_convolution_x86(inputDesc, input, dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, convParamSpec, algorithm, dwBiasDesc, dwBias, pwBiasDesc,
            pwBias, tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
            pointwiseActivationParamSpec, archInfo->arch);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_pointwise_convolution_arm(inputDesc, input, dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, convParamSpec, algorithm, dwBiasDesc, dwBias, pwBiasDesc,
            pwBias, tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
            pointwiseActivationParamSpec, archInfo->arch);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = depthwise_pointwise_convolution_mali(((MaliPara_t)(archInfo->archPara))->handle,
            inputDesc, (GCLMem_t)input, dwFilterDesc, pwFilterDesc, (GCLMem_t)dwFilter,
            (GCLMem_t)pwFilter, convParamSpec, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo,
            dwBiasDesc, pwBiasDesc, (GCLMem_t)dwBias, (GCLMem_t)pwBias, tmpBytes, (GCLMem_t)tmp,
            outputDesc, (GCLMem_t)output, depthwiseActivationParamSpec.mode,
            pointwiseActivationParamSpec.mode);
#endif
    }
    return ret;
}
