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
#ifdef _USE_GPU
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
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;

    U32 fhDilated = (fh - 1) * dilateH + 1;
    U32 fwDilated = (fw - 1) * dilateW + 1;

    oh = (ih + paddingT + paddingB - fhDilated) / strideH + 1;
    ow = (iw + paddingL + paddingR - fwDilated) / strideW + 1;
    if (ow <= 0 || oh <= 0) {
        CHECK_STATUS(NOT_MATCH);
    }

    DataFormat odf = DF_NCHWC8;
    if ((idt == DT_U8_Q || idf == DF_NCHWC16) && ic % 16 == 0) {
        odf = DF_NCHWC16;
    }

    *outputDesc = tensor4df(targetDataType, odf, in, fn2, oh, ow);
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

    CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size_cpu(
        inputDesc, dwFilterDesc, pwFilterDesc, convParamSpec, &outputDesc, targetDataType));
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        CHECK_STATUS(depthwise_pointwise_convolution_padding_input_mali(inputDesc, dwFilterDesc,
            pwFilterDesc, convParamSpec, &outputDesc, inputMem, outputMem));
#endif
    } else {
        U32 fn = pwFilterDesc.dims[pwFilterDesc.nDims - 1];
        if (fn % 8 != 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
#ifdef _USE_INT8
        if (IS_X86_AVX512(archInfo->arch) && (inputDesc.dt == DT_U8_Q))
        {
            outputDesc.df = DF_NCHWC16;
            if (fn % 16 != 0) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
        }
#endif
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
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
#if defined(_USE_NEON) || defined(_USE_GPU)
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
    } else if (IS_X86(arch)) {
        *algorithm = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT;
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_pointwise_convolution_infer_forward_algorithm_arm(inputDesc, dwFilterDesc,
            pwFilterDesc, outputDesc, convParamSpec, policy, algorithm, targetDataType);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = depthwise_pointwise_convolution_infer_forward_algorithm_mali(
            ((MaliPara_t)(archInfo->archPara))->handle, inputDesc, dwFilterDesc, pwFilterDesc,
            outputDesc, gclmemInputDesc, gclmemOutputDesc, convParamSpec, policy,
            depthwiseActivationParamSpec, pointwiseActivationParamSpec,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}

EE depthwise_pointwise_convolution_transform_filter_bytes(Tensor dwFilterTensor,
    Tensor pwFilterTensor,
    ConvolutionParamSpec convParamSpec,
    DepthwiseConvolutionForwardAlgorithm algorithm,
    void *dwBytes,
    void *pwBytes,
    ArchInfo_t archInfo)
{
    TensorDesc dwFilterDesc = dwFilterTensor.get_desc();
    TensorDesc pwFilterDesc = pwFilterTensor.get_desc();
    UNUSED(convParamSpec);
    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        U32 *size = (U32 *)dwBytes;
        *size = tensorNumBytes(dwFilterDesc);
        size = (U32 *)pwBytes;
        *size = tensorNumBytes(pwFilterDesc);
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        U32 *size = (U32 *)dwBytes;
        if (DT_I8 == dwFilterDesc.dt) {
            DataType fdt;
            DataFormat fdf;
            U32 fn, fc, fh, fw;
            CHECK_STATUS(tensor4dGet(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
            U32 alignSize = 4;
            U32 filterSize = (fh * fw + alignSize - 1) / alignSize * alignSize;
            *size = filterSize * fn * fc + 32 + fc * 4;
        } else {
            *size = tensorNumBytes(dwFilterDesc) + 32;
        }
        size = (U32 *)pwBytes;
        *size = tensorNumBytes(pwFilterDesc) + 32;
        ret = SUCCESS;
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        U32 *size = (U32 *)dwBytes;
        *size = tensorNumBytes(dwFilterDesc) + 32;
        size = (U32 *)pwBytes;
        *size = tensorNumBytes(pwFilterDesc) + 32;
        ret = SUCCESS;
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        GCLMemDesc_t gclmemFilterDesc = ((MaliPara_t)(archInfo->archPara))->gclmemFilterDesc;
        ret = depthwise_pointwise_convolution_transform_filter_bytes_mali(dwFilterDesc,
            pwFilterDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, (TensorDesc *)dwBytes,
            (TensorDesc *)pwBytes);
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
    } else if (IS_X86(arch)) {
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
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
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
    } else if (IS_X86(arch)) {
        ret = depthwise_convolution_infer_forward_tmp_bytes_x86(
            inputDesc, dwFilterDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_pointwise_convolution_infer_forward_tmp_bytes_arm(
            inputDesc, dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali(inputDesc, dwFilterDesc,
            pwFilterDesc, outputDesc, convParamSpec,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, bytes);
#endif
    }
    return ret;
}

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
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensors[0].get_desc();
    void *input = get_ptr_from_tensor(inputTensors[0], arch);
    U32 tmpBytes = tmpTensors[0].bytes();
    void *tmp = get_ptr_from_tensor(tmpTensors[0], arch);
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

    // process fused-add
    ActivationParamSpec eltwiseActDesc = pointwiseActivationParamSpec;
    void *eltwiseInput = nullptr;
    bool isEltwiseSeperate = true;
    TensorDesc eltwiseInputDesc;
    if (inputTensors.size() > 1) {
        eltwiseInput = get_ptr_from_tensor(inputTensors[1], arch);
        eltwiseInputDesc = inputTensors[1].get_desc();
        pointwiseActivationParamSpec.mode = ACTIVATION_NULL;
    }
#if defined(_USE_GENERAL) || defined(_USE_X86)
    if (tensorNumElements(eltwiseInputDesc) == tensorNumElements(outputDesc) &&
        eltwiseInputDesc.df == outputDesc.df) {
        isEltwiseSeperate = false;
        pointwiseActivationParamSpec = eltwiseActDesc;
    } else {
        eltwiseInput = nullptr;
    }
#endif

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = depthwise_pointwise_convolution_general(inputDesc, input, eltwiseInput, dwFilterDesc,
            dwFilter, pwFilterDesc, pwFilter, convParamSpec, dwBiasDesc, dwBias, pwBiasDesc, pwBias,
            tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
            pointwiseActivationParamSpec);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = depthwise_pointwise_convolution_x86(inputDesc, input, eltwiseInput, dwFilterDesc,
            dwFilter, pwFilterDesc, pwFilter, convParamSpec, algorithm, scale, dwBiasDesc, dwBias,
            pwBiasDesc, pwBias, tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
            pointwiseActivationParamSpec, archInfo->arch);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = depthwise_pointwise_convolution_arm(inputDesc, input, dwFilterDesc, dwFilter,
            pwFilterDesc, pwFilter, convParamSpec, algorithm, dwBiasDesc, dwBias, pwBiasDesc,
            pwBias, tmpBytes, tmp, outputDesc, output, depthwiseActivationParamSpec,
            pointwiseActivationParamSpec, archInfo->arch);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        std::vector<GCLMem_t> tmpVec(3, NULL);
        for (U32 i = 0; i < tmpTensors.size(); i++) {
            tmpVec[i] = (GCLMem_t)get_ptr_from_tensor(tmpTensors[i], arch);
        }
        ret = depthwise_pointwise_convolution_mali(((MaliPara_t)(archInfo->archPara))->handle,
            inputDesc, (GCLMem_t)input, dwFilterDesc, pwFilterDesc, (GCLMem_t)dwFilter,
            (GCLMem_t)pwFilter, convParamSpec, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo,
            dwBiasDesc, pwBiasDesc, (GCLMem_t)dwBias, (GCLMem_t)pwBias, tmpBytes, tmpVec, outputDesc,
            (GCLMem_t)output, depthwiseActivationParamSpec, pointwiseActivationParamSpec);
#endif
    }

    // process fused-add
#ifdef _USE_CPU
    if (inputTensors.size() > 1 && isEltwiseSeperate) {
        std::vector<Tensor> eltwiseInputTensors = {outputTensor, inputTensors[1]};
        EltwiseParamSpec eltwiseDesc;
        eltwiseDesc.mode = ELTWISE_SUM;
        eltwiseDesc.activation_type = eltwiseActDesc.mode;
        eltwiseDesc.activation_spec = convParamSpec.activation_spec;
        ret = eltwise(eltwiseInputTensors, eltwiseDesc, tmpTensors[0], outputTensor, archInfo);
    }
#endif

    return ret;
}
