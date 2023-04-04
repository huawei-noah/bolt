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

inline EE convolution_infer_output_size_cpu(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec p,
    TensorDesc *outputDesc,
    DataType targetDataType,
    Arch arch)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 in, ic, it, ih, iw;
    U32 fn, fc, ft, fh, fw;
    I32 ot, oh, ow;
    it = ft = ot = 1;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(filterDesc, &fdt, &fdf, &fn, &fc, &ft, &fh, &fw));
    } else {
        return NOT_SUPPORTED;
    }
    EE ret = SUCCESS;
    if (ft < 1 || fh < 1 || fw < 1) {
        ret = NOT_SUPPORTED;
    }

    U32 ftDilated = (ft - 1) * p.dilatedRate_t + 1;
    U32 fhDilated = (fh - 1) * p.dilatedRate_h + 1;
    U32 fwDilated = (fw - 1) * p.dilatedRate_w + 1;

    RoundMode rm = p.round_mode;
    switch (rm) {
        case ROUND_SAME_UPPER: {
            ot = (U32)(ceil((double(it) / p.stride_t)));
            oh = (U32)(ceil((double(ih) / p.stride_h)));
            ow = (U32)(ceil((double(iw) / p.stride_w)));
            break;
        }
        case ROUND_SAME_LOWER: {
            ot = (U32)(floor((double(it) / p.stride_t)));
            oh = (U32)(floor((double(ih) / p.stride_h)));
            ow = (U32)(floor((double(iw) / p.stride_w)));
            break;
        }
        default: {
            ot = (it + p.pad_before + p.pad_after - ftDilated) / p.stride_t + 1;
            oh = (ih + p.pad_top + p.pad_bottom - fhDilated) / p.stride_h + 1;
            ow = (iw + p.pad_left + p.pad_right - fwDilated) / p.stride_w + 1;
            break;
        }
    }

    if (ot < 0 || oh < 0 || ow < 0) {
        ret = NOT_MATCH;
    }

    DataFormat odf = DF_NCHW;
#if defined(_USE_LITE) && !defined(_USE_NEON)
    if (1) {
#else
    if (0) {
#endif
    } else if (IS_CPU(arch)) {
        if (fn % 8 != 0) {
            return NOT_SUPPORTED;
        }
        odf = DF_NCHWC8;
#ifdef _USE_INT8
        if (IS_X86_AVX512(arch) && (targetDataType == DT_U8_Q) && (fn != 8)) {
            odf = DF_NCHWC16;
        }
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        odf = DF_NCHW;
#endif
    }
    if (tensorIs4d(inputDesc)) {
        *outputDesc = tensor4df(targetDataType, odf, in, fn, oh, ow);
    } else if (tensorIs5d(inputDesc)) {
        *outputDesc = tensor5df(targetDataType, odf, in, fn, ot, oh, ow);
    }
    return ret;
}

EE convolution_infer_output_size(Tensor *inputTensor,
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
    CHECK_STATUS(convolution_infer_output_size_cpu(
        inputDesc, filterDesc, convParamSpec, &outputDesc, targetDataType, archInfo->arch));
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        CHECK_STATUS(convolution_padding_input_mali(
            inputDesc, filterDesc, convParamSpec, &outputDesc, inputMem, outputMem));
#endif
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE convolution_infer_forward_algorithm(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ConvolutionForwardAlgorithm *algorithm,
    DataType targetDataType,
    ActivationParamSpec activationDesc,
    ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();

    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = convolution_infer_forward_algorithm_x86(
            inputDesc, filterDesc, outputDesc, convParamSpec, policy, algorithm, targetDataType);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = convolution_infer_forward_algorithm_arm(
            inputDesc, filterDesc, outputDesc, convParamSpec, policy, algorithm, targetDataType);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = convolution_infer_forward_algorithm_mali(((MaliPara_t)(archInfo->archPara))->handle,
            inputDesc, filterDesc, convParamSpec, outputDesc, gclmemInputDesc, gclmemOutputDesc,
            policy, activationDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}

EE convolution_transform_filter_bytes(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    void *bytes,
    ArchInfo_t archInfo)
{
    TensorDesc filterDesc = filterTensor.get_desc();

    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        U32 *size = (U32 *)bytes;
        *size = tensorNumBytes(filterDesc);
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = convolution_transform_filter_bytes_x86(
            filterDesc, convParamSpec, algorithm, (U32 *)bytes);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = convolution_transform_filter_bytes_arm(
            filterDesc, convParamSpec, algorithm, (U32 *)bytes);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = convolution_transform_filter_bytes_mali(
            filterDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, (TensorDesc *)bytes);
#endif
    }
    return ret;
}

EE convolution_transform_filter(Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    Tensor tmpTensor,
    Tensor *ftmTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc filterDesc = filterTensor.get_desc();
    void *filter = get_ptr_from_tensor(filterTensor, arch);
    TensorDesc ftmDesc = ftmTensor->get_desc();
    void *filterTransformed = get_ptr_from_tensor(*ftmTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        UNI_MEMCPY(filterTransformed, filter, tensorNumBytes(filterDesc));
        ftmDesc = filterDesc;
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = convolution_transform_filter_x86(
            filterDesc, filter, convParamSpec, algorithm, &ftmDesc, filterTransformed);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = convolution_transform_filter_arm(
            filterDesc, filter, convParamSpec, algorithm, &ftmDesc, filterTransformed);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        ret = convolution_transform_filter_mali(((MaliPara_t)(archInfo->archPara))->handle,
            filterDesc, (GCLMem_t)filter, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo,
            (GCLMem_t)tmp, &ftmDesc, (GCLMem_t)filterTransformed);
#endif
    }
    ftmTensor->resize(ftmDesc);
    return ret;
}

EE convolution_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
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
        if (filterDesc.dt == DT_I8 && outputDesc.dt == DT_I8) {
            TensorDesc tmpDesc = outputDesc;
#ifdef _USE_FP16
            tmpDesc.dt = DT_F16;
#else
            tmpDesc.dt = DT_F32;
#endif
            *bytes += tensorNumBytes(tmpDesc);
        }
        if (filterDesc.dt == DT_I8 && inputDesc.dt != DT_I8) {
            *bytes += tensorNumBytes(inputDesc);
        }
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = convolution_infer_forward_tmp_bytes_x86(
            inputDesc, filterDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = convolution_infer_forward_tmp_bytes_arm(
            inputDesc, filterDesc, outputDesc, convParamSpec, algorithm, bytes);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc,
            convParamSpec, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, bytes);
#endif
    }
    if (inputDesc.df == DF_NHWC) {
        *bytes += tensorNumBytes(inputDesc);
    }
    return ret;
}

inline void convolution_process_bnn_scale(
    U8 **bias, U8 **scale, TensorDesc *biasDesc, TensorDesc *scaleDesc)
{
    U32 vecLen = tensorNumElements(*biasDesc) / 2;
    biasDesc->dims[0] = vecLen;
    *scaleDesc = *biasDesc;
    *scale = *bias;
    *bias += vecLen * bytesOf(DT_F16);
}

void autoPadding(TensorDesc inputDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec &p)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, it, ih, iw;
    U32 on, oc, ot, oh, ow;
    it = ot = 1;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
        CHECK_STATUS(tensor5dGet(outputDesc, &odt, &odf, &on, &oc, &ot, &oh, &ow));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    RoundMode rm = p.round_mode;
    if (rm == ROUND_SAME_LOWER || rm == ROUND_SAME_UPPER) {
        U32 tlane = ot * p.stride_t - it + p.kernel_t - p.stride_t;
        U32 hlane = oh * p.stride_h - ih + p.kernel_h - p.stride_h;
        U32 wlane = ow * p.stride_w - iw + p.kernel_w - p.stride_w;
        if (rm == ROUND_SAME_LOWER) {
            p.pad_before = (tlane + 1) / 2;
            p.pad_top = (hlane + 1) / 2;
            p.pad_left = (wlane + 1) / 2;
        } else {
            p.pad_before = tlane / 2;
            p.pad_top = hlane / 2;
            p.pad_left = wlane / 2;
        }
        p.pad_after = tlane - p.pad_before;
        p.pad_bottom = hlane - p.pad_top;
        p.pad_right = wlane - p.pad_left;
    }
}

EE convolution(std::vector<Tensor> inputTensors,
    Tensor filterTensor,
    ConvolutionParamSpec convParamSpec,
    ConvolutionForwardAlgorithm algorithm,
    void *scale,
    Tensor biasTensor,
    std::vector<Tensor> tmpTensors,
    Tensor outputTensor,
    ActivationParamSpec activationDesc,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensors[0].get_desc();

    void *input = get_ptr_from_tensor(inputTensors[0], arch);
    TensorDesc filterDesc = filterTensor.get_desc();
    void *filter = get_ptr_from_tensor(filterTensor, arch);
    TensorDesc biasDesc = biasTensor.get_desc();
    void *bias = get_ptr_from_tensor(biasTensor, arch);
    U32 tmpBytes = tmpTensors[0].bytes();
    void *tmp = get_ptr_from_tensor(tmpTensors[0], arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    TensorDesc scaleDesc = filterDesc;

    // process fused-add
    ActivationParamSpec eltwiseActDesc = activationDesc;
    void *eltwiseInput = nullptr;
    bool isEltwiseSeperate = true;
    TensorDesc eltwiseInputDesc;
    if (inputTensors.size() > 1) {
        eltwiseInput = get_ptr_from_tensor(inputTensors[1], arch);
        eltwiseInputDesc = inputTensors[1].get_desc();
        activationDesc.mode = ACTIVATION_NULL;
    }
#if defined(_USE_GENERAL) || defined(_USE_X86)
    if (tensorNumElements(eltwiseInputDesc) == tensorNumElements(outputDesc) &&
        eltwiseInputDesc.df == outputDesc.df) {
        isEltwiseSeperate = false;
        activationDesc = eltwiseActDesc;
    } else {
        eltwiseInput = nullptr;
    }
#endif

    EE ret = NOT_SUPPORTED;
#ifdef _USE_FP16
    if (IS_GENERAL(arch) || (IS_ARM(arch))) {
        if (filterDesc.dt == DT_BIN01 || filterDesc.dt == DT_BIN11) {
            U8 *biasPtr = (U8 *)get_ptr_from_tensor(biasTensor, arch);
            U8 *scalePtr = nullptr;
            convolution_process_bnn_scale(&biasPtr, &scalePtr, &biasDesc, &scaleDesc);
            bias = biasPtr;
            scale = scalePtr;
        }
    }
#endif

    autoPadding(inputDesc, outputDesc, convParamSpec);

    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = convolution_general(inputDesc, input, eltwiseInput, filterDesc, filter, convParamSpec,
            scaleDesc, scale, biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = convolution_x86(inputDesc, input, eltwiseInput, filterDesc, filter, convParamSpec,
            algorithm, scaleDesc, scale, biasDesc, bias, tmpBytes, tmp, outputDesc, output,
            activationDesc, archInfo->arch);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = convolution_arm(inputDesc, input, filterDesc, filter, convParamSpec, algorithm,
            scaleDesc, scale, biasDesc, bias, tmpBytes, tmp, outputDesc, output, activationDesc,
            archInfo->arch);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        std::vector<GCLMem_t> tmpVec(3, NULL);
        for (U32 i = 0; i < tmpTensors.size(); i++) {
            tmpVec[i] = (GCLMem_t)get_ptr_from_tensor(tmpTensors[i], arch);
        }
        ret = convolution_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, filterDesc, (GCLMem_t)filter, convParamSpec,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, scaleDesc, (GCLMem_t)scale, biasDesc,
            (GCLMem_t)bias, tmpBytes, tmpVec, outputDesc, (GCLMem_t)output, activationDesc);
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
