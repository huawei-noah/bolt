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

inline EE pooling_infer_output_size_cpu(
    TensorDesc inputDesc, PoolingParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, it, ih, iw;
    if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        it = 0;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }
    RoundMode rm = p.rm;
    U32 ot = 0, oh = 0, ow = 0;
    EE ret = SUCCESS;
    switch (rm) {
        case CEIL: {
            ot = (U32)(ceil(
                     (double(it + p.padding_before + p.padding_after - p.kernel_t) / p.stride_t))) +
                1;
            oh = (U32)(ceil(
                     (double(ih + p.padding_top + p.padding_bottom - p.kernel_h) / p.stride_h))) +
                1;
            ow = (U32)(ceil(
                     (double(iw + p.padding_left + p.padding_right - p.kernel_w) / p.stride_w))) +
                1;
            break;
        }
        case FLOOR: {
            ot = (U32)(floor(
                     (double(it + p.padding_before + p.padding_after - p.kernel_t) / p.stride_t))) +
                1;
            oh = (U32)(floor(
                     (double(ih + p.padding_top + p.padding_bottom - p.kernel_h) / p.stride_h))) +
                1;
            ow = (U32)(floor(
                     (double(iw + p.padding_left + p.padding_right - p.kernel_w) / p.stride_w))) +
                1;
            break;
        }
        case TF_SAME: {
            ot = (U32)(ceil((double(it) / p.stride_t)));
            oh = (U32)(ceil((double(ih) / p.stride_h)));
            ow = (U32)(ceil((double(iw) / p.stride_w)));
            break;
        }
        case TF_VALID: {
            ot = (U32)(ceil((double(it - p.kernel_t + 1) / p.stride_t)));
            oh = (U32)(ceil((double(ih - p.kernel_h + 1) / p.stride_h)));
            ow = (U32)(ceil((double(iw - p.kernel_w + 1) / p.stride_w)));
            break;
        }
        default: {
            ret = NOT_SUPPORTED;
            break;
        }
    }
    if (tensorIs4d(inputDesc)) {
        *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);
    } else if (tensorIs5d(inputDesc)) {
        *outputDesc = tensor5df(idt, idf, in, ic, ot, oh, ow);
    }
    return ret;
}

EE pooling_infer_output_size(
    Tensor *inputTensor, PoolingParamSpec poolingParamSpec, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    if (0 == poolingParamSpec.kernel_w) {
        poolingParamSpec.kernel_w = inputDesc.dims[0];
    }
    if (0 == poolingParamSpec.kernel_h) {
        poolingParamSpec.kernel_h = inputDesc.dims[1];
    }
    if (0 == poolingParamSpec.kernel_t) {
        poolingParamSpec.kernel_t = inputDesc.dims[2];
    }
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = pooling_infer_output_size_mali(
            inputDesc, poolingParamSpec, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        ret = pooling_infer_output_size_cpu(inputDesc, poolingParamSpec, &outputDesc);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE pooling(Tensor inputTensor,
    PoolingParamSpec poolingParamSpec,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    F32 scale[2] = {inputTensor.get_scale(), -1};
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);

    if (0 == poolingParamSpec.kernel_w) {
        poolingParamSpec.kernel_w = inputDesc.dims[0];
    }
    if (0 == poolingParamSpec.kernel_h) {
        poolingParamSpec.kernel_h = inputDesc.dims[1];
    }
    if (0 == poolingParamSpec.kernel_t) {
        poolingParamSpec.kernel_t = inputDesc.dims[2];
    }
    TensorDesc inDescCPU = inputDesc;
    U8 *inputCPU = (U8 *)input;
    TensorDesc outDescCPU = outputDesc;
    U8 *outputCPU = (U8 *)output;
    if (DF_NCHWC8 != inputDesc.df && IS_CPU(arch)) {
        int channelAxis = inputDesc.nDims - 2;
        U32 paddedC = (inputDesc.dims[channelAxis] + 7) / 8 * 8;
        inDescCPU.dims[channelAxis] = paddedC;
        inDescCPU.df = DF_NCHWC8;
        outDescCPU.dims[channelAxis] = paddedC;
        outDescCPU.df = DF_NCHWC8;
        inputCPU = (U8 *)tmp;
        outputCPU = inputCPU + tensorNumBytes(inDescCPU);
        transformNCHWToNCHWC8(inputDesc, input, inDescCPU, inputCPU);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = pooling_general(inDescCPU, inputCPU, poolingParamSpec, outDescCPU, outputCPU);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = pooling_x86(inDescCPU, inputCPU, poolingParamSpec, scale, outDescCPU, outputCPU);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = pooling_arm(inDescCPU, inputCPU, poolingParamSpec, scale, outDescCPU, outputCPU);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = pooling_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (const GCLMem_t)input, poolingParamSpec, scale, (GCLMem_t)tmp, outputDesc,
            (GCLMem_t)output);
#endif
    }
    if (DF_NCHWC8 != outputDesc.df && IS_CPU(arch)) {
        transformToNCHW(outDescCPU, outputCPU, outputDesc, output);
    }
    outputTensor.set_scale(scale[1]);
    return ret;
}

EE pooling_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor.get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        ret = pooling_infer_forward_tmp_bytes_mali(
            inputDesc, bytes, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        *bytes = 0;
        ret = SUCCESS;
        if (DF_NCHW == inputDesc.df) {
            int channelAxis = 0;
            if (tensorIs4d(inputDesc)) {
                channelAxis = 2;
            } else if (tensorIs5d(inputDesc)) {
                channelAxis = 3;
            } else {
                ret = NOT_SUPPORTED;
            }
            U32 paddedC = (inputDesc.dims[channelAxis] + 7) / 8 * 8;
            TensorDesc outputDesc = outputTensor.get_desc();
            inputDesc.dims[channelAxis] = paddedC;
            outputDesc.dims[channelAxis] = paddedC;
            *bytes = tensorNumBytes(inputDesc) + tensorNumBytes(outputDesc);
        }
    }
    return ret;
}
