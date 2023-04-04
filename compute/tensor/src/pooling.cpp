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

inline EE pooling_infer_output_size_cpu(
    TensorDesc inputDesc, PoolingParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, it, ih, iw;
    if (tensorIs3d(inputDesc)) {
        CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &in, &ic, &ih));
        it = iw = 1;
    } else if (tensorIs4d(inputDesc)) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        it = 1;
    } else if (tensorIs5d(inputDesc)) {
        CHECK_STATUS(tensor5dGet(inputDesc, &idt, &idf, &in, &ic, &it, &ih, &iw));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }
    RoundMode rm = p.round_mode;
    U32 ot = 0, oh = 0, ow = 0;
    EE ret = SUCCESS;
    switch (rm) {
        case ROUND_CEIL: {
            ot = (U32)(ceil((double(it + p.pad_before + p.pad_after - p.kernel_t) / p.stride_t))) +
                1;
            oh = (U32)(ceil((double(ih + p.pad_top + p.pad_bottom - p.kernel_h) / p.stride_h))) + 1;
            ow = (U32)(ceil((double(iw + p.pad_left + p.pad_right - p.kernel_w) / p.stride_w))) + 1;
            break;
        }
        case ROUND_FLOOR: {
            ot = (U32)(floor((double(it + p.pad_before + p.pad_after - p.kernel_t) / p.stride_t))) +
                1;
            oh = (U32)(floor((double(ih + p.pad_top + p.pad_bottom - p.kernel_h) / p.stride_h))) + 1;
            ow = (U32)(floor((double(iw + p.pad_left + p.pad_right - p.kernel_w) / p.stride_w))) + 1;
            break;
        }
        case ROUND_TF_SAME: {
            ot = (U32)(ceil((double(it) / p.stride_t)));
            oh = (U32)(ceil((double(ih) / p.stride_h)));
            ow = (U32)(ceil((double(iw) / p.stride_w)));
            break;
        }
        case ROUND_TF_VALID: {
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
    DataFormat odf = idf;
#ifdef _USE_AVX512_VNNI
    if (idt == DT_U8_Q) {
        odf = DF_NCHWC16;
    }
#endif
    if (tensorIs3d(inputDesc)) {
        *outputDesc = tensor3df(idt, odf, in, ic, oh);
    } else if (tensorIs4d(inputDesc)) {
        *outputDesc = tensor4df(idt, odf, in, ic, oh, ow);
    } else if (tensorIs5d(inputDesc)) {
        *outputDesc = tensor5df(idt, odf, in, ic, ot, oh, ow);
    }
    return ret;
}

static inline PoolingParamSpec update_param(TensorDesc inDesc, PoolingParamSpec poolingParamSpec)
{
    if (0 == poolingParamSpec.kernel_w) {
        if (inDesc.nDims > 3) {
            poolingParamSpec.kernel_w = inDesc.dims[0];
        } else {
            poolingParamSpec.kernel_w = 1;
        }
    }
    if (0 == poolingParamSpec.kernel_h) {
        if (inDesc.nDims > 3) {
            poolingParamSpec.kernel_h = inDesc.dims[1];
        } else {
            poolingParamSpec.kernel_h = inDesc.dims[0];
        }
    }
    if (0 == poolingParamSpec.kernel_t) {
        if (inDesc.nDims > 4) {
            poolingParamSpec.kernel_t = inDesc.dims[2];
        } else {
            poolingParamSpec.kernel_t = 1;
        }
    }
    return poolingParamSpec;
}

EE pooling_infer_output_size(
    Tensor *inputTensor, PoolingParamSpec poolingParamSpec, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc newInputDesc = transformDescTo4d(inputDesc);
    TensorDesc outputDesc = outputTensor->get_desc();
    poolingParamSpec = update_param(newInputDesc, poolingParamSpec);
    CHECK_STATUS(pooling_infer_output_size_cpu(inputDesc, poolingParamSpec, &outputDesc));
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        CHECK_STATUS(pooling_padding_input_mali(
            inputDesc, poolingParamSpec, &outputDesc, inputMem, outputMem));
#endif
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE pooling(Tensor inputTensor,
    PoolingParamSpec poolingParamSpec,
    Tensor tmpTensor,
    std::vector<Tensor> outputTensors,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = transformDescTo4d(inputTensor.get_desc());
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = transformDescTo4d(outputTensors[0].get_desc());
    void *output = get_ptr_from_tensor(outputTensors[0], arch);
    void *idx = nullptr;
    if (outputTensors.size() == 2) {
        idx = get_ptr_from_tensor(outputTensors[1], arch);
    }
    F32 scale[2] = {inputTensor.get_scale(), -1};
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    poolingParamSpec = update_param(inputDesc, poolingParamSpec);
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        ret = pooling_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            poolingParamSpec, scale, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
#ifdef _USE_GENERAL
    } else if (IS_GENERAL(arch)) {
        ret = pooling_general(inputDesc, input, poolingParamSpec, scale, outputDesc, output);
#endif
#if defined(_USE_NEON) || defined(_USE_X86)
    } else if (IS_CPU(arch)) {
        U8 *inputCPU = (U8 *)input;
        U8 *outputCPU = (U8 *)output;
        U8 *idxCPU = (U8 *)idx;
        TensorDesc inDescCPU = inputDesc;
        TensorDesc outDescCPU = outputDesc;
        DataFormat dstF = outputDesc.df;
        int channelAxis = inputDesc.nDims - 2;

        U32 cx = 8;
        if (IS_X86(arch)) {
            if (dstF == DF_NCHW || dstF == DF_MTK) {
                cx = 1;
            }
#ifdef _USE_AVX512_VNNI
            if (inputDesc.dt == DT_U8_Q) {
                dstF = DF_NCHWC16;  // padding to 16
                cx = 16;
            }
#endif
        } else {
            dstF = DF_NCHWC8;
        }

        U32 paddedC = (inputDesc.dims[channelAxis] + cx - 1) / cx * cx;
        if (paddedC != inputDesc.dims[channelAxis] || (inputDesc.df != dstF)) {
            inDescCPU.dims[channelAxis] = paddedC;
            inDescCPU.df = dstF;
            inputCPU = (U8 *)tmp;
            tmp = (U8 *)tmp + tensorNumBytes(inDescCPU);
            transformFormat(inputDesc, input, inDescCPU, inputCPU);
        }
        if (paddedC != inputDesc.dims[channelAxis] || (outputDesc.df != dstF)) {
            outDescCPU.dims[channelAxis] = paddedC;
            outDescCPU.df = dstF;
            outputCPU = (U8 *)tmp;
            if (idxCPU != nullptr) {
                idxCPU = (U8 *)tmp + tensorNumBytes(outDescCPU);
            }
        }
        if (0) {
#ifdef _USE_X86
        } else if (IS_X86(arch)) {
            ret = pooling_x86(
                inDescCPU, inputCPU, poolingParamSpec, scale, outDescCPU, outputCPU, idxCPU);
#endif
#ifdef _USE_NEON
        } else if (IS_ARM(arch)) {
            ret = pooling_arm(inDescCPU, inputCPU, poolingParamSpec, scale, outDescCPU, outputCPU);
#endif
        }

        if (paddedC != inputDesc.dims[channelAxis] || (outputDesc.df != outDescCPU.df)) {
            transformFormat(outDescCPU, outputCPU, outputDesc, output);
            if (idx != nullptr) {
                transformFormat(outDescCPU, idxCPU, outputDesc, idx);
            }
        }
#endif
    }
    outputTensors[0].set_scale(scale[1]);
    return ret;
}

EE pooling_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = transformDescTo4d(inputTensor.get_desc());
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        ret = pooling_infer_forward_tmp_bytes_mali(
            inputDesc, bytes, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else if (IS_GENERAL(archInfo->arch)) {
        *bytes = 0;
        ret = SUCCESS;
    } else {
        *bytes = 0;
        ret = SUCCESS;
        TensorDesc outputDesc = transformDescTo4d(outputTensor.get_desc());
        DataFormat dstF = outputDesc.df;
        int channelAxis = inputDesc.nDims - 2;
        U32 cx = 8;
        if (IS_X86(archInfo->arch)) {
            if (dstF == DF_NCHW || dstF == DF_MTK) {
                cx = 1;
            }
            if (inputDesc.dt == DT_U8_Q) {
                dstF = DF_NCHWC16;  // padding to 16
                cx = 16;
            }
        } else {
            dstF = DF_NCHWC8;
        }
        U32 paddedC = (inputDesc.dims[channelAxis] + cx - 1) / cx * cx;

        if (paddedC != inputDesc.dims[channelAxis] || (inputDesc.df != dstF)) {
            inputDesc.dims[channelAxis] = paddedC;
            *bytes += tensorNumBytes(inputDesc);
        }

        if (paddedC != outputDesc.dims[channelAxis] || (outputDesc.df != dstF)) {
            outputDesc.dims[channelAxis] = paddedC;
            *bytes += tensorNumBytes(outputDesc);
        }
    }
    return ret;
}
