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
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/tensor_computing_arm.h"
#endif
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE softmax_infer_output_size(
    Tensor *inputTensor, SoftmaxParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = inputDesc;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        CHECK_STATUS(softmax_padding_input_mali(inputDesc, &outputDesc, inputMem, outputMem));
#endif
    } else {
        int axis = (p.axis + inputDesc.nDims) % inputDesc.nDims;
        if (DF_NCHWC8 == outputDesc.df && axis == 1) {
            outputDesc.df = DF_NCHW;
        }
        outputTensor->resize(outputDesc);
    }
    return SUCCESS;
}

EE softmax_infer_forward_tmp_bytes(
    Tensor inputTensor, SoftmaxParamSpec p, U32 *bytes, ArchInfo_t archInfo)
{
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        TensorDesc inputDesc = inputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        ret = softmax_infer_forward_tmp_bytes_mali(inputDesc, gclmemInputDesc, p, bytes);
#endif
    } else {
        *bytes = 0;
        ret = SUCCESS;
    }
    return ret;
}

EE softmax(
    Tensor inputTensor, SoftmaxParamSpec p, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = softmax_general(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = softmax_x86(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = softmax_arm(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        ret = softmax_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            p, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}

EE logsoftmax(
    Tensor inputTensor, SoftmaxParamSpec p, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = logsoftmax_general(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = logsoftmax_x86(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = logsoftmax_arm(inputDesc, input, p, outputDesc, output);
#endif
    }
    return ret;
}
