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
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE argmax(
    Tensor inputTensor, ArgMaxParamSpec p, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#if defined(_USE_CPU)
        ret = argmax_cpu(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        ret = argmax_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}

EE argmax_infer_forward_tmp_bytes(
    Tensor inputTensor, ArgMaxParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = argmax_infer_forward_tmp_bytes_mali(inputDesc, p, outputDesc, bytes);
#endif
    } else {
        *bytes = 0;
        ret = SUCCESS;
    }
    return ret;
}

EE argmax_infer_output_size(
    Tensor *inputTensor, ArgMaxParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    outputDesc = inputDesc;
    int axis = p.axis;
    if (axis < 0) {
        axis += inputDesc.nDims;
    }
    axis = inputDesc.nDims - 1 - axis;
    for (int i = axis; i < (I32)(inputDesc.nDims) - 1; i++) {
        outputDesc.dims[i] = outputDesc.dims[i + 1];
    }
    outputDesc.nDims = inputDesc.nDims - 1;
    outputDesc.dt = DT_I32;
    if (outputDesc.nDims == 2) {
        outputDesc.df = DF_NORMAL;
    }
    if (outputDesc.nDims == 3) {
        outputDesc.df = DF_MTK;
    }
    if (outputDesc.nDims >= 4) {
        outputDesc.df = DF_NCHW;
    }
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        CHECK_STATUS(argmax_padding_input_mali(inputDesc, p, &outputDesc, inputMem, outputMem));
#endif
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}
