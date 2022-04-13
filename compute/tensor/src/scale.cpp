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

inline EE scale_infer_output_size_cpu(
    TensorDesc inputDesc, ScaleParamSpec p, U32 axisLen, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    I32 axis = p.axis;
    U32 nDims = inputDesc.nDims;
    axis = (axis + nDims) % nDims;
    axis = nDims - 1 - axis;
    if (inputDesc.dims[axis] != 1 && inputDesc.dims[axis] != axisLen) {
        CHECK_STATUS(NOT_MATCH);
    }
    *outputDesc = inputDesc;
    (*outputDesc).dims[axis] = axisLen;
    return SUCCESS;
}

EE scale_infer_output_size(
    Tensor *inputTensor, ScaleParamSpec p, U32 axisLen, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = scale_infer_output_size_cpu(inputDesc, p, axisLen, &outputDesc);
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        if (inputDesc.df != DF_NCHWC4) {
            U32 iw = inputDesc.dims[0];
            U32 pr = (iw + 3) / 4 * 4 - iw;
            OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
            inputMem->padding(0, pr, 0, 0);
        }
#endif
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE scale(Tensor inputTensor,
    void *alpha,
    void *beta,
    ScaleParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = scale_cpu(inputDesc, input, alpha, beta, p, outputDesc, output, arch);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = scale_mali(((MaliPara_t)(archInfo->archPara))->handle, (GCLMem_t)alpha,
            (GCLMem_t)beta, p, inputDesc, (GCLMem_t)input, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
