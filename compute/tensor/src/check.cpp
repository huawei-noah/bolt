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

EE check(Tensor inputTensorA,
    Tensor inputTensorB,
    CheckParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDescA = inputTensorA.get_desc();
    void *inputA = get_ptr_from_tensor(inputTensorA, arch);
    TensorDesc inputDescB = inputTensorB.get_desc();
    void *inputB = get_ptr_from_tensor(inputTensorB, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = check_cpu(inputDescA, inputA, inputDescB, inputB, p, outputDesc, output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = check_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDescA, (GCLMem_t)inputA,
            inputDescB, (GCLMem_t)inputB, p, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}

EE check_infer_output_size(Tensor *inputTensorA,
    Tensor *inputTensorB,
    CheckParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
{
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc desc0 = inputTensorA->get_desc();
    TensorDesc desc1 = inputTensorB->get_desc();
    TensorDesc outputDesc = desc0;
    for (U32 i = 0; i < UNI_MIN(desc0.nDims, desc1.nDims); i++) {
        int max_value = UNI_MAX(desc0.dims[i], desc1.dims[i]);
        int min_value = UNI_MIN(desc0.dims[i], desc1.dims[i]);
        if (min_value == 1) {
            desc0.dims[i] = max_value;
        } else {
            desc0.dims[i] = min_value;
        }
    }
    if (desc0.nDims < desc1.nDims) {
        for (U32 i = desc0.nDims; i < desc1.nDims; i++) {
            desc0.dims[i] = desc1.dims[i];
        }
        desc0.nDims = desc1.nDims;
    }
    outputDesc = desc0;
    outputDesc.dt = DT_U8;
    if (IS_GPU(archInfo->arch)) {
        outputDesc.dt = DT_I32;
    }
    EE ret = SUCCESS;
#ifdef _USE_CPU
    desc0 = inputTensorA->get_desc();
    if (IS_CPU(archInfo->arch) && tensorIsShape(desc0) && tensorIsShape(desc1) &&
        tensorIsShape(outputDesc)) {
        ret = check_cpu(desc0, desc0.dims + desc0.nDims, desc1, desc1.dims + desc1.nDims, p,
            outputDesc, outputDesc.dims + outputDesc.nDims);
    }
#endif
    outputTensor->resize(outputDesc);
    return ret;
}
