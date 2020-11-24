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
#ifdef _USE_MALI
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
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = check_general(inputDescA, inputA, inputDescB, inputB, p, outputDesc, output);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = check_x86(inputDescA, inputA, inputDescB, inputB, p, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = check_arm(inputDescA, inputA, inputDescB, inputB, p, outputDesc, output);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = check_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDescA, (GCLMem_t)inputA,
            inputDescB, (GCLMem_t)inputB, p, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}

EE check_infer_output_size(
    std::vector<Tensor *> inputTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    for (auto p : inputTensor) {
        if (p == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
    }
    TensorDesc inputDesc = inputTensor[0]->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDescA = ocl_get_desc(*(inputTensor[0]));
        GCLMemDesc gclmemInputDescB = ocl_get_desc(*(inputTensor[1]));
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = check_infer_output_size_mali(
            inputDesc, &outputDesc, &gclmemInputDescA, &gclmemInputDescB, &gclmemOutputDesc);
        ocl_set_desc(inputTensor[0], gclmemInputDescA);
        ocl_set_desc(inputTensor[1], gclmemInputDescB);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        outputDesc.dt = DT_I32;
        outputDesc.nDims = 1;
        outputDesc.dims[0] = inputDesc.dims[inputDesc.nDims - 1];
        ret = SUCCESS;
    }
    outputTensor->resize(outputDesc);
    return ret;
}
