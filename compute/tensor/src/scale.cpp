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

inline EE scale_infer_output_size_cpu(TensorDesc inputDesc, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }

    *outputDesc = inputDesc;
    return SUCCESS;
}

EE scale_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = scale_infer_output_size_mali(
            inputDesc, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        ret = scale_infer_output_size_cpu(inputDesc, &outputDesc);
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
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = scale_general(inputDesc, input, alpha, beta, p, outputDesc, output);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = scale_x86(inputDesc, input, alpha, beta, p, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = scale_arm(inputDesc, input, alpha, beta, p, outputDesc, output);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = scale_mali(((MaliPara_t)(archInfo->archPara))->handle, (GCLMem_t)alpha,
            (GCLMem_t)beta, inputDesc, (GCLMem_t)input, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
