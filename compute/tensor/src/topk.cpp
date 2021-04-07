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
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE topk(Tensor inputTensor,
    TopKParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    Tensor outputIndicesTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
#ifdef _USE_MALI
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    TensorDesc outputIndicesDesc = outputIndicesTensor.get_desc();
    void *outputIndices = get_ptr_from_tensor(outputIndicesTensor, arch);
    if (IS_MALI_GPU(arch)) {
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        ret = topk_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            (GCLMem_t)tmp, outputDesc, (GCLMem_t)output, outputIndicesDesc, (GCLMem_t)outputIndices);
    }
#endif
    return ret;
}

EE topk_infer_forward_tmp_bytes(
    Tensor inputTensor, TopKParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = topk_infer_forward_tmp_bytes_mali(inputDesc, p, outputDesc, bytes);
#endif
    } else {
        *bytes = 0;
        ret = SUCCESS;
    }
    return ret;
}

EE topk_infer_output_size(Tensor *inputTensor,
    TopKParamSpec p,
    Tensor *outputTensor,
    Tensor *outputIndicesTensor,
    ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputIndicesTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc outputDesc = outputTensor->get_desc();
    TensorDesc outputIndicesDesc = outputIndicesTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor->get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        GCLMemDesc gclmemOutputIndicesDesc = ocl_get_desc(*outputIndicesTensor);
        ret = topk_infer_output_size_mali(inputDesc, p, &outputDesc, &outputIndicesDesc,
            &gclmemInputDesc, &gclmemOutputDesc, &gclmemOutputIndicesDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
        ocl_set_desc(outputIndicesTensor, gclmemOutputIndicesDesc);
#endif
    }
    outputTensor->resize(outputDesc);
    outputIndicesTensor->resize(outputIndicesDesc);
    return ret;
}
