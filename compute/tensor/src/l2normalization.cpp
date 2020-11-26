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

inline EE l2normalization_infer_output_size_cpu(TensorDesc inputDesc, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (tensorIs2d(inputDesc) || tensorIs3d(inputDesc)) {
        *outputDesc = inputDesc;
    } else if (tensorIs4d(inputDesc) && inputDesc.dims[0] == 1 && inputDesc.dims[1] == 1) {
        *outputDesc = inputDesc;
    } else {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

EE l2normalization_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    UNUSED(archInfo);
    CHECK_STATUS(l2normalization_infer_output_size_cpu(inputDesc, &outputDesc));
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE l2normalization(Tensor inputTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = l2normalization_cpu(inputDesc, input, outputDesc, output, arch);
#endif
    }
    return ret;
}
