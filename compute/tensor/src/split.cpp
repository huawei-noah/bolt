// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>
#include "tensor_computing.h"
#if defined(_USE_GENERAL) || defined(_USE_NEON) || defined(_USE_X86)
#include "cpu/tensor_computing_cpu.h"
#endif

EE split_infer_output_size(Tensor *inputTensor, std::vector<Tensor *> output)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    for (auto p : output) {
        if (p == nullptr) {
            CHECK_STATUS(NULL_POINTER);
        }
        p->resize(inputDesc);
    }
    return SUCCESS;
}

EE split(Tensor inputTensor, std::vector<Tensor> outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    std::vector<TensorDesc> outputDesc = get_desc_from_tensors(outputTensor);
    std::vector<void *> output = get_data_from_tensors<void *>(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#if defined(_USE_GENERAL) || defined(_USE_NEON) || defined(_USE_X86)
        ret = split_cpu(inputDesc, input, outputDesc, &output);
#endif
    }
    return ret;
}
