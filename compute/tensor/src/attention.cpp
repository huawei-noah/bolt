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
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

EE attention(Tensor inputTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = attention_general(inputDesc, input, outputDesc, output);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = attention_x86(inputDesc, input, outputDesc, output);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = attention_arm(inputDesc, input, outputDesc, output);
#endif
    }
    return ret;
}

EE attention_infer_output_size(Tensor *inputTensor, AttentionParamSpec p, Tensor *outputTensor)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    DataType dt;
    DataFormat df;
    U32 batch, sequenceLength;
    CHECK_STATUS(tensor2dGet(inputDesc, &dt, &df, &batch, &sequenceLength));
    outputDesc =
        tensor4df(dt, DF_NCHW, batch, p.num_heads, p.from_sequence_length, p.to_sequence_length);
    outputTensor->resize(outputDesc);
    return SUCCESS;
}
