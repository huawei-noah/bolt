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
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif

EE instance_norm(Tensor inputTensor,
    Tensor tmpTensor,
    Tensor scaleTensor,
    Tensor biasTensor,
    InstanceNormParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    void *scale = (void *)get_ptr_from_tensor(scaleTensor, archInfo->arch);
    void *bias = (void *)get_ptr_from_tensor(biasTensor, archInfo->arch);
    void *input = get_ptr_from_tensor(inputTensor, archInfo->arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, archInfo->arch);
    void *output = get_ptr_from_tensor(outputTensor, archInfo->arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(archInfo->arch)) {
#ifdef _USE_CPU
        ret = instance_norm_cpu(
            inputTensor.get_desc(), input, tmp, scale, bias, p, output, archInfo->arch);
#endif
    }
    return ret;
}

EE instance_norm_infer_forward_tmp_bytes(
    TensorDesc inputDesc, InstanceNormParamSpec p, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(archInfo->arch)) {
#ifdef _USE_CPU
        if (inputDesc.df != DF_NCHWC8) {
            *bytes = 0;
            return SUCCESS;
        }
        if (IS_GENERAL(archInfo->arch) || IS_ARM(archInfo->arch)) {
#if defined(_USE_GENERAL) || defined(_USE_NEON)
            ret = instance_norm_infer_forward_tmp_bytes_cpu(inputDesc, p, bytes);
#endif
#ifdef _USE_X86
        } else if (IS_X86(archInfo->arch)) {
            ret = instance_norm_infer_forward_tmp_bytes_x86(inputDesc, p, bytes);
#endif
        }
#endif
    }
    return ret;
}
