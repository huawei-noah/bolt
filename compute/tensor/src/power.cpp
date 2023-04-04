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

inline EE power_infer_output_size_cpu(
    TensorDesc inputDesc, PowerParamSpec p, DataType tdt, TensorDesc *outputDesc, Arch arch)
{
    *outputDesc = inputDesc;
    if (outputDesc->dt == DT_U8) {
        if ((int)p.scale != p.scale || (int)p.shift != p.shift) {
            outputDesc->dt = tdt;
        }
    }
    EE ret = SUCCESS;
#ifdef _USE_CPU
    if (IS_CPU(arch) && tensorIsShape(inputDesc)) {
        float int_max = (float)INT_MAX;
        if (int_max - p.scale <= 1000) {
            p.scale = int_max - 100000;
        }
        if (int_max - p.shift <= 1000) {
            p.shift = int_max - 100000;
        }
        ret = power_cpu(inputDesc, inputDesc.dims + inputDesc.nDims, p, *outputDesc,
            outputDesc->dims + outputDesc->nDims, arch);
    }
#endif
    return ret;
}

EE power_infer_output_size(
    Tensor *inputTensor, PowerParamSpec p, DataType tdt, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return NULL_POINTER;
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = power_infer_output_size_cpu(inputDesc, p, tdt, &outputDesc, archInfo->arch);
    outputTensor->resize(outputDesc);
    return ret;
}

EE power(Tensor inputTensor, PowerParamSpec p, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = power_cpu(inputDesc, input, p, outputDesc, output, arch);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = power_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
