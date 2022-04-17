// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "image.h"
#ifdef _USE_CPU
#include "cpu/image_cpu.h"
#endif

EE grid_sample_infer_output_size(
    Tensor *inputTensor, Tensor *gridTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc gridDesc = gridTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
        ret = grid_sample_infer_output_size_cpu(inputDesc, gridDesc, &outputDesc);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE grid_sample_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor gridTensor,
    GridSampleParamSpec p,
    Tensor outputTensor,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    *bytes = 0;
    return SUCCESS;
}

EE grid_sample(Tensor inputTensor,
    Tensor gridTensor,
    GridSampleParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc gridDesc = gridTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *grid = get_ptr_from_tensor(gridTensor, arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
        ret = grid_sample_cpu(inputDesc, input, gridDesc, grid, p, tmp, outputDesc, output);
    }
    return ret;
}
