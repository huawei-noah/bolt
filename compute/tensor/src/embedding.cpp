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

EE embedding_infer_output_size(Tensor *inputTensor,
    EmbedParamSpec p,
    DataType outputDt,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
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
        ret = embedding_infer_output_size_mali(
            inputDesc, p, outputDt, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
#ifdef _USE_CPU
    } else {
        DataType dt;
        DataFormat df;
        U32 batch, step;
        bool inputOneDim = false;
        if (inputDesc.nDims == 1) {
            inputOneDim = true;
            inputDesc.nDims = 2;
            inputDesc.dims[1] = 1;
        }
        CHECK_REQUIREMENT(tensorIs2d(inputDesc));
        CHECK_STATUS(tensor2dGet(inputDesc, &dt, &df, &batch, &step));
        outputDesc = tensor3df(outputDt, DF_MTK, batch, step, p.num_output);
        if (inputOneDim) {
            outputDesc.nDims = 2;
            outputDesc.df = DF_NORMAL;
        }
        ret = SUCCESS;
#endif
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE embedding(Tensor inputTensor,
    Tensor weightTensor,
    EmbedParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *weight = get_ptr_from_tensor(weightTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        TensorDesc weightDesc = weightTensor.get_desc();
        ret = embedding_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            weightDesc, (GCLMem_t)weight, p, outputDesc, (GCLMem_t)output);
#endif
#ifdef _USE_CPU
    } else {
        ret = embedding_cpu(inputDesc, input, weight, p, outputDesc, output);
#endif
    }
    return ret;
}
