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
    outputDesc = tensor3df(outputDt, DF_MTK, batch, step, p.num_outputs);
    if (inputOneDim) {
        outputDesc.nDims = 2;
        outputDesc.df = DF_NORMAL;
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE embedding(Tensor inputTensor,
    Tensor weightTensor,
    EmbedParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc weightDesc = weightTensor.get_desc();
    void *weight = get_ptr_from_tensor(weightTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        ret = embedding_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            weightDesc, (GCLMem_t)weight, p, outputDesc, (GCLMem_t)output);
#endif
#ifdef _USE_CPU
    } else {
#ifdef _USE_INT8
        if (weightDesc.dt != outputDesc.dt) {
            output = tmp;
        }
#endif
        ret = embedding_cpu(inputDesc, input, weight, p, weightDesc, outputDesc, output);
#endif
    }

#ifdef _USE_INT8
    if (weightDesc.dt != outputDesc.dt) {
        TensorDesc qDesc = outputDesc;
        outputDesc.dt = weightDesc.dt;
        F32 scaleO = -1;
        if (DT_I8 == qDesc.dt || DT_U8_Q == qDesc.dt) {
            CHECK_STATUS(quantize_cpu(outputDesc, output, &qDesc,
                get_ptr_from_tensor(outputTensor, arch), &scaleO, arch));
            outputTensor.set_scale(scaleO);
        } else {
            ret = NOT_SUPPORTED;
        }
    }
#endif
    return ret;
}
