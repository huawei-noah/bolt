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

EE scatter(Tensor dataTensor,
    Tensor indexTensor,
    Tensor updateTensor,
    ScatterParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)

{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        TensorDesc dataDesc = dataTensor.get_desc();
        void *data = get_ptr_from_tensor(dataTensor, arch);
        TensorDesc indexDesc = indexTensor.get_desc();
        void *index = get_ptr_from_tensor(indexTensor, arch);
        TensorDesc updateDesc = updateTensor.get_desc();
        void *update = get_ptr_from_tensor(updateTensor, arch);
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        TensorDesc outputDesc = outputTensor.get_desc();
        void *output = get_ptr_from_tensor(outputTensor, arch);
        ret = scatter_cpu(
            dataDesc, data, indexDesc, index, updateDesc, update, p, tmp, outputDesc, output);
#endif
    }
    return ret;
}

EE scatter_infer_output_size(Tensor *dataTensor,
    Tensor *indexTensor,
    Tensor *updateTensor,
    ScatterParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
{
    if (dataTensor == nullptr || indexTensor == nullptr || updateTensor == nullptr ||
        outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;
    TensorDesc dataDesc = dataTensor->get_desc();
    TensorDesc indexDesc = indexTensor->get_desc();
    TensorDesc updateDesc = updateTensor->get_desc();
    TensorDesc outputDesc = dataDesc;
    if (outputDesc.df == DF_NCHWC8) {
        outputDesc.df = DF_NCHW;
    }
#ifdef _USE_CPU
    if (IS_CPU(archInfo->arch) && tensorIsShape(dataDesc) && tensorIsShape(indexDesc) && tensorIsShape(updateDesc)) {
        ret = scatter_cpu(dataDesc, dataDesc.dims + dataDesc.nDims, indexDesc,
            indexDesc.dims + indexDesc.nDims, updateDesc, updateDesc.dims + updateDesc.nDims, p,
            nullptr, outputDesc, outputDesc.dims + outputDesc.nDims);
    }
#endif
    outputTensor->resize(outputDesc);
    return ret;
}

EE scatter_infer_forward_tmp_bytes(
    Tensor dataTensor, Tensor updateTensor, U32 *bytes, ArchInfo_t archInfo)
{
    *bytes = 0;
    if (dataTensor.get_desc().df == DF_NCHWC8) {
        *bytes += dataTensor.bytes();
    }
    if (updateTensor.get_desc().df == DF_NCHWC8) {
        *bytes += updateTensor.bytes();
    }
    return SUCCESS;
}
