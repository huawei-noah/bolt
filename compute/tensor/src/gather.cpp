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

EE gather(Tensor dataTensor,
    Tensor indexTensor,
    GatherParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    EE ret = NOT_SUPPORTED;
    TensorDesc dataDesc = dataTensor.get_desc();
    void *data = get_ptr_from_tensor(dataTensor, arch);
    TensorDesc indexDesc = indexTensor.get_desc();
    void *index = get_ptr_from_tensor(indexTensor, arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = gather_cpu(dataDesc, data, indexDesc, index, p, tmp, outputDesc, output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = gather_mali(((MaliPara_t)(archInfo->archPara))->handle, dataDesc, (GCLMem_t)data,
            indexDesc, (GCLMem_t)index, p, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}

EE gather_infer_output_size(Tensor *dataTensor,
    Tensor *indexTensor,
    GatherParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    if (dataTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc dataDesc = dataTensor->get_desc();
    TensorDesc indexDesc = indexTensor->get_desc();
    int axis = (p.axis + dataDesc.nDims) % dataDesc.nDims;
    axis = dataDesc.nDims - 1 - axis;
    TensorDesc outputDesc = indexDesc;
    if (p.axis == INT_MAX) {
        int data_rank = dataDesc.nDims;
        int k = indexDesc.dims[indexDesc.nDims - 1];
        CHECK_REQUIREMENT(k <= data_rank);
        int e = data_rank - p.batch_dims - k;
        for (U32 i = 0; i < indexDesc.nDims; i++) {
            outputDesc.dims[e + i] = indexDesc.dims[i];
        }
        outputDesc.nDims = e + indexDesc.nDims;
    } else {
        outputDesc = dataDesc;
        if (!p.element_level) {
            if (tensorNumElements(indexDesc) == 1 && p.index_scalar) {
                for (int i = axis; i < (int)outputDesc.nDims - 1; i++) {
                    outputDesc.dims[i] = outputDesc.dims[i + 1];
                }
                outputDesc.nDims--;
            } else {
                for (int i = (int)outputDesc.nDims - 1; i > axis; i--) {
                    outputDesc.dims[i + indexDesc.nDims - 1] = outputDesc.dims[i];
                }
                for (U32 i = 0; i < indexDesc.nDims; i++) {
                    outputDesc.dims[axis + i] = indexDesc.dims[i];
                }
                outputDesc.nDims += indexDesc.nDims - 1;
            }
        }
    }
    if (IS_CPU(arch)) {
        if (outputDesc.df == DF_NCHWC8) {
            outputDesc.df = DF_NCHW;
        }
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        if (outputDesc.df == DF_NCHWC4) {
            outputDesc.df = DF_NCHW;
        }
#endif
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE gather_infer_forward_tmp_bytes(Tensor dataTensor,
    Tensor indexTensor,
    GatherParamSpec p,
    Tensor outputTensor,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;

    if (IS_CPU(arch)) {
        if (dataTensor.get_desc().df == DF_NCHWC8) {
            *bytes = dataTensor.bytes();
        } else {
            *bytes = 0;
        }
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        TensorDesc dataDesc = dataTensor.get_desc();
        TensorDesc indexDesc = indexTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        GCLMemDesc gclmemDataDesc = ocl_get_desc(dataTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        CHECK_STATUS(gather_infer_forward_tmp_bytes_mali(
            dataDesc, gclmemDataDesc, indexDesc, p, outputDesc, gclmemOutputDesc, bytes));
#endif
    }
    return SUCCESS;
}
