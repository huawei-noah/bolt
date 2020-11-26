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

inline EE reshape_infer_output_size_cpu(
    TensorDesc inputDesc, ReshapeParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        return NULL_POINTER;
    }

    I32 *shape = p.shape_dims;
    I32 shape_size = p.shape_size;

    int inputElementNum = tensorNumElements(inputDesc);
    int outputElementNum = 1;
    for (int i = 0; i < shape_size; i++) {
        outputElementNum *= shape[i];
    }
    int index_range = ((int)inputDesc.nDims > shape_size) ? shape_size : inputDesc.nDims;
    if (inputElementNum > 0 && outputElementNum > 0 && inputElementNum != outputElementNum) {
        for (int i = 0; i < index_range; i++) {
            if ((inputElementNum / (int)inputDesc.dims[inputDesc.nDims - 1 - i]) ==
                (outputElementNum / shape[i])) {
                shape[i] = inputDesc.dims[inputDesc.nDims - 1 - i];
                break;
            }
        }
    }

    *outputDesc = inputDesc;
    (*outputDesc).nDims = shape_size;
    if (shape_size == 2) {
        (*outputDesc).df = DF_NORMAL;
    }
    if (shape_size >= 4) {
        (*outputDesc).df = DF_NCHW;
    }

    U32 factor = 1;
    I32 count = 0;
    for (I32 i = 0; i < shape_size; i++) {
        I32 value = shape[i];
        if (value == 0) {
            value = inputDesc.dims[inputDesc.nDims - 1 - i];
        }
        if (value == -1) {
            value = 0;
            count++;
        } else {
            factor *= value;
        }

        (*outputDesc).dims[shape_size - 1 - i] = value;
    }
    if (count > 1) {
        return NOT_SUPPORTED;
    }

    for (I32 i = 0; i < shape_size; i++) {
        if ((*outputDesc).dims[i] == 0) {
            (*outputDesc).dims[i] = tensorNumElements(inputDesc) / factor;
        }
    }

    return SUCCESS;
}

EE reshape_infer_output_size(
    Tensor *inputTensor, ReshapeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
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
        ret = reshape_infer_output_size_mali(
            inputDesc, p, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
#ifdef _USE_CPU
    } else {
        ret = reshape_infer_output_size_cpu(inputDesc, p, &outputDesc);
#endif
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE reshape_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = reshape_infer_forward_tmp_bytes_mali(
            inputDesc, outputDesc, &gclmemInputDesc, &gclmemOutputDesc, bytes);
#endif
    } else {
        *bytes = UNI_MAX(inputTensor.bytes(), outputTensor.bytes());
        ret = SUCCESS;
    }
    return ret;
}

EE reshape(Tensor inputTensor, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        ret = reshape_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
#ifdef _USE_CPU
    } else {
        ret = reshape_cpu(inputDesc, input, outputDesc, output);
#endif
    }
    return ret;
}
