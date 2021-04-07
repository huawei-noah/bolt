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
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

inline void processInputDescs(std::vector<TensorDesc> *inputDesc, I32 axis)
{
    int inputNum = inputDesc->size();
    int axisInfo = (axis > 0) ? axis : ((*inputDesc)[0].nDims + axis);
    axisInfo = (*inputDesc)[0].nDims - 1 - axisInfo;
    for (int i = 0; i < (int)(*inputDesc)[0].nDims; i++) {
        if (i == axisInfo) {
            continue;
        }
        U32 minDim = (*inputDesc)[0].dims[i];
        for (int j = 1; j < inputNum; j++) {
            if ((*inputDesc)[j].dims[i] < minDim) {
                minDim = (*inputDesc)[j].dims[i];
            }
        }
        if (minDim == 0) {
            continue;
        }
        for (int j = 0; j < inputNum; j++) {
            (*inputDesc)[j].dims[i] = minDim;
        }
    }
}

inline EE concat_infer_output_size_cpu(
    std::vector<TensorDesc> inputDesc, ConcatParamSpec p, TensorDesc *outputDesc)
{
    if (inputDesc.size() < 1) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (inputDesc.size() == 1) {
        *outputDesc = inputDesc[0];
        return SUCCESS;
    }

    bool hasC8 = false;
    for (U32 i = 1; i < inputDesc.size(); i++) {
        if (inputDesc[i].nDims != 0) {
            *outputDesc = inputDesc[i];
        }
        if (inputDesc[i].df == DF_NCHWC8) {
            hasC8 = true;
        }
    }
    I32 dim = outputDesc->nDims;
    int axis = p.axis;
    axis = (axis + dim) % dim;
    axis = dim - 1 - axis;
    outputDesc->dims[axis] = 0;

    for (U32 i = 0; i < inputDesc.size(); i++) {
        if (inputDesc[i].nDims == 0) {
            continue;
        }

        if (inputDesc[i].nDims != (U32)dim) {
            return NOT_MATCH;
        }

        for (I32 j = 0; j < dim; j++) {
            if (j == axis) {
                outputDesc->dims[j] += inputDesc[i].dims[j];
            } else {
                outputDesc->dims[j] = UNI_MAX(inputDesc[i].dims[j], outputDesc->dims[j]);
                if (inputDesc[i].dims[j] != 0 && outputDesc->dims[j] != 0 &&
                    outputDesc->dims[j] != inputDesc[i].dims[j]) {
                    return NOT_MATCH;
                }
            }
        }
    }

    int channel = outputDesc->nDims - 2;
    if ((outputDesc->dims[channel] % 8 == 0) && hasC8) {
        outputDesc->df = DF_NCHWC8;
    }

    if ((outputDesc->df == DF_NCHWC8) && (outputDesc->dims[channel] % 8 != 0)) {
        outputDesc->df = DF_NCHW;
    }

    return SUCCESS;
}

EE concat_infer_output_size(
    std::vector<Tensor *> inputTensor, ConcatParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    std::vector<TensorDesc> inputDesc = get_desc_from_tensor_ptrs(inputTensor);
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        std::vector<GCLMemDesc> gclmemInputDescs = ocl_get_descs_ptr(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = concat_infer_output_size_mali(
            inputDesc, p, &outputDesc, gclmemInputDescs.data(), &gclmemOutputDesc);
        ocl_set_descs(inputTensor, gclmemInputDescs);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        processInputDescs(&inputDesc, p.axis);
        ret = concat_infer_output_size_cpu(inputDesc, p, &outputDesc);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE concat_infer_forward_tmp_bytes(std::vector<Tensor> inputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    std::vector<TensorDesc> inputDesc = get_desc_from_tensors(inputTensor);
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        std::vector<GCLMemDesc> gclmemInputDescs = ocl_get_descs(inputTensor);
        ret = concat_infer_forward_tmp_bytes_mali(inputDesc, gclmemInputDescs, bytes);
#endif
    } else {
        *bytes = 0;
        for (auto p : inputDesc) {
            *bytes += tensorNumBytes(p);
        }
        ret = SUCCESS;
    }
    return ret;
}

EE concat(std::vector<Tensor> inputTensor,
    ConcatParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> inputDesc = get_desc_from_tensors(inputTensor);
    std::vector<F32> inputScale = get_scale_from_tensors(inputTensor);
    std::vector<void *> input = get_data_from_tensors<void *>(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    F32 outputScale = outputTensor.get_scale();
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        processInputDescs(&inputDesc, p.axis);
        ret = concat_cpu(
            inputDesc, input, inputScale.data(), p, tmp, outputDesc, output, &outputScale);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = concat_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, input, NULL, p,
            (GCLMem_t)tmp, outputDesc, (GCLMem_t)output, NULL);
#endif
    }
    outputTensor.set_scale(outputScale);
    return ret;
}
