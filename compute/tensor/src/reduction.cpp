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

EE reduction(Tensor inputTensor,
    Tensor maskTensor,
    ReductionParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc maskDesc = maskTensor.get_desc();
    void *mask = get_ptr_from_tensor(maskTensor, arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = reduction_cpu(
            inputDesc, input, maskDesc, mask, p, tmpBytes, tmp, outputDesc, output, arch);
#endif
    } else if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        ret = reduction_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            maskDesc, (GCLMem_t)mask, p, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}

EE reduction_infer_forward_tmp_bytes(
    Tensor inputTensor, ReductionParamSpec p, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc outputDesc = outputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        CHECK_STATUS(reduction_infer_forward_tmp_bytes_mali(
            inputDesc, p, outputDesc, gclmemInputDesc, gclmemOutputDesc, bytes));
        return SUCCESS;
#endif
    }
    int factor = 0;
    if (p.axes_num > 1) {
        factor = 2;
    }
    if (inputDesc.df == DF_NCHWC8) {
        for (int i = 0; i < p.axes_num; i++) {
            // channel dimension
            if (p.axes[i] == 1 || p.axes[i] == -3) {
                factor = 2;
                break;
            }
        }
    }
    *bytes = UNI_MAX(inputTensor.bytes(), outputTensor.bytes()) * factor;
    return SUCCESS;
}

EE reduction_infer_output_size(Tensor *inputTensor,
    Tensor maskTensor,
    ReductionParamSpec p,
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
    TensorDesc maskDesc = maskTensor.get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        CHECK_STATUS(reduction_infer_output_size_mali(
            inputDesc, maskDesc, p, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc));
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
        outputTensor->resize(outputDesc);
        return SUCCESS;
#endif
    }
    int start = 0;
    TensorDesc tmpDesc = inputDesc;
    if (inputDesc.df == DF_NCHWC8) {
        for (int i = 0; i < p.axes_num; i++) {
            // channel dimension
            if (p.axes[i] == 1 || p.axes[i] == -3) {
                start = -1;
                break;
            }
        }
        for (int i = (int)tmpDesc.nDims - 1; i >= 0; i--) {
            tmpDesc.dims[i + 1] = tmpDesc.dims[i];
        }
        tmpDesc.dims[tmpDesc.nDims - 1] /= 8;
        tmpDesc.dims[0] = 8;
        tmpDesc.nDims += 1;
    }
    outputDesc = tmpDesc;
    for (int i = start; i < p.axes_num; i++) {
        int axis;
        if (i == -1) {
            axis = 4;
        } else {
            axis = p.axes[i];
        }
        if (axis < 0) {
            axis = tmpDesc.nDims + axis;
        }
        axis = tmpDesc.nDims - 1 - axis;
        if (tensorNumElements(maskDesc) == 0) {
            outputDesc.dims[axis] = 0;
        } else {
            int num = maskDesc.dims[1] > 1 ? maskDesc.dims[1] : 0;
            outputDesc.dims[axis] = num;
        }
    }
    if (p.keep_dim) {
        for (U32 i = 0; i < tmpDesc.nDims; i++) {
            if (outputDesc.dims[i] == 0) {
                outputDesc.dims[i] = 1;
            }
        }
        outputDesc.nDims = tmpDesc.nDims;
    } else {
        int index = 0;
        for (U32 i = 0; i < tmpDesc.nDims; i++) {
            if (outputDesc.dims[i] != 0) {
                outputDesc.dims[index++] = outputDesc.dims[i];
            }
        }
        outputDesc.nDims = index;
    }
    outputDesc.df = getTensorDefaultDataFormat(outputDesc.nDims);
    if (inputDesc.df == DF_NCHWC8) {
        if (start == 0) {
            outputDesc.df = DF_NCHWC8;
            for (int i = 0; i < (int)outputDesc.nDims - 1; i++) {
                outputDesc.dims[i] = outputDesc.dims[i + 1];
            }
            outputDesc.nDims -= 1;
            outputDesc.dims[outputDesc.nDims - 2] *= 8;
        }
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}
