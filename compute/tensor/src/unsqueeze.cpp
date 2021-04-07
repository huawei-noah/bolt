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
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif
#include <string.h>

EE unsqueeze(Tensor inputTensor, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        void *tmpbuf = get_ptr_from_tensor(tmpTensor, arch);
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = unsqueeze_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            (GCLMem_t)tmpbuf, outputDesc, (GCLMem_t)output);
#endif
#ifdef _USE_CPU
    } else {
        if (output != input) {
            memcpy(output, input, tensorNumBytes(inputDesc));
        }
        ret = SUCCESS;
#endif
    }
    return ret;
}

EE unsqueeze_infer_output_size_cpu(
    TensorDesc inputDesc, int *axes, int axesNum, TensorDesc *outputDesc)
{
    outputDesc->dt = inputDesc.dt;
    outputDesc->nDims = inputDesc.nDims + axesNum;
    outputDesc->df = getTensorDefaultDataFormat(outputDesc->nDims);
    for (U32 i = 0; i < outputDesc->nDims; i++) {
        outputDesc->dims[i] = 0;
    }
    for (int i = 0; i < axesNum; i++) {
        int axis = axes[i];
        if (axis < 0) {
            axis += outputDesc->nDims;
        }
        outputDesc->dims[outputDesc->nDims - 1 - axis] = 1;
    }
    U32 index = 0;
    for (U32 i = 0; i < outputDesc->nDims; i++) {
        if (outputDesc->dims[i] == 0) {
            outputDesc->dims[i] = inputDesc.dims[index++];
        }
    }
    CHECK_REQUIREMENT(index == inputDesc.nDims);
    return SUCCESS;
}

EE unsqueeze_infer_output_size(
    Tensor *inputTensor, UnsqueezeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
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
    ret = unsqueeze_infer_output_size_cpu(inputDesc, p.axes, p.axes_num, &outputDesc);
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = unsqueeze_infer_output_size_mali(
            inputDesc, outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE unsqueeze_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = SUCCESS;
    *bytes = 0;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = unsqueeze_infer_forward_tmp_bytes_mali(
            inputDesc, gclmemInputDesc, outputDesc, gclmemOutputDesc, bytes);
#endif
    }
    return ret;
}
