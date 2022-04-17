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
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE squeeze(Tensor inputTensor, Tensor tmpTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        void *tmpbuf = get_ptr_from_tensor(tmpTensor, arch);
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = squeeze_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input,
            (GCLMem_t)tmpbuf, outputDesc, (GCLMem_t)output);
#endif
#ifdef _USE_CPU
    } else {
        if ((inputDesc.df == DF_NCHWC8 || inputDesc.df == DF_NCHWC16) &&
            inputDesc.df != outputDesc.df) {
            TensorDesc nchwDesc = inputDesc;
            nchwDesc.df = DF_NCHW;
            transformToNCHW(inputDesc, input, nchwDesc, output);
        } else {
            UNI_MEMCPY(output, input, tensorNumBytes(inputDesc));
        }
        ret = SUCCESS;
#endif
    }
    return ret;
}

EE squeeze_infer_output_size_cpu(
    TensorDesc inputDesc, int *axes, int axesNum, TensorDesc *outputDesc)
{
    *outputDesc = inputDesc;
    if ((int)inputDesc.nDims == axesNum) {
        outputDesc->nDims = 1;
        outputDesc->df = DF_SCALAR;
        return SUCCESS;
    }
    for (int i = 0; i < axesNum; i++) {
        int axis = axes[i];
        if (axis < 0) {
            axis += inputDesc.nDims;
        }
        if (outputDesc->dims[inputDesc.nDims - 1 - axis] != 1) {
            UNI_ERROR_LOG(
                "try to squeeze non-one dimension in (%s).\n", tensorDesc2Str(inputDesc).c_str());
        }
        outputDesc->dims[inputDesc.nDims - 1 - axis] = INT_MAX;
    }
    U32 index = 0;
    for (U32 i = 0; i < inputDesc.nDims; i++) {
        if (outputDesc->dims[i] != INT_MAX) {
            outputDesc->dims[index++] = outputDesc->dims[i];
        }
    }
    CHECK_REQUIREMENT(index + axesNum == inputDesc.nDims);
    outputDesc->nDims = index;
    outputDesc->df = getTensorDefaultDataFormat(outputDesc->nDims);
    if (inputDesc.df == DF_NCHWC8 || inputDesc.df == DF_NCHWC16) {
        bool changeChannelAxis = false;
        for (int i = 0; i < axesNum; i++) {
            if (axes[i] < 1) {
                changeChannelAxis = true;
            }
        }
        if (!changeChannelAxis) {
            outputDesc->df = inputDesc.df;
        }
    }
    return SUCCESS;
}

EE squeeze_infer_output_size(
    Tensor *inputTensor, SqueezeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    CHECK_STATUS(squeeze_infer_output_size_cpu(inputDesc, p.axes, p.num_axes, &outputDesc));
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE squeeze_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = SUCCESS;
    *bytes = 0;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = squeeze_infer_forward_tmp_bytes_mali(
            inputDesc, gclmemInputDesc, outputDesc, gclmemOutputDesc, bytes);
#endif
    }
    return ret;
}
