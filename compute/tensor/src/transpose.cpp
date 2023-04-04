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
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif
#include <algorithm>

EE transpose_infer_output_size(
    Tensor *inputTensor, TransposeParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
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
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        ret = transpose_padding_input_mali(inputDesc, p, &outputDesc, inputMem, outputMem);
#endif
    } else {
        ret = transpose_infer_output_size_cpu(inputDesc, p, &outputDesc);
#ifdef _USE_CPU
        if (ret == SUCCESS && tensorIsShape(inputDesc)) {
            ret = transpose_cpu(inputDesc, inputDesc.dims, inputDesc.dims + inputDesc.nDims, p.axes,
                outputDesc, outputDesc.dims, outputDesc.dims + outputDesc.nDims);
        }
#endif
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE transpose_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        TensorDesc inputDesc = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        ret = transpose_infer_forward_tmp_bytes_mali(
            inputDesc, outputDesc, &gclmemInputDesc, &gclmemOutputDesc, bytes);
#endif
    } else {
        *bytes = 0;
        ret = SUCCESS;
    }
    return ret;
}

bool processC8Desc(DataFormat df, std::vector<U32> &dims, U32 cx)
{
    if (DF_NCHWC8 == df || DF_NCHWC16 == df) {
        CHECK_REQUIREMENT(dims[dims.size() - 2] % cx == 0);
        dims[dims.size() - 2] /= cx;
        dims.insert(dims.begin(), cx);
        return true;
    }
    return false;
}

EE transpose(Tensor inputTensor,
    TransposeParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    std::vector<U32> tmpDims(p.axes, p.axes + p.num_axes);
    std::vector<U32> inDims(inputDesc.dims, inputDesc.dims + inputDesc.nDims);
    std::vector<U32> outDims(outputDesc.dims, outputDesc.dims + outputDesc.nDims);
    if (IS_CPU(arch)) {
        U32 cx = 8;
        if (DF_NCHWC16 == inputDesc.df) {
            cx = 16;
        }
        bool padDimI = processC8Desc(inputDesc.df, inDims, cx);
        bool padDimO = processC8Desc(outputDesc.df, outDims, cx);
        if (p.axes[1] == 1 && padDimI && padDimO) {
            tmpDims.push_back(tmpDims.size());
        } else {
            if (padDimI) {
                auto ptr = std::find(tmpDims.begin(), tmpDims.end(), 1);
                U32 s = outputDesc.nDims - 1 - (ptr - tmpDims.begin());
                tmpDims.insert(ptr + 1, inputDesc.nDims);
                outDims[s] /= cx;
                outDims.insert(outDims.begin() + s, cx);
            }
            if (padDimO) {
                U32 s = p.axes[1];
                for (U32 &t : tmpDims) {
                    if (t > s) {
                        t += 1;
                    }
                }
                tmpDims.push_back(s + 1);
                s = inputDesc.nDims - 1 - p.axes[1];
                inDims[s] /= cx;
                inDims.insert(inDims.begin() + s, cx);
            }
        }
        inputDesc.nDims = inDims.size();
        outputDesc.nDims = outDims.size();
    }
    EE ret = NOT_SUPPORTED;
#ifdef _USE_CPU
    if (IS_CPU(arch)) {
        ret = transpose_cpu(
            inputDesc, inDims.data(), input, tmpDims.data(), outputDesc, outDims.data(), output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        ret = transpose_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, p, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
