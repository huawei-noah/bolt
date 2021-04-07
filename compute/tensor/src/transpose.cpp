// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include "tensor_computing.h"
#ifdef _USE_GENERAL
#include "cpu/general/tensor_computing_general.h"
#endif
#if defined(_USE_X86) || defined(_USE_NEON)
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

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
    std::vector<U32> tmpDims(p.trans_dims, p.trans_dims + p.trans_size);
    if (IS_CPU(arch)) {
        // Keep transDims unchanged so that input resize does not lead to error
        if (DF_NCHWC8 == inputDesc.df) {
            if (inputDesc.nDims == p.trans_size) {
                auto ptr = std::find(tmpDims.begin(), tmpDims.end(), 1);
                tmpDims.insert(ptr + 1, inputDesc.nDims);
            }
            inputDesc.nDims = inputDesc.nDims + 1;
            for (int i = inputDesc.nDims - 1; i > 0; i--) {
                inputDesc.dims[i] = inputDesc.dims[i - 1];
            }
            inputDesc.dims[0] = 8;
            inputDesc.dims[inputDesc.nDims - 2] /= 8;

            TensorDesc desc = outputDesc;
            desc.nDims = inputDesc.nDims;
            U32 idx = inputDesc.nDims - 1;
            for (int i = inputDesc.nDims - 2; i >= 0; i--) {
                if (1 == tmpDims[inputDesc.nDims - 2 - i]) {  // C
                    desc.dims[idx] = outputDesc.dims[i] / 8;
                    idx--;
                    desc.dims[idx] = 8;
                    idx--;
                } else {
                    desc.dims[idx] = outputDesc.dims[i];
                    idx--;
                }
            }
            outputDesc = desc;
        }
    }
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = transpose_general(inputDesc, input, tmpDims.data(), outputDesc, output);
#endif
#if defined(_USE_X86) || defined(_USE_NEON)
    } else if (IS_CPU(arch)) {
        ret = transpose_cpu(inputDesc, input, tmpDims.data(), outputDesc, output);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        void *tmp = get_ptr_from_tensor(tmpTensor, arch);
        ret = transpose_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (const GCLMem_t)input, p, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}

inline EE transpose_infer_output_size_cpu(
    TensorDesc inputDesc, TransposeParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }

    U32 *dim = p.trans_dims;
    *outputDesc = inputDesc;
    U32 inputDim = inputDesc.nDims;
    if (4 == inputDim) {
        (*outputDesc).df = DF_NCHW;
    }
    U32 outputDim = (*outputDesc).nDims;
    U32 index = 0;
    for (U32 i = 0; i < p.trans_size; i++) {
        // use 5-dim array to transpose a NCHWC8 tensor. skip c8 axis
        if (dim[i] >= inputDim) {
            continue;
        }
        // NOTE: TensorDesc.dims array is in [W H C N] order.
        // so if you want to transpose [N C H W] format data, we use (dims - 1 - *)
        // [5 6 7 8] + [0 3 2 1] = [5 8 7 6]
        // [8 7 6 5] + [0 3 2 1] = [6 7 8 5]
        (*outputDesc).dims[outputDim - 1 - index] = inputDesc.dims[inputDim - 1 - dim[i]];
        index++;
    }
    if ((*outputDesc).nDims >= 4) {
        (*outputDesc).df = DF_NCHW;
    }
    if ((*outputDesc).nDims == 4 && p.trans_size == 3 && (*outputDesc).dims[0] == 1) {
        (*outputDesc) = tensor3df(inputDesc.dt, DF_NCHW, (*outputDesc).dims[3],
            (*outputDesc).dims[2], (*outputDesc).dims[1]);
    }
    return SUCCESS;
}

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
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = transpose_infer_output_size_mali(
            inputDesc, p, &outputDesc, &gclmemInputDesc, &gclmemOutputDesc);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        ret = transpose_infer_output_size_cpu(inputDesc, p, &outputDesc);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE transpose_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
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
