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

inline EE slice_infer_output_size_kernel(
    TensorDesc inputDesc, SliceParamSpec p, std::vector<TensorDesc>& outputDesc, Arch arch)
{
    U32 num = outputDesc.size();
    int axis = (p.axis + inputDesc.nDims) % inputDesc.nDims;
    I32 *slice_points = p.slice_points;

    bool splitEqual = true;
    for (U32 i = 0; i < num; i++) {
        if (0 != slice_points[i]) {
            splitEqual = false;
            break;
        }
    }
    I32 target_axis = inputDesc.nDims - 1 - axis;
    I32 cDim = (I32)inputDesc.nDims - 2;
    if (splitEqual) {
        CHECK_REQUIREMENT(0 == inputDesc.dims[target_axis] % num);
        inputDesc.dims[target_axis] /= num;
    }
    for (U32 i = 0; i < num; i++) {
        outputDesc[i] = inputDesc;
        if (splitEqual) {
            continue;
        }

        I32 prev_point = 0;
        if (i > 0) {
            prev_point = slice_points[i - 1];
        }
        I32 next_point = inputDesc.dims[target_axis];
        if (i < num - 1) {
            next_point = slice_points[i];
        }
        if (i == 0 && num == 1 && p.num_slice == 1) {  // Could happen in onnx
            next_point = slice_points[0];
        }
        if (prev_point < 0) {
            prev_point = prev_point + inputDesc.dims[target_axis];
            if (prev_point < 0) {
                prev_point = 0;
            }
        }
        if (next_point < 0) {
            next_point = next_point + inputDesc.dims[target_axis];
            if (next_point < 0) {
                next_point = 0;
            }
        }
        outputDesc[i].dims[target_axis] = next_point - prev_point;
    }

    int cAlign = 8;
    if (IS_GPU(arch)) {
        cAlign = 4;
    }
    for (U32 i = 0; i < num; i++) {
        if ((cDim >= 0) && (outputDesc[i].dims[cDim] % cAlign != 0)) {
            if (outputDesc[i].nDims >= 4) {
                outputDesc[i].df = DF_NCHW;
            } else if (outputDesc[i].nDims == 3) {
                outputDesc[i].df = DF_MTK;
            } else if (outputDesc[i].nDims == 2) {
                outputDesc[i].df = DF_NORMAL;
            } else {
                return NOT_SUPPORTED;
            }
        }
    }

    EE ret = SUCCESS;
#ifdef _USE_CPU
    if (IS_CPU(arch) && tensorIsShape(inputDesc)) {
        std::vector<void *> output(num);
        for (U32 i = 0; i < num; i++) {
            output[i] = outputDesc[i].dims + outputDesc[i].nDims;
        }
        ret = slice_cpu(inputDesc, inputDesc.dims + inputDesc.nDims, p, outputDesc, output);
    }
#endif
    return ret;
}

EE slice_infer_output_size(
    Tensor *inputTensor, SliceParamSpec p, std::vector<Tensor *> outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        return NULL_POINTER;
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    std::vector<TensorDesc> outputDesc = get_desc_from_tensor_ptrs(outputTensor);
    auto arch = archInfo->arch;
    EE ret = slice_infer_output_size_kernel(inputDesc, p, outputDesc, arch);
#ifdef _USE_GPU
    if (IS_GPU(arch) && ret == SUCCESS) {
        OclMemory *inputMem = (OclMemory *)inputTensor->get_memory();
        std::vector<OclMemory *> outputMems;
        for (U32 i = 0; i < outputTensor.size(); i++) {
            outputMems.push_back((OclMemory *)outputTensor[i]->get_memory());
        }
        ret = slice_padding_input_mali(inputDesc, p, &outputDesc, inputMem, outputMems);
    }
#endif
    for (U32 i = 0; i < outputTensor.size(); i++) {
        outputTensor[i]->resize(outputDesc[i]);
    }
    return ret;
}

EE slice_infer_forward_tmp_bytes(Tensor inputTensor,
    SliceParamSpec p,
    std::vector<Tensor> outputTensor,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
        *bytes = 0;
        ret = SUCCESS;
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        TensorDesc inputDesc = inputTensor.get_desc();
        std::vector<TensorDesc> outputDesc = get_desc_from_tensors(outputTensor);
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        ret = slice_infer_forward_tmp_bytes_mali(inputDesc, gclmemInputDesc, p, outputDesc, bytes);
#endif
    }
    return ret;
}

EE slice(Tensor inputTensor,
    SliceParamSpec p,
    Tensor tmpTensor,
    std::vector<Tensor> outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    std::vector<TensorDesc> outputDesc = get_desc_from_tensors(outputTensor);
    std::vector<void *> output = get_data_from_tensors<void *>(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = slice_cpu(inputDesc, input, p, outputDesc, output);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        void *tmpbuf = get_ptr_from_tensor(tmpTensor, arch);
        ret = slice_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, (GCLMem_t)input, p,
            (GCLMem_t)tmpbuf, outputDesc, &output);
#endif
    }
    return ret;
}
