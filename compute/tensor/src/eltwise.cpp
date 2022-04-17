// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <set>
#include "tensor_computing.h"
#if defined(_USE_CPU)
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif

// [1, 10, 10] + [1, 10, 10] = [1, 10, 10]
// [1, 10, 1] + [1, 1, 10] = [1, 10, 10]
// [1, 20, 10] + [10] = [1, 20, 10]
inline EE eltwise_infer_output_size_cpu(std::vector<TensorDesc> inputDesc, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        return NULL_POINTER;
    }
    U32 num = inputDesc.size();
    if (num <= 1) {
        return NOT_MATCH;
    }

    U32 arrayDimMax = 0;
    for (U32 i = 1; i < num; i++) {
        if (inputDesc[i].nDims > inputDesc[arrayDimMax].nDims) {
            arrayDimMax = i;
        }
    }
    U32 nchwc8Count = 0;
    U32 nchwc16Count = 0;
    U32 nhwcCount = 0;
    bool sameDim = true;
    for (U32 i = 0; i < num; i++) {
        if (inputDesc[i].df == DF_NCHWC8) {
            nchwc8Count++;
        }
        if (inputDesc[i].df == DF_NCHWC16) {
            nchwc16Count++;
        }
        // Kaldi tdnn special case
        if (inputDesc[i].df == DF_NHWC && inputDesc[i].nDims == 3) {
            nhwcCount++;
            std::swap(inputDesc[i].dims[0], inputDesc[i].dims[1]);
        }
        if (tensorNumElements(inputDesc[i]) != tensorNumElements(inputDesc[0])) {
            sameDim = false;
        }
    }

    *outputDesc = inputDesc[arrayDimMax];
    for (U32 i = 0; i < outputDesc->nDims; i++) {
        for (U32 j = 0; j < num; j++) {
            if (inputDesc[j].nDims > i) {
                int max_value = UNI_MAX(outputDesc->dims[i], inputDesc[j].dims[i]);
                int min_value = UNI_MIN(outputDesc->dims[i], inputDesc[j].dims[i]);
                if (min_value == 1) {
                    outputDesc->dims[i] = max_value;
                } else {
                    outputDesc->dims[i] = min_value;
                }
            }
        }
    }
    if (nchwc8Count > 0 && nchwc8Count != num) {
        outputDesc->df = DF_NCHWC8;
    }
    if (nchwc16Count > 0 && nchwc16Count != num) {
        outputDesc->df = DF_NCHWC16;
    }
    if (!sameDim && (nchwc8Count > 0 || nchwc16Count > 0)) {
        outputDesc->df = DF_NCHW;
    }
    return SUCCESS;
}

EE eltwise_infer_output_size(
    std::vector<Tensor *> inputTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    std::vector<TensorDesc> inputDesc = get_desc_from_tensor_ptrs(inputTensor);
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        std::vector<OclMemory *> inputMems;
        for (U32 i = 0; i < inputTensor.size(); i++) {
            inputMems.push_back((OclMemory *)inputTensor[i]->get_memory());
        }
        OclMemory *outputMem = (OclMemory *)outputTensor->get_memory();
        ret = eltwise_padding_input_mali(inputDesc, &outputDesc, inputMems, outputMem);
#endif
    } else {
        ret = eltwise_infer_output_size_cpu(inputDesc, &outputDesc);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE eltwise_infer_forward_tmp_bytes(
    std::vector<Tensor> inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    std::vector<TensorDesc> inputDesc = get_desc_from_tensors(inputTensor);
    TensorDesc outputDesc = outputTensor.get_desc();
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        std::vector<GCLMemDesc> gclmemInputDesc = ocl_get_descs(inputTensor);
        CHECK_STATUS(eltwise_infer_forward_tmp_bytes_mali(inputDesc, gclmemInputDesc, bytes));
#endif
    } else {
        std::set<DataFormat> nchw = {DF_NORMAL, DF_MTK, DF_MKT, DF_NCHW};
        *bytes = 0;
        for (U32 i = 0; i < inputDesc.size(); i++) {
            if (inputDesc[i].nDims <= 2 ||
                (nchw.find(inputDesc[i].df) != nchw.end() && nchw.find(outputDesc.df) != nchw.end())) {
                continue;
            }
            if (inputDesc[i].df != outputDesc.df ||
                tensorNumElements(inputDesc[i]) != tensorNumElements(outputDesc)) {
                *bytes += tensorNumBytes(inputDesc[i]);
            }
        }
    }
    return SUCCESS;
}

EE eltwise(std::vector<Tensor> inputTensor,
    EltwiseParamSpec eltwiseDesc,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> inputDesc = get_desc_from_tensors(inputTensor);
    std::vector<void *> input = get_data_from_tensors<void *>(inputTensor, arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
#if defined(_USE_NEON) && defined(_USE_INT8)
        for (U32 i = 0; i < inputTensor.size(); i++) {
            if (inputDesc[i].dt == DT_I8) {
                F32 scale = inputTensor[i].get_scale();
                Tensor bTensor, dTensor;
                inputDesc[i].dt = outputDesc.dt;
                dTensor.resize(inputDesc[i]);
                std::shared_ptr<U8> shared_data((U8 *)tmp, [](U8 *ptr) {});
                ((CpuMemory *)(dTensor.get_memory()))->set_shared_ptr(shared_data);
                CHECK_STATUS(dequantize(inputTensor[i], &scale, bTensor, dTensor, archInfo));
                input[i] = tmp;
                tmp = (U8 *)tmp + dTensor.bytes();
            }
        }
#endif
        ret = eltwise_cpu(inputDesc, input, eltwiseDesc, tmpBytes, tmp, outputDesc, output, arch);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = eltwise_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, input,
            eltwiseDesc, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
