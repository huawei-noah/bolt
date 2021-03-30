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
#ifdef _USE_X86
#include "cpu/x86/tensor_computing_x86.h"
#endif
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE rnn_transform_filter(std::vector<Tensor> filterTensors,
    RNNParamSpec rnnParamSpec,
    std::vector<Tensor *> ftmTensors,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
    std::vector<void *> filters = get_data_from_tensors<void *>(filterTensors, arch);
    std::vector<TensorDesc> ftmDescs(ftmTensors.size());
    std::vector<void *> ftms = get_data_from_tensor_ptrs<void *>(ftmTensors, arch);

    EE ret = NOT_SUPPORTED;

    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = rnn_transform_filter_cpu(filterDescs.data(), (const void **)filters.data(),
            rnnParamSpec, ftmDescs.data(), ftms.data(), arch);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        GCLMem filterArray[2];
        GCLMem filterTranArray[2];
        filterArray[0] = *((GCLMem_t)filters[0]);
        filterTranArray[0] = *((GCLMem_t)ftms[0]);
        if (rnnParamSpec.numProjection > 0) {
            filterArray[1] = *((GCLMem_t)filters[1]);
            filterTranArray[1] = *((GCLMem_t)ftms[1]);
        }
        ret = rnn_transform_filter_mali(((MaliPara_t)(archInfo->archPara))->handle, filterDescs[0],
            filterArray, rnnParamSpec, ftmDescs.data(), filterTranArray,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    for (U32 i = 0; i < ftmTensors.size(); i++) {
        ftmTensors[i]->resize(ftmDescs[i]);
    }
    return ret;
}

EE rnn_transform_filter_bytes(
    std::vector<Tensor> filterTensors, RNNParamSpec rnnParamSpec, U32 *bytes, ArchInfo_t archInfo)
{
    std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;

    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = rnn_transform_filter_bytes_cpu(filterDescs.data(), rnnParamSpec, bytes);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = rnn_transform_filter_bytes_mali(filterDescs[0], rnnParamSpec,
            ((MaliPara_t)(archInfo->archPara))->gclmemFilterDesc, bytes,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}

EE rnn_infer_output_size(std::vector<Tensor *> inputTensors,
    RNNParamSpec rnnParamSpec,
    std::vector<Tensor *> outputTensors,
    ArchInfo_t archInfo)
{
    UNUSED(archInfo);
    if (inputTensors.size() == 0) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputTensors.size() == 0) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensors[0]->get_desc();
    TensorDesc outputDesc = outputTensors[0]->get_desc();
    DataType idt;
    DataFormat idf;
    U32 batch, step, xDim;
    CHECK_STATUS(tensor3dGet(inputDesc, &idt, &idf, &batch, &step, &xDim));
    U32 num = (rnnParamSpec.biDirection) ? 2 : 1;
    U32 hDim = num * rnnParamSpec.numOutput;
    outputDesc = tensor3df(idt, idf, batch, step, hDim);
    outputTensors[0]->resize(outputDesc);
    U32 column = (rnnParamSpec.numProjection > 0) ? rnnParamSpec.numProjection
                                                  : rnnParamSpec.numOutput;
    if (outputTensors.size() == 2) {
        if (rnnParamSpec.mode == RNN_LSTM) {
            outputDesc = tensor2df(idt, idf, batch, column + hDim);
        } else {
            outputDesc = tensor2df(idt, idf, batch, hDim);
        }
        outputTensors[1]->resize(outputDesc);
    } else if (outputTensors.size() == 3) {
        outputDesc = tensor2df(idt, idf, batch, column);
        outputTensors[1]->resize(outputDesc);
        outputDesc = tensor2df(idt, idf, batch, hDim);
        outputTensors[2]->resize(outputDesc);
    }
    return SUCCESS;
}

EE rnn_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();

    EE ret = NOT_SUPPORTED;

    if (IS_CPU(archInfo->arch)) {
#ifdef _USE_CPU
        ret = rnn_infer_forward_tmp_bytes_cpu(
            inputDesc, filterDesc, outputDesc, rnnParamSpec, bytes, archInfo->arch);
    }
#endif
    return ret;
}

EE rnn(Tensor inputTensor,
    std::vector<Tensor> filterTensors,
    std::vector<Tensor> biasTensors,
    RNNParamSpec rnnParamSpec,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
    std::vector<void *> filters = get_data_from_tensors<void *>(filterTensors, arch);
    std::vector<TensorDesc> biasDescs = get_desc_from_tensors(biasTensors);
    std::vector<void *> biases = get_data_from_tensors<void *>(biasTensors, arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = rnn_cpu(inputDesc, input, filterDescs.data(), (const void **)filters.data(),
            biasDescs.data(), (const void **)biases.data(), rnnParamSpec, tmpBytes, tmp, outputDesc,
            output, arch);
#endif
    }
    return ret;
}

EE rnncell_infer_output_size(std::vector<Tensor *> inputTensor,
    RNNParamSpec rnnParamSpec,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
{
    if (inputTensor[0] == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputTensor[1] == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor[0]->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    auto arch = archInfo->arch;
    DataType idt;
    DataFormat idf;
    U32 batch, xDim;
    CHECK_STATUS(tensor2dGet(inputDesc, &idt, &idf, &batch, &xDim));
    U32 hDim = rnnParamSpec.numOutput;
    outputDesc = tensor2df(idt, idf, batch, hDim);
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor[0]);
        GCLMemDesc gclmemStateDesc = ocl_get_desc(*inputTensor[1]);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        CHECK_STATUS(rnncell_infer_output_size_mali(inputDesc, rnnParamSpec, outputDesc,
            &gclmemInputDesc, &gclmemStateDesc, &gclmemOutputDesc));
        ocl_set_desc(inputTensor[0], gclmemInputDesc);
        ocl_set_desc(inputTensor[1], gclmemStateDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE rnncell_infer_forward_tmp_bytes(Tensor inputTensor,
    Tensor filterTensor,
    Tensor outputTensor,
    RNNParamSpec rnnParamSpec,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc filterDesc = filterTensor.get_desc();
    TensorDesc outputDesc = outputTensor.get_desc();

    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = rnncell_infer_forward_tmp_bytes_cpu(
            inputDesc, filterDesc, outputDesc, rnnParamSpec, bytes, archInfo->arch);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        ret = rnncell_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc, rnnParamSpec,
            bytes, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}

EE rnncell_infer_forward_algorithm(Tensor xTensor,
    Tensor filterTensor,
    Tensor biasTensor,
    Tensor stateTensor,
    RNNParamSpec rnncellDesc,
    U32 batchStrideX,
    U32 batchStrideH,
    Tensor hTensor,
    ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
#ifdef _USE_MALI
    if (IS_MALI_GPU(archInfo->arch)) {
        TensorDesc filterDesc = filterTensor.get_desc();
        TensorDesc biasDesc = biasTensor.get_desc();
        TensorDesc xDesc = xTensor.get_desc();
        TensorDesc hDesc = hTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(xTensor);
        GCLMemDesc gclmemStateDesc = ocl_get_desc(stateTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(hTensor);
        ret = rnncell_infer_forward_algorithm_mali(((MaliPara_t)(archInfo->archPara))->handle, xDesc,
            filterDesc, biasDesc, rnncellDesc, batchStrideX, batchStrideH, hDesc, gclmemInputDesc,
            gclmemStateDesc, gclmemOutputDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
    }
#endif
    return ret;
}

EE rnncell(Tensor xTensor,
    std::vector<Tensor> filterTensors,
    std::vector<Tensor> biasTensors,
    Tensor stateTensor,
    RNNParamSpec rnnParamSpec,
    U32 batchStrideX,
    U32 batchStrideH,
    U32 tmpOffset,
    Tensor tmpTensor,
    Tensor hTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc xDesc = xTensor.get_desc();
    void *currentX = get_ptr_from_tensor(xTensor, arch);
    std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
    std::vector<void *> filters = get_data_from_tensors<void *>(filterTensors, arch);
    std::vector<TensorDesc> biasDescs = get_desc_from_tensors(biasTensors);
    std::vector<void *> biases = get_data_from_tensors<void *>(biasTensors, arch);
    void *state = get_ptr_from_tensor(stateTensor, arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc hDesc = hTensor.get_desc();
    void *currentH = get_ptr_from_tensor(hTensor, arch);
    if (!IS_MALI_GPU(arch)) {
        tmp = (U8 *)tmp + tmpOffset;
    }
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = rnncell_cpu(xDesc, currentX, filterDescs.data(), (const void **)filters.data(),
            biasDescs.data(), (const void **)biases.data(), state, rnnParamSpec, batchStrideX,
            batchStrideH, tmpBytes, tmp, hDesc, currentH, archInfo->arch);
#endif
#ifdef _USE_MALI
    } else if (IS_MALI_GPU(arch)) {
        GCLMem filterArray[2];
        GCLMem biasArray[2];
        filterArray[0] = *((GCLMem_t)filters[0]);
        biasArray[0] = *((GCLMem_t)biases[0]);
        if (rnnParamSpec.numProjection > 0) {
            filterArray[1] = *((GCLMem_t)filters[1]);
            biasArray[1] = *((GCLMem_t)biases[1]);
        }
        ret = rnncell_mali(((MaliPara_t)(archInfo->archPara))->handle, xDesc, (GCLMem_t)currentX,
            filterDescs[0], filterArray, biasDescs[0], biasArray, (GCLMem_t)state, rnnParamSpec,
            batchStrideX, batchStrideH, tmpBytes, (GCLMem_t)tmp, hDesc, (GCLMem_t)currentH,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}
