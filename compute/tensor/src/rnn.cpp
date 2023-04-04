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
#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE rnn_transform_filter(std::vector<Tensor> filterTensors,
    RNNParamSpec rnnParamSpec,
    Tensor tmpTensor,
    std::vector<Tensor *> ftmTensors,
    ArchInfo_t archInfo,
    void *scale)
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
            rnnParamSpec, ftmDescs.data(), ftms.data(), (F32 *)scale, arch);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        std::vector<GCLMem> filterArray;
        std::vector<GCLMem> filterTranArray;
        for (U32 i = 0; i < filters.size(); i++) {
            filterArray.push_back(*((GCLMem_t)filters[i]));
        }
        for (U32 i = 0; i < ftms.size(); i++) {
            filterTranArray.push_back(*((GCLMem_t)ftms[i]));
        }
        GCLMem_t tmp = (GCLMem_t)get_ptr_from_tensor(tmpTensor, arch);
        ret = rnn_transform_filter_mali(((MaliPara_t)(archInfo->archPara))->handle, filterDescs[0],
            filterArray.data(), tmp, rnnParamSpec, ftmDescs.data(), filterTranArray.data(),
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    for (U32 i = 0; i < ftmTensors.size(); i++) {
        ftmTensors[i]->resize(ftmDescs[i]);
        if (scale != nullptr) {
            ftmTensors[i]->set_scale(((F32 *)scale)[i + 1]);
        }
    }
    return ret;
}

EE rnn_transform_filter_bytes(
    std::vector<Tensor> filterTensors, RNNParamSpec rnnParamSpec, void *bytes, ArchInfo_t archInfo)
{
    std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;

    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = rnn_transform_filter_bytes_cpu(filterDescs.data(), rnnParamSpec, (U32 *)bytes);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        ret = rnn_transform_filter_bytes_mali(filterDescs[0], rnnParamSpec,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, (TensorDesc *)bytes);
#endif
    }
    return ret;
}

EE rnn_infer_output_size(std::vector<Tensor *> inputTensors,
    RNNParamSpec rnnParamSpec,
    std::vector<Tensor *> outputTensors,
    ArchInfo_t archInfo)
{
    if (inputTensors.size() == 0) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputTensors.size() == 0) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensors[0]->get_desc();
    DataType idt = inputDesc.dt;
    DataFormat idf = inputDesc.df;
    U32 batch = inputDesc.dims[inputDesc.nDims - 1];
    U32 step = inputDesc.dims[inputDesc.nDims - 2];
    U32 xDim = inputDesc.dims[inputDesc.nDims - 3];
    for (U32 i = 0; i < inputDesc.nDims - 3; ++i) {
        xDim *= inputDesc.dims[i];
    }
    U32 num = (rnnParamSpec.bi_direction) ? 2 : 1;
    U32 hDim = num * rnnParamSpec.num_outputs;

    std::vector<TensorDesc> outputDescs;
    TensorDesc outputDesc = tensor3df(idt, DF_MTK, batch, step, hDim);
    outputDescs.push_back(outputDesc);
    U32 column = (rnnParamSpec.num_projection > 0) ? rnnParamSpec.num_projection
                                                   : rnnParamSpec.num_outputs;
    if (outputTensors.size() == 2) {
        if (rnnParamSpec.mode == RNN_LSTM) {
            outputDesc = tensor2df(idt, DF_NORMAL, batch, column + hDim);
        } else {
            outputDesc = tensor2df(idt, DF_NORMAL, batch, hDim);
        }
        outputDescs.push_back(outputDesc);
    } else if (outputTensors.size() == 3) {
        outputDesc = tensor2df(idt, DF_NORMAL, batch, hDim);
        outputDescs.push_back(outputDesc);
        outputDesc = tensor2df(idt, DF_NORMAL, batch, column);
        outputDescs.push_back(outputDesc);
    }
    for (U32 i = 0; i < outputDescs.size(); i++) {
        outputTensors[i]->resize(outputDescs[i]);
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
#endif
    } else if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        ret = rnn_infer_forward_tmp_bytes_mali(inputDesc, gclmemInputDesc, filterDesc, outputDesc,
            rnnParamSpec, bytes, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}

EE rnn_infer_forward_algorithm(Tensor inputTensor,
    std::vector<Tensor> filterTensors,
    std::vector<Tensor> biasTensors,
    RNNParamSpec rnnParamSpec,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
#ifdef _USE_GPU
    if (IS_GPU(archInfo->arch)) {
        TensorDesc inputDesc = inputTensor.get_desc();
        std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
        std::vector<TensorDesc> biasDescs = get_desc_from_tensors(biasTensors);
        TensorDesc outputDesc = outputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(outputTensor);
        ret = rnn_infer_forward_algorithm_mali(((MaliPara_t)(archInfo->archPara))->handle,
            inputDesc, filterDescs, biasDescs, rnnParamSpec, outputDesc, gclmemInputDesc,
            gclmemOutputDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
    }
#endif
    return ret;
}

EE rnn(std::vector<Tensor> inputTensors,
    std::vector<Tensor> filterTensors,
    std::vector<Tensor> biasTensors,
    RNNParamSpec rnnParamSpec,
    std::vector<Tensor> tmpTensors,
    std::vector<Tensor> outputTensors,
    void *scale,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> inputDescs = get_desc_from_tensors(inputTensors);
    std::vector<void *> inputs = get_data_from_tensors<void *>(inputTensors, arch);
    std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
    std::vector<void *> filters = get_data_from_tensors<void *>(filterTensors, arch);
    std::vector<TensorDesc> biasDescs = get_desc_from_tensors(biasTensors);
    std::vector<void *> biases = get_data_from_tensors<void *>(biasTensors, arch);
    std::vector<TensorDesc> outputDescs = get_desc_from_tensors(outputTensors);
    std::vector<void *> outputs = get_data_from_tensors<void *>(outputTensors, arch);
    U32 tmpBytes = tmpTensors[0].bytes();
    void *tmp = get_ptr_from_tensor(tmpTensors[0], arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = rnn_cpu(inputDescs[0], inputs[0], filterDescs.data(), (const void **)filters.data(),
            filterTensors.size(), biasDescs.data(), (const void **)biases.data(), (F32 *)scale, rnnParamSpec, tmpBytes,
            tmp, outputDescs[0], outputs[0], arch);
#endif
    } else if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        std::vector<GCLMem> input;
        std::vector<GCLMem> filter;
        std::vector<GCLMem> bias;
        std::vector<GCLMem> output;
        for (U32 i = 0; i < inputs.size(); i++) {
            input.push_back(*((GCLMem_t)inputs[i]));
        }
        for (U32 i = 0; i < filters.size(); i++) {
            filter.push_back(*((GCLMem_t)filters[i]));
        }
        for (U32 i = 0; i < biases.size(); i++) {
            bias.push_back(*((GCLMem_t)biases[i]));
        }
        for (U32 i = 0; i < outputs.size(); i++) {
            output.push_back(*((GCLMem_t)outputs[i]));
        }
        std::vector<GCLMem_t> tmpVec(2, NULL);
        for (U32 i = 0; i < tmpTensors.size(); i++) {
            tmpVec[i] = (GCLMem_t)get_ptr_from_tensor(tmpTensors[i], arch);
        }
        ret = rnn_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDescs, input.data(),
            filterDescs, filter.data(), biasDescs, bias.data(), rnnParamSpec, tmpVec, outputDescs,
            output.data(), ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
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
    U32 hDim = rnnParamSpec.num_outputs;
    outputDesc = tensor2df(idt, idf, batch, hDim);
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        U32 hDim_align = (hDim + 3) / 4 * 4;
        U32 pr = (hDim + 3) / 4 * 4 - hDim;
        OclMemory *stateMem = (OclMemory *)inputTensor[1]->get_memory();
        stateMem->padding(0, pr, 0, 0);
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
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
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
#ifdef _USE_GPU
    if (IS_GPU(archInfo->arch)) {
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

EE rnncell_transform_filter(std::vector<Tensor> filterTensors,
    RNNParamSpec rnnParamSpec,
    std::vector<Tensor *> ftmTensors,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
    std::vector<void *> filters = get_data_from_tensors<void *>(filterTensors, arch);
    std::vector<TensorDesc> ftmDescs(ftmTensors.size());
    std::vector<void *> ftms = get_data_from_tensor_ptrs<void *>(ftmTensors, arch);
    std::vector<float> scale(ftmTensors.size(), -1);

    EE ret = NOT_SUPPORTED;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        GCLMem filterArray[2];
        GCLMem filterTranArray[2];
        filterArray[0] = *((GCLMem_t)filters[0]);
        filterTranArray[0] = *((GCLMem_t)ftms[0]);
        if (rnnParamSpec.num_projection > 0) {
            filterArray[1] = *((GCLMem_t)filters[1]);
            filterTranArray[1] = *((GCLMem_t)ftms[1]);
        }
        ret = rnncell_transform_filter_mali(((MaliPara_t)(archInfo->archPara))->handle,
            filterDescs[0], filterArray, rnnParamSpec, ftmDescs.data(), filterTranArray,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    for (U32 i = 0; i < ftmTensors.size(); i++) {
        ftmTensors[i]->resize(ftmDescs[i]);
        ftmTensors[i]->set_scale(scale[i]);
    }
    return ret;
}

EE rnncell_transform_filter_bytes(
    std::vector<Tensor> filterTensors, RNNParamSpec rnnParamSpec, void *bytes, ArchInfo_t archInfo)
{
    std::vector<TensorDesc> filterDescs = get_desc_from_tensors(filterTensors);
    EE ret = NOT_SUPPORTED;
    auto arch = archInfo->arch;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        ret = rnncell_transform_filter_bytes_mali(filterDescs[0], rnnParamSpec,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo, (TensorDesc *)bytes);
#endif
    }
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
    void *scale,
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
    if (!IS_GPU(arch)) {
        tmp = (U8 *)tmp + tmpOffset;
    }
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = rnncell_cpu(xDesc, currentX, filterDescs.data(), (const void **)filters.data(),
            biasDescs.data(), (const void **)biases.data(), (F32 *)scale, state, rnnParamSpec,
            batchStrideX, batchStrideH, tmpBytes, tmp, hDesc, currentH, archInfo->arch);
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        GCLMem filterArray[2];
        GCLMem biasArray[2];
        filterArray[0] = *((GCLMem_t)filters[0]);
        biasArray[0] = *((GCLMem_t)biases[0]);
        if (rnnParamSpec.num_projection > 0) {
            filterArray[1] = *((GCLMem_t)filters[1]);
            //biasArray[1] = *((GCLMem_t)biases[1]);currently only init one bias
        }
        ret = rnncell_mali(((MaliPara_t)(archInfo->archPara))->handle, xDesc, (GCLMem_t)currentX,
            filterDescs[0], filterArray, biasDescs[0], biasArray, (GCLMem_t)state, rnnParamSpec,
            batchStrideX, batchStrideH, tmpBytes, (GCLMem_t)tmp, hDesc, (GCLMem_t)currentH,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    }
    return ret;
}
