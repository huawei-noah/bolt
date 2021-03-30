// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string.h>

#include "tensor_computing.h"
#include "blas_enhance.h"
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE multihead_attention_infer_output_size(Tensor *inputTensor,
    std::vector<Tensor> filterTensor,
    Tensor *outputTensor,
    U32 *firstFCSliceNum,
    ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    std::vector<TensorDesc> filterDesc = get_desc_from_tensors(filterTensor);
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemInputDesc = ocl_get_desc(*inputTensor);
        GCLMemDesc gclmemOutputDesc = ocl_get_desc(*outputTensor);
        ret = multihead_attention_infer_output_size_mali(inputDesc, filterDesc, &outputDesc,
            firstFCSliceNum, &gclmemInputDesc, &gclmemOutputDesc,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
        ocl_set_desc(inputTensor, gclmemInputDesc);
        ocl_set_desc(outputTensor, gclmemOutputDesc);
#endif
    } else {
        UNUSED(inputDesc);
        UNUSED(filterDesc);
        UNUSED(outputDesc);
        UNUSED(firstFCSliceNum);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE multihead_attention_infer_forward_algorithm(Tensor inputTensor,
    std::vector<Tensor> filterTensor,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    std::vector<TensorDesc> filterDesc = get_desc_from_tensors(filterTensor);
    TensorDesc outputDesc = outputTensor.get_desc();

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        ret = multihead_attention_infer_forward_algorithm_mali(
            ((MaliPara_t)(archInfo->archPara))->handle, inputDesc, filterDesc, multiplyAlpha,
            multiplyBeta, firstFCSliceNum, matmulSliceLen, eltwiseWithLayerNormIn, activation,
            outputDesc, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        UNUSED(inputDesc);
        UNUSED(filterDesc);
        UNUSED(multiplyAlpha);
        UNUSED(multiplyBeta);
        UNUSED(firstFCSliceNum);
        UNUSED(matmulSliceLen);
        UNUSED(eltwiseWithLayerNormIn);
        UNUSED(activation);
        UNUSED(outputDesc);
    }
    return ret;
}

EE multihead_attention_infer_forward_tmp_bytes(Tensor inputTensor,
    std::vector<Tensor> filterTensor,
    std::vector<bool> eltwiseWithLayerNormIn,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    TensorDesc inputDesc = inputTensor.get_desc();
    std::vector<TensorDesc> filterDesc = get_desc_from_tensors(filterTensor);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        ret = multihead_attention_infer_forward_tmp_bytes_mali(inputDesc, filterDesc,
            eltwiseWithLayerNormIn, firstFCSliceNum, matmulSliceLen, bytes,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        UNUSED(inputDesc);
        UNUSED(filterDesc);
        UNUSED(eltwiseWithLayerNormIn);
        UNUSED(firstFCSliceNum);
        UNUSED(matmulSliceLen);
        UNUSED(bytes);
    }
    return ret;
}

EE multihead_attention_transform_filter_bytes(
    std::vector<Tensor> filterTensor, U32 *bytes, ArchInfo_t archInfo)
{
    std::vector<TensorDesc> filterDesc = get_desc_from_tensors(filterTensor);

    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        CHECK_STATUS(multihead_attention_transform_filter_bytes_mali(filterDesc,
            ((MaliPara_t)(archInfo->archPara))->gclmemFilterDesc, bytes,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo));
#endif
    } else {
        UNUSED(filterTensor);
        UNUSED(bytes);
        UNUSED(archInfo);
    }
    return SUCCESS;
}

EE multihead_attention_transform_filter(
    std::vector<Tensor> filterTensor, std::vector<Tensor *> ftmTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> filterDesc = get_desc_from_tensors(filterTensor);
    std::vector<void *> filter = get_data_from_tensors<void *>(filterTensor, arch);
    std::vector<TensorDesc> ftmDesc = get_desc_from_tensor_ptrs(ftmTensor);
    std::vector<void *> filterTransformed = get_data_from_tensor_ptrs<void *>(ftmTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        ret = multihead_attention_transform_filter_mali(((MaliPara_t)(archInfo->archPara))->handle,
            filterDesc, filter, &ftmDesc, filterTransformed,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        UNUSED(filterTensor);
        UNUSED(ftmTensor);
        UNUSED(archInfo);
    }
    for (U32 i = 0; i < ftmTensor.size(); i++) {
        ftmTensor[i]->resize(ftmDesc[i]);
    }
    return ret;
}

EE multihead_attention(Tensor inputTensor,
    std::vector<Tensor> filterTensor,
    std::vector<Tensor> biasTensor,
    std::vector<Tensor> layerNormAlphaTensor,
    std::vector<Tensor> layerNormBetaTensor,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    void *input = get_ptr_from_tensor(inputTensor, arch);
    std::vector<TensorDesc> filterDesc = get_desc_from_tensors(filterTensor);
    std::vector<void *> filter = get_data_from_tensors<void *>(filterTensor, arch);
    std::vector<void *> layerNormAlpha = get_data_from_tensors<void *>(layerNormAlphaTensor, arch);
    ;
    std::vector<void *> layerNormBeta = get_data_from_tensors<void *>(layerNormBetaTensor, arch);
    std::vector<TensorDesc> biasDesc = get_desc_from_tensors(biasTensor);
    std::vector<void *> bias = get_data_from_tensors<void *>(biasTensor, arch);
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_MALI_GPU(arch)) {
#ifdef _USE_MALI
        ret = multihead_attention_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc,
            (GCLMem_t)input, filterDesc, filter, biasDesc, bias, layerNormAlpha, layerNormBeta,
            multiplyAlpha, multiplyBeta, firstFCSliceNum, matmulSliceLen, eltwiseWithLayerNormIn,
            activation, tmpBytes, (GCLMem_t)tmp, outputDesc, (GCLMem_t)output,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        UNUSED(inputDesc);
        UNUSED(filterDesc);
        UNUSED(biasTensor);
        UNUSED(layerNormAlpha);
        UNUSED(layerNormBeta);
        UNUSED(multiplyAlpha);
        UNUSED(multiplyBeta);
        UNUSED(firstFCSliceNum);
        UNUSED(matmulSliceLen);
        UNUSED(eltwiseWithLayerNormIn);
        UNUSED(activation);
        UNUSED(tmpTensor);
        UNUSED(outputDesc);
        UNUSED(output);
    }
    return ret;
}
