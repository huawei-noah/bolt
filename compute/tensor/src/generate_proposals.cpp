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

EE generate_proposals_infer_output_size(Tensor *deltaTensor,
    Tensor *logitTensor,
    GenerateProposalsParamSpec generateProposalsParam,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
{
    if (deltaTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (logitTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc deltaDesc = deltaTensor->get_desc();
    TensorDesc logitDesc = logitTensor->get_desc();
    U32 blockDim = tensorNumElements(deltaDesc) / tensorNumElements(logitDesc);
    U32 postNmsTop = generateProposalsParam.post_nms_topN;
    TensorDesc outputDesc = tensor2df(deltaDesc.dt, DF_NORMAL, postNmsTop, blockDim);
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE generate_proposals_infer_forward_tmp_bytes(Tensor deltaTensor,
    Tensor logitTensor,
    GenerateProposalsParamSpec generateProposalsParam,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc deltaDesc = deltaTensor.get_desc();
    TensorDesc logitDesc = logitTensor.get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        GCLMemDesc gclMemLogitDesc = ocl_get_desc(logitTensor);
        ret = generate_proposals_infer_forward_tmp_bytes_mali(
            deltaDesc, logitDesc, gclMemLogitDesc, generateProposalsParam, bytes);
#endif
    }
    return ret;
}

EE generate_proposals(Tensor deltaTensor,
    Tensor logitTensor,
    Tensor imgInfoTensor,
    Tensor anchorTensor,
    GenerateProposalsParamSpec generateProposalsParam,
    std::vector<Tensor> tmpTensors,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc deltaDesc = deltaTensor.get_desc();
    void *delta = get_ptr_from_tensor(deltaTensor, arch);
    TensorDesc logitDesc = logitTensor.get_desc();
    void *logit = get_ptr_from_tensor(logitTensor, arch);
    TensorDesc imgInfoDesc = imgInfoTensor.get_desc();
    void *imgInfo = get_ptr_from_tensor(imgInfoTensor, arch);
    TensorDesc anchorDesc = anchorTensor.get_desc();
    void *anchor = get_ptr_from_tensor(anchorTensor, arch);
    void *tmp = get_ptr_from_tensor(tmpTensors[0], arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
#endif
#ifdef _USE_GPU
    } else if (IS_GPU(arch)) {
        void *tmpCpu = get_ptr_from_tensor(tmpTensors[1], CPU_GENERAL);
        ret = generate_proposals_mali(((MaliPara_t)(archInfo->archPara))->handle, deltaDesc,
            (GCLMem_t)delta, logitDesc, (GCLMem_t)logit, imgInfoDesc, (GCLMem_t)imgInfo, anchorDesc,
            (GCLMem_t)anchor, generateProposalsParam, (GCLMem_t)tmp, (U8 *)tmpCpu, outputDesc,
            (GCLMem_t)output);
#endif
    }
    return ret;
}
