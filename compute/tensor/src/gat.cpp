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

EE gat(Tensor nodeFeature0,
    Tensor node0,
    Tensor nodeFeature1,
    Tensor node1,
    Tensor edgeFeature,
    GATParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc nodeFeatureDesc = nodeFeature0.get_desc();
    void *nodeFeaturePtr0 = get_ptr_from_tensor(nodeFeature0, arch);
    TensorDesc nodeDesc = node0.get_desc();
    void *nodePtr0 = get_ptr_from_tensor(node0, arch);
    void *nodeFeaturePtr1 = get_ptr_from_tensor(nodeFeature1, arch);
    void *nodePtr1 = get_ptr_from_tensor(node1, arch);
    TensorDesc edgeFeatureDesc = edgeFeature.get_desc();
    void *edgeFeaturePtr = get_ptr_from_tensor(edgeFeature, arch);
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;

    if (IS_CPU(arch)) {
        ret = gat_cpu(nodeFeatureDesc, nodeDesc, edgeFeatureDesc, nodeFeaturePtr0, nodePtr0,
            nodeFeaturePtr1, nodePtr1, edgeFeaturePtr, p, tmp, outputDesc, output, arch);
    }
    return ret;
}

EE gat_infer_output_size(
    Tensor *nodeFeatureTensor, GATParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (nodeFeatureTensor == nullptr || outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc nodeFeatureDesc = nodeFeatureTensor->get_desc();
    TensorDesc outputDesc;
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(archInfo->arch)) {
        ret = gat_infer_output_size_cpu(nodeFeatureDesc, p, &outputDesc);
    }
    outputTensor->resize(outputDesc);
    return ret;
}

EE gat_infer_forward_tmp_bytes(Tensor nodeFeatureTensor,
    Tensor edgeFeatureTensor,
    GATParamSpec p,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(archInfo->arch)) {
        TensorDesc nodeFeatureDesc = nodeFeatureTensor.get_desc();
        TensorDesc edgeFeatureDesc = edgeFeatureTensor.get_desc();
        ret = gat_infer_forward_tmp_bytes_cpu(nodeFeatureDesc, edgeFeatureDesc, p, bytes);
    }
    return ret;
}
