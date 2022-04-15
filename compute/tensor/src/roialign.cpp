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

inline EE roialign_infer_output_size_cpu(
    std::vector<TensorDesc> inputDesc, RoIAlignParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    CHECK_REQUIREMENT(inputDesc.size() >= 2);
    DataType idt0, idt1, idt2;
    DataFormat idf0, idf1, idf2;
    U32 in0, ic0, ih0, iw0;
    U32 ih1, iw1;
    U32 ilens2;
    // feature map
    CHECK_STATUS(tensor4dGet(inputDesc[0], &idt0, &idf0, &in0, &ic0, &ih0, &iw0));
    // rois
    CHECK_STATUS(tensor2dGet(inputDesc[1], &idt1, &idf1, &ih1, &iw1));
    // bacth indices
    if (inputDesc.size() == 3) {
        CHECK_STATUS(tensor1dGet(inputDesc[2], &idt2, &idf2, &ilens2));
        CHECK_REQUIREMENT(ih1 == ilens2);
    }
    CHECK_REQUIREMENT(iw1 == 4);
    // output size
    U32 on, oc, oh, ow;
    // on = num_rois, oc = ic, oh = output_h, ow = output_w
    on = ih1;
    oc = ic0;
    oh = p.output_h;
    ow = p.output_w;
    *outputDesc = tensor4df(idt0, DF_NCHW, on, oc, oh, ow);
    return SUCCESS;
}

EE roialign_infer_output_size(
    std::vector<Tensor *> inputTensor, RoIAlignParamSpec p, Tensor *outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    std::vector<TensorDesc> inputDesc = get_desc_from_tensor_ptrs(inputTensor);
    TensorDesc outputDesc = outputTensor->get_desc();
    CHECK_STATUS(roialign_infer_output_size_cpu(inputDesc, p, &outputDesc));
    if (IS_GPU(arch)) {
        TensorDesc desc = inputDesc[0];
        U32 ic = desc.dims[desc.nDims - 2];
        if (desc.df == DF_NCHWC4 && (ic & 3) == 0) {
            outputDesc.df = DF_NCHWC4;
        }
    }
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE roialign_infer_forward_tmp_bytes(
    Tensor inputTensor, Tensor outputTensor, U32 *bytes, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        TensorDesc inputDesc = inputTensor.get_desc();
        GCLMemDesc gclmemInputDesc = ocl_get_desc(inputTensor);
        TensorDesc outputDesc = outputTensor.get_desc();
        CHECK_STATUS(
            roialign_infer_forward_tmp_bytes_mali(inputDesc, gclmemInputDesc, outputDesc, bytes));
#endif
    } else {
        *bytes = 0;
    }
    return SUCCESS;
}

EE roialign(std::vector<Tensor> inputTensor,
    RoIAlignParamSpec p,
    Tensor tmpTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> inputDesc = get_desc_from_tensors(inputTensor);
    std::vector<void *> input = get_data_from_tensors<void *>(inputTensor, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = roialign_cpu(inputDesc, input, p, outputDesc, output);
#endif
    } else if (IS_GPU(arch)) {
#ifdef _USE_GPU
        void *tmpbuf = get_ptr_from_tensor(tmpTensor, arch);
        ret = roialign_mali(((MaliPara_t)(archInfo->archPara))->handle, inputDesc, input, p,
            (GCLMem_t)tmpbuf, outputDesc, (GCLMem_t)output);
#endif
    }
    return ret;
}
