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

inline EE priorbox_infer_output_size_cpu(
    std::vector<TensorDesc> inputDesc, PriorBoxParamSpec priorBoxParamSpec, TensorDesc *outputDesc)
{
    std::vector<F32> minsizes;
    for (int i = 0; i < 2; i++) {
        if (priorBoxParamSpec.min_sizes[i] == 0) {
            break;
        }
        minsizes.push_back(priorBoxParamSpec.min_sizes[i]);
    }
    std::vector<F32> maxsizes;
    for (int i = 0; i < 2; i++) {
        if (priorBoxParamSpec.max_sizes[i] == 0) {
            break;
        }
        maxsizes.push_back(priorBoxParamSpec.max_sizes[i]);
    }
    std::vector<F32> ars;
    for (int i = 0; i < 2; i++) {
        if (priorBoxParamSpec.aspect_ratios[i] == 0) {
            break;
        }
        ars.push_back(priorBoxParamSpec.aspect_ratios[i]);
    }
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt;
    DataFormat idf;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensor4dGet(inputDesc[0], &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_REQUIREMENT(!ars.empty());
    U32 num_priorboxs = ars.size();
    if (priorBoxParamSpec.flip) {
        num_priorboxs = num_priorboxs * 2;
    }
    CHECK_REQUIREMENT(!minsizes.empty());
    U32 num_minsize = minsizes.size();
    num_priorboxs = num_priorboxs * num_minsize + num_minsize;
    if (!maxsizes.empty()) {
        U32 num_maxsize = maxsizes.size();
        CHECK_REQUIREMENT(num_minsize == num_maxsize);
        num_priorboxs = num_priorboxs + num_maxsize;
    }
    UNI_DEBUG_LOG("Number of priorboxs per pixel: %u\n", num_priorboxs);
    // on = 1, oc = 2, ol= 4*num_priorboxs*ih*iw
    if (DT_I8 == idt) {
        idt = DT_F16;
    }
    *outputDesc = tensor3d(idt, 1, 2, 4 * num_priorboxs * ih * iw);
    return SUCCESS;
}

EE priorbox_infer_output_size(std::vector<Tensor *> inputTensor,
    PriorBoxParamSpec priorBoxParamSpec,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
{
    UNUSED(archInfo);
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    std::vector<TensorDesc> inputDesc = get_desc_from_tensor_ptrs(inputTensor);
    TensorDesc outputDesc = outputTensor->get_desc();
    CHECK_STATUS(priorbox_infer_output_size_cpu(inputDesc, priorBoxParamSpec, &outputDesc));
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE priorbox(std::vector<Tensor> inputTensor,
    PriorBoxParamSpec priorBoxParamSpec,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    std::vector<TensorDesc> inputDesc = get_desc_from_tensors(inputTensor);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
#ifdef _USE_CPU
        ret = priorbox_cpu(inputDesc, priorBoxParamSpec, outputDesc, output, arch);
#endif
    }
    return ret;
}
