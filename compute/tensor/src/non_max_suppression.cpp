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

inline EE non_max_suppression_infer_output_size_cpu(
    std::vector<TensorDesc> inputDesc, NonMaxSuppressionParamSpec p, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt0, idt1;
    DataFormat idf0, idf1;
    U32 in0, ic0, ilens0;
    U32 in1, ic1, ilens1;
    // boxes
    CHECK_STATUS(tensor3dGet(inputDesc[0], &idt0, &idf0, &in0, &ic0, &ilens0));
    // scores
    CHECK_STATUS(tensor3dGet(inputDesc[1], &idt1, &idf1, &in1, &ic1, &ilens1));
    CHECK_REQUIREMENT(ilens0 == 4);
    CHECK_REQUIREMENT(ic0 == ilens1);
    CHECK_REQUIREMENT(p.max_output_boxes_per_class != 0);
    // output size
    U32 oh, ow;
    // oh = the first box for saving the number of available boxes(1) + the maximum number of dectected boxes(max_output_boxes_per_class * num_class)
    U32 max_output_boxes_per_class = p.max_output_boxes_per_class;
    U32 num_class = ic1;
    U32 num_detected_max = max_output_boxes_per_class * num_class;
    oh = num_detected_max + 1;
    // Each width is a 3 dimension vector, which stores [batch_index, class_index, box_index] -> 3
    // The first box is [ number of available boxes, 0, 0 ]
    ow = 3;
    *outputDesc = tensor2d(idt0, oh, ow);
    return SUCCESS;
}

EE non_max_suppression_infer_output_size(std::vector<Tensor *> inputTensor,
    NonMaxSuppressionParamSpec p,
    Tensor *outputTensor,
    ArchInfo_t archInfo)
{
    UNUSED(archInfo);
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    std::vector<TensorDesc> inputDesc = get_desc_from_tensor_ptrs(inputTensor);
    TensorDesc outputDesc = outputTensor->get_desc();
    CHECK_STATUS(non_max_suppression_infer_output_size_cpu(inputDesc, p, &outputDesc));
    outputTensor->resize(outputDesc);
    return SUCCESS;
}

EE non_max_suppression(std::vector<Tensor> inputTensor,
    NonMaxSuppressionParamSpec p,
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
        ret = non_max_suppression_cpu(inputDesc, input, p, outputDesc, output);
#endif
    }
    return ret;
}
