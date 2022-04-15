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

template <typename T1, typename T2>
void select_kernel(U8 *choice, T1 *a, T1 *b, T1 *output, U32 length)
{
    for (U32 i = 0; i < length; i++) {
        output[i] = choice[i] ? a[i] : b[i];
    }
}

EE select(Tensor boolChoice, Tensor a, Tensor b, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc choiceDesc = boolChoice.get_desc();
    void *choice = get_ptr_from_tensor(boolChoice, arch);
    void *a_ptr = get_ptr_from_tensor(a, arch);
    void *b_ptr = get_ptr_from_tensor(b, arch);
    TensorDesc outputDesc = outputTensor.get_desc();
    void *output = get_ptr_from_tensor(outputTensor, arch);

    U32 length = tensorNumElements(outputDesc);
    EE ret = NOT_SUPPORTED;
    CHECK_REQUIREMENT(choiceDesc.dt == DT_U8);
    if (IS_CPU(arch)) {
        switch (outputDesc.dt) {
            case DT_F32:
                select_kernel<F32, F32>(
                    (U8 *)choice, (F32 *)a_ptr, (F32 *)b_ptr, (F32 *)output, length);
                ret = SUCCESS;
                break;
#ifdef _USE_FP16
            case DT_F16:
                select_kernel<F16, F16>(
                    (U8 *)choice, (F16 *)a_ptr, (F16 *)b_ptr, (F16 *)output, length);
                ret = SUCCESS;
                break;
#endif
            default:
                break;
        }
    }
    return ret;
}

inline EE select_infer_output_size_cpu(TensorDesc inputDesc, TensorDesc *outputDesc)
{
    if (nullptr == outputDesc) {
        CHECK_STATUS(NULL_POINTER);
    }
    *outputDesc = inputDesc;
    return SUCCESS;
}

EE select_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    if (inputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (outputTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc inputDesc = inputTensor->get_desc();
    TensorDesc outputDesc = outputTensor->get_desc();
    EE ret = NOT_SUPPORTED;
    if (IS_CPU(archInfo->arch)) {
        ret = select_infer_output_size_cpu(inputDesc, &outputDesc);
    }
    outputTensor->resize(outputDesc);
    return ret;
}
