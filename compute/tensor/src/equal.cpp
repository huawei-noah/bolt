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

EE equal_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    auto inDesc = inputTensor->get_desc();
    auto outDesc = inDesc;
    outDesc.dt = DT_U8;
    outputTensor->resize(outDesc);
    return SUCCESS;
}

// attention: comparision ptr will be fixed in mt
template <typename T>
static EE diffSourceEqual(
    U32 inputLen, U32 comparisonLen, T *inputPtr, F32 *comparisionPtr, U8 *outputPtr)
{
    if (inputLen == comparisonLen) {
        for (U32 i = 0; i < inputLen; ++i) {
            if (inputPtr[i] == (T)(comparisionPtr[i])) {
                outputPtr[i] = 1;
            } else {
                outputPtr[i] = 0;
            }
        }
    } else if (comparisonLen == 1) {
        F32 compF = comparisionPtr[0];
        for (U32 i = 0; i < inputLen; ++i) {
            if (inputPtr[i] == (T)compF) {
                outputPtr[i] = 1;
            } else {
                outputPtr[i] = 0;
            }
        }
    } else {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE equal(Tensor inputTensor, Tensor compareTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *comparision = get_ptr_from_tensor(compareTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);
    TensorDesc inputDesc = inputTensor.get_desc();
    U32 inputLen = tensorNumElements(inputDesc);
    U32 comparisonLen = tensorNumElements(compareTensor.get_desc());

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = diffSourceEqual<F32>(
                inputLen, comparisonLen, (F32 *)input, (F32 *)comparision, (U8 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = diffSourceEqual<F16>(
                inputLen, comparisonLen, (F16 *)input, (F32 *)comparision, (U8 *)output);
            break;
        }
#endif
        case DT_I32: {
            ret = diffSourceEqual<I32>(
                inputLen, comparisonLen, (I32 *)input, (F32 *)comparision, (U8 *)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
