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

EE where_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    auto inDesc = inputTensor->get_desc();
    auto outDesc = inDesc;
    outputTensor->resize(outDesc);
    return SUCCESS;
}

bool tensorDescEqual(TensorDesc a, TensorDesc b)
{
    if (a.nDims != b.nDims) {
        return false;
    } else {
        for (int i = 0; i < (int)(a.nDims); i++) {
            if (a.dims[i] != b.dims[i]) {
                return false;
            }
        }
    }
    return true;
}

int brocastIndex(TensorDesc inputDesc, TensorDesc conditionDesc)
{
    if (inputDesc.nDims != conditionDesc.nDims) {
        return -1;
    }

    for (int i = 2; i < (int)(inputDesc.nDims); i++) {
        if (inputDesc.dims[i] != conditionDesc.dims[i]) {
            return i;
        }
    }
    return -1;
}

template <typename T>
static EE diffSourceWhere(TensorDesc inputDesc,
    TensorDesc conditionDesc,
    TensorDesc yDesc,
    T *inputPtr,
    U8 *conditionPtr,
    T *yPtr,
    T *outputPtr)
{
    if (tensorDescEqual(inputDesc, conditionDesc)) {
        for (int i = 0; i < (int)(tensorNumElements(inputDesc)); i++) {
            if (tensorNumElements(yDesc) == 1) {
                outputPtr[i] = (conditionPtr[i] > 0) ? inputPtr[i] : yPtr[0];
            } else if (tensorNumElements(inputDesc) == tensorNumElements(yDesc)) {
                outputPtr[i] = (conditionPtr[i] > 0) ? inputPtr[i] : yPtr[i];
            } else {
                return NOT_SUPPORTED;
            }
        }
    } else {
        int bIndex = brocastIndex(inputDesc, conditionDesc);
        if (bIndex == -1) {
            return NOT_SUPPORTED;
        }
        int batchNum = 1;
        for (int i = 0; i < bIndex; i++) {
            batchNum *= inputDesc.dims[i];
        }
        for (int i = 0; i < (int)(inputDesc.dims[bIndex]); i++) {
            for (int j = 0; j < (int)(inputDesc.dims[1]); j++) {
                for (int k = 0; k < (int)(inputDesc.dims[0]); k++) {
                    if (tensorNumElements(yDesc) == 1) {
                        outputPtr[i * batchNum + j * inputDesc.dims[0] + k] =
                            conditionPtr[j * conditionDesc.dims[0] + k] > 0
                            ? inputPtr[i * batchNum + j * inputDesc.dims[0] + k]
                            : yPtr[0];
                    } else if (tensorNumElements(inputDesc) == tensorNumElements(yDesc)) {
                        outputPtr[i * batchNum + j * inputDesc.dims[0] + k] =
                            conditionPtr[j * conditionDesc.dims[0] + k] > 0
                            ? inputPtr[i * batchNum + j * inputDesc.dims[0] + k]
                            : yPtr[i * batchNum + j * inputDesc.dims[0] + k];
                    } else {
                        return NOT_SUPPORTED;
                    }
                }
            }
        }
    }
    return SUCCESS;
}

// replaceF -> yTensor
EE where(Tensor inputTensor,
    Tensor conditionTensor,
    Tensor yTensor,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *condition = get_ptr_from_tensor(conditionTensor, arch);
    void *yPtr = get_ptr_from_tensor(yTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);
    TensorDesc inputDesc = inputTensor.get_desc();
    TensorDesc conditionDesc = conditionTensor.get_desc();
    TensorDesc yDesc = yTensor.get_desc();

    if (inputDesc.dims[1] == 1) {
        memcpy(output, input, tensorNumBytes(inputDesc));
        return SUCCESS;
    }

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = diffSourceWhere(inputDesc, conditionDesc, yDesc, (F32 *)input, (U8 *)condition,
                (F32 *)yPtr, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = diffSourceWhere(inputDesc, conditionDesc, yDesc, (F16 *)input, (U8 *)condition,
                (F16 *)yPtr, (F16 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
