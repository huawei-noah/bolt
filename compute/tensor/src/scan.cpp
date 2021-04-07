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

EE scan_infer_output_size(Tensor *inputTensor, Tensor *outputTensor, ArchInfo_t archInfo)
{
    auto inDesc = inputTensor->get_desc();
    auto outDesc = inDesc;
    outputTensor->resize(outDesc);
    return SUCCESS;
}

template <typename T>
static EE diffSourceScan(TensorDesc inputDesc, T *inputPtr, T *outputPtr)
{
    for (int i = 0; i < (int)(tensorNumElements(inputDesc)); i++) {
        outputPtr[i] = (T)1;
    }
    return SUCCESS;
}

EE scan(Tensor inputTensor, Tensor outputTensor, ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    void *input = get_ptr_from_tensor(inputTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);
    TensorDesc inputDesc = inputTensor.get_desc();

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = diffSourceScan(inputDesc, (F32 *)input, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = diffSourceScan(inputDesc, (F16 *)input, (F16 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
