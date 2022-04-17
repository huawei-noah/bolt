// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#if 0
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
template <typename T1, typename T2>
static EE equal_kernel(T1 *a1, int len1, T2 *a2, int len2, bool not_equal, U8 *out)
{
    U8 equal_flag, notequal_flag;
    if (not_equal) {
        equal_flag = 0;
        notequal_flag = 1;
    } else {
        equal_flag = 1;
        notequal_flag = 0;
    }
    EE ret = SUCCESS;
    if (len1 == len2) {
        for (int i = 0; i < len1; ++i) {
            if (a1[i] == (T1)(a2[i])) {
                out[i] = equal_flag;
            } else {
                out[i] = notequal_flag;
            }
        }
    } else if (len2 == 1) {
        for (int i = 0; i < len1; ++i) {
            if (a1[i] == (T1)(a2[0])) {
                out[i] = equal_flag;
            } else {
                out[i] = notequal_flag;
            }
        }
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}

EE equal(Tensor inputTensor,
    Tensor compareTensor,
    EqualParamSpec p,
    Tensor outputTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    TensorDesc inputDesc = inputTensor.get_desc();
    U32 inputLen = tensorNumElements(inputDesc);
    void *input = get_ptr_from_tensor(inputTensor, arch);
    TensorDesc compareDesc = compareTensor.get_desc();
    U32 compareLen = tensorNumElements(compareDesc);
    void *compare = get_ptr_from_tensor(compareTensor, arch);
    void *output = get_ptr_from_tensor(outputTensor, arch);

    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = equal_kernel<F32, F32>(
                (F32 *)input, inputLen, (F32 *)compare, compareLen, p.invert, (U8 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            if (compareDesc.dt == DT_F32) {
                ret = equal_kernel<F16, F32>(
                    (F16 *)input, inputLen, (F32 *)compare, compareLen, p.invert, (U8 *)output);
            } else {
                ret = equal_kernel<F16, F16>(
                    (F16 *)input, inputLen, (F16 *)compare, compareLen, p.invert, (U8 *)output);
            }
            break;
        }
#endif
        case DT_I32: {
            ret = equal_kernel<I32, I32>(
                (I32 *)input, inputLen, (I32 *)compare, compareLen, p.invert, (U8 *)output);
            break;
        }
        default:
            break;
    }
    return ret;
}
#endif
