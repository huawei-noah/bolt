// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/tensor_computing_x86.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/tensor_computing_fp32.h"
#endif

EE attention_x86(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    DataType dt;
    DataFormat df;
    U32 batch, numHeads, fromSequenceLength, toSequenceLength;
    CHECK_REQUIREMENT(tensorIs2d(inputDesc));
    CHECK_REQUIREMENT(tensorIs4d(outputDesc));
    CHECK_STATUS(tensor4dGet(
        outputDesc, &dt, &df, &batch, &numHeads, &fromSequenceLength, &toSequenceLength));

    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = attention_fp32(batch, numHeads, fromSequenceLength, toSequenceLength,
                (const F32 *)input, (F32 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
