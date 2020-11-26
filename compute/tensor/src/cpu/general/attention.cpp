// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/general/tensor_computing_general.h"
#include "cpu/general/general_functions.h"

template <typename T>
EE attention(
    U32 batch, U32 numHeads, U32 fromSequenceLength, U32 toSequenceLength, const T *input, T *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    T minValue = -10000.0;
    U32 count = array_sum_template<T>(input, toSequenceLength);
    U32 valid = UNI_MIN(count, fromSequenceLength);
    for (U32 n = 0; n < batch; n++) {
        for (U32 i = 0; i < numHeads; i++) {
            for (U32 j = 0; j < valid; j++) {
                for (U32 k = 0; k < toSequenceLength; k++) {
                    T value = input[n * toSequenceLength + k];
                    U32 index =
                        (((n * numHeads + i) * fromSequenceLength + j) * toSequenceLength + k);
                    output[index] = (1 - value) * minValue;
                }
            }
            for (U32 j = valid; j < fromSequenceLength; j++) {
                for (U32 k = 0; k < toSequenceLength; k++) {
                    U32 index =
                        (((n * numHeads + i) * fromSequenceLength + j) * toSequenceLength + k);
                    output[index] = minValue;
                }
            }
        }
    }
    return SUCCESS;
}

EE attention_general(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
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
#ifdef _USE_FP16
        case DT_F16: {
            ret = attention<F16>(batch, numHeads, fromSequenceLength, toSequenceLength,
                (const F16 *)input, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = attention<F32>(batch, numHeads, fromSequenceLength, toSequenceLength,
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
