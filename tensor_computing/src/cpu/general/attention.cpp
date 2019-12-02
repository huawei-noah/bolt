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

EE attention(U32 batch, U32 attentionNum, I32 sequenceLength, const F16 *input, F16 *output)
{
    for (U32 n = 0; n < batch; n++) {
        for (U32 i = 0; i < attentionNum; i++) {
            for (I32 j = 0; j < sequenceLength; j++) {
                for (I32 k = 0; k < sequenceLength; k++) {
                    F16 value = input[k];

                    U32 index = (((n * attentionNum + i)*sequenceLength + j)*sequenceLength + k);
                    output[index] = (1 - value) * -10000.0;
                }
            }
        }
    }
    return SUCCESS;
}

EE attention_general(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);

    DataType dt;
    DataFormat df;
    U32 batch, attentionNum, fromSequenceLength, toSequenceLength;
    CHECK_REQUIREMENT(tensorIs2d(inputDesc));
    CHECK_REQUIREMENT(tensorIs4d(outputDesc));
    CHECK_STATUS_WITH_RETURN(tensor4dGet(outputDesc, &dt, &df, &batch, &attentionNum, &fromSequenceLength, &toSequenceLength));
    CHECK_REQUIREMENT(fromSequenceLength == toSequenceLength);

    EE ret = SUCCESS;
    switch (dt) {
        case DT_F16: {
            ret = attention(batch, attentionNum, fromSequenceLength, (const F16*)input, (F16*)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
