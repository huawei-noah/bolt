// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <arm_neon.h>
#include <string.h>
#include "cpu/arm/tensor_computing_arm.h"

EE attention_fp16(U32 batch, U32 attentionNum, I32 sequenceLength, const F16 *input, F16 *output)
{
    F16 mask_s = -10000.0;
    float16x8_t mask_v = vdupq_n_f16(float16_t(mask_s));
    float16x8_t one_v = vdupq_n_f16(float16_t(1.0));
    for(U32 n = 0; n < batch; n++){
        I32 i = 0;
        for (; i < sequenceLength-7; i+=8) {
            float16x8_t in_v = vld1q_f16(input + i);
            float16x8_t tmp_v = vsubq_f16(one_v, in_v);
            tmp_v = vmulq_f16(tmp_v, mask_v);
            vst1q_f16(output+i, tmp_v);
        }
        for (; i < sequenceLength; i++) {
            F16 value = (1 - input[i]) * mask_s;
            output[i] = value;
        }

        for (i = 1; i < sequenceLength; i++) {
            memcpy(output+i*sequenceLength, output, sequenceLength*sizeof(F16));
        }

        // expand dim
        for (U32 i = 1; i < attentionNum; i++) {
            memcpy(output+i*sequenceLength*sequenceLength, output, sequenceLength*sequenceLength*sizeof(F16));
        }

        input += sequenceLength;
        output += attentionNum * sequenceLength * sequenceLength;
    }
    return SUCCESS;
}


EE attention_arm(TensorDesc inputDesc, const void *input, TensorDesc outputDesc, void *output)
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
            ret = attention_fp16(batch, attentionNum, fromSequenceLength, (const F16*)input, (F16*)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
