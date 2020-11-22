// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/fp16/tensor_computing_fp16.h"

EE check_fp16(TensorDesc inputDescA,
    const F16 *inputA,
    TensorDesc inputDescB,
    const F16 *inputB,
    CheckMode checkMode,
    TensorDesc outputDesc,
    I32 *output)
{
    if (nullptr == inputA || nullptr == inputB || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    if (tensorNumElements(inputDescA) != tensorNumElements(inputDescB)) {
        CHECK_STATUS(NOT_MATCH);
    }

    U32 size = tensorNumElements(inputDescA);
    U32 loopOuter = inputDescA.dims[inputDescA.nDims - 1];
    I32 length = size / loopOuter;
    if (tensorNumElements(outputDesc) != loopOuter) {
        CHECK_STATUS(NOT_MATCH);
    }
    for (U32 j = 0; j < loopOuter; j++) {
        const F16 *arrayA = inputA + j * length;
        const F16 *arrayB = inputB + j * length;
        switch (checkMode) {
            case CHECK_GREAT: {
                uint16x8_t count_v = vdupq_n_u16(0);
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    float16x8_t a = vld1q_f16(arrayA + i);
                    float16x8_t b = vld1q_f16(arrayA + i);
                    count_v = vaddq_u16(count_v, vcgtq_f16(a, b));
                }
                I32 count = vaddvq_u16(count_v);
                for (; i < length; i++) {
                    if (arrayA[i] > arrayB[i]) {
                        count++;
                    }
                }
                output[j] = (count == length);
                break;
            }
            case CHECK_GREATEQUAL: {
                uint16x8_t count_v = vdupq_n_u16(0);
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    float16x8_t a = vld1q_f16(arrayA + i);
                    float16x8_t b = vld1q_f16(arrayA + i);
                    count_v = vaddq_u16(count_v, vcgeq_f16(a, b));
                }
                I32 count = vaddvq_u16(count_v);
                for (; i < length; i++) {
                    if (arrayA[i] >= arrayB[i]) {
                        count++;
                    }
                }
                output[j] = (count == length);
                break;
            }
            case CHECK_EQUAL: {
                uint16x8_t count_v = vdupq_n_u16(0);
                I32 i = 0;
                for (; i < length - 7; i += 8) {
                    float16x8_t a = vld1q_f16(arrayA + i);
                    float16x8_t b = vld1q_f16(arrayA + i);
                    count_v = vaddq_u16(count_v, vceqq_f16(a, b));
                }
                I32 count = vaddvq_u16(count_v);
                for (; i < length; i++) {
                    if (arrayA[i] == arrayB[i]) {
                        count++;
                    }
                }
                output[j] = (count == length);
                break;
            }
            default:
                CHECK_STATUS(NOT_SUPPORTED);
                break;
        }
    }
    return SUCCESS;
}
