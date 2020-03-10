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
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif

EE check_u32(TensorDesc inputDescA, const U32* inputA,
    TensorDesc inputDescB, const U32* inputB,
    CheckMode checkMode,
    TensorDesc outputDesc, I32* output)
{
    if (nullptr == inputA || nullptr == inputB || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    if (tensorNumElements(inputDescA) != tensorNumElements(inputDescB))
        CHECK_STATUS(NOT_MATCH);

    U32 size = tensorNumElements(inputDescA);
    U32 loopOuter = inputDescA.dims[inputDescA.nDims-1];
    if (tensorNumElements(outputDesc) != loopOuter)
        CHECK_STATUS(NOT_MATCH);
    I32 length = size / loopOuter;
    for (U32 j = 0; j < loopOuter; j++) {
        const U32 *arrayA = inputA + j * length;
        const U32 *arrayB = inputB + j * length;
        switch (checkMode) {
            case CHECK_EQUAL: {
                uint32x4_t count_v = vdupq_n_u32(0);
                I32 i = 0;
                for (; i < length-3; i+=4) {
                    uint32x4_t a = vld1q_u32(arrayA + i);
                    uint32x4_t b = vld1q_u32(arrayA + i);
                    count_v = vaddq_u32(count_v, vceqq_u32(a, b));
                }
                I32 count = vaddvq_u32(count_v);
                for (; i < length; i++)
                    if (arrayA[i] == arrayB[i])
                        count ++;
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

EE check_arm(TensorDesc inputDescA, const void* inputA,
    TensorDesc inputDescB, const void* inputB,
    CheckMode checkMode,
    TensorDesc outputDesc, void* output)
{
    DataType idt = inputDescA.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = check_fp32(inputDescA, (const F32*)inputA,
                             inputDescB, (const F32*)inputB,
                             checkMode,
                             outputDesc, (I32*)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = check_fp16(inputDescA, (const F16*)inputA,
                             inputDescB, (const F16*)inputB,
                             checkMode,
                             outputDesc, (I32*)output);
            break;
        }
#endif
        case DT_U32: {
            ret = check_u32(inputDescA, (const U32*)inputA,
                            inputDescB, (const U32*)inputB,
                            checkMode,
                            outputDesc, (I32*)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
