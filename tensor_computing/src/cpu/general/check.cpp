// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <math.h>
#include "cpu/general/tensor_computing_general.h"

template<typename T>
EE check(TensorDesc inputDescA, const T* inputA,
    TensorDesc inputDescB, const T* inputB,
    CheckMode checkMode,
    TensorDesc outputDesc, I32* output)
{
    UNUSED(inputDescB);
    UNUSED(outputDesc);

    if (nullptr == inputA || nullptr == inputB || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    U32 size = tensorNumElements(inputDescA);
    U32 loopOuter = inputDescA.dims[inputDescA.nDims-1];
    U32 loopInner = size / loopOuter;
    
    for (U32 i = 0; i < loopOuter; i++) {
        U32 count = 0;
        for (U32 j = 0; j < loopInner; j++) {
            U32 index = i * loopInner + j;
            switch (checkMode) {
                case CHECK_EQUAL: {
                    if (inputA[index] == inputB[index])
                        count ++;
                    break;
                }
                default:
                    CHECK_STATUS(NOT_SUPPORTED);
                    break;
            }
        }
        switch (checkMode) {
            case CHECK_EQUAL: {
                if (count == loopInner)
                    output[i] = 1;
                else
                    output[i] = 0;
                break;
            }
            default:
                break;
        }
    }
    return SUCCESS;
}

EE check_general(TensorDesc inputDescA, const void* inputA,
    TensorDesc inputDescB, const void* inputB,
    CheckMode checkMode,
    TensorDesc outputDesc, void* output)
{
    DataType idt = inputDescA.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = check<F16>(inputDescA, (const F16*)inputA,
                             inputDescB, (const F16*)inputB,
                             checkMode, outputDesc, (I32*)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = check<F32>(inputDescA, (const F32*)inputA,
                             inputDescB, (const F32*)inputB,
                             checkMode, outputDesc, (I32*)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
