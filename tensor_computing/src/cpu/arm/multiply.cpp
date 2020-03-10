// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/arm_neon_expand_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/arm_neon_expand_fp16.h"
#endif

template<typename T>
inline void array_scale(T *input, T *output, I32 len, F32 alpha, F32 beta) {
    for (I32 i = 0; i < len; i++) {
        output[i] = alpha * input[i] + beta;
    }
}

EE multiply_arm(void *alpha, void *beta, TensorDesc inputDesc, void* input, TensorDesc outputDesc, void *output)
{
    UNUSED(outputDesc);

    if (nullptr == alpha
        || nullptr == beta
        || nullptr == input
        || nullptr == output)
        CHECK_STATUS(NULL_POINTER);

    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            array_scale_f32((F32 *)input, (F32 *)output, tensorNumElements(inputDesc), *((F32 *)alpha), *((F32 *)beta));
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            array_scale_f16((F16 *)input, (F16 *)output, tensorNumElements(inputDesc), *((F32 *)alpha), *((F32 *)beta));
            break;
        }
#endif
        case DT_I32: {
            array_scale<I32>((I32 *)input, (I32 *)output, tensorNumElements(inputDesc), *((F32 *)alpha), *((F32 *)beta));
            break;
        }
        case DT_U32: {
            array_scale<U32>((U32 *)input, (U32 *)output, tensorNumElements(inputDesc), *((F32 *)alpha), *((F32 *)beta));
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
