// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"

template <typename T>
static inline U32 non_zero_kernel(TensorDesc inputDesc, T *input, TensorDesc outputDesc, I32 *output)
{
    U32 count = 0;
    for (U32 i = 0; i < tensorNumElements(inputDesc); i++) {
        if (input[i] != 0) {
            count++;
        }
    }
    U32 length = count;
    count = 0;
    for (U32 i = 0; i < tensorNumElements(inputDesc); i++) {
        if (input[i] != 0) {
            std::vector<U32> id = calculateLocalIndex(i, inputDesc.dims, inputDesc.nDims);
            for (U32 j = 0; j < inputDesc.nDims; j++) {
                output[j * length + count] = id[inputDesc.nDims - 1 - j];
            }
            count++;
        }
    }
    return length;
}

EE non_zero_cpu(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output, U32 *length)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32:
            *length = non_zero_kernel<F32>(inputDesc, (F32 *)input, outputDesc, (I32 *)output);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            *length = non_zero_kernel<F16>(inputDesc, (F16 *)input, outputDesc, (I32 *)output);
            break;
#endif
        case DT_U32:
            *length = non_zero_kernel<U32>(inputDesc, (U32 *)input, outputDesc, (I32 *)output);
            break;
        case DT_I32:
            *length = non_zero_kernel<I32>(inputDesc, (I32 *)input, outputDesc, (I32 *)output);
            break;
        case DT_U8:
            *length = non_zero_kernel<U8>(inputDesc, (UINT8 *)input, outputDesc, (I32 *)output);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
