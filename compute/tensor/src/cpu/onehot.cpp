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

template <typename IT, typename OT>
static inline EE onehot_kernel(
    TensorDesc inputDesc, IT *input, OneHotParamSpec p, TensorDesc outputDesc, OT *output)
{
    UNI_INIT(tensorNumElements(outputDesc), outputDesc.dt, p.values[0], output);
    int axis = (p.axis + outputDesc.nDims) % outputDesc.nDims;
    axis = outputDesc.nDims - 1 - axis;
    int loopInner = 1, loopOuter = 1;
    for (int i = 0; i < axis; i++) {
        loopInner *= outputDesc.dims[i];
    }
    for (U32 i = axis + 1; i < outputDesc.nDims; i++) {
        loopOuter *= outputDesc.dims[i];
    }
    for (int i = 0, k = 0; i < loopOuter; i++) {
        for (int j = 0; j < loopInner; j++, k++) {
            int index = input[k] >= 0 ? input[k] : input[k] + p.depth;
            int id = (i * p.depth + index) * loopInner + j;
            output[id] = p.values[1];
        }
    }
    return SUCCESS;
}

EE onehot_cpu(
    TensorDesc inputDesc, void *input, OneHotParamSpec p, TensorDesc outputDesc, void *output)
{
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.dt != DT_I32) {
        return NOT_SUPPORTED;
    }
    EE ret = NOT_SUPPORTED;
    switch (outputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32:
            ret = onehot_kernel<I32, F32>(inputDesc, (I32 *)input, p, outputDesc, (F32 *)output);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = onehot_kernel<I32, F16>(inputDesc, (I32 *)input, p, outputDesc, (F16 *)output);
            break;
#endif
        case DT_I8:
            ret = onehot_kernel<I32, I8>(inputDesc, (I32 *)input, p, outputDesc, (I8 *)output);
            break;
        default:
            break;
    }
    return ret;
}
