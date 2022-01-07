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
#include <algorithm>

template <typename T, bool increase, bool order>
inline static void topk_kernel(
    const TensorDesc &inputDesc, T *input, const TopKParamSpec &p, int *tmp, T *output, int *index)
{
    int axis = inputDesc.nDims - 1 - (p.axis + inputDesc.nDims) % inputDesc.nDims;
    int loopInner = 1, loops = inputDesc.dims[axis], loopOuter = 1;
    for (int i = 0; i < axis; i++) {
        loopInner *= inputDesc.dims[i];
    }
    for (U32 i = axis + 1; i < inputDesc.nDims; i++) {
        loopOuter *= inputDesc.dims[i];
    }
    int num = UNI_MIN(loops, p.topk);
    int *tmpEnd = tmp + loops;
    for (int i = 0; i < loopOuter; i++) {
        int offset = i * loops * loopInner;
        for (int j = 0; j < loopInner; j++, offset++) {
            for (int k = 0; k < loops; k++) {
                tmp[k] = k;
            }
            if (increase) {
                std::sort(tmp, tmpEnd, [&input, &offset, &loopInner](int i1, int i2) {
                    return input[offset + i1 * loopInner] < input[offset + i2 * loopInner];
                });
            } else {
                std::sort(tmp, tmpEnd, [&input, &offset, &loopInner](int i1, int i2) {
                    return input[offset + i1 * loopInner] > input[offset + i2 * loopInner];
                });
            }
            if (!order) {
                std::sort(tmp, tmp + num);
            }
            for (int k = 0; k < num; k++) {
                int id = (i * p.topk + k) * loopInner + j;
                index[id] = tmp[k];
                output[id] = input[offset + tmp[k] * loopInner];
            }
        }
    }
}

template <typename T, bool increase>
inline static void topk_wrapper0(
    const TensorDesc &inputDesc, T *input, const TopKParamSpec &p, int *tmp, T *output, int *index)
{
    if (p.sorted) {
        topk_kernel<T, increase, true>(inputDesc, input, p, tmp, output, index);
    } else {
        topk_kernel<T, increase, false>(inputDesc, input, p, tmp, output, index);
    }
}
template <typename T>
inline static void topk_wrapper1(
    const TensorDesc &inputDesc, T *input, const TopKParamSpec &p, int *tmp, T *output, int *index)
{
    if (p.largest) {
        topk_wrapper0<T, false>(inputDesc, input, p, tmp, output, index);
    } else {
        topk_wrapper0<T, true>(inputDesc, input, p, tmp, output, index);
    }
}

EE topk_cpu(TensorDesc inputDesc,
    void *input,
    TopKParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    TensorDesc indexDesc,
    void *index)
{
    if (nullptr == input || nullptr == output || nullptr == index) {
        CHECK_STATUS(NULL_POINTER);
    }
    EE ret;
    switch (inputDesc.dt) {
        case DT_F32:
            topk_wrapper1<F32>(inputDesc, (F32 *)input, p, (I32 *)tmp, (F32 *)output, (I32 *)index);
            ret = SUCCESS;
            break;
#ifdef _USE_FP16
        case DT_F16:
            topk_wrapper1<F16>(inputDesc, (F16 *)input, p, (I32 *)tmp, (F16 *)output, (I32 *)index);
            ret = SUCCESS;
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
