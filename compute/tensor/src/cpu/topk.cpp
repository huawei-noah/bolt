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

template <typename T, bool increase>
inline static bool cmp(T *data, const I32 &a, const I32 &b)
{
    if (increase) {
        return (data[a] < data[b]) || (data[a] == data[b] && a < b);
    } else {
        return (data[a] > data[b]) || (data[a] == data[b] && a < b);
    }
}

template <typename T, bool increase>
static void heap(I32 *buffer, I32 i, I32 k, T *data)
{
    while (true) {
        I32 left = 2 * i + 1;
        I32 right = left + 1;
        if (right < k) {
            bool replace = cmp<T, increase>(data, buffer[i], buffer[left]);
            if (replace && cmp<T, increase>(data, buffer[right], buffer[left])) {
                auto tmp = buffer[i];
                buffer[i] = buffer[left];
                buffer[left] = tmp;
                i = left;
            } else if (replace || cmp<T, increase>(data, buffer[i], buffer[right])) {
                auto tmp = buffer[i];
                buffer[i] = buffer[right];
                buffer[right] = tmp;
                i = right;
            } else
                break;
        } else if ((left < k) && cmp<T, increase>(data, buffer[i], buffer[left])) {
            auto tmp = buffer[i];
            buffer[i] = buffer[left];
            buffer[left] = tmp;
            i = left;
        } else
            break;
    }
}

template <typename T, bool increase, bool order>
inline static void topk_kernel(
    const TensorDesc &inputDesc, T *input, const TopKParamSpec &p, I32 *tmp, T *output, I32 *index)
{
    I32 axis = inputDesc.nDims - 1 - (p.axis + inputDesc.nDims) % inputDesc.nDims;
    I32 loopInner = 1, loops = inputDesc.dims[axis], loopOuter = 1;
    for (I32 i = 0; i < axis; i++) {
        loopInner *= inputDesc.dims[i];
    }
    for (U32 i = axis + 1; i < inputDesc.nDims; i++) {
        loopOuter *= inputDesc.dims[i];
    }
    I32 num = loops;
    if (p.k > 0 && p.k < num) {
        num = p.k;
    }
    I32 *tmpEnd = tmp + loops;
    for (I32 i = 0; i < loopOuter; i++) {
        I32 offset = i * loops * loopInner;
        for (I32 j = 0; j < loopInner; j++, offset++) {
#if 0
            for (I32 k = 0; k < loops; k++) {
                tmp[k] = offset + k * loopInner;
            }
            if (increase) {
                std::stable_sort(
                    tmp, tmpEnd, [&input](I32 i1, I32 i2) { return input[i1] < input[i2]; });
            } else {
                std::stable_sort(
                    tmp, tmpEnd, [&input](I32 i1, I32 i2) { return input[i1] > input[i2]; });
            }
            if (!order) {
                std::sort(tmp, tmp + num);
            }
            for (I32 k = 0; k < num; k++) {
                I32 id = (i * num + k) * loopInner + j;
                index[id] = (tmp[k] - offset) / loopInner;
                output[id] = input[tmp[k]];
            }
#else
            I32 l = 0;
            I32 cur_idx = offset;
            for (; l < num; ++l) {
                tmp[num - l - 1] = cur_idx;
                heap<T, increase>(tmp, num - l - 1, num, input);
                cur_idx += loopInner;
            }

            auto top = tmp[0];
            for (; l < loops; ++l) {
                if (cmp<T, increase>(input, cur_idx, top)) {
                    tmp[0] = cur_idx;
                    heap<T, increase>(tmp, 0, num, input);
                    top = tmp[0];
                }
                cur_idx += loopInner;
            }
            if (order) {
                for (l = 0; l < num; ++l) {
                    I32 id = (i * num + (num - l - 1)) * loopInner + j;
                    index[id] = (tmp[0] - offset) / loopInner;
                    output[id] = input[tmp[0]];
                    tmp[0] = tmp[num - l - 1];
                    heap<T, increase>(tmp, 0, num - l - 1, input);
                }
            } else {
                for (l = 0; l < num; ++l) {
                    I32 id = (i * num + l) * loopInner + j;
                    index[id] = (tmp[l] - offset) / loopInner;
                    output[id] = input[tmp[l]];
                }
            }
#endif
        }
    }
}

template <typename T, bool increase>
inline static void topk_wrapper0(
    const TensorDesc &inputDesc, T *input, const TopKParamSpec &p, I32 *tmp, T *output, I32 *index)
{
    if (p.sorted) {
        topk_kernel<T, increase, true>(inputDesc, input, p, tmp, output, index);
    } else {
        topk_kernel<T, increase, false>(inputDesc, input, p, tmp, output, index);
    }
}
template <typename T>
inline static void topk_wrapper1(
    const TensorDesc &inputDesc, T *input, const TopKParamSpec &p, I32 *tmp, T *output, I32 *index)
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
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
#ifdef _USE_FP32
        case DT_F32:
            topk_wrapper1<F32>(inputDesc, (F32 *)input, p, (I32 *)tmp, (F32 *)output, (I32 *)index);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            topk_wrapper1<F16>(inputDesc, (F16 *)input, p, (I32 *)tmp, (F16 *)output, (I32 *)index);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
