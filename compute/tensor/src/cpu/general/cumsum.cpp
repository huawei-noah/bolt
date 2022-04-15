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

template <typename T>
static void cumsum(
    TensorDesc inputDesc, const T *input, CumSumParamSpec p, TensorDesc outputDesc, T *output)
{
    int axis = (p.axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;
    int loopOuter = 1, loopInner = 1;
    for (int i = 0; i < axis; i++) {
        loopInner *= inputDesc.dims[i];
    }
    int loops = inputDesc.dims[axis];
    for (U32 i = axis + 1; i < inputDesc.nDims; i++) {
        loopOuter *= inputDesc.dims[i];
    }
    int id, id1;
    for (int i = 0; i < loopOuter; i++) {
        for (int j = 0; j < loopInner; j++) {
            if (p.reverse) {
                id = (i * loops + loops - 1) * loopInner + j;
                if (p.exclusive) {
                    output[id] = 0;
                    id1 = id;
                    id -= loopInner;
                } else {
                    output[id] = input[id];
                    id1 = id - loopInner;
                    id = id1;
                }
                for (int k = loops - 2; k >= 0; k--, id -= loopInner, id1 -= loopInner) {
                    output[id] = output[id + loopInner] + input[id1];
                }
            } else {
                id = i * loops * loopInner + j;
                if (p.exclusive) {
                    output[id] = 0;
                    id1 = id;
                    id += loopInner;
                } else {
                    output[id] = input[id];
                    id1 = id + loopInner;
                    id = id1;
                }
                for (int k = 1; k < loops; k++, id += loopInner, id1 += loopInner) {
                    output[id] = output[id - loopInner] + input[id1];
                }
            }
        }
    }
}

EE cumsum_general(
    TensorDesc inputDesc, const void *input, CumSumParamSpec p, TensorDesc outputDesc, void *output)
{
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP16
        case DT_F16: {
            cumsum<F16>(inputDesc, (const F16 *)input, p, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            cumsum<F32>(inputDesc, (const F32 *)input, p, outputDesc, (F32 *)output);
            break;
        }
#endif
        case DT_I32: {
            cumsum<I32>(inputDesc, (const I32 *)input, p, outputDesc, (I32 *)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
