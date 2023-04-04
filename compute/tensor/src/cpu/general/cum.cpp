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

template <typename IT, typename OT, EltwiseMode mode>
static void cum(
    TensorDesc inputDesc, const IT *input, CumParamSpec p, TensorDesc outputDesc, OT *output)
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
    float v;
    for (int i = 0; i < loopOuter; i++) {
        for (int j = 0; j < loopInner; j++) {
            if (p.reverse) {
                id = (i * loops + loops - 1) * loopInner + j;
                if (p.exclusive) {
                    v = (mode == ELTWISE_SUM) ? 0 : 1;
                    output[id] = v;
                    id1 = id;
                    id -= loopInner;
                } else {
                    v = input[id];
                    output[id] = v;
                    id1 = id - loopInner;
                    id = id1;
                }
                for (int k = loops - 2; k >= 0; k--, id -= loopInner, id1 -= loopInner) {
                    if (mode == ELTWISE_SUM) {
                        v = v + input[id1];
                    } else {
                        v = v * input[id1];
                    }
                    output[id] = v;
                }
            } else {
                id = i * loops * loopInner + j;
                if (p.exclusive) {
                    v = (mode == ELTWISE_SUM) ? 0 : 1;
                    output[id] = v;
                    id1 = id;
                    id += loopInner;
                } else {
                    v = input[id];
                    output[id] = v;
                    id1 = id + loopInner;
                    id = id1;
                }
                for (int k = 1; k < loops; k++, id += loopInner, id1 += loopInner) {
                    if (mode == ELTWISE_SUM) {
                        v = v + input[id1];
                    } else {
                        v = v * input[id1];
                    }
                    output[id] = v;
                }
            }
        }
    }
}

template <typename IT, typename OT>
static void cum(
    TensorDesc inputDesc, const IT *input, CumParamSpec p, TensorDesc outputDesc, OT *output)
{
    if (p.mode == ELTWISE_SUM) {
        cum<IT, OT, ELTWISE_SUM>(inputDesc, input, p, outputDesc, output);
    } else {
        cum<IT, OT, ELTWISE_PROD>(inputDesc, input, p, outputDesc, output);
    }
}

template <typename IT>
static EE cum(
    TensorDesc inputDesc, const IT *input, CumParamSpec p, TensorDesc outputDesc, void *output)
{
    EE ret = SUCCESS;
    switch (outputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16: {
            cum<IT, F16>(inputDesc, input, p, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            cum<IT, F32>(inputDesc, input, p, outputDesc, (F32 *)output);
            break;
        }
#endif
        case DT_I32: {
            cum<IT, I32>(inputDesc, input, p, outputDesc, (I32 *)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE cum_general(
    TensorDesc inputDesc, const void *input, CumParamSpec p, TensorDesc outputDesc, void *output)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = cum<F16>(inputDesc, (const F16 *)input, p, outputDesc, output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = cum<F32>(inputDesc, (const F32 *)input, p, outputDesc, output);
            break;
        }
#endif
        case DT_I32: {
            ret = cum<I32>(inputDesc, (const I32 *)input, p, outputDesc, output);
            break;
        }
        default:
            break;
    }
    return ret;
}
