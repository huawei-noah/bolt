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

template <typename TA, typename TB>
static inline EE check_kernel(
    TensorDesc aDesc, TA *a, TensorDesc bDesc, TB *b, CheckParamSpec p, TensorDesc outDesc, U8 *out)
{
    int aLen = tensorNumElements(aDesc);
    int bLen = tensorNumElements(bDesc);
    int len = tensorNumElements(outDesc);
    EE ret = SUCCESS;
    for (int i = 0; i < len; i++) {
        TA va = a[i % aLen];
        TB vb = b[i % bLen];
        switch (p.mode) {
            case CHECK_GREATER: {
                out[i] = (va > (TA)vb) ? 1 : 0;
                break;
            }
            case CHECK_GREATER_EQUAL: {
                out[i] = (va >= (TA)vb) ? 1 : 0;
                break;
            }
            case CHECK_EQUAL: {
                out[i] = (va == (TA)vb) ? 1 : 0;
                break;
            }
            case CHECK_NOT_EQUAL: {
                out[i] = (va != (TA)vb) ? 1 : 0;
                break;
            }
            case CHECK_LESS: {
                out[i] = (va < (TA)vb) ? 1 : 0;
                break;
            }
            case CHECK_LESS_EQUAL: {
                out[i] = (va <= (TA)vb) ? 1 : 0;
                break;
            }
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    }
    return ret;
}

template <typename TA>
EE check_wrapper(TensorDesc inputDescA,
    TA *inputA,
    TensorDesc inputDescB,
    void *inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    U8 *output)
{
    EE ret = SUCCESS;
    switch (inputDescB.dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = check_kernel<TA, F32>(
                inputDescA, inputA, inputDescB, (F32 *)inputB, p, outputDesc, output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = check_kernel<TA, F16>(
                inputDescA, inputA, inputDescB, (F16 *)inputB, p, outputDesc, output);
            break;
#endif
        case DT_U32: {
            ret = check_kernel<TA, U32>(
                inputDescA, inputA, inputDescB, (U32 *)inputB, p, outputDesc, output);
            break;
        }
        case DT_I32: {
            ret = check_kernel<TA, I32>(
                inputDescA, inputA, inputDescB, (I32 *)inputB, p, outputDesc, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE check_cpu(TensorDesc inputADesc,
    void *inputA,
    TensorDesc inputBDesc,
    void *inputB,
    CheckParamSpec p,
    TensorDesc outputDesc,
    void *output)
{
    EE ret = NOT_SUPPORTED;
    switch (inputADesc.dt) {
        case DT_U32:
            ret = check_wrapper<U32>(
                inputADesc, (U32 *)inputA, inputBDesc, inputB, p, outputDesc, (U8 *)output);
            break;
        case DT_I32:
            ret = check_wrapper<I32>(
                inputADesc, (I32 *)inputA, inputBDesc, inputB, p, outputDesc, (U8 *)output);
            break;
#ifdef _USE_FP32
        case DT_F32:
            ret = check_wrapper<F32>(
                inputADesc, (F32 *)inputA, inputBDesc, inputB, p, outputDesc, (U8 *)output);
            break;
#endif
#ifdef _USE_FP16
        case DT_F16:
            ret = check_wrapper<F16>(
                inputADesc, (F16 *)inputA, inputBDesc, inputB, p, outputDesc, (U8 *)output);
            break;
#endif
        default:
            break;
    }
    return ret;
}
