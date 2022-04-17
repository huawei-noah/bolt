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

template <typename TI, typename TO>
static void cast_kernel(U32 len, TI *input, TO *output)
{
    for (U32 i = 0; i < len; ++i) {
        output[i] = (TO)(input[i]);
    }
}

template <typename T>
static EE cast_kernel(U32 len, DataType odt, T *input, void *output)
{
    EE ret = SUCCESS;
    switch (odt) {
        case DT_I32: {
            cast_kernel<T, I32>(len, input, (I32 *)output);
            break;
        }
        case DT_U32: {
            cast_kernel<T, U32>(len, input, (U32 *)output);
            break;
        }
        case DT_F32: {
            cast_kernel<T, F32>(len, input, (F32 *)output);
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            cast_kernel<T, F16>(len, input, (F16 *)output);
            break;
        }
#endif
        case DT_U8: {
            cast_kernel<T, U8>(len, input, (U8 *)output);
            break;
        }
        case DT_I8: {
            cast_kernel<T, INT8>(len, input, (INT8 *)output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE cast_cpu(TensorDesc inputDesc, void *input, TensorDesc outputDesc, void *output)
{
    DataType idt = inputDesc.dt;
    DataType odt = outputDesc.dt;
    U32 len = tensorNumElements(inputDesc);
    EE ret;
    switch (idt) {
        case DT_F32: {
            ret = cast_kernel<F32>(len, odt, (F32 *)input, output);
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            ret = cast_kernel<F16>(len, odt, (F16 *)input, output);
            break;
        }
#endif
        case DT_U32: {
            ret = cast_kernel<U32>(len, odt, (U32 *)input, output);
            break;
        }
        case DT_I32: {
            ret = cast_kernel<I32>(len, odt, (I32 *)input, output);
            break;
        }
        case DT_U8: {
            ret = cast_kernel<U8>(len, odt, (U8 *)input, output);
            break;
        }
        case DT_I8: {
            ret = cast_kernel<INT8>(len, odt, (INT8 *)input, output);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
