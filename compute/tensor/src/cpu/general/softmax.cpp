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
#include "tensor_transpose.h"

template <typename T>
static F32 array_max(const T *input, U32 len, U32 stride)
{
    F32 tmp = input[0];
    for (U32 i = 1; i < len; i++) {
        if (input[i * stride] > tmp) {
            tmp = input[i * stride];
        }
    }
    return tmp;
}

template <typename T>
static EE softmax(TensorDesc inputDesc, const T *input, int axis, TensorDesc outputDesc, T *output)
{
    UNUSED(outputDesc);
    if (nullptr == input || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }

    U32 size = tensorNumElements(inputDesc);
    axis = (axis + inputDesc.nDims) % inputDesc.nDims;
    axis = inputDesc.nDims - 1 - axis;
    std::vector<T> buffer;
    if (inputDesc.df == DF_NCHWC8) {
        if (axis == 2) {
            if (inputDesc.dims[0] != 1 || inputDesc.dims[1] != 1) {
                buffer = std::vector<T>(size);
                TensorDesc tmpInputDesc = inputDesc;
                tmpInputDesc.df = DF_NCHW;
                transformToNCHW(inputDesc, input, tmpInputDesc, buffer.data());
                input = (const T *)(buffer.data());
            }
        } else {
            for (I32 i = (int)inputDesc.nDims; i > 0; i--) {
                inputDesc.dims[i] = inputDesc.dims[i - 1];
            }
            inputDesc.dims[inputDesc.nDims - 1] /= 8;
            inputDesc.dims[0] = 8;
            inputDesc.nDims += 1;
            axis += 1;
        }
    }
    U32 loops = inputDesc.dims[axis];

    U32 loop_inner = 1;
    for (int i = 0; i < axis; i++) {
        loop_inner *= inputDesc.dims[i];
    }
    U32 loop_outer = size / loops / loop_inner;

    for (U32 i = 0; i < loop_outer; i++) {
        for (U32 j = 0; j < loop_inner; j++) {
            const T *in = input + i * loops * loop_inner + j;
            T *out = output + i * loops * loop_inner + j;
            F32 max_value = array_max<T>(in, loops, loop_inner);
            F32 sum = 0;
            for (U32 i = 0; i < loops; i++) {
                F32 tmp = exp(in[i * loop_inner] - max_value);
                sum += tmp;
                out[i * loop_inner] = tmp;
            }
            sum = 1 / sum;
            for (U32 i = 0; i < loops; i++) {
                out[i * loop_inner] *= sum;
            }
        }
    }
    return SUCCESS;
}

EE softmax_general(
    TensorDesc inputDesc, const void *input, SoftmaxParamSpec p, TensorDesc outputDesc, void *output)
{
    DataType idt = inputDesc.dt;
    EE ret = SUCCESS;
    switch (idt) {
#ifdef _USE_FP16
        case DT_F16: {
            ret = softmax<F16>(inputDesc, (const F16 *)input, p.axis, outputDesc, (F16 *)output);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = softmax<F32>(inputDesc, (const F32 *)input, p.axis, outputDesc, (F32 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}
