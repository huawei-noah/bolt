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
T getFloatScalar(void *input, int inputSize, int index)
{
    int local = index % inputSize;
    return ((T *)input)[local];
}

template <typename T>
EE eltwise_general_kernel(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    T *output_ptr = (T *)output;
    for (U32 i = 0; i < len; i++) {
        F32 tmp_s = getFloatScalar<T>(input[0], inputSize[0], i);
        for (U32 j = 1; j < num; j++) {
            F32 value_s = getFloatScalar<T>(input[j], inputSize[j], i);
            switch (eltwiseMode) {
                case ELTWISE_SUM:
                    tmp_s = value_s + tmp_s;
                    break;
                case ELTWISE_MAX:
                    tmp_s = (value_s > tmp_s) ? value_s : tmp_s;
                    break;
                case ELTWISE_PROD:
                    tmp_s *= value_s;
                    break;
                case ELTWISE_SUB:
                    tmp_s -= value_s;
                    break;
                case ELTWISE_DIV:
                    tmp_s /= value_s;
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        output_ptr[i] = tmp_s;
    }
    return SUCCESS;
}

EE eltwise_general(DataType dataType,
    std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    EE ret = SUCCESS;
    switch (dataType) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = eltwise_general_kernel<F32>(input, inputSize, num, len, output, eltwiseMode);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = eltwise_general_kernel<F16>(input, inputSize, num, len, output, eltwiseMode);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
