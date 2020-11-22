// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/cpu_functions.h"

EE eltwise_fp32(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    F32 buffer[8];
    U32 len_tail = len % 8;
    U32 len_main = len - len_tail;

    F32 *tmp = buffer;
    F32 *output_ptr = (F32 *)output;
    for (U32 i = 0; i < len_main; i += 8) {
        get_vector<F32>((F32 *)input[0], inputSize[0], &tmp, 8, i, 8, buffer);
        __m256 tmp_v = _mm256_loadu_ps(tmp);
        for (U32 j = 1; j < num; j++) {
            get_vector<F32>((F32 *)input[j], inputSize[j], &tmp, 8, i, 8, buffer);
            __m256 value_v = _mm256_loadu_ps(tmp);
            switch (eltwiseMode) {
                case ELTWISE_SUM:
                    tmp_v = _mm256_add_ps(value_v, tmp_v);
                    break;
                case ELTWISE_MAX:
                    tmp_v = _mm256_max_ps(value_v, tmp_v);
                    break;
                case ELTWISE_PROD:
                    tmp_v = _mm256_mul_ps(value_v, tmp_v);
                    break;
                case ELTWISE_SUB:
                    tmp_v = _mm256_sub_ps(tmp_v, value_v);
                    break;
                case ELTWISE_DIV:
                    tmp_v = _mm256_div_ps(tmp_v, value_v);
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        _mm256_storeu_ps(output_ptr + i, tmp_v);
    }

    for (U32 i = len_main; i < len; i++) {
        get_vector<F32>((F32 *)input[0], inputSize[0], &tmp, 8, i, 1, buffer);
        F32 tmp_s = tmp[0];
        for (U32 j = 1; j < num; j++) {
            get_vector<F32>((F32 *)input[j], inputSize[j], &tmp, 8, i, 1, buffer);
            F32 value_s = tmp[0];
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
                default:
                    return NOT_SUPPORTED;
            }
        }
        output_ptr[i] = tmp_s;
    }
    return SUCCESS;
}
