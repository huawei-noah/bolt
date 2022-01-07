// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>
#include "cpu/cpu_functions.h"
#include "cpu/arm/tensor_computing_arm.h"
#ifdef _USE_FP32
#include "cpu/arm/fp32/tensor_computing_fp32.h"
#endif
#ifdef _USE_FP16
#include "cpu/arm/fp16/tensor_computing_fp16.h"
#endif

EE eltwise_i32(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    I32 buffer[4];
    U32 len_tail = len % 4;
    U32 len_main = len - len_tail;

    I32 *tmp = buffer;
    I32 *output_ptr = (I32 *)output;
    for (U32 i = 0; i < len_main; i += 4) {
        get_vector<I32>((I32 *)input[0], inputSize[0], &tmp, 4, i, 4, buffer);
        int32x4_t tmp_v = vld1q_s32(tmp);
        for (U32 j = 1; j < num; j++) {
            get_vector<I32>((I32 *)input[j], inputSize[j], &tmp, 4, i, 4, buffer);
            int32x4_t value_v = vld1q_s32(tmp);
            switch (eltwiseMode) {
                case ELTWISE_SUM:
                    tmp_v = vaddq_s32(tmp_v, value_v);
                    break;
                case ELTWISE_MAX:
                    tmp_v = vmaxq_s32(tmp_v, value_v);
                    break;
                case ELTWISE_PROD:
                    tmp_v = vmulq_s32(tmp_v, value_v);
                    break;
                case ELTWISE_SUB:
                    tmp_v = vsubq_s32(tmp_v, value_v);
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        vst1q_s32(output_ptr + i, tmp_v);
    }
    for (U32 i = len_main; i < len; i++) {
        get_vector<I32>((I32 *)input[0], inputSize[0], &tmp, 4, i, 1, buffer);
        I32 tmp_s = tmp[0];
        for (U32 j = 1; j < num; j++) {
            get_vector<I32>((I32 *)input[j], inputSize[j], &tmp, 4, i, 1, buffer);
            I32 value_s = tmp[0];
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
                default:
                    return NOT_SUPPORTED;
            }
        }
        output_ptr[i] = tmp_s;
    }
    return SUCCESS;
}

EE eltwise_u8(std::vector<void *> input,
    std::vector<int> inputSize,
    U32 num,
    U32 len,
    void *output,
    EltwiseMode eltwiseMode)
{
    U8 buffer[16];
    U32 len_tail = len % 16;
    U32 len_main = len - len_tail;

    U8 *tmp = buffer;
    U8 *output_ptr = (U8 *)output;
    for (U32 i = 0; i < len_main; i += 16) {
        get_vector<U8>((U8 *)input[0], inputSize[0], &tmp, 16, i, 16, buffer);
        uint8x16_t tmp_v = vld1q_u8(tmp);
        for (U32 j = 1; j < num; j++) {
            get_vector<U8>((U8 *)input[j], inputSize[j], &tmp, 16, i, 16, buffer);
            uint8x16_t value_v = vld1q_u8(tmp);
            switch (eltwiseMode) {
                case ELTWISE_AND:
                    tmp_v = vandq_u8(tmp_v, value_v);
                    break;
                case ELTWISE_OR:
                    tmp_v = vorrq_u8(tmp_v, value_v);
                    break;
                case ELTWISE_XOR:
                    tmp_v = veorq_u8(tmp_v, value_v);
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        vst1q_u8(output_ptr + i, tmp_v);
    }
    for (U32 i = len_main; i < len; i++) {
        get_vector<U8>((U8 *)input[0], inputSize[0], &tmp, 16, i, 1, buffer);
        U8 tmp_s = tmp[0];
        for (U32 j = 1; j < num; j++) {
            get_vector<U8>((U8 *)input[j], inputSize[j], &tmp, 16, i, 1, buffer);
            U8 value_s = tmp[0];
            switch (eltwiseMode) {
                case ELTWISE_AND:
                    tmp_s = value_s & tmp_s;
                    break;
                case ELTWISE_OR:
                    tmp_s = value_s | tmp_s;
                    break;
                case ELTWISE_XOR:
                    tmp_s = value_s ^ tmp_s;
                    break;
                default:
                    return NOT_SUPPORTED;
            }
        }
        output_ptr[i] = tmp_s;
    }
    return SUCCESS;
}

EE eltwise_arm(DataType dataType,
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
            ret = eltwise_fp32(input, inputSize, num, len, output, eltwiseMode);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            ret = eltwise_fp16(input, inputSize, num, len, output, eltwiseMode);
            break;
        }
#endif
        case DT_I32: {
            ret = eltwise_i32(input, inputSize, num, len, output, eltwiseMode);
            break;
        }
        case DT_U8:
            ret = eltwise_u8(input, inputSize, num, len, output, eltwiseMode);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
