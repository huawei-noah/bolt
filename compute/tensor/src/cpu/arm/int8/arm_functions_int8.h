// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_ARM_FUNCTIONS_INT8
#define _H_ARM_FUNCTIONS_INT8

#include "cpu/cpu_functions_template.h"
#include "arm_neon_expand.h"

inline F32 array_sum_int8(const INT8 *data, I32 len)
{
    if (len <= 0) {
        return 0;
    }

    I32 i = 0;
    F32 sum_s = 0;
    int8x16_t sum_v = vdupq_n_s8(0);
    for (i = 0; i < len - 15; i += 16) {
        int8x16_t in = vld1q_s8(data + i);
        sum_v = vaddq_s8(sum_v, in);
    }
    sum_s += vaddvq_s8(sum_v);
    for (; i < len; i++) {
        sum_s += data[i];
    }
    return sum_s;
}

inline void array_add_int8(const INT8 *inputA, const INT8 *inputB, INT8 *output, I32 len)
{
    I32 i = 0;
    for (i = 0; i < len - 15; i += 16) {
        int8x16_t a = vld1q_s8(inputA + i);
        int8x16_t b = vld1q_s8(inputB + i);
        int8x16_t c = vaddq_s8(a, b);
        vst1q_s8(output + i, c);
    }

    for (; i < len; i++) {
        output[i] = inputA[i] + inputB[i];
    }
}

inline EE activation_int8(INT8 *input, U32 len, ActivationParamSpec activationDesc, INT8 *output)
{
    int8x16_t zero = vdupq_n_s8(0);
    U32 loops = len / 16 * 16;
    EE ret = SUCCESS;
    switch (activationDesc.mode) {
        case ACTIVATION_NULL: {
            if (output != input) {
                UNI_MEMCPY(output, input, sizeof(INT8) * len);
            }
            loops = len;
            break;
        }
        case ACTIVATION_RELU: {
            if (activationDesc.value[0] != 0) {
                ret = NOT_SUPPORTED;
            } else {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
                for (U32 i = 0; i < loops; i += 16) {
                    int8x16_t in = vld1q_s8(input + i);
                    int8x16_t out = vmaxq_s8(zero, in);
                    vst1q_s8(output + i, out);
                }
            }
            break;
        }
        case ACTIVATION_ABS: {
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
            for (U32 i = 0; i < loops; i += 16) {
                int8x16_t in = vld1q_s8(input + i);
                int8x16_t out = vabsq_s8(in);
                vst1q_s8(output + i, out);
            }
            break;
        }
        case ACTIVATION_SIGN: {
            loops = 0;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    if (ret == SUCCESS) {
        for (U32 i = loops; i < len; i++) {
            ret = activation_template<INT8>(activationDesc, input[i], output + i);
        }
    }
    return ret;
}
#endif
