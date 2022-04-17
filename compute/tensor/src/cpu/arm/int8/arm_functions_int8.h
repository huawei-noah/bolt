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
                for (U32 i = 0; i < loops; i += 16) {
                    int8x16_t in = vld1q_s8(input + i);
                    int8x16_t out = vmaxq_s8(zero, in);
                    vst1q_s8(output + i, out);
                }
            }
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
