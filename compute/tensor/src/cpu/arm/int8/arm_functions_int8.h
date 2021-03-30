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

#include "arm_neon_expand.h"
#include "parameter_spec.h"

inline EE activation_int8(INT8 *input, U32 len, ActivationParamSpec activationDesc, INT8 *output)
{
    int8x16_t in, out;
    int8x16_t zero = vdupq_n_s8(0);
    U32 len_main = len / 16;
    U32 len_tail = len % 16;

    switch (activationDesc.mode) {
        case ACTIVATION_NULL: {
            break;
        }
        case ACTIVATION_RELU: {
            if (activationDesc.value[0] != 0) {
                return NOT_SUPPORTED;
            }
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_s8(input);
                out = vmaxq_s8(zero, in);
                vst1q_s8(output, out);
                input += 16;
                output += 16;
            }
            for (U32 i = 0; i < len_tail; i++) {
                output[i] = (input[i] < 0) ? 0 : input[i];
            }
            break;
        }
        default:
            return NOT_SUPPORTED;
    }

    return SUCCESS;
}

inline INT8 round_towards_zero(F32 num, bool clamp = true)
{
    INT8 ret;
    if (clamp) {
        if (num > 127.0) {
            return 127;
        } else if (num < -127.0) {
            return -127;
        }
    }
    if (num > 0) {
        ret = floor(num);
    } else {
        ret = ceil(num);
    }
    return ret;
}
#endif
