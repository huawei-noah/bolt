// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <arm_neon.h>
#include <math.h>
#include "arm_neon_expand.h"
#include "error.h"
#include "cpu/arm/common_arm.h"

EE activation_fp16(F16* data, U32 len, ActivationMode activationMode)
{
    float16x8_t in, out;
    float16x8_t zero  = vdupq_n_f16(float16_t(0.));
    float16x8_t one   = vdupq_n_f16(float16_t(1.));
    float16x8_t three = vdupq_n_f16(float16_t(3.));
    float16x8_t six   = vdupq_n_f16(float16_t(6.));
    U32 len_main = len / 8;
    U32 len_tail = len % 8;

    F16 value;
    switch (activationMode){
        case ACTIVATION_NULL: {
            break;
        }
        case ACTIVATION_RELU: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f16(data);
                out = vmaxq_f16(zero, in);
                vst1q_f16(data, out);
                data += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                data[i] = (data[i] < 0) ? 0 : data[i];
            }
            break;
        }
        case ACTIVATION_RELU6: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f16(data);
                out = vmaxq_f16(zero, in);
                out = vminq_f16(six, out);
                vst1q_f16(data, out);
                data += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = (data[i] < 0) ? 0 : data[i];
                if (value > 6) {
                    value = 6;
                }
                data[i] = value;
            }
            break;
        }
        case ACTIVATION_H_SIGMOID: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f16(data);
                out = vaddq_f16(in, three);
                out = vmaxq_f16(out, zero);
                out = vminq_f16(out, six);
                out = vdivq_f16(out, six);
                vst1q_f16(data, out);
                data += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = data[i] + 3;
                value = (value < 0) ? 0 : value;
                value = (value > 6) ? 6 : value;
                value = value / 6;
                data[i] = value;
            }
            break;
        }
        case ACTIVATION_H_SWISH: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f16(data);
                out = vaddq_f16(in, three);
                out = vmaxq_f16(out, zero);
                out = vminq_f16(out, six);
                out = vdivq_f16(out, six);
                out = vmulq_f16(out, in);
                vst1q_f16(data, out);
                data += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = data[i] + 3;
                value = (value < 0) ? 0 : value;
                value = (value > 6) ? 6 : value;
                value = data[i] * value;
                value = value / 6;
                data[i] = value;
            }
            break;
        }
        case ACTIVATION_GELU: {
            F16 two_div_PI_sqrt = sqrt(2 / 3.14159265358979323846);
            float16x8_t vec0  = vdupq_n_f16(two_div_PI_sqrt);
            float16x8_t vec1  = vdupq_n_f16(float16_t(0.044715));
            float16x8_t vec2  = vdupq_n_f16(float16_t(0.5));
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f16(data);
                out = vmulq_f16(in, in);
                out = vmulq_f16(out, in);
                out = vfmaq_f16(in, vec1, out);
                out = vmulq_f16(vec0, out);
                out = vtanhq_f16(out);
                out = vaddq_f16(one, out);
                out = vmulq_f16(vec2, out);
                out = vmulq_f16(in, out);
                vst1q_f16(data, out);
                data += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = data[i];
                value = two_div_PI_sqrt * (value + 0.044715 * pow(value, 3));
                value = 1.0 - 2.0 / (exp(2.0 * value) + 1.0);
                value = 0.5 * (1.0 + value);
                value = data[i] * value;
                data[i] = value;
            }
            break;
        }
        case ACTIVATION_TANH: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f16(data);
                out = vtanhq_f16(in);
                vst1q_f16(data, out);
                data += 8;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = 1.0 - 2.0 / (exp(2.0 * data[i]) + 1.0);
                data[i] = value;
            }
            break;
        }
        default:
            return NOT_SUPPORTED;
    }

    return SUCCESS;
}
