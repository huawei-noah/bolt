// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_ARM_NEON_EXPAND_FP32
#define _H_ARM_NEON_EXPAND_FP32
#include "cpu/arm/arm_neon_expand.h"

// array sum
inline F32 array_sum_f32(const F32 *data, I32 len) {
    if(len <= 0) return 0;

    I32 i = 0;
    F32 sum_s = 0;
    float32x4_t sum_v = vdupq_n_f32(0);
    for(i = 0; i < len - 3; i+=4){
        float32x4_t in = vld1q_f32(data + i);
        sum_v = vaddq_f32(sum_v, in);
    }
    sum_s += vaddvq_f32(sum_v);
    for(; i < len; i++){
        sum_s += data[i];
    }
    return sum_s;
}

// array mean
inline F32 array_mean_f32(const F32 *data, I32 len) {
    if(len <= 0) return 0;
    return array_sum_f32(data, len) / len;
}

// array var
inline F32 array_var_f32(const F32 *data, I32 len, F32 mean) {
    if(len <= 0) return 0;

    I32 i = 0;
    F32 sum_s = 0;
    float32x4_t mean_v = vdupq_n_f32(mean);
    for(i = 0; i < len - 3; i+=4){
        float32x4_t in = vld1q_f32(data + i);
        float32x4_t tmp_v = vsubq_f32(in, mean_v);
        float32x4_t sum_v = vmulq_f32(tmp_v, tmp_v);
        sum_s += vaddvq_f32(sum_v);
    }
    for(; i < len; i++){
        F32 in = data[i];
        F32 tmp = in - mean;
        sum_s += tmp * tmp;
    }
    return sum_s / len;
}

// array max
inline F32 array_max_f32(const F32* data, I32 len) {
    F32 max_s = data[0];
    I32 i = 0;
    if(len >= 4){
        float32x4_t max_v, tmp_v;
        max_v = vld1q_f32(data);
        for (i = 4; i < len - 3; i+=4) {
            tmp_v = vld1q_f32(data + i);
            max_v = vmaxq_f32(tmp_v, max_v);
        }
        max_s = vmaxvq_f32(max_v);
    }

    for (; i < len; i++) {
        if(data[i] > max_s)
            max_s = data[i];
    }

    return max_s;
}

inline void array_scale_f32(F32 *input, F32 *output, I32 len, F32 alpha, F32 beta) {
    I32 i = 0;
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);
    for (i = 0; i < len-3; i+=4) {
        float32x4_t in = vld1q_f32(input + i);
        float32x4_t tmp_v = vfmaq_f32(beta_v, alpha_v, in);
        vst1q_f32(output+i, tmp_v);
    }
    for (; i < len; i++) {
        output[i] = alpha * input[i] + beta;
    }
}

inline EE activation_fp32(F32* data, U32 len, ActivationMode activationMode)
{
    float32x4_t in, out;
    float32x4_t zero  = vdupq_n_f32(0.);
    float32x4_t one   = vdupq_n_f32(1.);
    float32x4_t three = vdupq_n_f32(3.);
    float32x4_t six   = vdupq_n_f32(6.);
    U32 len_main = len / 4;
    U32 len_tail = len % 4;

    F32 value;
    switch (activationMode){
        case ACTIVATION_NULL: {
            break;
        }
        case ACTIVATION_RELU: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f32(data);
                out = vmaxq_f32(zero, in);
                vst1q_f32(data, out);
                data += 4;
            }
            for (U32 i = 0; i < len_tail; i++) {
                data[i] = (data[i] < 0) ? 0 : data[i];
            }
            break;
        }
        case ACTIVATION_RELU6: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f32(data);
                out = vmaxq_f32(zero, in);
                out = vminq_f32(six, out);
                vst1q_f32(data, out);
                data += 4;
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
                in = vld1q_f32(data);
                out = vaddq_f32(in, three);
                out = vmaxq_f32(out, zero);
                out = vminq_f32(out, six);
                out = vdivq_f32(out, six);
                vst1q_f32(data, out);
                data += 4;
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
                in = vld1q_f32(data);
                out = vaddq_f32(in, three);
                out = vmaxq_f32(out, zero);
                out = vminq_f32(out, six);
                out = vdivq_f32(out, six);
                out = vmulq_f32(out, in);
                vst1q_f32(data, out);
                data += 4;
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
            F32 two_div_PI_sqrt = sqrt(2 / 3.14159265358979323846);
            float32x4_t vec0  = vdupq_n_f32(two_div_PI_sqrt);
            float32x4_t vec1  = vdupq_n_f32(0.044715);
            float32x4_t vec2  = vdupq_n_f32(0.5);
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f32(data);
                out = vmulq_f32(in, in);
                out = vmulq_f32(out, in);
                out = vfmaq_f32(in, vec1, out);
                out = vmulq_f32(vec0, out);
                out = vtanhq_f32(out);
                out = vaddq_f32(one, out);
                out = vmulq_f32(vec2, out);
                out = vmulq_f32(in, out);
                vst1q_f32(data, out);
                data += 4;
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
                in = vld1q_f32(data);
                out = vtanhq_f32(in);
                vst1q_f32(data, out);
                data += 4;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = 1.0 - 2.0 / (exp(2.0 * data[i]) + 1.0);
                data[i] = value;
            }
            break;
        }
        case ACTIVATION_SIGMOID: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f32(data);
                out = vsigmoidq_f32(in);
                vst1q_f32(data, out);
                data += 4;
            }
            for (U32 i = 0; i < len_tail; i++) {
                value = 1.0 / (1.0 + exp(-1.0 * data[i]));
                data[i] = value;
            }
            break;
        }
        default:
            return NOT_SUPPORTED;
    }

    return SUCCESS;
}
#endif
