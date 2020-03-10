// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_ARM_NEON_EXPAND_FP16
#define _H_ARM_NEON_EXPAND_FP16

#ifdef _USE_FP16
#include "cpu/arm/arm_neon_expand.h"

inline F32 vaddvq_f16(float16x8_t x)
{
    float32x4_t a = vcvt_f32_f16(vget_high_f16(x));
    float32x4_t b = vcvt_f32_f16(vget_low_f16(x));
    F32 sum = vaddvq_f32(vaddq_f32(a, b));
    return sum;
}

inline float16x8_t vaddq_f16_f32(float16x8_t a, float16x8_t b)
{
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t a0 = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a1 = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b0 = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b1 = vcvt_f32_f16(vget_high_f16(b));
    return vcombine_f16(vcvt_f16_f32(vaddq_f32(a0, b0)), vcvt_f16_f32(vaddq_f32(a1, b1)));
#else
    return vaddq_f16(a, b);
#endif
}

inline float16x8_t vtaylor_polyq_f16(float16x8_t x, const std::array<float16x8_t, 8> &coeffs)
{
    float16x8_t A   = vfmaq_f16(coeffs[0], coeffs[4], x);
    float16x8_t B   = vfmaq_f16(coeffs[2], coeffs[6], x);
    float16x8_t C   = vfmaq_f16(coeffs[1], coeffs[5], x);
    float16x8_t D   = vfmaq_f16(coeffs[3], coeffs[7], x);
    float16x8_t x2  = vmulq_f16(x, x);
    float16x8_t x4  = vmulq_f16(x2, x2);
    float16x8_t res = vfmaq_f16(vfmaq_f16(A, B, x2),
                                vfmaq_f16(C, D, x2),
                                x4);
    return res;
}

inline float16x8_t vexpq_f16_03_percent_error(float16x8_t x)
{
    const std::array<float16x8_t, 8> exp_tab =
    {
        {
            vdupq_n_f16(1.f),
            vdupq_n_f16(0.0416598916054f),
            vdupq_n_f16(0.500000596046f),
            vdupq_n_f16(0.0014122662833f),
            vdupq_n_f16(1.00000011921f),
            vdupq_n_f16(0.00833693705499f),
            vdupq_n_f16(0.166665703058f),
            vdupq_n_f16(0.000195780929062f),
        }
    };

    x = vminq_f16(x, vdupq_n_f16(11.0898664884f));

    static const float16x8_t CONST_LN2          = vdupq_n_f16(0.6931471805f);
    static const float16x8_t CONST_INV_LN2      = vdupq_n_f16(1.4426950408f);
    static const float16x8_t CONST_0            = vdupq_n_f16(0.f);
    static const int16x8_t   CONST_NEGATIVE_14 = vdupq_n_s16(-14);

    int16x8_t   m   = vcvtq_s16_f16(vmulq_f16(x, CONST_INV_LN2));
    float16x8_t val = vfmsq_f16(x, vcvtq_f16_s16(m), CONST_LN2);

    float16x8_t poly = vtaylor_polyq_f16(val, exp_tab);

    poly = vreinterpretq_f16_s16(vqaddq_s16(vreinterpretq_s16_f16(poly), vqshlq_n_s16(m, 10)));
    poly = vbslq_f16(vcltq_s16(m, CONST_NEGATIVE_14), CONST_0, poly);

    return poly;
}

inline float16x8_t vexpq_f16_4_percent_error_half_time(float16x8_t x)
{
    x = vminq_f16(x, vdupq_n_f16(11.0898664884f));
    static const float16x8_t CONST_Y = vdupq_n_f16(1477.3197217792);
    static const float16x8_t CONST_B = vdupq_n_f16(15301.3197217792);
    float16x8_t in1, in3;
    int16x8_t in2;
    x = vmaxq_f16(x, vdupq_n_f16(-10));
    in1 = vfmaq_f16(CONST_B, CONST_Y, x);
    in2 = vcvtq_s16_f16(in1);
    in3 = vreinterpretq_f16_s16(in2);
    return in3;
}


inline float16x8_t vexpq_f16_f32(float16x8_t a)
{
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t a0 = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a1 = vcvt_f32_f16(vget_high_f16(a));
    return vcombine_f16(vcvt_f16_f32(vexpq_f32_03_percent_error(a0)), vcvt_f16_f32(vexpq_f32_03_percent_error(a1)));
#else
    return vexpq_f16_03_percent_error(a);
#endif
}

inline float16x8_t vsigmoidq_f16(float16x8_t x)
{
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t x0 = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t x1 = vcvt_f32_f16(vget_high_f16(x));
    float16x8_t y = vcombine_f16(vcvt_f16_f32(vsigmoidq_f32(x0)),
                        vcvt_f16_f32(vsigmoidq_f32(x1)));
    return y;
#else
    float16x8_t one_v = vdupq_n_f16(1.f);
    return vrecpeq_f16(vaddq_f16_f32(vexpq_f16_03_percent_error(vnegq_f16(x)), one_v));
#endif
}

inline float16x8_t vtanhq_f16(float16x8_t x)
{
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t x0 = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t x1 = vcvt_f32_f16(vget_high_f16(x));
    float16x8_t y = vcombine_f16(vcvt_f16_f32(vtanhq_f32(x0)),
                        vcvt_f16_f32(vtanhq_f32(x1)));
    return y;
#else
    float16x8_t one_v = vdupq_n_f16(1.f);
    float16x8_t two_v = vdupq_n_f16(2.f);
    float16x8_t e_2G_v = vexpq_f16_03_percent_error(vmulq_f16(two_v, x));
    return vfmsq_f16(one_v, two_v, vrecpeq_f16(vaddq_f16(e_2G_v, one_v)));
#endif
}

// array sum
inline F32 array_sum_f16(const F16 *data, I32 len) {
    if(len <= 0) return 0;

    I32 i = 0;
    F32 sum_s = 0;
    float16x8_t sum_v = vdupq_n_f16(0);
    for(i = 0; i < len - 7; i+=8){
        float16x8_t in = vld1q_f16(data + i);
        sum_v = vaddq_f16(sum_v, in);
    }
    sum_s += vaddvq_f16(sum_v);
    for(; i < len; i++){
        sum_s += data[i];
    }
    return sum_s;
}

// array mean
inline F32 array_mean_f16(const F16 *data, I32 len) {
    if(len <= 0) return 0;
    return array_sum_f16(data, len) / len;
}

// array var
inline F32 array_var_f16(const F16 *data, I32 len, F32 mean) {
    if(len <= 0) return 0;

    I32 i = 0;
    F32 sum_s = 0;
    float32x4_t mean_v = vdupq_n_f32(mean);
    for(i = 0; i < len - 3; i+=4){
        float16x4_t in = vld1_f16(data + i);
        float32x4_t in_f32 = vcvt_f32_f16(in);
        float32x4_t tmp_v = vsubq_f32(in_f32, mean_v);
        float32x4_t sum_v = vmulq_f32(tmp_v, tmp_v);
        sum_s += vaddvq_f32(sum_v);
    }
    for(; i < len; i++){
        F16 in = data[i];
        F32 tmp = in - mean;
        sum_s += tmp * tmp;
    }
    return sum_s / len;
}

// array max
inline F16 array_max_f16(const F16* data, I32 len) {
    F16 max_s = data[0];
    I32 i = 0;
    if(len >= 8){
        float16x8_t max_v, tmp_v;
        max_v = vld1q_f16(data);
        for(i = 8; i < len - 7; i+=8){
            tmp_v = vld1q_f16(data + i);
            max_v = vmaxq_f16(tmp_v, max_v);
        }
        max_s = vmaxvq_f16(max_v);
    }

    for(; i < len; i++){
        if(data[i] > max_s)
            max_s = data[i];
    }

    return max_s;
}

inline void array_scale_f16(F16 *input, F16 *output, I32 len, F32 alpha, F32 beta) {
    I32 i = 0;
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t alpha_v = vdupq_n_f32(alpha);
    float32x4_t beta_v  = vdupq_n_f32(beta);
    for(i = 0; i < len - 3; i+=4){
        float16x4_t in = vld1_f16(input + i);
        float32x4_t in_f32 = vcvt_f32_f16(in);
        float32x4_t result = vfmaq_f32(beta_v, alpha_v, in_f32);
        vst1_f16(output + i, vcvt_f16_f32(result));
    }
#else
    float16x8_t alpha_v = vdupq_n_f16(alpha);
    float16x8_t beta_v  = vdupq_n_f16(beta);
    for (i = 0; i < len - 7; i += 8) {
        float16x8_t in = vld1q_f16(input + i);
        float16x8_t tmp_v = vfmaq_f16(beta_v, alpha_v, in);
        vst1q_f16(output+i, tmp_v);
    }
#endif
    for (; i < len; i++) {
        output[i] = alpha * input[i] + beta;
    }
}

inline EE activation_fp16(F16* data, U32 len, ActivationMode activationMode)
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
        case ACTIVATION_SIGMOID: {
            for (U32 i = 0; i < len_main; i++) {
                in = vld1q_f16(data);
                out = vsigmoidq_f16(in);
                vst1q_f16(data, out);
                data += 8;
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
#endif
