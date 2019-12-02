// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_ARM_NEON_EXPAND
#define _H_ARM_NEON_EXPAND
#include <math.h>
#include <arm_neon.h>
#include <array>

#include "type.h"

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
    for (U32 idx = 0; idx < 8; idx++) {
        F16 val = vgetq_lane_f16(x, idx);
        val = (val < 11.0898664884f) ? val : 11.0898664884f;
        x = vsetq_lane_f16(val, x, idx);
    }

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
    for (U32 idx = 0; idx < 8; idx++) {
        F16 val = vgetq_lane_f16(x, idx);
        val = (val < 11.0898664884f) ? val : 11.0898664884f;
        x = vsetq_lane_f16(val, x, idx);
    }
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

// sigmoid function
inline float16x8_t vsigmoidq_f16(float16x8_t x)
{
    float16x8_t one_v = vdupq_n_f16(1.f);
    return vrecpeq_f16(vaddq_f16(vexpq_f16_03_percent_error(vnegq_f16(x)), one_v));
}


// tanh function
inline float16x8_t vtanhq_f16(float16x8_t x)
{
    float16x8_t one_v = vdupq_n_f16(1.f);
    float16x8_t two_v = vdupq_n_f16(2.f);
    float16x8_t e_2G_v = vexpq_f16_03_percent_error(vmulq_f16(two_v, x));
    return vfmsq_f16(one_v, two_v, vrecpeq_f16(vaddq_f16(e_2G_v, one_v)));
}


// array sum
inline F16 array_sum_f16(F16 *data, I32 len) {
    if(len <= 0) return 0;

    F16 buffer[8];

    I32 i = 0;
    float16x8_t sum_v = vdupq_n_f16(0);
    for(i = 0; i < len - 7; i += 8){
        float16x8_t in = vld1q_f16(data + i);
        sum_v = vaddq_f16(sum_v, in);
    }
    vst1q_f16(buffer, sum_v);

    F16 sum_s = 0;
    for(U32 j = 0; j < 8; j++) {
        sum_s += buffer[j];
    }
    for(; i < len; i++){
        sum_s += data[i];
    }
    return sum_s;
}


// array mean
inline F16 array_mean_f16(F16 *data, I32 len) {
    if(len <= 0) return 0;
    return array_sum_f16(data, len) / len;
}


// array var
inline F16 array_var_f16(F16 *data, I32 len, F16 mean) {
    if(len <= 0) return 0;
    F16 buffer[8];

    I32 i = 0;
    float16x8_t sum_v  = vdupq_n_f16(0);
    float16x8_t mean_v = vdupq_n_f16(mean);
    for(i = 0; i < len - 7; i += 8){
        float16x8_t in = vld1q_f16(data + i);
        float16x8_t tmp_v = vsubq_f16(in, mean_v);
        sum_v = vfmaq_f16(sum_v, tmp_v, tmp_v);
    }
    vst1q_f16(buffer, sum_v);

    F16 sum_s = 0;
    for(U32 j = 0; j < 8; j++) {
        sum_s += buffer[j];
    }
    for(; i < len; i++){
        F16 in = data[i];
        F16 tmp = in - mean;
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
#endif
