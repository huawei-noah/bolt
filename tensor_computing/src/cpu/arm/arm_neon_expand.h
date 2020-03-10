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
#include "error.h"

inline float32x4_t vtaylor_polyq_f32(float32x4_t x, const std::array<float32x4_t, 8> &coeffs)
{
    float32x4_t A   = vfmaq_f32(coeffs[0], coeffs[4], x);
    float32x4_t B   = vfmaq_f32(coeffs[2], coeffs[6], x);
    float32x4_t C   = vfmaq_f32(coeffs[1], coeffs[5], x);
    float32x4_t D   = vfmaq_f32(coeffs[3], coeffs[7], x);
    float32x4_t x2  = vmulq_f32(x, x);
    float32x4_t x4  = vmulq_f32(x2, x2);
    float32x4_t res = vfmaq_f32(vfmaq_f32(A, B, x2),
                                vfmaq_f32(C, D, x2),
                                x4);
    return res;
}

inline float32x4_t vexpq_f32_03_percent_error(float32x4_t x)
{
    const std::array<float32x4_t, 8> exp_tab =
    {
        {
            vdupq_n_f32(1.f),
            vdupq_n_f32(0.0416598916054f),
            vdupq_n_f32(0.500000596046f),
            vdupq_n_f32(0.0014122662833f),
            vdupq_n_f32(1.00000011921f),
            vdupq_n_f32(0.00833693705499f),
            vdupq_n_f32(0.166665703058f),
            vdupq_n_f32(0.000195780929062f),
        }
    };

    x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));

    static const float32x4_t CONST_LN2          = vdupq_n_f32(0.6931471805f);
    static const float32x4_t CONST_INV_LN2      = vdupq_n_f32(1.4426950408f);
    static const float32x4_t CONST_0            = vdupq_n_f32(0.f);
    static const int32x4_t   CONST_NEGATIVE_14 = vdupq_n_s32(-14);

    int32x4_t   m   = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
    float32x4_t val = vfmsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);

    float32x4_t poly = vtaylor_polyq_f32(val, exp_tab);

    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_14), CONST_0, poly);

    return poly;
}

inline float32x4_t vsigmoidq_f32(float32x4_t x)
{
    float32x4_t one_v = vdupq_n_f32(1.f);
    return vrecpeq_f32(vaddq_f32(vexpq_f32_03_percent_error(vnegq_f32(x)), one_v));
}

inline float32x4_t vtanhq_f32(float32x4_t x)
{
    float32x4_t one_v = vdupq_n_f32(1.f);
    float32x4_t two_v = vdupq_n_f32(2.f);
    float32x4_t e_2G_v = vexpq_f32_03_percent_error(vmulq_f32(two_v, x));
    return vfmsq_f32(one_v, two_v, vrecpeq_f32(vaddq_f32(e_2G_v, one_v)));
}
#endif
