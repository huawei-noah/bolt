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

#include <arm_neon.h>
#include <array>
#include <math.h>
#include "error.h"

#ifndef __aarch64__
inline float32x4_t vrndaq_f32(float32x4_t a)
{
    int32x4_t b = vcvtq_n_s32_f32(a, 1);
    b = vsraq_n_s32(b, b, 31);
    b = vrshrq_n_s32(b, 1);
    return vcvtq_f32_s32(b);
}

inline int32x4_t vcvtaq_s32_f32(float32x4_t a)
{
    int32x4_t b = vcvtq_n_s32_f32(a, 1);
    b = vsraq_n_s32(b, b, 31);
    return vrshrq_n_s32(b, 1);
}

inline int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b)
{
    int32x2_t low = vpadd_s32(vget_low_s32(a), vget_low_s32(b));
    int32x2_t high = vpadd_s32(vget_high_s32(a), vget_high_s32(b));
    int32x2x2_t v = vzip_s32(low, high);
    return vcombine_s32(v.val[0], v.val[1]);
}

inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b)
{
    float32x4_t b_recip = vrecpeq_f32(b);
    b_recip = vmulq_f32(vrecpsq_f32(b, b_recip), b_recip);
    return vmulq_f32(a, b_recip);
}

inline int vminvq_s32(int32x4_t x)
{
    int32x2_t min = vmin_s32(vget_low_s32(x), vget_high_s32(x));
    min = vpmin_s32(min, min);
    return vget_lane_s32(min, 0);
}

inline int vmaxvq_s32(int32x4_t x)
{
    int32x2_t max = vmax_s32(vget_low_s32(x), vget_high_s32(x));
    max = vpmax_s32(max, max);
    return vget_lane_s32(max, 0);
}

inline float vminvq_f32(float32x4_t x)
{
    float32x2_t min = vmin_f32(vget_low_f32(x), vget_high_f32(x));
    min = vpmin_f32(min, min);
    return vget_lane_f32(min, 0);
}

inline float vmaxvq_f32(float32x4_t x)
{
    float32x2_t max = vmax_f32(vget_low_f32(x), vget_high_f32(x));
    max = vpmax_f32(max, max);
    return vget_lane_f32(max, 0);
}

#if !defined(__ANDROID__) && !defined(__APPLE__) && !defined(__WIN32)
#ifndef __ARM_FEATURE_FMA
inline float32x4_t vfmaq_f32(float32x4_t c, float32x4_t a, float32_t b)
{
    return vmlaq_f32(c, a, vdupq_n_f32(b));
}
#endif

inline float32x4_t vfmaq_n_f32(float32x4_t c, float32x4_t a, float32_t b)
{
    return vfmaq_f32(c, a, vdupq_n_f32(b));
}
#endif

inline float vaddvq_f32(float32x4_t x)
{
    float32x2_t sum = vadd_f32(vget_low_f32(x), vget_high_f32(x));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}

inline int vaddvq_s32(int32x4_t x)
{
    int32x2_t sum = vadd_s32(vget_low_s32(x), vget_high_s32(x));
    sum = vpadd_s32(sum, sum);
    return vget_lane_s32(sum, 0);
}

inline int vaddvq_s8(int8x16_t x)
{
    int16x8_t y0 = vaddl_s8(vget_low_s8(x), vget_high_s8(x));
    int32x4_t y1 = vaddl_s16(vget_low_s16(y0), vget_high_s16(y0));
    return vaddvq_s32(y1);
}
#endif

inline float32x4_t vtaylor_polyq_f32(float32x4_t x, const std::array<float32x4_t, 8> &coeffs)
{
    float32x4_t A = vfmaq_f32(coeffs[0], coeffs[4], x);
    float32x4_t B = vfmaq_f32(coeffs[2], coeffs[6], x);
    float32x4_t C = vfmaq_f32(coeffs[1], coeffs[5], x);
    float32x4_t D = vfmaq_f32(coeffs[3], coeffs[7], x);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t res = vfmaq_f32(vfmaq_f32(A, B, x2), vfmaq_f32(C, D, x2), x4);
    return res;
}

inline float32x4_t vexpq_f32_03_percent_error(float32x4_t x)
{
    const std::array<float32x4_t, 8> exp_tab = {{
        vdupq_n_f32(1.f),
        vdupq_n_f32(0.0416598916054f),
        vdupq_n_f32(0.500000596046f),
        vdupq_n_f32(0.0014122662833f),
        vdupq_n_f32(1.00000011921f),
        vdupq_n_f32(0.00833693705499f),
        vdupq_n_f32(0.166665703058f),
        vdupq_n_f32(0.000195780929062f),
    }};

    x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));

    static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f);
    static const float32x4_t CONST_INV_LN2 = vdupq_n_f32(1.4426950408f);
    static const float32x4_t CONST_0 = vdupq_n_f32(0.f);
    static const int32x4_t CONST_NEGATIVE_14 = vdupq_n_s32(-14);

    int32x4_t m = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
    float32x4_t val = vfmsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);

    float32x4_t poly = vtaylor_polyq_f32(val, exp_tab);

    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_14), CONST_0, poly);

    return poly;
}

inline float32x4_t vlogq_f32(float32x4_t x)
{
    uint32x4_t ux = vreinterpretq_u32_f32(x);
    float32x4_t fx = vcvtq_f32_u32(ux);
    // fx * (1.0f / (1 << 23))
    fx = vmulq_f32(fx, vdivq_f32(vdupq_n_f32(1.0f), vcvtq_f32_u32(vshlq_n_u32(vdupq_n_u32(1), 23))));

    uint32x4_t umx =
        vorrq_u32(vandq_u32(ux, vdupq_n_u32(0x007FFFFF)), vshlq_n_u32(vdupq_n_u32(0x7e), 23));
    float32x4_t mx = vreinterpretq_f32_u32(umx);

    const float32x4_t c_124_22551499 = vdupq_n_f32(124.22551499f);
    const float32x4_t c_1_498030302 = vdupq_n_f32(1.498030302f);
    const float32x4_t c_1_725877999 = vdupq_n_f32(1.72587999f);
    const float32x4_t c_0_3520087068 = vdupq_n_f32(0.3520887068f);

    float32x4_t tmp = vdivq_f32(c_1_725877999, vaddq_f32(c_0_3520087068, mx));
    tmp = vaddq_f32(c_124_22551499, tmp);
    tmp = vfmaq_f32(tmp, c_1_498030302, mx);
    const float32x4_t c_0_69314718 = vdupq_n_f32(0.69314718f);
    float32x4_t result_v = vmulq_f32(vsubq_f32(fx, tmp), c_0_69314718);
    result_v = vbslq_f32(vcltq_f32(x, vdupq_n_f32(0)), vdupq_n_f32(NAN), result_v);
    result_v = vbslq_f32(vceqq_f32(x, vdupq_n_f32(0)), vdupq_n_f32(-INFINITY), result_v);
    return result_v;
}

inline float32x4_t vsigmoidq_f32(float32x4_t x)
{
    float32x4_t one_v = vdupq_n_f32(1.f);
    float32x4_t tmp_v = vaddq_f32(vexpq_f32_03_percent_error(vnegq_f32(x)), one_v);
    float32x4_t out_v = vrecpeq_f32(tmp_v);
    out_v = vmulq_f32(vrecpsq_f32(tmp_v, out_v), out_v);
    return out_v;
}

inline float32x4_t vtanhq_f32(float32x4_t x)
{
    float32x4_t one_v = vdupq_n_f32(1.f);
    float32x4_t two_v = vdupq_n_f32(2.f);
    float32x4_t e_2G_v = vexpq_f32_03_percent_error(vmulq_f32(two_v, x));
    // float32x4_t result_v = vfmsq_f32(one_v, two_v, vrecpeq_f32(vaddq_f32(e_2G_v, one_v)));
    float32x4_t result_v = vsubq_f32(one_v, vdivq_f32(two_v, vaddq_f32(one_v, e_2G_v)));
    return result_v;
}

inline void vsincosq_f32(float32x4_t x, float32x4_t *ysin, float32x4_t *ycos)
{
    const float32x4_t c_dp = vdupq_n_f32(-0.7853981633974483);
    const float32x4_t c_4_div_pi = vdupq_n_f32(1.27323954473516);
    const float32x4_t c_sin_p0 = vdupq_n_f32(-1.9515295891E-4);
    const float32x4_t c_sin_p1 = vdupq_n_f32(8.3321608736E-3);
    const float32x4_t c_sin_p2 = vdupq_n_f32(-1.6666654611E-1);
    const float32x4_t c_cos_p0 = vdupq_n_f32(2.443315711809948E-005);
    const float32x4_t c_cos_p1 = vdupq_n_f32(-1.388731625493765E-003);
    const float32x4_t c_cos_p2 = vdupq_n_f32(4.166664568298827E-002);
    const uint32x4_t c_2 = vdupq_n_u32(2);
    const uint32x4_t c_4 = vdupq_n_u32(4);

    uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(0));
    x = vabsq_f32(x);

    float32x4_t y = vmulq_f32(x, c_4_div_pi);

    uint32x4_t emm2 = vcvtq_u32_f32(y);
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);

    uint32x4_t poly_mask = vtstq_u32(emm2, c_2);

    x = vfmaq_f32(x, y, c_dp);

    uint32x4_t sin_sign = veorq_u32(mask, vtstq_u32(emm2, c_4));
    uint32x4_t cos_sign = vtstq_u32(vsubq_u32(emm2, c_2), c_4);

    float32x4_t z = vmulq_f32(x, x);
    float32x4_t y1 = vfmaq_f32(c_cos_p1, z, c_cos_p0);
    float32x4_t y2 = vfmaq_f32(c_sin_p1, z, c_sin_p0);
    y1 = vfmaq_f32(c_cos_p2, y1, z);
    y2 = vfmaq_f32(c_sin_p2, y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, z);
    y1 = vmulq_f32(y1, z);
    y2 = vmulq_f32(y2, x);
    y1 = vsubq_f32(y1, vmulq_f32(z, vdupq_n_f32(0.5f)));
    y2 = vaddq_f32(y2, x);
    y1 = vaddq_f32(y1, vdupq_n_f32(1));

    float32x4_t ys = vbslq_f32(poly_mask, y1, y2);
    float32x4_t yc = vbslq_f32(poly_mask, y2, y1);
    *ysin = vbslq_f32(sin_sign, vnegq_f32(ys), ys);
    *ycos = vbslq_f32(cos_sign, yc, vnegq_f32(yc));
}

inline float32x4_t vsinq_f32(float32x4_t x)
{
    float32x4_t ysin, ycos;
    vsincosq_f32(x, &ysin, &ycos);
    return ysin;
}

inline float32x4_t vcosq_f32(float32x4_t x)
{
    float32x4_t ysin, ycos;
    vsincosq_f32(x, &ysin, &ycos);
    return ycos;
}

#ifdef _USE_FP16
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
    float16x8_t A = vfmaq_f16(coeffs[0], coeffs[4], x);
    float16x8_t B = vfmaq_f16(coeffs[2], coeffs[6], x);
    float16x8_t C = vfmaq_f16(coeffs[1], coeffs[5], x);
    float16x8_t D = vfmaq_f16(coeffs[3], coeffs[7], x);
    float16x8_t x2 = vmulq_f16(x, x);
    float16x8_t x4 = vmulq_f16(x2, x2);
    float16x8_t res = vfmaq_f16(vfmaq_f16(A, B, x2), vfmaq_f16(C, D, x2), x4);
    return res;
}

inline float16x8_t vexpq_f16_03_percent_error(float16x8_t x)
{
    const std::array<float16x8_t, 8> exp_tab = {{
        vdupq_n_f16(1.f),
        vdupq_n_f16(0.0416598916054f),
        vdupq_n_f16(0.500000596046f),
        vdupq_n_f16(0.0014122662833f),
        vdupq_n_f16(1.00000011921f),
        vdupq_n_f16(0.00833693705499f),
        vdupq_n_f16(0.166665703058f),
        vdupq_n_f16(0.000195780929062f),
    }};

    x = vminq_f16(x, vdupq_n_f16(11.0898664884f));

    static const float16x8_t CONST_LN2 = vdupq_n_f16(0.6931471805f);
    static const float16x8_t CONST_INV_LN2 = vdupq_n_f16(1.4426950408f);
    static const float16x8_t CONST_0 = vdupq_n_f16(0.f);
    static const int16x8_t CONST_NEGATIVE_14 = vdupq_n_s16(-14);

    int16x8_t m = vcvtq_s16_f16(vmulq_f16(x, CONST_INV_LN2));
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

inline void vsincosq_f16(float16x8_t x, float16x8_t *ysin, float16x8_t *ycos)
{
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t a1 = vcvt_f32_f16(vget_high_f16(x));
    float32x4_t s0, s1, c0, c1;
    vsincosq_f32(a0, &s0, &c0);
    vsincosq_f32(a1, &s1, &c1);
    *ysin = vcombine_f16(vcvt_f16_f32(s0), vcvt_f16_f32(s1));
    *ycos = vcombine_f16(vcvt_f16_f32(c0), vcvt_f16_f32(c1));
#else
    const float16x8_t c_dp = vdupq_n_f16(-0.7853981633974483);
    const float16x8_t c_4_div_pi = vdupq_n_f16(1.27323954473516);
    const float16x8_t c_sin_p0 = vdupq_n_f16(-1.9515295891E-4);
    const float16x8_t c_sin_p1 = vdupq_n_f16(8.3321608736E-3);
    const float16x8_t c_sin_p2 = vdupq_n_f16(-1.6666654611E-1);
    const float16x8_t c_cos_p0 = vdupq_n_f16(2.443315711809948E-005);
    const float16x8_t c_cos_p1 = vdupq_n_f16(-1.388731625493765E-003);
    const float16x8_t c_cos_p2 = vdupq_n_f16(4.166664568298827E-002);
    const uint16x8_t c_2 = vdupq_n_u16(2);
    const uint16x8_t c_4 = vdupq_n_u16(4);

    uint16x8_t mask = vcltq_f16(x, vdupq_n_f16(0));
    x = vabsq_f16(x);

    float16x8_t y = vmulq_f16(x, c_4_div_pi);

    uint16x8_t emm2 = vcvtq_u16_f16(y);
    emm2 = vaddq_u16(emm2, vdupq_n_u16(1));
    emm2 = vandq_u16(emm2, vdupq_n_u16(~1));
    y = vcvtq_f16_u16(emm2);

    uint16x8_t poly_mask = vtstq_u16(emm2, c_2);

    x = vfmaq_f16(x, y, c_dp);

    uint16x8_t sin_sign = veorq_u16(mask, vtstq_u16(emm2, c_4));
    uint16x8_t cos_sign = vtstq_u16(vsubq_u16(emm2, c_2), c_4);

    float16x8_t z = vmulq_f16(x, x);
    float16x8_t y1 = vfmaq_f16(c_cos_p1, z, c_cos_p0);
    float16x8_t y2 = vfmaq_f16(c_sin_p1, z, c_sin_p0);
    y1 = vfmaq_f16(c_cos_p2, y1, z);
    y2 = vfmaq_f16(c_sin_p2, y2, z);
    y1 = vmulq_f16(y1, z);
    y2 = vmulq_f16(y2, z);
    y1 = vmulq_f16(y1, z);
    y2 = vmulq_f16(y2, x);
    y1 = vsubq_f16(y1, vmulq_f16(z, vdupq_n_f16(0.5f)));
    y2 = vaddq_f16(y2, x);
    y1 = vaddq_f16(y1, vdupq_n_f16(1));

    float16x8_t ys = vbslq_f16(poly_mask, y1, y2);
    float16x8_t yc = vbslq_f16(poly_mask, y2, y1);
    *ysin = vbslq_f16(sin_sign, vnegq_f16(ys), ys);
    *ycos = vbslq_f16(cos_sign, yc, vnegq_f16(yc));
#endif
}

inline float16x8_t vexpq_f16_f32(float16x8_t a)
{
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t a0 = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a1 = vcvt_f32_f16(vget_high_f16(a));
    return vcombine_f16(
        vcvt_f16_f32(vexpq_f32_03_percent_error(a0)), vcvt_f16_f32(vexpq_f32_03_percent_error(a1)));
#else
    return vexpq_f16_03_percent_error(a);
#endif
}

inline float16x8_t vlogq_f16(float16x8_t x)
{
    float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t a1 = vcvt_f32_f16(vget_high_f16(x));
    return vcombine_f16(vcvt_f16_f32(vlogq_f32(a0)), vcvt_f16_f32(vlogq_f32(a1)));
}

inline float16x8_t vsigmoidq_f16(float16x8_t x)
{
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t x0 = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t x1 = vcvt_f32_f16(vget_high_f16(x));
    float16x8_t y = vcombine_f16(vcvt_f16_f32(vsigmoidq_f32(x0)), vcvt_f16_f32(vsigmoidq_f32(x1)));
    return y;
#else
    float16x8_t one_v = vdupq_n_f16(1.f);
    float16x8_t tmp_v = vaddq_f16_f32(vexpq_f16_03_percent_error(vnegq_f16(x)), one_v);
    float16x8_t out_v = vrecpeq_f16(tmp_v);
    out_v = vmulq_f16(vrecpsq_f16(tmp_v, out_v), out_v);
    return out_v;
#endif
}

inline float16x8_t vtanhq_f16(float16x8_t x)
{
#ifdef _USE_F16_MIX_PRECISION
    float32x4_t x0 = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t x1 = vcvt_f32_f16(vget_high_f16(x));
    float16x8_t y = vcombine_f16(vcvt_f16_f32(vtanhq_f32(x0)), vcvt_f16_f32(vtanhq_f32(x1)));
    return y;
#else
    float16x8_t one_v = vdupq_n_f16(1.f);
    float16x8_t two_v = vdupq_n_f16(2.f);
    float16x8_t e_2G_v = vexpq_f16_03_percent_error(vmulq_f16(two_v, x));
    // float16x8_t result_v = vfmsq_f16(one_v, two_v, vrecpeq_f16(vaddq_f16(e_2G_v, one_v)));
    float16x8_t result_v = vsubq_f16(one_v, vdivq_f16(two_v, vaddq_f16(one_v, e_2G_v)));
    return result_v;
#endif
}

inline float vaddvq_f16(float16x8_t x)
{
    float32x4_t a = vcvt_f32_f16(vget_high_f16(x));
    float32x4_t b = vcvt_f32_f16(vget_low_f16(x));
    float sum = vaddvq_f32(vaddq_f32(a, b));
    return sum;
}

inline float16x8_t vsinq_f16(float16x8_t x)
{
    float16x8_t ysin, ycos;
    vsincosq_f16(x, &ysin, &ycos);
    return ysin;
}

inline float16x8_t vcosq_f16(float16x8_t x)
{
    float16x8_t ysin, ycos;
    vsincosq_f16(x, &ysin, &ycos);
    return ycos;
}

inline void vst1q_lane_f16_builtin(__fp16 *address, float16x8_t vec, const int laneId)
{
    switch (laneId) {
        case 0:
            vst1q_lane_f16(address, vec, 0);
            break;
        case 1:
            vst1q_lane_f16(address, vec, 1);
            break;
        case 2:
            vst1q_lane_f16(address, vec, 2);
            break;
        case 3:
            vst1q_lane_f16(address, vec, 3);
            break;
        case 4:
            vst1q_lane_f16(address, vec, 4);
            break;
        case 5:
            vst1q_lane_f16(address, vec, 5);
            break;
        case 6:
            vst1q_lane_f16(address, vec, 6);
            break;
        case 7:
            vst1q_lane_f16(address, vec, 7);
            break;
        default:
            CHECK_REQUIREMENT(0);
    }
}
#endif

#ifdef _USE_INT8
#ifdef _USE_FP16
inline int32x4_t vdotq_laneq_s32_builtin(int32x4_t c, int8x16_t a, int8x16_t b, const int laneId)
{
    int32x4_t ret;
    switch (laneId) {
        case 0:
            ret = vdotq_laneq_s32(c, a, b, 0);
            break;
        case 1:
            ret = vdotq_laneq_s32(c, a, b, 1);
            break;
        case 2:
            ret = vdotq_laneq_s32(c, a, b, 2);
            break;
        case 3:
            ret = vdotq_laneq_s32(c, a, b, 3);
            break;
        default:
            CHECK_REQUIREMENT(0);
            ret = vdupq_n_s32(0);
            break;
    }
    return ret;
}
#endif
#endif
#endif
