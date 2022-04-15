// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef VEC_FUNCTIONS_
#define VEC_FUNCTIONS_

#if defined(__aarch64__)

#include <arm_neon.h>

static inline float32x4_t exp4_approx(float32x4_t& x)
{
    int32x4_t i;
    float32x4_t xf;

    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(10.f)), vdupq_n_f32(-10.f));

    /* express exp(x) as exp2(x/log(2)), add 127 for the exponent later */
    x = vmlaq_f32(vdupq_n_f32(127.f), x, vdupq_n_f32(1.44269504f));

    /* split into integer and fractional parts */
    i = vcvtq_s32_f32(x);
    xf = vcvtq_f32_s32(i);
    x = vsubq_f32(x, xf);

    float32x4_t K0 = vdupq_n_f32(0.99992522f);
    float32x4_t K1 = vdupq_n_f32(0.69583354f);
    float32x4_t K2 = vdupq_n_f32(0.22606716f);
    float32x4_t K3 = vdupq_n_f32(0.078024523f);
    float32x4_t Y = vmlaq_f32(K0, x, vmlaq_f32(K1, x, vmlaq_f32(K2, K3, x)));

    /* compute 2^i */
    float32x4_t exponent = vreinterpretq_f32_s32(vshlq_n_s32(i, 23));

    Y = vmulq_f32(Y, exponent);
    return Y;
}

static float celt_exp(float x)
{
    float32x4_t X, Y;
    X = vdupq_n_f32(x);
    Y = exp4_approx(X);
    return Y[0];
}

static inline void vec_tanh(float* __restrict y, const float* __restrict x, int N)
{
    int i;
#pragma unroll(4)
    for (i = 0; i < N - 3; i += 4)
    {
        static const float32x4_t two = vdupq_n_f32(2.f);
        static const float32x4_t one = vdupq_n_f32(1.f);
        float32x4_t X, Y;
        X = vld1q_f32(&x[i]);
        X = vmulq_f32(X, two);
        Y = exp4_approx(X);
#ifdef __aarch64__
        Y = vdivq_f32(vsubq_f32(Y, one), vaddq_f32(Y, one));
#else
        Y = vmulq_f32(vsubq_f32(Y, one), vrecpeq_f32(vaddq_f32(Y, one)));
#endif
        vst1q_f32(&y[i], Y);
    }
    for (; i < N; i++)
    {
        float ex2;
        ex2 = celt_exp(2 * x[i]);
        y[i] = (ex2 - 1) / (ex2 + 1);
    }
}

static inline float32x4_t sigmoid4(float32x4_t X)
{
    static const float32x4_t one = vdupq_n_f32(1.f);
    auto Y = exp4_approx(X);
#ifdef __aarch64__
    Y = vdivq_f32(Y, vaddq_f32(Y, one));
#else
    Y = vmulq_f32(Y, vrecpeq_f32(vaddq_f32(Y, one)));
#endif
    return Y;
}

static inline void vec_sigmoid(float* __restrict y, const float* __restrict x, int N)
{
    int i;
#pragma unroll(4)
    for (i = 0; i < N - 3; i += 4)
    {
        static const float32x4_t one = vdupq_n_f32(1.f);
        float32x4_t X, Y;
        X = vld1q_f32(&x[i]);
        Y = exp4_approx(X);
#ifdef __aarch64__
        Y = vdivq_f32(Y, vaddq_f32(Y, one));
#else
        Y = vmulq_f32(Y, vrecpeq_f32(vaddq_f32(Y, one)));
#endif
        vst1q_f32(&y[i], Y);
    }
    for (; i < N; i++)
    {
        float ex;
        ex = celt_exp(x[i]);
        y[i] = (ex) / (ex + 1);
    }
}

#endif // arm64

#endif