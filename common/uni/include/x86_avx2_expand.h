// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_X86_AVX2_EXPAND
#define _H_X86_AVX2_EXPAND

//horizontal add u32
inline unsigned int _mm256_hadd_u32(__m256i x)
{
    __m128i low = _mm256_extracti128_si256(x, 0);
    __m128i high = _mm256_extracti128_si256(x, 1);
    __m128i sum = _mm_add_epi32(low, high);
    int one = _mm_extract_epi32(sum, 0);
    int two = _mm_extract_epi32(sum, 1);
    int three = _mm_extract_epi32(sum, 2);
    int four = _mm_extract_epi32(sum, 3);

    return (one + two + three + four);
}

inline __m256 _mm256_log_ps(__m256 x)
{
    static const __m256 CONST_one = _mm256_set1_ps(1.0f);
    static const __m256 CONST_two = _mm256_set1_ps(2.0f);
    static const __m256 CONST_neg_one = _mm256_set1_ps(-1.0f);
    F32 i = 30;
    __m256 n = _mm256_set1_ps(i);
    __m256 nk = _mm256_add_ps(_mm256_mul_ps(CONST_two, n), CONST_one);
    x = _mm256_div_ps(_mm256_add_ps(x, CONST_neg_one), _mm256_add_ps(x, CONST_one));
    __m256 xx = _mm256_mul_ps(x, x);
    __m256 y = _mm256_div_ps(CONST_one, nk);
    for (; i > 0; i--) {
        nk = _mm256_sub_ps(nk, CONST_two);
        y = _mm256_add_ps(_mm256_div_ps(CONST_one, nk), _mm256_mul_ps(xx, y));
    }

    y = _mm256_mul_ps(CONST_two, _mm256_mul_ps(x, y));
    return y;
}

inline __m256 _mm256_exp_ps(__m256 x)
{
    // the max and min x in exp(x) in 32-bit float range
    __m256 max_upper_bound = _mm256_set1_ps(88.3762626647949f);
    __m256 min_lower_bound = _mm256_set1_ps(-87.3365447504019f);

    x = _mm256_min_ps(x, max_upper_bound);
    x = _mm256_max_ps(x, min_lower_bound);

    __m256 t, f, p, r;
    __m256i i, j;

    const __m256 l2e = _mm256_set1_ps(1.442695041f);    /* log2(e) */
    const __m256 l2h = _mm256_set1_ps(-6.93145752e-1f); /* -log(2)_hi */
    const __m256 l2l = _mm256_set1_ps(-1.42860677e-6f); /* -log(2)_lo */
    const __m256 c0 = _mm256_set1_ps(0.008301110f);
    const __m256 c1 = _mm256_set1_ps(0.041906696f);
    const __m256 c2 = _mm256_set1_ps(0.166674897f);
    const __m256 c3 = _mm256_set1_ps(0.499990642f);
    const __m256 c4 = _mm256_set1_ps(0.999999762f);
    const __m256 c5 = _mm256_set1_ps(1.000000000f);

    /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
    t = _mm256_mul_ps(x, l2e);                                             /* t = log2(e) * x */
    r = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); /* r = rint (t) */

    f = _mm256_fmadd_ps(r, l2h, x); /* x - log(2)_hi * r */
    f = _mm256_fmadd_ps(r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */

    i = _mm256_cvtps_epi32(t); /* i = (int)rint(t) */

    /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
    p = c0;                        /* c0 */
    p = _mm256_fmadd_ps(p, f, c1); /* c0*f+c1 */
    p = _mm256_fmadd_ps(p, f, c2); /* (c0*f+c1)*f+c2 */
    p = _mm256_fmadd_ps(p, f, c3); /* ((c0*f+c1)*f+c2)*f+c3 */
    p = _mm256_fmadd_ps(p, f, c4); /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
    p = _mm256_fmadd_ps(p, f, c5); /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
    /* exp(x) = 2^i * p */
    j = _mm256_slli_epi32(i, 23);                                         /* i << 23 */
    r = _mm256_castsi256_ps(_mm256_add_epi32(j, _mm256_castps_si256(p))); /* r = p * 2^i */

    return r;
}

inline __m256 _mm256_sigmod_ps(__m256 x)
{
    __m256 one_v = _mm256_set1_ps(1.f);
    __m256 neg_one_v = _mm256_set1_ps(-1.f);
    return _mm256_rcp_ps(_mm256_add_ps(_mm256_exp_ps(_mm256_mul_ps(x, neg_one_v)), one_v));
}

inline __m256 _mm256_tanh_ps(__m256 x)
{
    __m256 one_v = _mm256_set1_ps(1.f);
    __m256 two_v = _mm256_set1_ps(2.f);
    __m256 e_2G_v = _mm256_exp_ps(_mm256_mul_ps(two_v, x));
    __m256 result_v = _mm256_sub_ps(one_v, _mm256_div_ps(two_v, _mm256_add_ps(one_v, e_2G_v)));
    return result_v;
}

// horizontal add, sum array to f32
inline F32 _mm256_sum_ps(__m256 x)
{
    __m128 low = _mm256_extractf128_ps(x, 0);
    __m128 high = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_hadd_ps(low, high);
    low = _mm_hadd_ps(sum, sum);
    high = _mm_permute_ps(low, 0b01);
    sum = _mm_add_ss(low, high);
    return _mm_cvtss_f32(sum);
}

// horizontal max
inline F32 _mm256_hmax_ps(__m256 x)
{
    __m128 low = _mm256_extractf128_ps(x, 0);
    __m128 high = _mm256_extractf128_ps(x, 1);
    __m128 max = _mm_max_ps(low, high);
    high = _mm_permute_ps(max, 0b1110);
    low = _mm_max_ps(max, high);
    high = _mm_permute_ps(low, 0b01);
    max = _mm_max_ss(low, high);
    return _mm_cvtss_f32(max);
}
#endif  // _H_X86_AVX2_EXPAND
