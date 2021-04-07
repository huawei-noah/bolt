// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"
#define MANGLE_NAME_LMPL(base, AM, FM, BM, LM, LN) base##AM##FM##BM##LM##LN
#define MANGLE_NAME(base, AM, FM, BM, LM, LN) MANGLE_NAME_LMPL(base, AM, FM, BM, LM, LN)

#define BM
#define FM

#if defined(USE_POINTWISE_NCWHC4)
#define FM pointwise_ncwhc4_
#endif

#if defined(NO_BIAS)
#define BM nobias_
#elif defined(USE_BIAS_MATCH_A)
#define BM biasA_
#elif defined(USE_BIAS_MATCH_B)
#define BM biasB_
#endif

#if defined(USE_POINTWISE_NCWHC4)
#define STORE_REG_V4(c0, c1, c2, c3, off, buf) \
    {                                          \
        T4 tmp;                                \
        tmp.x = c0[0];                         \
        tmp.y = c1[0];                         \
        tmp.z = c2[0];                         \
        tmp.w = c3[0];                         \
        ACTIVATION_V4(tmp);                    \
        vstore4(tmp, c_off, buf);              \
        UPDATE_REG(c0);                        \
        UPDATE_REG(c1);                        \
        UPDATE_REG(c2);                        \
        UPDATE_REG(c3);                        \
    }

#if (LM == 4)
#define STORE_REG(iy, oc, c, str, off, buf)             \
    {                                                   \
        STORE_REG_V4(c[0], c[1], c[2], c[3], off, buf); \
    }
#elif (LM == 8)
#define STORE_REG(iy, oc, c, str, off, buf)             \
    {                                                   \
        STORE_REG_V4(c[0], c[1], c[2], c[3], off, buf); \
        if (iy + 4 >= oc) {                             \
            continue;                                   \
        }                                               \
        off += str;                                     \
        STORE_REG_V4(c[4], c[5], c[6], c[7], off, buf); \
    }
#endif

#define STORE_C()                                                        \
    {                                                                    \
        int c_base = (idz * by * (LM >> 2) + (iy >> 2)) * C_str + C_off; \
        for (uchar i = 0; i < LN; ++i) {                                 \
            int oxh = (ix + i) % oh;                                     \
            int oxw = (ix + i) / oh;                                     \
            if (oxw >= ow) {                                             \
                break;                                                   \
            }                                                            \
            int c_off = c_base + oxw * ow_str + oxh;                     \
            STORE_REG(iy, oc, c, C_str, c_off, C)                        \
        }                                                                \
    }
#else
#define STORE_C()                                           \
    {                                                       \
        int c_off = idz * C_str + iy * ow_str + ix + C_off; \
        char ex = (ix + LN <= ow) ? LN : (ow % LN);         \
        char ey = (iy + LM <= oh) ? LM : (oh % LM);         \
        GEMM_STORE_C(c, c_off, ow_str, ex, ey, C);          \
    }
#endif

__kernel void MANGLE_NAME(gemm_tn_, AM, FM, BM, LM, LN)(const int M,
    const int N,
    const int K,
    const int A_str,
    const int B_str,
    const int C_str,
    const int A_off,
    const int B_off,
    const int C_off,
    const int ow_str,
    const int ow,
    const int oh,
    const int oc,
    const int bx,
    const int by,
    __global const T *A,
    __global const T *B,
#if !defined(NO_BIAS)
    __global const T *bias,
#endif
    __global T *C)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    if (idx >= bx || idy >= by) {
        return;
    }
    const int ix = idx * LN;
    const int iy = idy * LM;

    T a[LM];
    T b[LN];
    T c[LM][LN];
    int a_off = idz * A_str + iy + A_off;
    int b_off = idz * B_str + ix + B_off;

#if defined(USE_BIAS_MATCH_A)
    GEMM_LOAD_A(a, iy, bias);
    GEMM_SET_C_BIAS_A(a, c);
#elif defined(USE_BIAS_MATCH_B)
    GEMM_LOAD_B(b, ix, bias);
    GEMM_SET_C_BIAS_B(b, c);
#else
    GEMM_SET_C_ZERO(c);
#endif
    for (int i = 0; i < K; ++i) {
        GEMM_LOAD_A(a, a_off, A);
        GEMM_LOAD_B(b, b_off, B);
        GEMM_CALCORE(a, b, c);
        a_off += M;
        b_off += N;
    }
    STORE_C();
}
