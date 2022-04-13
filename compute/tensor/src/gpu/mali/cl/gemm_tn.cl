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
#define MANGLE_NAME_LMPL(base, IAM, IBM, OM, AM, FM, BM, LM, LN) \
    base##IAM##IBM##OM##AM##FM##BM##LM##LN
#define MANGLE_NAME(base, IAM, IBM, OM, AM, FM, BM, LM, LN) \
    MANGLE_NAME_LMPL(base, IAM, IBM, OM, AM, FM, BM, LM, LN)

#define BM
#define FM

#if defined(USE_POINTWISE_NCHWC4)
#define FM pointwise_nchwc4_
#endif

#if defined(NO_BIAS)
#define BM nobias_
#elif defined(USE_BIAS_MATCH_A)
#define BM biasA_
#elif defined(USE_BIAS_MATCH_B)
#define BM biasB_
#endif

#if defined(USE_INPUT_A_IMG)
#define IAM am_
#define READ_ONLY_A_MEM __read_only image3d_t
#else
#define IAM
#define READ_ONLY_A_MEM __global const T *
#endif

#if defined(USE_INPUT_B_IMG)
#define IBM bm_
#define READ_ONLY_B_MEM __read_only image3d_t
#else
#define IBM
#define READ_ONLY_B_MEM __global const T *
#endif

#if defined(USE_OUTPUT_IMG)
#define OM cm_
#else
#define OM
#endif

#if defined(USE_POINTWISE_NCHWC4)
#define STORE_REG_V4(c0, c1, c2, c3, off, mem)     \
    {                                              \
        T4 tmp = (T4)(c0[0], c1[0], c2[0], c3[0]); \
        ACTIVATION_V4(tmp);                        \
        STORE_MEM_V4(tmp, off, mem);               \
        UPDATE_REG(c0);                            \
        UPDATE_REG(c1);                            \
        UPDATE_REG(c2);                            \
        UPDATE_REG(c3);                            \
    }
#if (LM == 4)
#define STORE_REG(c, off, mem)                          \
    {                                                   \
        STORE_REG_V4(c[0], c[1], c[2], c[3], off, mem); \
    }
#elif (LM == 8)
#if defined(USE_OUTPUT_IMG)
#define ADD_C_OFF(off) \
    {                  \
        off.z += 1;    \
    }
#else
#define ADD_C_OFF(off) \
    {                  \
        off += C_str;  \
    }
#endif
#define STORE_REG(c, off, mem)                              \
    {                                                       \
        STORE_REG_V4(c[0], c[1], c[2], c[3], off, mem);     \
        if (iy + 4 < oc) {                                  \
            ADD_C_OFF(off);                                 \
            STORE_REG_V4(c[4], c[5], c[6], c[7], off, mem); \
        }                                                   \
    }
#endif
#if defined(USE_OUTPUT_IMG)
#define STORE_C()                                     \
    {                                                 \
        int z_off = idz * by * (LM >> 2) + (iy >> 2); \
        for (uchar i = 0; i < LN; ++i) {              \
            int oxw = (ix + i) % ow;                  \
            int oxh = (ix + i) / ow;                  \
            if (oxh >= oh) {                          \
                break;                                \
            }                                         \
            int4 c_off = (int4)(oxw, oxh, z_off, 0);  \
            STORE_REG(c, c_off, C);                   \
        }                                             \
    }
#else
#define STORE_C()                                                        \
    {                                                                    \
        int c_base = (idz * by * (LM >> 2) + (iy >> 2)) * C_str + C_off; \
        for (uchar i = 0; i < LN; ++i) {                                 \
            int oxw = (ix + i) % ow;                                     \
            int oxh = (ix + i) / ow;                                     \
            if (oxh >= oh) {                                             \
                break;                                                   \
            }                                                            \
            int c_off = c_base + oxh * ow_str + oxw;                     \
            STORE_REG(c, c_off, C);                                      \
        }                                                                \
    }
#endif
#else
#if defined(USE_OUTPUT_IMG)
#define STORE_C()                                   \
    {                                               \
        char ey = (iy + LM <= oh) ? LM : (oh % LM); \
        int4 c_off = (int4)(ix >> 2, iy, idz, 0);   \
        GEMM_STORE_C(c, c_off, 0, 0, ey, C);        \
    }
#else
#define STORE_C()                                           \
    {                                                       \
        char ex = (ix + LN <= ow) ? LN : (ow % LN);         \
        char ey = (iy + LM <= oh) ? LM : (oh % LM);         \
        int c_off = idz * C_str + iy * ow_str + ix + C_off; \
        GEMM_STORE_C(c, c_off, ow_str, ex, ey, C);          \
    }
#endif
#endif

__kernel void MANGLE_NAME(gemm_tn_, IAM, IBM, OM, AM, FM, BM, LM, LN)(const int M,
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
    READ_ONLY_A_MEM A,
    READ_ONLY_B_MEM B,
    __global const T *bias,
    KERNEL_MEM C)
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
#if defined(USE_INPUT_A_IMG)
    int4 a_off = (int4)(iy >> 2, 0, idz, 0);
#else
    int a_off = idz * A_str + iy + A_off;
#endif

#if defined(USE_INPUT_B_IMG)
    int4 b_off = (int4)(ix >> 2, 0, idz, 0);
#else
    int b_off = idz * B_str + ix + B_off;
#endif

#if defined(USE_BIAS_MATCH_A)
    GEMM_LOAD_BIAS_MATCH_A(a, iy, bias);
    GEMM_SET_C_BIAS_A(a, c);
#elif defined(USE_BIAS_MATCH_B)
    GEMM_LOAD_BIAS_MATCH_B(b, ix, bias);
    GEMM_SET_C_BIAS_B(b, c);
#else
    GEMM_SET_C_ZERO(c);
#endif
    for (int i = 0; i < K; ++i) {
        GEMM_LOAD_A(a, a_off, A);
        GEMM_LOAD_B(b, b_off, B);
        GEMM_CALCORE(a, b, c);
#if defined(USE_INPUT_A_IMG)
        a_off.y += 1;
#else
        a_off += M;
#endif
#if defined(USE_INPUT_B_IMG)
        b_off.y += 1;
#else
        b_off += N;
#endif
    }
    STORE_C();
}
