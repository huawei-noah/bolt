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

#if (LM == 4)
#define LMV 1
#elif (LM == 8)
#define LMV 2
#endif

#if (LN == 4)
#define LNV 1
#elif (LN == 8)
#define LNV 2
#endif

#if defined(USE_INPUT_A_IMG)
#if (LM == 4)
#define GEMM_LOAD_A(a, a_off, A) {\
    a[0] = READ_IMAGE(A, sampler, a_off);\
}
#elif (LM == 8)
#define GEMM_LOAD_A(a, a_off, A) {\
    a[0] = READ_IMAGE(A, sampler, a_off);\
    a[1] = READ_IMAGE(A, sampler, (int4)(a_off.x + 1, a_off.y, a_off.z, a_off.w));\
}
#endif
#else
#if (LM == 4)
#define GEMM_LOAD_A(a, a_off, A) {\
    a[0] = vload4(0, A + a_off);\
}
#elif (LM == 8)
#define GEMM_LOAD_A(a, a_off, A) {\
    T8 tv = vload8(0, A + a_off);\
    a[0].x = tv.s0;\
    a[0].y = tv.s1;\
    a[0].z = tv.s2;\
    a[0].w = tv.s3;\
    a[1].x = tv.s4;\
    a[1].y = tv.s5;\
    a[1].z = tv.s6;\
    a[1].w = tv.s7;\
}
#endif
#endif

#if defined(USE_INPUT_B_IMG)
#if (LN == 4)
#define GEMM_LOAD_B(b, b_off, B) {\
    b[0] = READ_IMAGE(B, sampler, b_off);\
}
#elif (LN == 8)
#define GEMM_LOAD_B(b, b_off, B) {\
    b[0] = READ_IMAGE(B, sampler, b_off);\
    b[1] = READ_IMAGE(B, sampler, (int4)(b_off.x + 1, b_off.y, b_off.z, b_off.w));\
}
#endif
#else
#if (LM == 4)
#define GEMM_LOAD_B(b, b_off, B) {\
    b[0] = vload4(0, B + b_off);\
}
#elif (LM == 8)
#define GEMM_LOAD_B(b, b_off, B) {\
    T8 tv = vload8(0, B + b_off);\
    b[0].x = tv.s0;\
    b[0].y = tv.s1;\
    b[0].z = tv.s2;\
    b[0].w = tv.s3;\
    b[1].x = tv.s4;\
    b[1].y = tv.s5;\
    b[1].z = tv.s6;\
    b[1].w = tv.s7;\
}
#endif
#endif

#if (LN == 4 && LM == 4)
#define GEMM_CALCORE(a, b, c) {\
    c[0][0] += a[0].x * b[0];\
    c[1][0] += a[0].y * b[0];\
    c[2][0] += a[0].z * b[0];\
    c[3][0] += a[0].w * b[0];\
}
#elif (LN == 4 && LM == 8)
#define GEMM_CALCORE(a, b, c) {\
    c[0][0] += a[0].x * b[0];\
    c[1][0] += a[0].y * b[0];\
    c[2][0] += a[0].z * b[0];\
    c[3][0] += a[0].w * b[0];\
    c[4][0] += a[1].x * b[0];\
    c[5][0] += a[1].y * b[0];\
    c[6][0] += a[1].z * b[0];\
    c[7][0] += a[1].w * b[0];\
}
#elif (LN == 8 && LM == 4)
#define GEMM_CALCORE(a, b, c) {\
    c[0][0] += a[0].x * b[0];\
    c[1][0] += a[0].y * b[0];\
    c[2][0] += a[0].z * b[0];\
    c[3][0] += a[0].w * b[0];\
    c[0][1] += a[0].x * b[1];\
    c[1][1] += a[0].y * b[1];\
    c[2][1] += a[0].z * b[1];\
    c[3][1] += a[0].w * b[1];\
}
#elif (LN == 8 && LM == 8)
#define GEMM_CALCORE(a, b, c) {\
    c[0][0] += a[0].x * b[0];\
    c[1][0] += a[0].y * b[0];\
    c[2][0] += a[0].z * b[0];\
    c[3][0] += a[0].w * b[0];\
    c[4][0] += a[1].x * b[0];\
    c[5][0] += a[1].y * b[0];\
    c[6][0] += a[1].z * b[0];\
    c[7][0] += a[1].w * b[0];\
    c[0][1] += a[0].x * b[1];\
    c[1][1] += a[0].y * b[1];\
    c[2][1] += a[0].z * b[1];\
    c[3][1] += a[0].w * b[1];\
    c[4][1] += a[1].x * b[1];\
    c[5][1] += a[1].y * b[1];\
    c[6][1] += a[1].z * b[1];\
    c[7][1] += a[1].w * b[1];\
}
#endif

#if defined(USE_OUTPUT_IMG)
#if (LN == 4)
#define STORE_C() {\
    ix = ix >> 2;\
    for (char i = 0; i < LM; i++) {\
        if (iy + i < oh) {\
            WRITE_IMAGE(C, (int4)(ix, iy + i, idz, 0), c[i][0]);\
        }\
    }\
}
#elif (LN == 8)
#define STORE_C() {\
    ix = ix >> 2;\
    for (char i = 0; i < LM; i++) {\
        if (iy + i < oh) {\
            WRITE_IMAGE(C, (int4)(ix, iy + i, idz, 0), c[i][0]);\
            WRITE_IMAGE(C, (int4)(ix + 1, iy + i, idz, 0), c[i][1]);\
        }\
    }\
}
#endif
#else
#if (LN == 4)
#define STORE_C() {\
    int c_off = idz * C_str + iy * ow_str + ix + C_off;\
    char ex = (ix + 4 <= ow) ? 4 : (ow & 3);\
    for (char i = 0; i < LM; i++) {\
        if (iy + i < oh) {\
            STORE_MEM_V4_C1(c[i][0], c_off, ex, C);\
            c_off += ow_str;\
        }\
    }\
}
#elif (LN == 8)
#define STORE_C() {\
    int c_off = idz * C_str + iy * ow_str + ix + C_off;\
    char ex0 = (ix + 4 <= ow) ? 4 : (ow & 3);\
    char ex1 = (ix + 8 <= ow) ? 4 : (ow & 3);\
    for (char i = 0; i < LM; i++) {\
        if (iy + i < oh) {\
            STORE_MEM_V4_C1(c[i][0], c_off, ex0, C);\
            STORE_MEM_V4_C1(c[i][1], c_off + 4, ex1, C);\
            c_off += ow_str;\
        }\
    }\
}
#endif
#endif

__kernel void MANGLE_NAME(gemm_tn_qc_, IAM, IBM, OM, AM, FM, BM, LM, LN)(const int M,
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

    T4 a[LMV];
    T4 b[LNV];
    T4 c[LM][LNV];
    int ix = idx * LN;
    int iy = idy * LM;
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
    for (char i = 0; i < LMV; i++) {
        a[i] = vload4(i, bias + iy);
    }
    c[0][0] = (T4)a[0].x;
    c[1][0] = (T4)a[0].y;
    c[2][0] = (T4)a[0].z;
    c[3][0] = (T4)a[0].w;
#if (LM == 8)
    c[4][0] = (T4)a[1].x;
    c[5][0] = (T4)a[1].y;
    c[6][0] = (T4)a[1].z;
    c[7][0] = (T4)a[1].w;
#endif
#if (LN == 8)
    for (char i = 0; i < LM; i++) {
        c[i][1] = c[i][0];
    }
#endif
#elif defined(USE_BIAS_MATCH_B)
    for (char i = 0; i < LNV; i++) {
        c[0][i] = vload4(i, bias + ix);
    }
    for (char i = 1; i < LM; i++) {
        c[i][0] = c[0][0];
    }
#if (LN == 8)
    for (char i = 1; i < LM; i++) {
        c[i][1] = c[0][1];
    }
#endif
#else
    for (char i = 0; i < LM; i++) {
        for (char j = 0; j < LN; j++) {
            c[i][j] = 0;
        }
    }
#endif

    for (int i = 0; i < K; i++) {
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
