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
#define MANGLE_NAME_LMPL(base, LM, LN, LK) base##LM##LN##LK
#define MANGLE_NAME(base, LM, LN, LK) MANGLE_NAME_LMPL(base, LM, LN, LK)

#if defined(NO_BIAS)
__kernel void MANGLE_NAME(gemm_nt_nobias_, LM, LN, LK)(const int KA,
    const int KB,
    const int K,
    const int ow_str,
    const int A_str,
    const int B_str,
    const int C_str,
    const int A_off,
    const int B_off,
    const int ow,
    const int oh,
    const int bx,
    const int by,
    __global const T *A,
    __global const T *B,
    __global T *C)
#else
__kernel void MANGLE_NAME(gemm_nt_, LM, LN, LK)(const int KA,
    const int KB,
    const int K,
    const int ow_str,
    const int A_str,
    const int B_str,
    const int C_str,
    const int A_off,
    const int B_off,
    const int ow,
    const int oh,
    const int bx,
    const int by,
    __global const T *A,
    __global const T *B,
    __global const T *bias,
    __global T *C)
#endif
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    const int ix = idx * LN;
    const int iy = idy * LM;
    const int L = K >> LK;
    const int VN = 1 << LK;

    T c[LM][LN];
#if (LK == 0)
    T a[LM];
    T b[LN];
#elif (LK == 1)
    T2 a[LM];
    T2 b[LN];
#elif (LK == 2)
    T4 a[LM];
    T4 b[LN];
#elif (LK == 3)
    T8 a[LM];
    T8 b[LN];
#elif (LK == 4)
    T16 a[LM];
    T16 b[LN];
#endif

#if defined(NO_BIAS)
    GEMM_SET_C_ZERO(c);
#else
    GEMM_LOAD_A(a, iy, bias);
    GEMM_SET_C_BIAS_A(a, c);
#endif

    int a_off = iy * KA + idz * A_str + A_off;
    int b_off = ix * KB + idz * B_str + B_off;
    for (int i = 0; i < L; ++i) {
        GEMM_NT_LOAD_A(a, a_off, KA, A);
        GEMM_NT_LOAD_B(b, b_off, KB, B);
        GEMM_CALCORE(a, b, c);
        a_off += VN;
        b_off += VN;
    }
    int c_off = iy * ow_str + ix + idz * C_str;
    char ex = (ix + LN <= ow) ? LN : (ow % LN);
    char ey = (iy + LM <= oh) ? LM : (oh % LM);
    GEMM_STORE_C(c, c_off, ow_str, ex, ey, C);
}
