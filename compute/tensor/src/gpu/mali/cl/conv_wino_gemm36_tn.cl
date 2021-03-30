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
#define MANGLE_NAME_LMPL(base, LM, LN) base##LM##LN
#define MANGLE_NAME(base, LM, LN) MANGLE_NAME_LMPL(base, LM, LN)

__kernel void MANGLE_NAME(conv_wino_gemm36_tn_, LM, LN)(int M,
    int N,
    int K,
    int a_str,
    int b_str,
    int c_str,
    const int bx,
    const int by,
    __global const T *A,
    __global const T *B,
    global T *C)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    const int ix = idx * LN;
    const int iy = idy * LM;

    T a[LM];
    T b[LN];
    T c[LM][LN];
    GEMM_SET_C_ZERO(c);

    int a_off = iy + a_str;
    int b_off = ix + b_str;
    for (int i = 0; i < K; i++) {
        GEMM_LOAD_A(a, a_off, A);
        GEMM_LOAD_B(b, b_off, B);
        GEMM_CALCORE(a, b, c);
        a_off += M;
        b_off += N;
    }

    int c_off = iy * N + ix + c_str;
    GEMM_MUL_C((float)(0.1111111111), 0, c);
    GEMM_STORE_C(c, c_off, N, LN, LM, C);
}
