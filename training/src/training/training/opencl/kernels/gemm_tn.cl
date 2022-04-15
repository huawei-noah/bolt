R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

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

#if defined(USE_NCWHC4)
#if defined(USE_RELU)
__kernel void MANGLE_NAME(gemm_tn_relu_ncwhc4_, LM, LN)
#elif defined(USE_GELU)
__kernel void MANGLE_NAME(gemm_tn_gelu_ncwhc4_, LM, LN)
#elif defined(USE_ELTWISE_NCHW)
__kernel void MANGLE_NAME(gemm_tn_eltwise1_ncwhc4_, LM, LN)
#elif defined(USE_ELTWISE_NCWHC4)
__kernel void MANGLE_NAME(gemm_tn_eltwise4_ncwhc4_, LM, LN)
#else
__kernel void MANGLE_NAME(gemm_tn_ncwhc4_, LM, LN)
#endif
    (const int M,
        const int N,
        const int K,
        const int oh,
        const int ow,
        const int oc,
        const int oh_str,
        const int ow_str,
        const int ohw_str,
        const int oh_off,
        const int ow_off,
        const int bx,
        const int by,
        __global const T *A,
        __global const T *B,
        __global const T *bias,
        __global T *C
#if defined(USE_ELTWISE_NCHW)
        ,
        const int ew_str,
        const int ew_off,
        const int eh_off,
        __global const T *eltVal
#endif
#if defined(USE_ELTWISE_NCWHC4)
        ,
        const int eh_str,
        const int ew_str,
        const int ehw_str,
        const int eh_off,
        const int ew_off,
        __global const T *eltVal
#endif
    )
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
    int a_off = iy;
    int b_off = ix;
    GEMM_LOAD_A(a, iy, bias);
    GEMM_SET_C_BIAS(a, c);
#if defined(USE_ELTWISE_NCHW)
    int c_off = (iy + eh_off) * ew_str + ix + ew_off;
    ADD_ELTWISE_NCHW(c, c_off, ew_str, eltVal);
#endif

    for (int i = 0; i < K; ++i) {
        GEMM_LOAD_A(a, a_off, A);
        GEMM_LOAD_B(b, b_off, B);
        GEMM_CALCORE(a, b, c);
        a_off += M;
        b_off += N;
    }

    /*LM = 4 or LM = 8*/
    int c_base = (iy >> 2) * ohw_str;
#if defined(USE_ELTWISE_NCWHC4)
    int e_base = (iy >> 2) * ehw_str;
#endif
    for (uchar i = 0; i < LN; ++i) {
        int oxh = (ix + i) % oh;
        int oxw = (ix + i) / oh;
        if (oxw >= ow) {
            break;
        }
        int c_off = c_base + (oxw + ow_off) * oh_str + oxh + oh_off;
        T4 tmp;
#if defined(USE_ELTWISE_NCWHC4)
        int e_off = e_base + (oxw + ew_off) * eh_str + oxh + eh_off;
        tmp = vload4(e_off, eltVal);
        tmp.x += c[0][0];
        tmp.y += c[1][0];
        tmp.z += c[2][0];
        tmp.w += c[3][0];
#else
        tmp.x = c[0][0];
        tmp.y = c[1][0];
        tmp.z = c[2][0];
        tmp.w = c[3][0];
        ACTIVATION_V4(tmp);
#endif
        vstore4(tmp, c_off, C);
        UPDATE_REG(c[0]);
        UPDATE_REG(c[1]);
        UPDATE_REG(c[2]);
        UPDATE_REG(c[3]);
#if (LM == 8)
        if (iy + 4 >= oc) {
            continue;
        }
        c_off += ohw_str;
#if defined(USE_ELTWISE_NCWHC4)
        e_off += ohw_str;
        tmp = vload4(e_off, eltVal);
        tmp.x += c[4][0];
        tmp.y += c[5][0];
        tmp.z += c[6][0];
        tmp.w += c[7][0];
#else
        tmp.x = c[4][0];
        tmp.y = c[5][0];
        tmp.z = c[6][0];
        tmp.w = c[7][0];
        ACTIVATION_V4(tmp);
#endif
        vstore4(tmp, c_off, C);
        UPDATE_REG(c[4]);
        UPDATE_REG(c[5]);
        UPDATE_REG(c[6]);
        UPDATE_REG(c[7]);
#endif
    }
}

#elif defined(NO_BIAS)
__kernel void MANGLE_NAME(gemm_tn_nobias_, LM, LN)(const int M,
    const int N,
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
    float alp,
    float bet,
    __global const T *A,
    __global const T *B,
    __global T *C,
	const int C_off)
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
    int a_off = iy + A_off;
    int b_off = ix + B_off;
    a_off += idz * A_str;
    b_off += idz * B_str;
    GEMM_SET_C_ZERO(c);

    for (int i = 0; i < K; ++i) {
        GEMM_LOAD_A(a, a_off, A);
        GEMM_LOAD_B(b, b_off, B);
        GEMM_CALCORE(a, b, c);
        a_off += M;
        b_off += N;
    }
	
    int c_off = iy * ow_str + ix + C_off;
    c_off += idz * C_str;
    GEMM_MUL_C(alp, 0, c);
    char ex = (ix + LN <= ow) ? LN : (ow % LN);
    char ey = (iy + LM <= oh) ? LM : (oh % LM);

	GEMM_ADD_STORE_C(c, c_off, ow_str, ex, ey, bet, C);
}

#else
#if defined(USE_RELU)
__kernel void MANGLE_NAME(gemm_tn_relu_, LM, LN)
#elif defined(USE_GELU)
__kernel void MANGLE_NAME(gemm_tn_gelu_, LM, LN)
#else
__kernel void MANGLE_NAME(gemm_tn_, LM, LN)
#endif
    (const int M,
        const int N,
        const int K,
        const int ow_str,
        const int ow,
        const int oh,
        const int bx,
        const int by,
        __global const T *A,
        __global const T *B,
        __global const T *bias,
        __global T *C)
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
    int a_off = iy;
    int b_off = ix;
    GEMM_LOAD_A(a, iy, bias);
    GEMM_SET_C_BIAS(a, c);

    for (int i = 0; i < K; ++i) {
        GEMM_LOAD_A(a, a_off, A);
        GEMM_LOAD_B(b, b_off, B);
        GEMM_CALCORE(a, b, c);
        a_off += M;
        b_off += N;
    }

    int c_off = iy * ow_str + ix;
    char ex = (ix + LN <= ow) ? LN : (ow % LN);
    char ey = (iy + LM <= oh) ? LM : (oh % LM);
    GEMM_STORE_C(c, c_off, ow_str, ex, ey, C);
}
#endif
)"