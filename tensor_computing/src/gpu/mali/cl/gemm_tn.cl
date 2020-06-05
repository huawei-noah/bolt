// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





#include"kernel_def.h"
#define MANGLE_NAME_LMPL(base, LM, LN) base ## LM ## LN
#define MANGLE_NAME(base, LM, LN) MANGLE_NAME_LMPL(base, LM, LN)

#if defined(USE_NCWHC4)
#if defined(USE_RELU)
__kernel void MANGLE_NAME(gemm_tn_relu_ncwhc4_, LM, LN) 
#else
__kernel void MANGLE_NAME(gemm_tn_ncwhc4_, LM, LN) 
#endif
#else
#if defined(USE_RELU)
__kernel void MANGLE_NAME(gemm_tn_relu_, LM, LN)
#elif defined(NO_BIAS)
__kernel void MANGLE_NAME(gemm_tn_nobias_, LM, LN)
#else
__kernel void MANGLE_NAME(gemm_tn_, LM, LN) 
#endif
#endif
#if defined(USE_NCWHC4)
(const int M, const int N, const int K, const int oh, const int ow, const int oc, const int oh_str, const int ow_str, const int ohw_str, 
    const int oh_off, const int ow_off, const int bx, const int by, __global const T* A, __global const T* B, __global const T* bias, __global T* C)
#else
#if defined(NO_BIAS)
(const int M, const int N, const int K, const int ow_str, const int A_str, const int B_str, const int C_str, const int bx, const int by, __global const T* A, __global const T* B, __global T* C)
#else
(const int M, const int N, const int K, const int bx, const int by, __global const T* A, __global const T* B, __global const T* bias, __global T* C)
#endif
#endif
{
    const int idx = get_global_id(0);    
    const int idy = get_global_id(1);
    if(idx >= bx || idy >= by) return;
    const int ix  = idx * LN;
    const int iy  = idy * LM;

    T a[LM];
    T b[LN];
    T c[LM][LN];
    int a_off = iy;
    int b_off = ix;
#if defined(NO_BIAS)
    const int idz = get_global_id(2);
    a_off += idz * A_str;
    b_off += idz * B_str;
    GEMM_SET_C_ZERO(c); 
#else
    GEMM_LOAD_A(a, iy, bias);
    GEMM_SET_C_BIAS(a, c);
#endif    

    for(int i = 0; i < K; ++i) {
        GEMM_LOAD_A(a, a_off, A);
        GEMM_LOAD_B(b, b_off, B);
        GEMM_CALCORE(a, b, c);
        a_off += M;
        b_off += N;
    }

#if defined(USE_NCWHC4)
    /*LM = 4 or LM = 8*/
    int c_base = (iy >> 2) * ohw_str;
    for(uchar i = 0; i < LN; ++i) {
        int oxh = (ix + i) % oh;
        int oxw = (ix + i) / oh;
        if(oxw >= ow) break;
        int c_off = c_base + (oxw + ow_off) * oh_str + oxh + oh_off;
        T4 tmp;
        tmp.x = c[0][0];
        tmp.y = c[1][0];
        tmp.z = c[2][0];
        tmp.w = c[3][0];
        ACTIVATION_V4(tmp);
        vstore4(tmp, c_off, C);
        UPDATE_REG(c[0]);
        UPDATE_REG(c[1]);
        UPDATE_REG(c[2]);
        UPDATE_REG(c[3]);
#if (LM == 8)
        if(iy + 4 >= oc) continue;
        c_off += ohw_str;
        tmp.x = c[4][0];
        tmp.y = c[5][0];
        tmp.z = c[6][0];
        tmp.w = c[7][0];
        ACTIVATION_V4(tmp);
        vstore4(tmp, c_off, C);
        UPDATE_REG(c[4]);
        UPDATE_REG(c[5]);
        UPDATE_REG(c[6]);
        UPDATE_REG(c[7]);
#endif
    }
#else
    int c_off = iy * ow_str + ix;
#if defined(NO_BIAS)
    c_off += idz * C_str;
#endif
    GEMM_STORE_C(c, c_off, ow_str, C);
#endif    
}
