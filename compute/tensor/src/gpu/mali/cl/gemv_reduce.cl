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

#define MANGLE_NAME_IMPL(base, AM, OFM, BM, OC) base##AM##OFM##BM##OC
#define MANGLE_NAME(base, AM, OFM, BM, OC) MANGLE_NAME_IMPL(base, AM, OFM, BM, OC)

#define OFM
#define BM
#if defined(NO_BIAS)
#define BM nobias_
#endif
#if defined(USE_OUTPUT_NCHWC4)
#define OFM oc4_
#endif

#define CAL_EDGE(ov, rc, v_off, mv, vec)    \
    {                                       \
        if (rc >= 4) {                      \
            T4 vv = vload4(0, vec + v_off); \
            ov += vv.x * mv.x;              \
            ov += vv.y * mv.y;              \
            ov += vv.z * mv.z;              \
            ov += vv.w * mv.w;              \
        } else if (rc == 1) {               \
            T vv = vec[v_off];              \
            ov += vv * mv.x;                \
        } else if (rc == 2) {               \
            T2 vv = vload2(0, vec + v_off); \
            ov += vv.x * mv.x;              \
            ov += vv.y * mv.y;              \
        } else if (rc == 3) {               \
            T3 vv = vload3(0, vec + v_off); \
            ov += vv.x * mv.x;              \
            ov += vv.y * mv.y;              \
            ov += vv.z * mv.z;              \
        }                                   \
    }

#if (OC == 8)
#define GET_LOOP_INFO(col, bc, rc) \
    {                              \
        bc = col >> 3;             \
        rc = col & 7;              \
    }
#define CALCORE_EDGE(ov, bc, rc, v_off, m_off, vec, mat)    \
    {                                                       \
        if (rc > 0) {                                       \
            T8 mv = vload8(bc, mat + m_off);                \
            v_off += (bc << 3);                             \
            CAL_EDGE(ov, rc, v_off, mv, vec);               \
            CAL_EDGE(ov, rc - 4, v_off + 4, mv.s4567, vec); \
        }                                                   \
    }
#endif
__kernel void MANGLE_NAME(gemv_reduce_, AM, OFM, BM, OC)(const int row,
    const int col,
    const int ow_str,
    const int oh_str,
    const int on_str,
    const int o_off,
    const int bx,
    const int by,
    __global const T *vec,
    __global const T *mat,
    __global const T *bias,
    __global T *tmp,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);  //batch
    if (idx >= bx || idy >= by) {
        return;
    }
    T8 vv;
    T8 mv;
    T res = 0;
    int mat_off = idy * ((col + 7) >> 3) << 3;
    int vec_off = idz * col;
    int bc;
    char rc;
    GET_LOOP_INFO(col, bc, rc);
    for (int i = idx; i < bc; i += 32) {
        mv = vload8(i, mat + mat_off);
        vv = vload8(i, vec + vec_off);
        res += mv.s0 * vv.s0 + mv.s1 * vv.s1 + mv.s2 * vv.s2 + mv.s3 * vv.s3;
        res += mv.s4 * vv.s4 + mv.s5 * vv.s5 + mv.s6 * vv.s6 + mv.s7 * vv.s7;
    }
    if (idx == 0) {
        CALCORE_EDGE(res, bc, rc, vec_off, mat_off, vec, mat);
    }
    tmp[(idy << 5) + idx] = res;
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (idx == 0) {
#if defined(NO_BIAS)
        res = 0;
#else
        res = bias[idy];
#endif
        for (uchar i = 0; i < 2; i++) {
            T16 tv = vload16(i, tmp + (idy << 5));
            res += tv.s0 + tv.s1 + tv.s2 + tv.s3;
            res += tv.s4 + tv.s5 + tv.s6 + tv.s7;
            res += tv.s8 + tv.s9 + tv.sa + tv.sb;
            res += tv.sc + tv.sd + tv.se + tv.sf;
        }
        ACTIVATION_V1(res);
#if defined(USE_OUTPUT_NCHWC4)
        const int idc = idy >> 2;
        const int lane = idy & 3;
        int out_off = idc * ow_str * oh_str + o_off;
        out[out_off * 4 + idz * on_str + lane] = res;
#else
        out[idz * on_str + idy + o_off] = res;
#endif
    }
}
