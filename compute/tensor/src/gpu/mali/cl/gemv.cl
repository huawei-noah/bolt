// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NOCINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTIOC OF COCTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN COCNECTIOC WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

#if (OC == 4)
#define GET_LOOP_INFO(col, bc, rc) \
    {                              \
        bc = col >> 2;             \
        rc = col & 3;              \
    }
#define CALCORE(ov, v_off, m_off, vec, mat) \
    {                                       \
        T4 vv = vload4(v_off, vec);         \
        T4 mv = vload4(m_off, mat);         \
        DOT_A4B4C1(vv, mv, ov);             \
    }
#define CALCORE_EDGE(ov, bc, rc, v_off, m_off, vec, mat) \
    {                                                    \
        if (rc > 0) {                                    \
            T4 mv = vload4(m_off, mat);                  \
            v_off += (bc << 2);                          \
            CAL_EDGE(ov, rc, v_off, mv, vec);            \
        }                                                \
    }
#endif

#if (OC == 8)
#define GET_LOOP_INFO(col, bc, rc) \
    {                              \
        bc = col >> 3;             \
        rc = col & 7;              \
    }
#define CALCORE(ov, v_off, m_off, vec, mat) \
    {                                       \
        T8 vv = vload8(v_off, vec);         \
        T8 mv = vload8(m_off, mat);         \
        DOT_A8B8C1(vv, mv, ov);             \
    }
#define CALCORE_EDGE(ov, bc, rc, v_off, m_off, vec, mat)    \
    {                                                       \
        if (rc > 0) {                                       \
            T8 mv = vload8(m_off, mat);                     \
            v_off += (bc << 3);                             \
            CAL_EDGE(ov, rc, v_off, mv, vec);               \
            CAL_EDGE(ov, rc - 4, v_off + 4, mv.s4567, vec); \
        }                                                   \
    }
#endif

#if (OC == 16)
#define GET_LOOP_INFO(col, bc, rc) \
    {                              \
        bc = col >> 4;             \
        rc = col & 15;             \
    }
#define CALCORE(ov, v_off, m_off, vec, mat) \
    {                                       \
        T16 vv = vload16(v_off, vec);       \
        T16 mv = vload16(m_off, mat);       \
        DOT_A16B16C1(vv, mv, ov);           \
    }
#define CALCORE_EDGE(ov, bc, rc, v_off, m_off, vec, mat)      \
    {                                                         \
        if (rc > 0) {                                         \
            T16 mv = vload16(m_off, mat);                     \
            v_off += (bc << 4);                               \
            CAL_EDGE(ov, rc, v_off, mv, vec);                 \
            CAL_EDGE(ov, rc - 4, v_off + 4, mv.s4567, vec);   \
            CAL_EDGE(ov, rc - 8, v_off + 8, mv.s89ab, vec);   \
            CAL_EDGE(ov, rc - 12, v_off + 12, mv.scdef, vec); \
        }                                                     \
    }
#endif

__kernel void MANGLE_NAME(gemv_, AM, OFM, BM, OC)(const int row,
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
    if (idx >= bx || idy >= by) {
        return;
    }
#if defined(NO_BIAS)
    T out_val = 0;
#else
    T out_val = bias[idx];
#endif

    int vec_off = idy * col;
    int mat_off = idx;
    int bc;
    char rc;
    GET_LOOP_INFO(col, bc, rc);

    for (int i = 0; i < bc; ++i) {
        CALCORE(out_val, i, mat_off, vec + vec_off, mat);
        mat_off += row;
    }
    CALCORE_EDGE(out_val, bc, rc, vec_off, mat_off, vec, mat);
    ACTIVATION_V1(out_val);

#if defined(USE_OUTPUT_NCHWC4)
    const int idc = idx >> 2;
    const int lane = idx & 3;
    int out_off = idc * ow_str * oh_str + o_off;
    out[out_off * 4 + idy * on_str + lane] = out_val;
#else
    int out_off = idy * on_str + idx + o_off;
    out[out_off] = out_val;
#endif
}
