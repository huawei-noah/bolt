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
#define MANGLE_NAME_IMPL(base, IOM, FM, BM, AXIS, ALPHA, BETA) base##IOM##FM##BM##AXIS##ALPHA##BETA
#define MANGLE_NAME(base, IOM, FM, BM, AXIS, ALPHA, BETA) \
    MANGLE_NAME_IMPL(base, IOM, FM, BM, AXIS, ALPHA, BETA)

#define FM
#define BM
#define AXIS
#define ALPHA
#define BETA

#if defined(USE_NCHW)
#define FM nchw_
#endif

#if defined(USE_BROADCAST_MODE)
#define BM broad_
#endif

#if defined(SCALE_ON_AXIS_W)
#define AXIS w_
#elif defined(SCALE_ON_AXIS_H)
#define AXIS h_
#elif defined(SCALE_ON_AXIS_C)
#define AXIS c_
#endif

#if defined(USE_ALPHA)
#define ALPHA alpha_
#endif

#if defined(USE_BETA)
#define BETA beta_
#endif

#define LOAD_ALPHA
#define LOAD_BETA

#if defined(USE_NCHW)

#if !defined(USE_BROADCAST_MODE)
#define LOAD_INPUT                                                                   \
    {                                                                                \
        LOAD_MEM_V4_C1_COMMON(val, idx, idy, idz, iw_str, ih_str, i_off, iw, input); \
    }
#endif

#if defined(SCALE_ON_AXIS_W)

#if defined(USE_ALPHA)
#define LOAD_ALPHA alp = vload4(idx, alpha)
#endif

#if defined(USE_BETA)
#define LOAD_BETA bet = vload4(idx, beta)
#endif

#if defined(USE_BROADCAST_MODE)
#define LOAD_INPUT                                                                 \
    {                                                                              \
        LOAD_MEM_V4_C1_COMMON(val, 0, idy, idz, iw_str, ih_str, i_off, iw, input); \
        val.y = val.x;                                                             \
        val.z = val.x;                                                             \
        val.w = val.x;                                                             \
    }
#endif

#elif defined(SCALE_ON_AXIS_H)

#if defined(USE_ALPHA)
#define LOAD_ALPHA alp = alpha[idy]
#endif

#if defined(USE_BETA)
#define LOAD_BETA bet = beta[idy]
#endif

#if defined(USE_BROADCAST_MODE)
#define LOAD_INPUT                                                                 \
    {                                                                              \
        LOAD_MEM_V4_C1_COMMON(val, idx, 0, idz, iw_str, ih_str, i_off, iw, input); \
    }
#endif

#elif defined(SCALE_ON_AXIS_C)

#if defined(USE_ALPHA)
#define LOAD_ALPHA alp = alpha[idc]
#endif

#if defined(USE_BETA)
#define LOAD_BETA bet = beta[idc]
#endif

#if defined(USE_BROADCAST_MODE)
#define LOAD_INPUT                                                                   \
    {                                                                                \
        LOAD_MEM_V4_C1_COMMON(val, idx, idy, idn, iw_str, ih_str, i_off, iw, input); \
    }
#endif
#endif

#else
#if !defined(USE_BROADCAST_MODE)
#define LOAD_INPUT                                                            \
    {                                                                         \
        LOAD_MEM_V4_COMMON(val, idx, idy, idz, iw_str, ih_str, i_off, input); \
    }
#endif

#if defined(SCALE_ON_AXIS_W)

#if defined(USE_ALPHA)
#define LOAD_ALPHA alp = alpha[idx]
#endif

#if defined(USE_BETA)
#define LOAD_BETA bet = beta[idx]
#endif

#if defined(USE_BROADCAST_MODE)
#define LOAD_INPUT                                                          \
    {                                                                       \
        LOAD_MEM_V4_COMMON(val, 0, idy, idz, iw_str, ih_str, i_off, input); \
    }
#endif

#elif defined(SCALE_ON_AXIS_H)

#if defined(USE_ALPHA)
#define LOAD_ALPHA alp = alpha[idy]
#endif

#if defined(USE_BETA)
#define LOAD_BETA bet = beta[idy]
#endif

#if defined(USE_BROADCAST_MODE)
#define LOAD_INPUT                                                          \
    {                                                                       \
        LOAD_MEM_V4_COMMON(val, idx, 0, idz, iw_str, ih_str, i_off, input); \
    }
#endif

#elif defined(SCALE_ON_AXIS_C)

#if defined(USE_ALPHA)
#define LOAD_ALPHA alp = vload4(idc, alpha)
#endif

#if defined(USE_BETA)
#define LOAD_BETA bet = vload4(idc, beta)
#endif

#if defined(USE_BROADCAST_MODE)
#define LOAD_INPUT                                                            \
    {                                                                         \
        LOAD_MEM_V4_COMMON(val, idx, idy, idn, iw_str, ih_str, i_off, input); \
        val.y = val.x;                                                        \
        val.z = val.x;                                                        \
        val.w = val.x;                                                        \
    }
#endif
#endif
#endif

#if defined(USE_NCHW)
#if defined(SCALE_ON_AXIS_W)
#define USE_V4
#endif
#else
#if defined(SCALE_ON_AXIS_C)
#define USE_V4
#endif
#endif

__kernel void MANGLE_NAME(scale_, IOM, FM, BM, AXIS, ALPHA, BETA)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int iw,
    const int ih,
    const int ic,
    const int ow,
    const int oh,
    const int oc,
    const int bx,
    const int by,
    __global const T *alpha,
    __global const T *beta,
    READ_ONLY_KERNEL_MEM input,
    KERNEL_MEM output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
#if !defined(USE_NCHW)
    int idc = idz % ((oc + 3) >> 2);
    int idn = idz / ((oc + 3) >> 2);
#else
    int idc = idz % oc;
    int idn = idz / oc;
#endif
    if (idx >= bx || idy >= by) {
        return;
    }

    T4 val;
#if defined(USE_V4)
    T4 alp = (T4)1.0;
    T4 bet = (T4)0.0;
#else
    T alp = 1.0;
    T bet = 0.0;
#endif

    LOAD_ALPHA;
    LOAD_BETA;
    LOAD_INPUT;

#if defined(USE_V4)
    val.s0 = val.s0 * alp.x + bet.x;
    val.s1 = val.s1 * alp.y + bet.y;
    val.s2 = val.s2 * alp.z + bet.z;
    val.s3 = val.s3 * alp.w + bet.w;
#else
    val.s0 = val.s0 * alp + bet;
    val.s1 = val.s1 * alp + bet;
    val.s2 = val.s2 * alp + bet;
    val.s3 = val.s3 * alp + bet;
#endif

#if defined(USE_NCHW)
    STORE_MEM_V4_C1_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, ow, output);
#else
    STORE_MEM_V4_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, output);
#endif
}
