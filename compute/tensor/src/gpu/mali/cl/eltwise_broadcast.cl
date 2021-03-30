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
#define MANGLE_NAME_IMPL(base, AM, EM, FM, AXIS_NAME) base##AM##EM##FM##AXIS_NAME
#define MANGLE_NAME(base, AM, EM, FM, AXIS_NAME) MANGLE_NAME_IMPL(base, AM, EM, FM, AXIS_NAME)

#define FM
#define AXIS_NAME common
#if defined(USE_NCHW)
#if defined(AXIS_W1)
#define AXIS_NAME axis_w1
#endif
#define FM nchw_
#define LOAD_INPUT(v)                                                              \
    {                                                                              \
        int in_off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off; \
        if (iew == 4) {                                                            \
            v = vload4(0, in + in_off);                                            \
        } else {                                                                   \
            if (iew == 3) {                                                        \
                v.xyz = vload3(0, in + in_off);                                    \
            } else if (iew == 2) {                                                 \
                v.xy = vload2(0, in + in_off);                                     \
            } else if (iew == 1) {                                                 \
                v.x = in[in_off];                                                  \
            }                                                                      \
        }                                                                          \
    }

#define STORE_OUTPUT(v)                                                             \
    {                                                                               \
        int out_off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off; \
        if (iew == 4) {                                                             \
            vstore4(v, 0, out + out_off);                                           \
        } else {                                                                    \
            if (iew == 3) {                                                         \
                vstore3(v.xyz, 0, out + out_off);                                   \
            } else if (iew == 2) {                                                  \
                vstore2(v.xy, 0, out + out_off);                                    \
            } else if (iew == 1) {                                                  \
                out[out_off] = v.x;                                                 \
            }                                                                       \
        }                                                                           \
    }
#else
#if defined(AXIS_Z1)
#define AXIS_NAME axis_z1
#endif
#define LOAD_INPUT(v)                                                       \
    {                                                                       \
        int in_off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off; \
        v = vload4(in_off, in);                                             \
    }
#define STORE_OUTPUT(v)                                                      \
    {                                                                        \
        int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off; \
        vstore4(v, out_off, out);                                            \
    }
#endif

__kernel void MANGLE_NAME(eltwise_broadcast_, AM, EM, FM, AXIS_NAME)(const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int bh_str,
    const int bw_str,
    const int bh_off,
    const int bw_off,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const int iw,
    const int bh,
    const int bw,
    const int bc,
    const int bx,
    const int by,
    __global const T *in,
    __global const T *broad,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
#if defined(USE_NCHW)
    char iew = ((idx << 2) + 4 <= iw) ? 4 : (iw & 3);
#endif
    T4 val = 0;
    T4 res;
    LOAD_INPUT(res);

#if defined(USE_NCHW)
    const int bidy = (idy < bh) ? idy : bh - 1;
    const int bidz = (idz < bc) ? idz : bc - 1;
    const int b_off = (bidz * bh_str + bidy + bh_off) * bw_str + bw_off;
#if defined(AXIS_W1)
    val.x = broad[b_off];
    val.y = val.x;
    val.z = val.x;
    val.w = val.x;
#else
    char ew = ((idx << 2) + 4 <= bw) ? 4 : (bw & 3);
    if ((idx << 2) >= bw) {
        ew = 1;
    }
    if (ew == 4) {
        val = vload4(idx, broad + b_off);
    } else {
        if (ew == 3) {
            val.xyz = vload3(0, broad + b_off + (idx << 2));
            val.w = val.z;
        } else if (ew == 2) {
            val.xy = vload2(0, broad + b_off + (idx << 2));
            val.z = val.y;
            val.w = val.y;
        } else if (ew == 1) {
            val.x = broad[b_off + bw - 1];
            val.y = val.x;
            val.z = val.x;
            val.w = val.x;
        }
    }
#endif
#else
    const int bidx = (idx < bh) ? idx : bh - 1;
    const int bidy = (idy < bw) ? idy : bw - 1;
    int b_off = (bidy + bw_off) * bh_str + bidx + bh_off;
#if defined(AXIS_Z1)
    val.x = broad[b_off * 4];
    val.y = val.x;
    val.z = val.x;
    val.w = val.x;
#else
    char ec = ((idz << 2) + 4 <= bc) ? 4 : (bc & 3);
    if ((idz << 2) >= bc) {
        ec = 1;
    }
    if (ec == 4) {
        val = vload4(idz * bw_str * bh_str + b_off, broad);
    } else {
        b_off = b_off << 2;
        int b_str = bw_str * bh_str * 4;
        if (ec == 3) {
            val.xyz = vload3(0, broad + b_off + idz * b_str);
            val.w = val.z;
        } else if (ec == 2) {
            val.xy = vload2(0, broad + b_off + idz * b_str);
            val.z = val.y;
            val.w = val.y;
        } else if (ec == 1) {
            int z_off = (bc >> 2) * bw_str * bh_str * 4 + (bc & 3) - 1;
            val.x = broad[b_off + z_off];
            val.y = val.x;
            val.z = val.x;
            val.w = val.x;
        }
    }
#endif
#endif
    ELTWISE_V4(val, res);
    ACTIVATION_V4(res);
    STORE_OUTPUT(res);
}
