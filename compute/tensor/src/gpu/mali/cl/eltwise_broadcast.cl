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
#define MANGLE_NAME_IMPL(base, IOM, AM, EM, SI, FM, AXIS_NAME) base##IOM##AM##EM##SI##FM##AXIS_NAME
#define MANGLE_NAME(base, IOM, AM, EM, SI, FM, AXIS_NAME) \
    MANGLE_NAME_IMPL(base, IOM, AM, EM, SI, FM, AXIS_NAME)

#if defined(USE_INPUT_IMG1)
#define READ_ONLY_KERNEL_MEM_1 __read_only image3d_t
#else
#define READ_ONLY_KERNEL_MEM_1 __global const T *
#endif

#define SI
#define FM
#define AXIS_NAME common
#if defined(SWAP_INPUT)
#define SI si_
#endif

#if defined(USE_NCHW)
#define FM nchw_

#if defined(USE_INPUT_IMG1)
#define LOAD_BROAD(val)                                                            \
    {                                                                              \
        char ew = ((idx << 2) + 4 <= bw) ? 4 : (bw & 3);                           \
        if ((idx << 2) >= bw) {                                                    \
            ew = 1;                                                                \
        }                                                                          \
        const int bidx = (ew == 1) ? (bw >> 2) : idx;                              \
        val = READ_IMAGE(broad, sampler, (int4)(bidx, bidy, bidn * bc + bidc, 0)); \
        if (ew == 3) {                                                             \
            val.w = val.z;                                                         \
        } else if (ew == 2) {                                                      \
            val.z = val.y;                                                         \
            val.w = val.y;                                                         \
        } else if (ew == 1) {                                                      \
            val.y = val.x;                                                         \
            val.z = val.x;                                                         \
            val.w = val.x;                                                         \
        }                                                                          \
    }
#else
#define LOAD_BROAD(val)                                                     \
    {                                                                       \
        char ew = ((idx << 2) + 4 <= bw) ? 4 : (bw & 3);                    \
        if ((idx << 2) >= bw) {                                             \
            ew = 1;                                                         \
        }                                                                   \
        int b_off = ((bidn * bc + bidc) * bh_str + bidy) * bw_str + ib_off; \
        if (ew == 4) {                                                      \
            val = vload4(idx, broad + b_off);                               \
        } else {                                                            \
            if (ew == 3) {                                                  \
                val.xyz = vload3(0, broad + b_off + (idx << 2));            \
                val.w = val.z;                                              \
            } else if (ew == 2) {                                           \
                val.xy = vload2(0, broad + b_off + (idx << 2));             \
                val.z = val.y;                                              \
                val.w = val.y;                                              \
            } else if (ew == 1) {                                           \
                val.x = broad[b_off + bw - 1];                              \
                val.y = val.x;                                              \
                val.z = val.x;                                              \
                val.w = val.x;                                              \
            }                                                               \
        }                                                                   \
    }
#endif
#else
#if defined(USE_INPUT_IMG1)
#define LOAD_BROAD(val)                                                              \
    {                                                                                \
        const int bcd4 = (bc + 3) >> 2;                                              \
        const int bwh_str = bw_str * bh_str;                                         \
        char ec = ((idc << 2) + 4 <= bc) ? 4 : (bc & 3);                             \
        if ((idc << 2) >= bc) {                                                      \
            ec = 1;                                                                  \
        }                                                                            \
        const int bidc = (ec == 1) ? (bc >> 2) : idc;                                \
        val = READ_IMAGE(broad, sampler, (int4)(bidx, bidy, bidn * bcd4 + bidc, 0)); \
        if (ec == 3) {                                                               \
            val.w = val.z;                                                           \
        } else if (ec == 2) {                                                        \
            val.z = val.y;                                                           \
            val.w = val.y;                                                           \
        } else if (ec == 1) {                                                        \
            val.y = val.x;                                                           \
            val.z = val.x;                                                           \
            val.w = val.x;                                                           \
        }                                                                            \
    }
#else
#define LOAD_BROAD(val)                                                     \
    {                                                                       \
        const int bcd4 = (bc + 3) >> 2;                                     \
        const int bwh_str = bw_str * bh_str;                                \
        char ec = ((idc << 2) + 4 <= bc) ? 4 : (bc & 3);                    \
        if ((idc << 2) >= bc) {                                             \
            ec = 1;                                                         \
        }                                                                   \
        int b_off = (bidn * bcd4 * bh_str + bidy) * bw_str + bidx + ib_off; \
        if (ec == 4) {                                                      \
            val = vload4(idc * bwh_str + b_off, broad);                     \
        } else {                                                            \
            b_off = b_off << 2;                                             \
            if (ec == 3) {                                                  \
                val.xyz = vload3(0, broad + b_off + (idc << 2) * bwh_str);  \
                val.w = val.z;                                              \
            } else if (ec == 2) {                                           \
                val.xy = vload2(0, broad + b_off + (idc << 2) * bwh_str);   \
                val.z = val.y;                                              \
                val.w = val.y;                                              \
            } else if (ec == 1) {                                           \
                val.x = broad[b_off + (bc - 1) * bwh_str];                  \
                val.y = val.x;                                              \
                val.z = val.x;                                              \
                val.w = val.x;                                              \
            }                                                               \
        }                                                                   \
    }
#endif
#endif

#if defined(AXIS_W1)
#define AXIS_NAME axis_w1
#if defined(USE_INPUT_IMG1)
#define LOAD_BROAD(val)                                                         \
    {                                                                           \
        val = READ_IMAGE(broad, sampler, (int4)(0, bidy, bidn * bc + bidc, 0)); \
        val.y = val.x;                                                          \
        val.z = val.x;                                                          \
        val.w = val.x;                                                          \
    }
#else
#define LOAD_BROAD(val)                                                     \
    {                                                                       \
        int b_off = ((bidn * bc + bidc) * bh_str + bidy) * bw_str + ib_off; \
        val.x = broad[b_off];                                               \
        val.y = val.x;                                                      \
        val.z = val.x;                                                      \
        val.w = val.x;                                                      \
    }
#endif
#endif

#if defined(AXIS_C1)
#define AXIS_NAME axis_c1
#if defined(USE_INPUT_IMG1)
#define LOAD_BROAD(val)                                                \
    {                                                                  \
        val = READ_IMAGE(broad, sampler, (int4)(bidx, bidy, bidn, 0)); \
        val.y = val.x;                                                 \
        val.z = val.x;                                                 \
        val.w = val.x;                                                 \
    }
#else
#define LOAD_BROAD(val)                                                     \
    {                                                                       \
        const int bcd4 = (bc + 3) >> 2;                                     \
        int b_off = (bidn * bcd4 * bh_str + bidy) * bw_str + bidx + ib_off; \
        val.x = broad[b_off * 4];                                           \
        val.y = val.x;                                                      \
        val.z = val.x;                                                      \
        val.w = val.x;                                                      \
    }
#endif
#endif

__kernel void MANGLE_NAME(eltwise_broadcast_, IOM, AM, EM, SI, FM, AXIS_NAME)(const int iw_str,
    const int ih_str,
    const int bw_str,
    const int bh_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int ib_off,
    const int o_off,
    const int iw,
    const int ic,
    const int bw,
    const int bh,
    const int bc,
    const int bn,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    READ_ONLY_KERNEL_MEM_1 broad,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
#if defined(USE_NCHW)
    const int idc = idz % ic;
    const int idn = idz / ic;
#else
    const int idc = idz % ((ic + 3) >> 2);
    const int idn = idz / ((ic + 3) >> 2);
#endif
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 val = 0;
    T4 res = 0;

#if defined(USE_NCHW)
    LOAD_MEM_V4_C1_COMMON(res, idx, idy, idz, iw_str, ih_str, i_off, iw, in);
    const int bidy = (idy < bh) ? idy : bh - 1;
    const int bidc = (idc < bc) ? idc : bc - 1;
    const int bidn = (idn < bn) ? idn : bn - 1;
#else
    LOAD_MEM_V4_COMMON(res, idx, idy, idz, iw_str, ih_str, i_off, in);
    const int bidx = (idx < bw) ? idx : bw - 1;
    const int bidy = (idy < bh) ? idy : bh - 1;
    const int bidn = (idn < bn) ? idn : bn - 1;
#endif

    LOAD_BROAD(val);
#if defined(SWAP_INPUT)
    ELTWISE_V4(res, val);
    res = val;
#else
    ELTWISE_V4(val, res);
#endif
    ACTIVATION_V4(res);

#if defined(USE_NCHW)
    STORE_MEM_V4_C1_COMMON(res, idx, idy, idz, ow_str, oh_str, o_off, iw, out);
#else
    STORE_MEM_V4_COMMON(res, idx, idy, idz, ow_str, oh_str, o_off, out);
#endif
}
