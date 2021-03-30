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
#define MANGLE_NAME_IMPL(base, AM, EM, FM, N) base##AM##EM##FM##N
#define MANGLE_NAME(base, AM, EM, FM, N) MANGLE_NAME_IMPL(base, AM, EM, FM, N)
#define FM
#if defined(USE_NCHW)
#define FM nchw_
#endif

#if defined(USE_NCHW)
#define LOAD_VAL(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val)   \
    {                                                                           \
        int off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off; \
        val = 0;                                                                \
        if (ew == 4) {                                                          \
            val = vload4(0, buf + off);                                         \
        } else {                                                                \
            if (ew == 1)                                                        \
                val.x = buf[off];                                               \
            if (ew == 2) {                                                      \
                T2 tmp = vload2(0, buf + off);                                  \
                val.x = tmp.x;                                                  \
                val.y = tmp.y;                                                  \
            }                                                                   \
            if (ew == 3) {                                                      \
                T3 tmp = vload3(0, buf + off);                                  \
                val.x = tmp.x;                                                  \
                val.y = tmp.y;                                                  \
                val.z = tmp.z;                                                  \
            }                                                                   \
        }                                                                       \
    }
#define STORE_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, buf, val)  \
    {                                                                           \
        int off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off; \
        if (ew == 4) {                                                          \
            vstore4(val, 0, buf + off);                                         \
        } else {                                                                \
            if (ew == 1)                                                        \
                buf[off] = val.x;                                               \
            if (ew == 2) {                                                      \
                vstore2((T2)(val.x, val.y), 0, buf + off);                      \
            }                                                                   \
            if (ew == 3) {                                                      \
                vstore3((T3)(val.x, val.y, val.z), 0, buf + off);               \
            }                                                                   \
        }                                                                       \
    }
#else
#define LOAD_VAL(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val) \
    {                                                                         \
        int off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;      \
        val = vload4(off, buf);                                               \
    }
#define STORE_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, buf, val) \
    {                                                                          \
        int off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;       \
        vstore4(val, off, buf);                                                \
    }
#endif

__kernel void MANGLE_NAME(eltwise_, AM, EM, FM, N)(const int h,
    const int w,
    const int c,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const int bx,
    const int by,
    const int ih0_str,
    const int iw0_str,
    const int ih0_off,
    const int iw0_off,
    __global const T *in0,
#if (N > 1)
    const int ih1_str,
    const int iw1_str,
    const int ih1_off,
    const int iw1_off,
    __global const T *in1,
#endif
#if (N > 2)
    const int ih2_str,
    const int iw2_str,
    const int ih2_off,
    const int iw2_off,
    __global const T *in2,
#endif
#if (N > 3)
    const int ih3_str,
    const int iw3_str,
    const int ih3_off,
    const int iw3_off,
    __global const T *in3,
#endif
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    char ew = 0;
#if defined(USE_NCHW)
    ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
#endif

    T4 val;
    T4 res;
    LOAD_VAL(ew, idx, idy, idz, ih0_str, iw0_str, ih0_off, iw0_off, in0, res);
#if (N > 1)
    LOAD_VAL(ew, idx, idy, idz, ih1_str, iw1_str, ih1_off, iw1_off, in1, val);
    ELTWISE_V4(val, res);
#endif
#if (N > 2)
    LOAD_VAL(ew, idx, idy, idz, ih2_str, iw2_str, ih2_off, iw2_off, in2, val);
    ELTWISE_V4(val, res);
#endif
#if (N > 3)
    LOAD_VAL(ew, idx, idy, idz, ih3_str, iw3_str, ih3_off, iw3_off, in3, val);
    ELTWISE_V4(val, res);
#endif
    ACTIVATION_V4(res);
    STORE_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, out, res);
}
