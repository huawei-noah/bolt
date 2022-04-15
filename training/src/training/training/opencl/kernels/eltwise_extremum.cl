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

__kernel void eltwise_extremum(const int h,
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
    const T index,
    __global const T *in0,
    __global T *indexes,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    char ew = 0;
    ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);

    T4 val;
    T4 res;
    T4 ids;
    LOAD_VAL(ew, idx, idy, idz, ih0_str, iw0_str, ih0_off, iw0_off, in0, res);
    LOAD_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, indexes, ids);
    LOAD_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, out, val);
#if defined(USE_FORWARD)
#if defined(USE_MAX)
    ids.s0 = (res.s0 >= val.s0) ? index : ids.s0;
    ids.s1 = (res.s1 >= val.s1) ? index : ids.s1;
    ids.s2 = (res.s2 >= val.s2) ? index : ids.s2;
    ids.s3 = (res.s3 >= val.s3) ? index : ids.s3;
#else
    ids.s0 = (res.s0 <= val.s0) ? index : ids.s0;
    ids.s1 = (res.s1 <= val.s1) ? index : ids.s1;
    ids.s2 = (res.s2 <= val.s2) ? index : ids.s2;
    ids.s3 = (res.s3 <= val.s3) ? index : ids.s3;
#endif
    STORE_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, indexes, ids);

    ELTWISE_V4(val, res);
#else
    res.s0 = (ids.s0 == index) * res.s0;
    res.s1 = (ids.s1 == index) * res.s1;
    res.s2 = (ids.s2 == index) * res.s2;
    res.s3 = (ids.s3 == index) * res.s3;

    res.s0 += val.s0;
    res.s1 += val.s1;
    res.s2 += val.s2;
    res.s3 += val.s3;
#endif
    STORE_VAL(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, out, res);
}
)"