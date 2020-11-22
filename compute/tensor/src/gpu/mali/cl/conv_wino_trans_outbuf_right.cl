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
#define loadR(val, str, off, in)    \
    {                               \
        val[0] = in[off];           \
        val[1] = in[off + str];     \
        val[2] = in[off + str * 2]; \
        val[3] = in[off + str * 3]; \
        val[4] = in[off + str * 4]; \
        val[5] = in[off + str * 5]; \
    }

#define calCore(s, t, tmp)                      \
    {                                           \
        t[0] = s[1] + s[2];                     \
        t[1] = s[3] + s[4];                     \
        t[2] = s[1] - s[2];                     \
        t[3] = s[3] - s[4];                     \
        tmp[0] = s[0] + t[0] + t[1];            \
        tmp[1] = t[2] + (T)(2.0) * t[3];        \
        tmp[2] = t[0] + (T)(4.0) * t[1];        \
        tmp[3] = t[2] + (T)(8.0) * t[3] + s[5]; \
    }

__kernel void conv_wino_trans_outbuf_right(const int iw_str,
    const int iwh_str,
    const int wino_h,
    const int wino_w,
    const int wino_h6,
    const int wino_hw,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= wino_hw) {
        return;
    }

    int in_off = idz * iwh_str * 6 + (idy << 2) * iw_str + idx;
    T s[6];
    T4 res[4];
    for (int ii = 0; ii < 4; ++ii) {
        loadR(s, iwh_str, in_off, in);
        res[0] = res[1];
        res[1] = res[2];
        res[2] = res[3];
        res[3].x = s[0] + s[1] + s[2] + s[3] + s[4];
        res[3].y = s[1] - s[2] + (T)(2) * (s[3] - s[4]);
        res[3].z = s[1] + s[2] + (T)(4) * (s[3] + s[4]);
        res[3].w = s[1] - s[2] + (T)(8) * (s[3] - s[4]) + s[5];
        in_off += iw_str;
    }

    const int idx_i = idx % wino_h;
    const int idx_j = idx / wino_h;
    const int out_off = (idy * 24 * wino_w + idx_j * 24 + idz) * wino_h + idx_i;
    vstore4((T4)(res[0].x, res[1].x, res[2].x, res[3].x), out_off, out);
    vstore4((T4)(res[0].y, res[1].y, res[2].y, res[3].y), out_off + wino_h6, out);
    vstore4((T4)(res[0].z, res[1].z, res[2].z, res[3].z), out_off + wino_h6 * 2, out);
    vstore4((T4)(res[0].w, res[1].w, res[2].w, res[3].w), out_off + wino_h6 * 3, out);
}
