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
#define MANGLE_NAME_IMPL(base, ON) base##ON
#define MANGLE_NAME(base, ON) MANGLE_NAME_IMPL(base, ON)

__kernel void MANGLE_NAME(conv_wino_trans_picbuf_left_, ON)(const int ih_str,
    const int iw_str,
    const int ic_str,
    const int oh_str,
    const int ow_str,
    const int ohw_str,
    const int ohwc_str,
    const int bx,
    const int by,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    const int idzx = idz % ic_str;
    const int idzy = idz / ic_str;
    if (idx * ON >= oh_str) {
        return;
    }
    T in_val[6];
    T out_val0[ON];
    T out_val1[ON];
    T out_val2[ON];
    T out_val3[ON];
    T out_val4[ON];
    T out_val5[ON];
    T p[2];

    int in_off = (idz * iw_str + idy) * ih_str + (idx << 2) * ON;

    LOAD_BUF_ARRAY2(in_val, in_off, in);

    for (uchar i = 0; i < ON; ++i) {
        T4 tmp = vload4(0, in + in_off + 2);
        in_val[2] = tmp.x;
        in_val[3] = tmp.y;
        in_val[4] = tmp.z;
        in_val[5] = tmp.w;
        UPDATE_REG(out_val0);
        UPDATE_REG(out_val1);
        UPDATE_REG(out_val2);
        UPDATE_REG(out_val3);
        UPDATE_REG(out_val4);
        UPDATE_REG(out_val5);
        p[0] = -4;
        p[1] = 1;
        for (uchar j = 0; j < 2; ++j) {
            out_val1[UN] = out_val2[UN];
            out_val2[UN] = p[0] * in_val[1] - (T)(4.0) * in_val[2] + p[1] * in_val[3] + in_val[4];
            p[0] = -p[0];
            p[1] = -p[1];
        }

        p[0] = -2;
        p[1] = 2;
        for (uchar j = 0; j < 2; ++j) {
            out_val3[UN] = out_val4[UN];
            out_val4[UN] = p[0] * in_val[1] - in_val[2] + p[1] * in_val[3] + in_val[4];
            p[0] = -p[0];
            p[1] = -p[1];
        }

        p[0] = 4;
        p[1] = -5;
        for (uchar j = 0; j < 2; j++) {
            out_val0[UN] = out_val5[UN];
            out_val5[UN] = p[0] * in_val[0] + p[1] * in_val[2] + in_val[4];
            in_val[0] = in_val[1];
            in_val[2] = in_val[3];
            in_val[3] = in_val[4];
            in_val[4] = in_val[5];
        }

        in_val[0] = in_val[2];
        in_val[1] = in_val[3];
        in_off += 4;
    }

    int out_off = idzy * ohwc_str + idzx * ohw_str + idy * oh_str + idx * ON;
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val0, out_off, 1, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val1, out_off + 6 * ohwc_str, 1, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val2, out_off + 12 * ohwc_str, 1, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val3, out_off + 18 * ohwc_str, 1, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val4, out_off + 24 * ohwc_str, 1, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val5, out_off + 30 * ohwc_str, 1, out);
}
