// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define get_max(val, dim)      \
    {                          \
        dim.s0 = 0;            \
        dim.s1 = 1;            \
        dim.s2 = 2;            \
        dim.s3 = 3;            \
        if (val.s4 > val.s0) { \
            val.s0 = val.s4;   \
            dim.s0 = 4;        \
        }                      \
        if (val.s5 > val.s1) { \
            val.s1 = val.s5;   \
            dim.s1 = 5;        \
        }                      \
        if (val.s6 > val.s2) { \
            val.s2 = val.s6;   \
            dim.s2 = 6;        \
        }                      \
        if (val.s7 > val.s3) { \
            val.s3 = val.s7;   \
            dim.s3 = 7;        \
        }                      \
        if (val.s2 > val.s0) { \
            val.s0 = val.s2;   \
            dim.s0 = dim.s2;   \
        }                      \
        if (val.s3 > val.s1) { \
            val.s1 = val.s3;   \
            dim.s1 = dim.s3;   \
        }                      \
        if (val.s1 > val.s0) { \
            val.s0 = val.s1;   \
            dim.s0 = dim.s1;   \
        }                      \
    }

#if defined(USE_INDEX)
__kernel void argmax_x_index
#else
__kernel void argmax_x
#endif
    (const int iw_str,
        const int ih_str,
        const int iw_off,
        const int ih_off,
        const int ow_str,
        const int oh_str,
        const int ow_off,
        const int oh_off,
        const int len,
        const int bx,
        const int by,
        __global const T *in,
        __global const uint *ini,
        __global T *outv,
        __global uint *outi)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    int bn = len >> 3;
    int en = len & 7;
    T8 val;
    uchar4 dim;
    T maxval = -65504;
    uint maxIndex = 1;
    const int in_off = (idz * ih_str + idy + ih_off) * iw_str + iw_off;
    for (int i = idx; i < bn; i += bx) {
        val = vload8(i, in + in_off);
        get_max(val, dim);
        if (val.s0 > maxval) {
            maxval = val.s0;
            maxIndex = (i << 3) + dim.s0;
        }
    }

    if (en != 0 && idx == bx - 1) {
        int be = len - 8;
        int rx = 0;
        if (be < 0) {
            be = 0;
            rx = -be;
        }
        val = vload8(0, in + in_off + be);
        if (rx > 0) {
            val.s7 = -65504;
            if (rx > 1) {
                val.s6 = -65504;
            }
            if (rx > 2) {
                val.s5 = -65504;
            }
            if (rx > 3) {
                val.s4 = -65504;
            }
            if (rx > 4) {
                val.s3 = -65504;
            }
            if (rx > 5) {
                val.s2 = -65504;
            }
            if (rx > 6) {
                val.s1 = -65504;
            }
        }
        get_max(val, dim);
        if (val.s0 > maxval) {
            maxval = val.s0;
            maxIndex = be + dim.s0;
        }
    }
    int out_off = (idz * oh_str + idy + oh_off) * ow_str + idx + ow_off;
#if defined(USE_INDEX)
    maxIndex = ini[maxIndex];
#endif
    if (bx > 1) {
        outv[out_off] = maxval;
    }
    outi[out_off] = maxIndex;
}
