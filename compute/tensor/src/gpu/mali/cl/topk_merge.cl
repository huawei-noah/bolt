// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define SWAP_DN(x, y, ix, iy) \
    {                         \
        if (x > y) {          \
            T z = y;          \
            y = x;            \
            x = z;            \
            ushort iz = iy;   \
            iy = ix;          \
            ix = iz;          \
        }                     \
    }

#if defined(USE_MAX)
#define SORT_TOP16(va, vb, ia, ib)           \
    {                                        \
        SWAP_DN(va.s0, vb.s0, ia.s0, ib.s0); \
        SWAP_DN(va.s1, vb.s1, ia.s1, ib.s1); \
        SWAP_DN(va.s2, vb.s2, ia.s2, ib.s2); \
        SWAP_DN(va.s3, vb.s3, ia.s3, ib.s3); \
        SWAP_DN(va.s4, vb.s4, ia.s4, ib.s4); \
        SWAP_DN(va.s5, vb.s5, ia.s5, ib.s5); \
        SWAP_DN(va.s6, vb.s6, ia.s6, ib.s6); \
        SWAP_DN(va.s7, vb.s7, ia.s7, ib.s7); \
        SWAP_DN(va.s8, vb.s8, ia.s8, ib.s8); \
        SWAP_DN(va.s9, vb.s9, ia.s9, ib.s9); \
        SWAP_DN(va.sa, vb.sa, ia.sa, ib.sa); \
        SWAP_DN(va.sb, vb.sb, ia.sb, ib.sb); \
        SWAP_DN(va.sc, vb.sc, ia.sc, ib.sc); \
        SWAP_DN(va.sd, vb.sd, ia.sd, ib.sd); \
        SWAP_DN(va.se, vb.se, ia.se, ib.se); \
        SWAP_DN(va.sf, vb.sf, ia.sf, ib.sf); \
                                             \
        SWAP_DN(vb.s0, vb.s8, ib.s0, ib.s8); \
        SWAP_DN(vb.s1, vb.s9, ib.s1, ib.s9); \
        SWAP_DN(vb.s2, vb.sa, ib.s2, ib.sa); \
        SWAP_DN(vb.s3, vb.sb, ib.s3, ib.sb); \
        SWAP_DN(vb.s4, vb.sc, ib.s4, ib.sc); \
        SWAP_DN(vb.s5, vb.sd, ib.s5, ib.sd); \
        SWAP_DN(vb.s6, vb.se, ib.s6, ib.se); \
        SWAP_DN(vb.s7, vb.sf, ib.s7, ib.sf); \
                                             \
        SWAP_DN(vb.s0, vb.s4, ib.s0, ib.s4); \
        SWAP_DN(vb.s1, vb.s5, ib.s1, ib.s5); \
        SWAP_DN(vb.s2, vb.s6, ib.s2, ib.s6); \
        SWAP_DN(vb.s3, vb.s7, ib.s3, ib.s7); \
        SWAP_DN(vb.s8, vb.sc, ib.s8, ib.sc); \
        SWAP_DN(vb.s9, vb.sd, ib.s9, ib.sd); \
        SWAP_DN(vb.sa, vb.se, ib.sa, ib.se); \
        SWAP_DN(vb.sb, vb.sf, ib.sb, ib.sf); \
                                             \
        SWAP_DN(vb.s0, vb.s2, ib.s0, ib.s2); \
        SWAP_DN(vb.s1, vb.s3, ib.s1, ib.s3); \
        SWAP_DN(vb.s4, vb.s6, ib.s4, ib.s6); \
        SWAP_DN(vb.s5, vb.s7, ib.s5, ib.s7); \
        SWAP_DN(vb.s8, vb.sa, ib.s8, ib.sa); \
        SWAP_DN(vb.s9, vb.sb, ib.s9, ib.sb); \
        SWAP_DN(vb.sc, vb.se, ib.sc, ib.se); \
        SWAP_DN(vb.sd, vb.sf, ib.sd, ib.sf); \
                                             \
        SWAP_DN(vb.s0, vb.s1, ib.s0, ib.s1); \
        SWAP_DN(vb.s2, vb.s3, ib.s2, ib.s3); \
        SWAP_DN(vb.s4, vb.s5, ib.s4, ib.s5); \
        SWAP_DN(vb.s6, vb.s7, ib.s6, ib.s7); \
        SWAP_DN(vb.s8, vb.s9, ib.s8, ib.s9); \
        SWAP_DN(vb.sa, vb.sb, ib.sa, ib.sb); \
        SWAP_DN(vb.sc, vb.sd, ib.sc, ib.sd); \
        SWAP_DN(vb.se, vb.sf, ib.se, ib.sf); \
    }

#define STORE_EDGE(val, num, off, buf) \
    {                                  \
        buf[off] = val.sf;             \
        \ 
    if (num > 1)                       \
        {                              \
            buf[off + 1] = val.se;     \
        }                              \
        if (num > 2) {                 \
            buf[off + 2] = val.sd;     \
        }                              \
        if (num > 3) {                 \
            buf[off + 3] = val.sc;     \
        }                              \
        if (num > 4) {                 \
            buf[off + 4] = val.sb;     \
        }                              \
        if (num > 5) {                 \
            buf[off + 5] = val.sa;     \
        }                              \
        if (num > 6) {                 \
            buf[off + 6] = val.s9;     \
        }                              \
        if (num > 7) {                 \
            buf[off + 7] = val.s8;     \
        }                              \
        if (num > 8) {                 \
            buf[off + 8] = val.s7;     \
        }                              \
        if (num > 9) {                 \
            buf[off + 9] = val.s6;     \
        }                              \
        if (num > 10) {                \
            buf[off + 10] = val.s5;    \
        }                              \
        if (num > 11) {                \
            buf[off + 11] = val.s4;    \
        }                              \
        if (num > 12) {                \
            buf[off + 12] = val.s3;    \
        }                              \
        if (num > 13) {                \
            buf[off + 13] = val.s2;    \
        }                              \
        if (num > 14) {                \
            buf[off + 14] = val.s1;    \
        }                              \
    }
#elif defined(USE_MIN)
#define SORT_TOP16(va, vb, ia, ib)           \
    {                                        \
        SWAP_DN(va.s0, vb.s0, ia.s0, ib.s0); \
        SWAP_DN(va.s1, vb.s1, ia.s1, ib.s1); \
        SWAP_DN(va.s2, vb.s2, ia.s2, ib.s2); \
        SWAP_DN(va.s3, vb.s3, ia.s3, ib.s3); \
        SWAP_DN(va.s4, vb.s4, ia.s4, ib.s4); \
        SWAP_DN(va.s5, vb.s5, ia.s5, ib.s5); \
        SWAP_DN(va.s6, vb.s6, ia.s6, ib.s6); \
        SWAP_DN(va.s7, vb.s7, ia.s7, ib.s7); \
        SWAP_DN(va.s8, vb.s8, ia.s8, ib.s8); \
        SWAP_DN(va.s9, vb.s9, ia.s9, ib.s9); \
        SWAP_DN(va.sa, vb.sa, ia.sa, ib.sa); \
        SWAP_DN(va.sb, vb.sb, ia.sb, ib.sb); \
        SWAP_DN(va.sc, vb.sc, ia.sc, ib.sc); \
        SWAP_DN(va.sd, vb.sd, ia.sd, ib.sd); \
        SWAP_DN(va.se, vb.se, ia.se, ib.se); \
        SWAP_DN(va.sf, vb.sf, ia.sf, ib.sf); \
                                             \
        SWAP_DN(va.s0, va.s8, ia.s0, ia.s8); \
        SWAP_DN(va.s1, va.s9, ia.s1, ia.s9); \
        SWAP_DN(va.s2, va.sa, ia.s2, ia.sa); \
        SWAP_DN(va.s3, va.sb, ia.s3, ia.sb); \
        SWAP_DN(va.s4, va.sc, ia.s4, ia.sc); \
        SWAP_DN(va.s5, va.sd, ia.s5, ia.sd); \
        SWAP_DN(va.s6, va.se, ia.s6, ia.se); \
        SWAP_DN(va.s7, va.sf, ia.s7, ia.sf); \
                                             \
        SWAP_DN(va.s0, va.s4, ia.s0, ia.s4); \
        SWAP_DN(va.s1, va.s5, ia.s1, ia.s5); \
        SWAP_DN(va.s2, va.s6, ia.s2, ia.s6); \
        SWAP_DN(va.s3, va.s7, ia.s3, ia.s7); \
        SWAP_DN(va.s8, va.sc, ia.s8, ia.sc); \
        SWAP_DN(va.s9, va.sd, ia.s9, ia.sd); \
        SWAP_DN(va.sa, va.se, ia.sa, ia.se); \
        SWAP_DN(va.sb, va.sf, ia.sb, ia.sf); \
                                             \
        SWAP_DN(va.s0, va.s2, ia.s0, ia.s2); \
        SWAP_DN(va.s1, va.s3, ia.s1, ia.s3); \
        SWAP_DN(va.s4, va.s6, ia.s4, ia.s6); \
        SWAP_DN(va.s5, va.s7, ia.s5, ia.s7); \
        SWAP_DN(va.s8, va.sa, ia.s8, ia.sa); \
        SWAP_DN(va.s9, va.sb, ia.s9, ia.sb); \
        SWAP_DN(va.sc, va.se, ia.sc, ia.se); \
        SWAP_DN(va.sd, va.sf, ia.sd, ia.sf); \
                                             \
        SWAP_DN(va.s0, va.s1, ia.s0, ia.s1); \
        SWAP_DN(va.s2, va.s3, ia.s2, ia.s3); \
        SWAP_DN(va.s4, va.s5, ia.s4, ia.s5); \
        SWAP_DN(va.s6, va.s7, ia.s6, ia.s7); \
        SWAP_DN(va.s8, va.s9, ia.s8, ia.s9); \
        SWAP_DN(va.sa, va.sb, ia.sa, ia.sb); \
        SWAP_DN(va.sc, va.sd, ia.sc, ia.sd); \
        SWAP_DN(va.se, va.sf, ia.se, ia.sf); \
    }

#define STORE_EDGE(val, num, off, buf) \
    {                                  \
        buf[off] = val.s0;             \
        \ 
    if (num > 1)                       \
        {                              \
            buf[off + 1] = val.s1;     \
        }                              \
        if (num > 2) {                 \
            buf[off + 2] = val.s2;     \
        }                              \
        if (num > 3) {                 \
            buf[off + 3] = val.s3;     \
        }                              \
        if (num > 4) {                 \
            buf[off + 4] = val.s4;     \
        }                              \
        if (num > 5) {                 \
            buf[off + 5] = val.s5;     \
        }                              \
        if (num > 6) {                 \
            buf[off + 6] = val.s6;     \
        }                              \
        if (num > 7) {                 \
            buf[off + 7] = val.s7;     \
        }                              \
        if (num > 8) {                 \
            buf[off + 8] = val.s8;     \
        }                              \
        if (num > 9) {                 \
            buf[off + 9] = val.s9;     \
        }                              \
        if (num > 10) {                \
            buf[off + 10] = val.sa;    \
        }                              \
        if (num > 11) {                \
            buf[off + 11] = val.sb;    \
        }                              \
        if (num > 12) {                \
            buf[off + 12] = val.sc;    \
        }                              \
        if (num > 13) {                \
            buf[off + 13] = val.sd;    \
        }                              \
        if (num > 14) {                \
            buf[off + 14] = val.se;    \
        }                              \
    }
#endif

#define SWAP_VEC16(a, b) \
    {                    \
        b.s0 = a.sf;     \
        b.s1 = a.se;     \
        b.s2 = a.sd;     \
        b.s3 = a.sc;     \
        b.s4 = a.sb;     \
        b.s5 = a.sa;     \
        b.s6 = a.s9;     \
        b.s7 = a.s8;     \
        b.s8 = a.s7;     \
        b.s9 = a.s6;     \
        b.sa = a.s5;     \
        b.sb = a.s4;     \
        b.sc = a.s3;     \
        b.sd = a.s2;     \
        b.se = a.s1;     \
        b.sf = a.s0;     \
    }

__kernel void
#if defined(USE_MAX)
topk_merge_max
#elif defined(USE_MIN)
topk_merge_min
#endif
    (const int total_group_num,
        const int out_val_num,
        const int out_off,
        const int bx,
        __global const T *in,
        __global const ushort *in_index,
        __global T *out,
        __global ushort *out_index)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }

    T16 res = vload16(idx, in);
    ushort16 res_id = vload16(idx, in_index);
    T16 va, vb;
    ushort16 ia, ib;
    for (int i = idx + bx; i < total_group_num; i += bx) {
        va = vload16(i, in);
        ia = vload16(i, in_index);
#if defined(USE_MAX)
        SWAP_VEC16(res, vb);
        SWAP_VEC16(res_id, ib);
        SORT_TOP16(va, vb, ia, ib);
        res = vb;
        res_id = ib;
#elif defined(USE_MIN)
        SWAP_VEC16(va, vb);
        SWAP_VEC16(ia, ib);
        SORT_TOP16(res, vb, res_id, ib);
#endif
    }

    vstore16(res_id, idx, out_index);
    if (out_val_num == 16) {
        vstore16(res, idx, out + out_off);
    } else {
        STORE_EDGE(res, out_val_num, out_off, out);
    }
}
