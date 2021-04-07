// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define SWAP_UP(x, y, ix, iy) \
    {                         \
        if (x < y) {          \
            T z = y;          \
            y = x;            \
            x = z;            \
            uchar iz = iy;    \
            iy = ix;          \
            ix = iz;          \
        }                     \
    }

#define SWAP_DN(x, y, ix, iy) \
    {                         \
        if (x > y) {          \
            T z = y;          \
            y = x;            \
            x = z;            \
            uchar iz = iy;    \
            iy = ix;          \
            ix = iz;          \
        }                     \
    }

#define SORT_VAL(v, id)                    \
    {                                      \
        SWAP_DN(v.s0, v.s1, id.s0, id.s1); \
        SWAP_UP(v.s2, v.s3, id.s2, id.s3); \
        SWAP_DN(v.s4, v.s5, id.s4, id.s5); \
        SWAP_UP(v.s6, v.s7, id.s6, id.s7); \
        SWAP_DN(v.s8, v.s9, id.s8, id.s9); \
        SWAP_UP(v.sa, v.sb, id.sa, id.sb); \
        SWAP_DN(v.sc, v.sd, id.sc, id.sd); \
        SWAP_UP(v.se, v.sf, id.se, id.sf); \
                                           \
        \    
    SWAP_DN(v.s0, v.s2, id.s0, id.s2);     \
        SWAP_DN(v.s1, v.s3, id.s1, id.s3); \
        SWAP_UP(v.s4, v.s6, id.s4, id.s6); \
        SWAP_UP(v.s5, v.s7, id.s5, id.s7); \
        SWAP_DN(v.s8, v.sa, id.s8, id.sa); \
        SWAP_DN(v.s9, v.sb, id.s9, id.sb); \
        SWAP_UP(v.sc, v.se, id.sc, id.se); \
        SWAP_UP(v.sd, v.sf, id.sd, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s1, id.s0, id.s1); \
        SWAP_DN(v.s2, v.s3, id.s2, id.s3); \
        SWAP_UP(v.s4, v.s5, id.s4, id.s5); \
        SWAP_UP(v.s6, v.s7, id.s6, id.s7); \
        SWAP_DN(v.s8, v.s9, id.s8, id.s9); \
        SWAP_DN(v.sa, v.sb, id.sa, id.sb); \
        SWAP_UP(v.sc, v.sd, id.sc, id.sd); \
        SWAP_UP(v.se, v.sf, id.se, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s4, id.s0, id.s4); \
        SWAP_DN(v.s1, v.s5, id.s1, id.s5); \
        SWAP_DN(v.s2, v.s6, id.s2, id.s6); \
        SWAP_DN(v.s3, v.s7, id.s3, id.s7); \
        SWAP_UP(v.s8, v.sc, id.s8, id.sc); \
        SWAP_UP(v.s9, v.sd, id.s9, id.sd); \
        SWAP_UP(v.sa, v.se, id.sa, id.se); \
        SWAP_UP(v.sb, v.sf, id.sb, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s2, id.s0, id.s2); \
        SWAP_DN(v.s1, v.s3, id.s1, id.s3); \
        SWAP_DN(v.s4, v.s6, id.s4, id.s6); \
        SWAP_DN(v.s5, v.s7, id.s5, id.s7); \
        SWAP_UP(v.s8, v.sa, id.s8, id.sa); \
        SWAP_UP(v.s9, v.sb, id.s9, id.sb); \
        SWAP_UP(v.sc, v.se, id.sc, id.se); \
        SWAP_UP(v.sd, v.sf, id.sd, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s1, id.s0, id.s1); \
        SWAP_DN(v.s2, v.s3, id.s2, id.s3); \
        SWAP_DN(v.s4, v.s5, id.s4, id.s5); \
        SWAP_DN(v.s6, v.s7, id.s6, id.s7); \
        SWAP_UP(v.s8, v.s9, id.s8, id.s9); \
        SWAP_UP(v.sa, v.sb, id.sa, id.sb); \
        SWAP_UP(v.sc, v.sd, id.sc, id.sd); \
        SWAP_UP(v.se, v.sf, id.se, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s8, id.s0, id.s8); \
        SWAP_DN(v.s1, v.s9, id.s1, id.s9); \
        SWAP_DN(v.s2, v.sa, id.s2, id.sa); \
        SWAP_DN(v.s3, v.sb, id.s3, id.sb); \
        SWAP_DN(v.s4, v.sc, id.s4, id.sc); \
        SWAP_DN(v.s5, v.sd, id.s5, id.sd); \
        SWAP_DN(v.s6, v.se, id.s6, id.se); \
        SWAP_DN(v.s7, v.sf, id.s7, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s4, id.s0, id.s4); \
        SWAP_DN(v.s1, v.s5, id.s1, id.s5); \
        SWAP_DN(v.s2, v.s6, id.s2, id.s6); \
        SWAP_DN(v.s3, v.s7, id.s3, id.s7); \
        SWAP_DN(v.s8, v.sc, id.s8, id.sc); \
        SWAP_DN(v.s9, v.sd, id.s9, id.sd); \
        SWAP_DN(v.sa, v.se, id.sa, id.se); \
        SWAP_DN(v.sb, v.sf, id.sb, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s2, id.s0, id.s2); \
        SWAP_DN(v.s1, v.s3, id.s1, id.s3); \
        SWAP_DN(v.s4, v.s6, id.s4, id.s6); \
        SWAP_DN(v.s5, v.s7, id.s5, id.s7); \
        SWAP_DN(v.s8, v.sa, id.s8, id.sa); \
        SWAP_DN(v.s9, v.sb, id.s9, id.sb); \
        SWAP_DN(v.sc, v.se, id.sc, id.se); \
        SWAP_DN(v.sd, v.sf, id.sd, id.sf); \
                                           \
        SWAP_DN(v.s0, v.s1, id.s0, id.s1); \
        SWAP_DN(v.s2, v.s3, id.s2, id.s3); \
        SWAP_DN(v.s4, v.s5, id.s4, id.s5); \
        SWAP_DN(v.s6, v.s7, id.s6, id.s7); \
        SWAP_DN(v.s8, v.s9, id.s8, id.s9); \
        SWAP_DN(v.sa, v.sb, id.sa, id.sb); \
        SWAP_DN(v.sc, v.sd, id.sc, id.sd); \
        SWAP_DN(v.se, v.sf, id.se, id.sf); \
    }

#if defined(USE_MAX)
#define PRO_VAL_EDGE(idx, len, val)                            \
    {                                                          \
        uchar rx = ((idx * 16 + 16) <= len) ? 16 : (len & 15); \
        if (rx < 16) {                                         \
            val.sf = -65535;                                   \
            if (rx < 15) {                                     \
                val.se = -65535;                               \
            }                                                  \
            if (rx < 14) {                                     \
                val.sd = -65535;                               \
            }                                                  \
            if (rx < 13) {                                     \
                val.sc = -65535;                               \
            }                                                  \
            if (rx < 12) {                                     \
                val.sb = -65535;                               \
            }                                                  \
            if (rx < 11) {                                     \
                val.sa = -65535;                               \
            }                                                  \
            if (rx < 10) {                                     \
                val.s9 = -65535;                               \
            }                                                  \
            if (rx < 9) {                                      \
                val.s8 = -65535;                               \
            }                                                  \
            if (rx < 8) {                                      \
                val.s7 = -65535;                               \
            }                                                  \
            if (rx < 7) {                                      \
                val.s6 = -65535;                               \
            }                                                  \
            if (rx < 6) {                                      \
                val.s5 = -65535;                               \
            }                                                  \
            if (rx < 5) {                                      \
                val.s4 = -65535;                               \
            }                                                  \
            if (rx < 4) {                                      \
                val.s3 = -65535;                               \
            }                                                  \
            if (rx < 3) {                                      \
                val.s2 = -65535;                               \
            }                                                  \
            if (rx < 2) {                                      \
                val.s1 = -65535;                               \
            }                                                  \
        }                                                      \
    }
#elif defined(USE_MIN)
#define PRO_VAL_EDGE(idx, len, val)                            \
    {                                                          \
        uchar rx = ((idx * 16 + 16) <= len) ? 16 : (len & 15); \
        if (rx < 16) {                                         \
            val.sf = 65535;                                    \
            if (rx < 15) {                                     \
                val.se = 65535;                                \
            }                                                  \
            if (rx < 14) {                                     \
                val.sd = 65535;                                \
            }                                                  \
            if (rx < 13) {                                     \
                val.sc = 65535;                                \
            }                                                  \
            if (rx < 12) {                                     \
                val.sb = 65535;                                \
            }                                                  \
            if (rx < 11) {                                     \
                val.sa = 65535;                                \
            }                                                  \
            if (rx < 10) {                                     \
                val.s9 = 65535;                                \
            }                                                  \
            if (rx < 9) {                                      \
                val.s8 = 65535;                                \
            }                                                  \
            if (rx < 8) {                                      \
                val.s7 = 65535;                                \
            }                                                  \
            if (rx < 7) {                                      \
                val.s6 = 65535;                                \
            }                                                  \
            if (rx < 6) {                                      \
                val.s5 = 65535;                                \
            }                                                  \
            if (rx < 5) {                                      \
                val.s4 = 65535;                                \
            }                                                  \
            if (rx < 4) {                                      \
                val.s3 = 65535;                                \
            }                                                  \
            if (rx < 3) {                                      \
                val.s2 = 65535;                                \
            }                                                  \
            if (rx < 2) {                                      \
                val.s1 = 65535;                                \
            }                                                  \
        }                                                      \
    }
#endif

#define build_id(id) \
    {                \
        id.s1 += 1;  \
        id.s2 += 2;  \
        id.s3 += 3;  \
        id.s4 += 4;  \
        id.s5 += 5;  \
        id.s6 += 6;  \
        id.s7 += 7;  \
        id.s8 += 8;  \
        id.s9 += 9;  \
        id.sa += 10; \
        id.sb += 11; \
        id.sc += 12; \
        id.sd += 13; \
        id.se += 14; \
        id.sf += 15; \
    }

__kernel void
#if defined(USE_MAX)
topk_sort_max
#elif defined(USE_MIN)
topk_sort_min
#endif
    (const int len, const int bx, __global const T *in, __global T *out, __global ushort *out_id)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    T16 val = vload16(idx, in);
    uchar16 id = 0;

    build_id(id);
    PRO_VAL_EDGE(idx, len, val);
    SORT_VAL(val, id);
    vstore16(val, idx, out);

    ushort16 id_final = (ushort16)(idx << 4);
    id_final.s0 += id.s0;
    id_final.s1 += id.s1;
    id_final.s2 += id.s2;
    id_final.s3 += id.s3;
    id_final.s4 += id.s4;
    id_final.s5 += id.s5;
    id_final.s6 += id.s6;
    id_final.s7 += id.s7;
    id_final.s8 += id.s8;
    id_final.s9 += id.s9;
    id_final.sa += id.sa;
    id_final.sb += id.sb;
    id_final.sc += id.sc;
    id_final.sd += id.sd;
    id_final.se += id.se;
    id_final.sf += id.sf;
    vstore16(id_final, idx, out_id);
}
