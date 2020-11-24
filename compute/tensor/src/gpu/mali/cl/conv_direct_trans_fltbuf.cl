// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, C, K) base##C##K
#define MANGLE_NAME(base, C, K) MANGLE_NAME_IMPL(base, C, K)
#if (C == 1)
#define loadFltval(off, str, flt, val) \
    {                                  \
        val = flt[off];                \
    }

#define loadFltvalEdge(off, str, flt, val, edge) \
    {}
#endif

#if (C == 2)
#define loadFltval(off, str, flt, val) \
    {                                  \
        val.x = flt[off];              \
        val.y = flt[off + str];        \
    }

#define loadFltvalEdge(off, str, flt, val, edge) \
    {                                            \
        val.x = flt[off];                        \
    }
#endif

#if (C == 3)
#define loadFltval(off, str, flt, val) \
    {                                  \
        val.x = flt[off];              \
        val.y = flt[off + str];        \
        val.z = flt[off + str * 2];    \
    }

#define loadFltvalEdge(off, str, flt, val, edge) \
    {                                            \
        val.x = flt[off];                        \
        if (edge > 1)                            \
            val.y = flt[off + str];              \
    }
#endif

#if (C == 4)
#define loadFltval(off, str, flt, val) \
    {                                  \
        val.x = flt[off];              \
        val.y = flt[off + str];        \
        val.z = flt[off + str * 2];    \
        val.w = flt[off + str * 3];    \
    }

#define loadFltvalEdge(off, str, flt, val, edge) \
    {                                            \
        val.x = flt[off];                        \
        if (edge > 1)                            \
            val.y = flt[off + str];              \
        if (edge > 2)                            \
            val.z = flt[off + str * 2];          \
    }
#endif

#if (C == 8)
#define loadFltval(off, str, flt, val) \
    {                                  \
        val.s0 = flt[off];             \
        val.s1 = flt[off + str];       \
        val.s2 = flt[off + str * 2];   \
        val.s3 = flt[off + str * 3];   \
        val.s4 = flt[off + str * 4];   \
        val.s5 = flt[off + str * 5];   \
        val.s6 = flt[off + str * 6];   \
        val.s7 = flt[off + str * 7];   \
    }
#define loadFltvalEdge(off, str, flt, val, edge) \
    {                                            \
        val.s0 = flt[off];                       \
        if (edge > 1)                            \
            val.s1 = flt[off + str];             \
        if (edge > 2)                            \
            val.s2 = flt[off + str * 2];         \
        if (edge > 3)                            \
            val.s3 = flt[off + str * 3];         \
        if (edge > 4)                            \
            val.s4 = flt[off + str * 4];         \
        if (edge > 5)                            \
            val.s5 = flt[off + str * 5];         \
        if (edge > 6)                            \
            val.s6 = flt[off + str * 6];         \
    }
#endif

#if (C == 16)
#define loadFltval(off, str, flt, val) \
    {                                  \
        val.s0 = flt[off];             \
        val.s1 = flt[off + str];       \
        val.s2 = flt[off + str * 2];   \
        val.s3 = flt[off + str * 3];   \
        val.s4 = flt[off + str * 4];   \
        val.s5 = flt[off + str * 5];   \
        val.s6 = flt[off + str * 6];   \
        val.s7 = flt[off + str * 7];   \
        val.s8 = flt[off + str * 8];   \
        val.s9 = flt[off + str * 9];   \
        val.sa = flt[off + str * 10];  \
        val.sb = flt[off + str * 11];  \
        val.sc = flt[off + str * 12];  \
        val.sd = flt[off + str * 13];  \
        val.se = flt[off + str * 14];  \
        val.sf = flt[off + str * 15];  \
    }
#define loadFltvalEdge(off, str, flt, val, edge) \
    {                                            \
        val.s0 = flt[off];                       \
        if (edge > 1)                            \
            val.s1 = flt[off + str];             \
        if (edge > 2)                            \
            val.s2 = flt[off + str * 2];         \
        if (edge > 3)                            \
            val.s3 = flt[off + str * 3];         \
        if (edge > 4)                            \
            val.s4 = flt[off + str * 4];         \
        if (edge > 5)                            \
            val.s5 = flt[off + str * 5];         \
        if (edge > 6)                            \
            val.s6 = flt[off + str * 6];         \
        if (edge > 7)                            \
            val.s7 = flt[off + str * 7];         \
        if (edge > 8)                            \
            val.s8 = flt[off + str * 8];         \
        if (edge > 9)                            \
            val.s9 = flt[off + str * 9];         \
        if (edge > 10)                           \
            val.sa = flt[off + str * 10];        \
        if (edge > 11)                           \
            val.sb = flt[off + str * 11];        \
        if (edge > 12)                           \
            val.sc = flt[off + str * 12];        \
        if (edge > 13)                           \
            val.sd = flt[off + str * 13];        \
        if (edge > 14)                           \
            val.se = flt[off + str * 14];        \
    }
#endif

__kernel void MANGLE_NAME(conv_direct_trans_fltbuf_, C, K)(
    const int fwh, const int fc, const int fn, __global const T *fltdata, __global T *fltbuf)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    short ec = ((idy + 1) * C <= fc) ? C : (fc % C);

    const int flt_off = (idz * fc + idy * C) * fwh + idx;
#if (C == 1)
    T val = 0;
#elif (C == 2)
    T2 val = 0;
#elif (C == 3)
    T3 val = 0;
#elif (C == 4)
    T4 val = 0;
#elif (C == 8)
    T8 val = 0;
#elif (C == 16)
    T16 val = 0;
#endif
    if (idz < fn) {
        if (ec == C) {
            loadFltval(flt_off, fwh, fltdata, val);
        } else {
            loadFltvalEdge(flt_off, fwh, fltdata, val, ec);
        }
    }
    const int bc = (fc + C - 1) / C;
    int out_off;
#if (K == 0)
    out_off = (idy * fwh + idx) * fn + idz;
#else
    out_off = (idz / K * bc + idy) * fwh * K + idx * K + (idz % K);
#endif
#if (C == 1)
    fltbuf[out_off] = val;
#elif (C == 2)
    vstore2(val, out_off, fltbuf);
#elif (C == 3)
    vstore3(val, out_off, fltbuf);
#elif (C == 4)
    vstore4(val, out_off, fltbuf);
#elif (C == 8)
    vstore8(val, out_off, fltbuf);
#elif (C == 16)
    vstore16(val, out_off, fltbuf);
#endif
}
