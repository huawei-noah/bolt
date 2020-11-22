// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define loadG(val, str, off, flt)    \
    {                                \
        val[0] = flt[off];           \
        val[1] = flt[off + str];     \
        val[2] = flt[off + str * 2]; \
    }

#define setReg6(reg0, reg1) \
    {                       \
        reg1[0] = reg0[0];  \
        reg1[1] = reg0[1];  \
        reg1[2] = reg0[2];  \
        reg1[3] = reg0[3];  \
        reg1[4] = reg0[4];  \
        reg1[5] = reg0[5];  \
    }

#define addReg6(reg0, reg1) \
    {                       \
        reg1[0] += reg0[0]; \
        reg1[1] += reg0[1]; \
        reg1[2] += reg0[2]; \
        reg1[3] += reg0[3]; \
        reg1[4] += reg0[4]; \
        reg1[5] += reg0[5]; \
    }

#define minReg6(reg0, reg1) \
    {                       \
        reg1[0] -= reg0[0]; \
        reg1[1] -= reg0[1]; \
        reg1[2] -= reg0[2]; \
        reg1[3] -= reg0[3]; \
        reg1[4] -= reg0[4]; \
        reg1[5] -= reg0[5]; \
    }

#define mulReg6(s, reg0, reg1) \
    {                          \
        reg1[0] = s * reg0[0]; \
        reg1[1] = s * reg0[1]; \
        reg1[2] = s * reg0[2]; \
        reg1[3] = s * reg0[3]; \
        reg1[4] = s * reg0[4]; \
        reg1[5] = s * reg0[5]; \
    }

#define calCore(g, t)                                                    \
    {                                                                    \
        t[0] = (T)(0.75) * g[0];                                         \
        t[1] = (g[0] + g[1] + g[2]) * (T)(-0.5);                         \
        t[2] = (g[0] - g[1] + g[2]) * (T)(-0.5);                         \
        t[3] = ((T)(0.125) * g[0] + (T)(0.25) * g[1] + (T)(0.5) * g[2]); \
        t[4] = ((T)(0.125) * g[0] - (T)(0.25) * g[1] + (T)(0.5) * g[2]); \
        t[5] = (T)(3.0) * g[2];                                          \
    }

#define storeReg6(reg, off, str, flt) \
    {                                 \
        flt[off] = reg[0];            \
        flt[off + str] = reg[1];      \
        flt[off + str * 2] = reg[2];  \
        flt[off + str * 3] = reg[3];  \
        flt[off + str * 4] = reg[4];  \
        flt[off + str * 5] = reg[5];  \
    }

__kernel void conv_wino_trans_fltbuf_3x3(
    const int fn, const int fc, const int fnc, __global const T *fltbuf, __global T *flttran)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int in_off = idy * fn + idx;

    T g[3];
    T h0[6], h1[6], h2[6], h3[6], h4[6], h5[6], t[6], tmp[6];
    loadG(g, fnc, in_off, fltbuf);
    calCore(g, tmp);
    mulReg6((T)(0.75), tmp, h0);
    mulReg6((T)(-0.5), tmp, t);
    setReg6(t, h1);
    setReg6(t, h2);
    mulReg6((T)(0.125), tmp, t);
    setReg6(t, h3);
    setReg6(t, h4);

    loadG(g, fnc, in_off + 3 * fnc, fltbuf);
    calCore(g, tmp);
    mulReg6((T)(0.5), tmp, t);
    minReg6(t, h1);
    addReg6(t, h2);
    mulReg6((T)(0.25), tmp, t);
    addReg6(t, h3);
    minReg6(t, h4);

    loadG(g, fnc, in_off + 6 * fnc, fltbuf);
    calCore(g, tmp);
    mulReg6((T)(0.5), tmp, t);
    minReg6(t, h1);
    minReg6(t, h2);
    addReg6(t, h3);
    addReg6(t, h4);
    mulReg6((T)(3.0), tmp, h5);

    storeReg6(h0, in_off, fnc, flttran);
    storeReg6(h1, in_off + 6 * fnc, fnc, flttran);
    storeReg6(h2, in_off + 12 * fnc, fnc, flttran);
    storeReg6(h3, in_off + 18 * fnc, fnc, flttran);
    storeReg6(h4, in_off + 24 * fnc, fnc, flttran);
    storeReg6(h5, in_off + 30 * fnc, fnc, flttran);
}
