// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define maxvec4(x, y)                       \
    {                                       \
        x.s0 = (x.s0 > y.s0) ? x.s0 : y.s0; \
        x.s1 = (x.s1 > y.s1) ? x.s1 : y.s1; \
        x.s2 = (x.s2 > y.s2) ? x.s2 : y.s2; \
        x.s3 = (x.s3 > y.s3) ? x.s3 : y.s3; \
    }

__kernel void pooling_max(const int ih,
    const int iw,
    const int ih_off,
    const int iw_off,
    const int ih_str,
    const int iw_str,
    const int oh,
    const int ow,
    const int oh_off,
    const int ow_off,
    const int oh_str,
    const int ow_str,
    const int sh,
    const int sw,
    const int ph,
    const int pw,
    const int kh,
    const int kw,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= oh || idy >= ow) {
        return;
    }

    int bh = idx * sh - ph;
    int bw = idy * sw - pw;
    int eh = bh + kh;
    int ew = bw + kw;
    bh = (bh < 0) ? 0 : bh;
    bw = (bw < 0) ? 0 : bw;
    eh = (eh < ih) ? eh : ih;
    ew = (ew < iw) ? ew : iw;

    bh += ih_off;
    bw += iw_off;
    eh += ih_off;
    ew += iw_off;
    int in_off = (idz * iw_str + bw) * ih_str;

    T4 val = -FLT_MAX;
    T4 maxval = -FLT_MAX;
    for (int i = bw; i < ew; ++i) {
        for (int j = bh; j < eh; ++j) {
            val = vload4(in_off + j, in);
            maxvec4(maxval, val);
        }
        in_off += ih_str;
    }
    int out_off = (idz * ow_str + ow_off + idy) * oh_str + oh_off + idx;
    vstore4(maxval, out_off, out);
}
