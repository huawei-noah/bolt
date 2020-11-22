// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define sumvec4(x, y)        \
    {                        \
        x.s0 += (float)y.s0; \
        x.s1 += (float)y.s1; \
        x.s2 += (float)y.s2; \
        x.s3 += (float)y.s3; \
    }

__kernel void pooling_mean(const int ih,
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
    float psize = (eh - bh) * (ew - bw);

    bh += ih_off;
    bw += iw_off;
    eh += ih_off;
    ew += iw_off;
    int in_off = (idz * iw_str + bw) * ih_str;

    T4 val;
    float4 sum = 0;
    for (int i = bw; i < ew; ++i) {
        for (int j = bh; j < eh; ++j) {
            val = vload4(in_off + j, in);
            sumvec4(sum, val);
        }
        in_off += ih_str;
    }
    sum = sum / psize;
    int out_off = (idz * ow_str + ow_off + idy) * oh_str + oh_off + idx;
    vstore4((T4)(sum.x, sum.y, sum.z, sum.w), out_off, out);
}
