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

__kernel void pooling_global_mean_w(const int ih_str,
    const int ihw_str,
    const int ih_off,
    const int iw_off,
    const int ih,
    const int iw,
    const int bx,
    const int by,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }

    int in_off = idy * ihw_str + iw_off * ih_str + idx + ih_off;

    T4 val;
    float4 sum = 0;
    for (int i = 0; i < iw; ++i) {
        val = vload4(in_off + ih_str * i, in);
        sumvec4(sum, val);
    }
    sum = sum / (float)(iw);
    int out_off = (idy * ih) + idx;
    vstore4((T4)(sum.x, sum.y, sum.z, sum.w), out_off, out);
}
