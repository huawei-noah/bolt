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

__kernel void pooling_global_mean_h(const int ih,
    const int oh_str,
    const int ohw_str,
    const int oh_off,
    const int ow_off,
    const int bx,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    const int in_off = idx * ih;

    T4 val;
    float4 sum = 0;
    for (int i = 0; i < ih; ++i) {
        val = vload4(in_off + i, in);
        sumvec4(sum, val);
    }
    sum = sum / ((float)(ih));
    int out_off = idx * ohw_str + ow_off * oh_str + oh_off;
    vstore4((T4)(sum.x, sum.y, sum.z, sum.w), out_off, out);
}
