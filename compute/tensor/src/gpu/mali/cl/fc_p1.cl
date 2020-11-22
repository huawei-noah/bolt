// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define calCore(iv, fv, res)                                                \
    {                                                                       \
        res.x += iv.x * fv.s0 + iv.y * fv.s1 + iv.z * fv.s2 + iv.w * fv.s3; \
        res.y += iv.x * fv.s4 + iv.y * fv.s5 + iv.z * fv.s6 + iv.w * fv.s7; \
        res.z += iv.x * fv.s8 + iv.y * fv.s9 + iv.z * fv.sa + iv.w * fv.sb; \
        res.w += iv.x * fv.sc + iv.y * fv.sd + iv.z * fv.se + iv.w * fv.sf; \
    }
__kernel void fc_p1(const int item_y,
    const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int ihy_str,
    const int ihw_str,
    const int fh,
    const int fw,
    const int fc,
    const int fn,
    const int fhy_str,
    const int fhw_str,
    const int fwc_str,
    __global const T *flt,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= fh || idy >= item_y) {
        return;
    }

    T4 in_val;
    T16 flt_val;
    T4 sum = 0;
    int in_off = (idy + iw_off) * ih_str + idx + ih_off;
    int flt_off = (idz * fwc_str + idy) * fh + idx;

    for (int i = 0; i < fc; i++) {
        int k = 0;
        for (int j = idy; j < fw; j += item_y) {
            in_val = vload4(in_off + k * ihy_str, in);
            flt_val = vload16(flt_off + k * fhy_str, flt);
            calCore(in_val, flt_val, sum);
            k++;
        }
        in_off += ihw_str;
        flt_off += fhw_str;
    }

    const int out_off = (idy * fh + idx) * fn + idz;
    vstore4(sum, out_off, out);
}
