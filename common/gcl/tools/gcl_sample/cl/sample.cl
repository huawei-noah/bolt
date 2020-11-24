// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void sample(const int iw_str,
    const int iwh_str,
    const int fc,
    const int flt_str,
    const int ow_str,
    const int oh_str,
    const int bx,
    const int by,
    __global const T *in,
    __global const T *flt,
    __global const T *bias,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T3 flt_val;
    T3 in_val;
    T out_val;

    out_val = bias[idz];
    int flt_off = idz * flt_str;
    int in_off = idy * iw_str + idx;
    for (int i = 0; i < fc; ++i) {
        for (uchar j = 0; j < 3; ++j) {
            flt_val = vload3(0, flt + flt_off + j * 3);
            in_val = vload3(0, in + in_off + j * iw_str);
            out_val += flt_val.x * in_val.x;
            out_val += flt_val.y * in_val.y;
            out_val += flt_val.z * in_val.z;
        }
        flt_off += 9;
        in_off += iwh_str;
    }

    int out_off = (idz * oh_str + idy) * ow_str + idx;
    out[out_off] = out_val;
}
