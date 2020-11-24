// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void padding_edge(const int ih,
    const int iw,
    const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh,
    const int ow,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const int ph,
    const int pb,
    const int pw,
    const int pr,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= oh || idy >= ow) {
        return;
    }

    int in_off = idz * iw_str * ih_str;
    if (idx < ph) {
        if (idy < pw) {
            in_off = in_off + iw_off * ih_str + ih_off;
        } else if (idy >= pw + iw) {
            in_off = in_off + (iw_off + iw - 1) * ih_str + ih_off;
        } else {
            in_off = in_off + (idy + iw_off - pw) * ih_str + ih_off;
        }
    } else if (idx >= ph + ih) {
        in_off = in_off + iw_off * ih_str + ih_off + ih - 1;
        if (idy < pw) {
            in_off = in_off;
        } else if (idy >= pw + iw) {
            in_off = in_off + (iw - 1) * ih_str;
        } else {
            in_off = in_off + (idy - pw) * ih_str;
        }
    } else {
        in_off = in_off + iw_off * ih_str + ih_off;
        if (idy < pw) {
            in_off = in_off + idx - ph;
        } else if (idy >= pw + iw) {
            in_off = in_off + idx - ph + (iw - 1) * ih_str;
        } else {
            in_off = in_off + (idy - pw) * ih_str + idx - ph;
        }
    }
    T4 val;
    val = vload4(in_off, in);
    int out_off = (idz * ow_str + idy + ow_off) * oh_str + oh_off + idx;
    vstore4(val, out_off, out);
}
