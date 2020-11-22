// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void padding_constant(const int ih,
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
    if (idx < ph || idx >= ph + ih) {
        return;
    }
    if (idy < pw || idy >= pw + iw) {
        return;
    }

    int in_off = (idz * iw_str + idy - pw + iw_off) * ih_str + ih_off + idx - ph;
    int out_off = (idz * ow_str + idy + ow_off) * oh_str + oh_off + idx;
    T4 val;
    val = vload4(in_off, in);
    vstore4(val, out_off, out);
}
