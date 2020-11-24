// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void mem_trans_ncwhc4_to_mtk(const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int k,
    const int offset,
    const int bx,
    const int by,
    __global T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    uchar ek = ((idy << 2) + 4 <= k) ? 4 : (k & 3);
    const int in_off = (idy * iw_str + iw_off) * ih_str + idx + ih_off;
    T4 val = vload4(in_off, in);
    const int out_off = idx * k + (idy << 2) + offset;
    if (ek == 4) {
        vstore4(val, 0, out + out_off);
    } else {
        if (ek == 1) {
            out[out_off] = val.x;
        }
        if (ek == 2) {
            vstore2((T2)(val.x, val.y), 0, out + out_off);
        }
        if (ek == 3) {
            vstore3((T3)(val.x, val.y, val.z), 0, out + out_off);
        }
    }
}
