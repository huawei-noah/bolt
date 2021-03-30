// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void embedding(const int iw_str,
    const int iw_off,
    const int ih_off,
    const int fw_str,
    const int fw_off,
    const int fh_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
    const int ow,
    const int bx,
    const int by,
    __global const unsigned int *input,
    __global const T *weight,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 val = 0;
    int in_off = (idz + ih_off) * iw_str + (idy + iw_off);
    unsigned int index = input[in_off];
    const int wei_off = (index + fh_off) * fw_str + (idx << 2) + fw_off;
    uchar rw = ((idx << 2) + 4 <= ow) ? 4 : (ow & 3);
    if (rw == 4) {
        val = vload4(0, weight + wei_off);
    } else {
        if (rw == 1) {
            val.x = weight[wei_off];
        }
        if (rw == 2) {
            val.xy = vload2(0, weight + wei_off);
        }
        if (rw == 3) {
            val.xyz = vload3(0, weight + wei_off);
        }
    }
    const int out_off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off;
    if (rw == 4) {
        vstore4(val, 0, output + out_off);
    } else {
        if (rw == 1) {
            output[out_off] = val.x;
        }
        if (rw == 2) {
            vstore2(val.xy, 0, output + out_off);
        }
        if (rw == 3) {
            vstore3(val.xyz, 0, output + out_off);
        }
    }
}
