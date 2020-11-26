// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void embedding(const int step,
    const int on,
    const int on_d4,
    const int oh_str,
    const int oh_off,
    const int ow_off,
    __global const unsigned int *input,
    __global const T *weight,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if (idx >= on_d4 || idy >= step) {
        return;
    }
    T4 val = 0;
    unsigned int index = input[idy];
    const int wei_off = index * on + (idx << 2);
    uchar rn = ((idx << 2) + 4 <= on) ? 0 : (on & 3);
    if (rn == 0) {
        val = vload4(0, weight + wei_off);
    } else {
        if (rn == 1) {
            val.x = weight[wei_off];
        }
        if (rn == 2) {
            T2 tmp = vload2(0, weight + wei_off);
            val.x = tmp.x;
            val.y = tmp.y;
        }
        if (rn == 3) {
            T3 tmp = vload3(0, weight + wei_off);
            val.x = tmp.x;
            val.y = tmp.y;
            val.z = tmp.z;
        }
    }
    const int out_off = (idx + ow_off) * oh_str + idy + oh_off;
    vstore4(val, out_off, output);
}
