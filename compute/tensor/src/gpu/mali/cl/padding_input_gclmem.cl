// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void padding_input_gclmem(const int iw,
    const int ih,
    const int pw,
    const int ph,
    const int ow,
    const int oh,
    const int in_offset,
    const int out_offset,
    const __global const T *in,
    __global T *out)
{
    int idx = get_global_id(0) << 2;
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= ow || idy >= oh) {
        return;
    }

    int in_y = idy - ph;
    int be_x = idx - pw;
    int en_x = be_x + 4;
    T4 val = 0;
    if (in_y >= 0 && in_y < ih) {
        int in_off = (idz * ih + in_y) * iw + in_offset;
        if (be_x >= 0 && en_x <= iw) {
            val = vload4(0, in + in_off + be_x);
        } else {
            if (be_x >= 0 && be_x < iw) {
                val.x = in[in_off + be_x];
            }
            if (be_x + 1 >= 0 && be_x + 1 < iw) {
                val.y = in[in_off + be_x + 1];
            }
            if (be_x + 2 >= 0 && be_x + 2 < iw) {
                val.z = in[in_off + be_x + 2];
            }
            if (be_x + 3 >= 0 && be_x + 3 < iw) {
                val.w = in[in_off + be_x + 3];
            }
        }
    }

    int out_off = (idz * oh + idy) * ow + idx + out_offset;
    if (idx + 3 >= ow) {
        out[out_off] = val.x;
        if (idx + 1 < ow) {
            out[out_off + 1] = val.y;
        }
        if (idx + 2 < ow) {
            out[out_off + 2] = val.z;
        }
    } else {
        vstore4(val, 0, out + out_off);
    }
}
