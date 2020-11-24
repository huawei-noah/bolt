// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void mem_trans_nchw_to_nchw(const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
    const int iw,
    const int ih,
    const int ic,
    const int ow,
    const int oh,
    const int oc,
    const int offset_in,
    const int offset_out,
    const __global T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= ((ow + 3) >> 2) || idy >= oh) {
        return;
    }
    char ie = (((idx << 2) + 4) <= iw) ? 4 : (iw & 3);
    char oe = (((idx << 2) + 4) <= ow) ? 4 : (ow & 3);
    if (idx >= ((iw + 3) >> 2) || idy >= ih || idz >= ic) {
        ie = 0;
    }

    T4 val = 0;
    const int in_off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off + offset_in;
    if (ie == 4) {
        val = vload4(0, in + in_off);
    } else {
        if (ie == 1) {
            val.x = in[in_off];
        }
        if (ie == 2) {
            T2 tmp = vload2(0, in + in_off);
            val.x = tmp.x;
            val.y = tmp.y;
        }
        if (ie == 3) {
            T3 tmp = vload3(0, in + in_off);
            val.x = tmp.x;
            val.y = tmp.y;
            val.z = tmp.z;
        }
    }
    const int out_off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off + offset_out;
    if (oe == 4) {
        vstore4(val, 0, out + out_off);
    } else {
        if (oe == 1) {
            out[out_off] = val.x;
        }
        if (oe == 2) {
            vstore2(val.xy, 0, out + out_off);
        }
        if (oe == 3) {
            vstore3(val.xyz, 0, out + out_off);
        }
    }
}
