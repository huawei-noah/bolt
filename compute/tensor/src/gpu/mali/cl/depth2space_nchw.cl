// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void depth2space_nchw_(const int blockSize,
    const int iw_str,
    const int ihw_str,
    const int ow_str,
    const int ohw_str,
    const int i_off,
    const int o_off,
    const int iw,
    const int ih,
    const int ic,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= iw || idy >= ih) {
        return;
    }
    const int idz = get_global_id(2);
    const int bs2 = blockSize * blockSize;
    const int z_group = idz / bs2;
    const int z_group_lane = idz % bs2;
    const int z_group_lane_x = z_group_lane % blockSize;
    const int z_group_lane_y = z_group_lane / blockSize;

    const int z_off = z_group * (bs2 << 2) + z_group_lane;
    int in_off = z_off * ihw_str + idy * iw_str + idx + i_off;
    T4 val = 0;
    val.x = in[in_off];
    if (z_off + bs2 < ic) {
        val.y = in[in_off + bs2 * ihw_str];
    }
    if (z_off + bs2 * 2 < ic) {
        val.z = in[in_off + bs2 * 2 * ihw_str];
    }
    if (z_off + bs2 * 3 < ic) {
        val.w = in[in_off + bs2 * 3 * ihw_str];
    }

    int out_off = idx * blockSize + z_group_lane_x + o_off;
    out_off += (idy * blockSize + z_group_lane_y) * ow_str;
    out_off += z_group * ohw_str;
    vstore4(val, out_off, out);
}
