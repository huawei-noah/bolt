// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"

__kernel void KERNEL_NAME(__global const uchar *_srcptr,
    __global T *_dstptr,
    const int src_chw,
    const int src_row,
    const int src_step,
    const int dst_n,
    const int dst_h,
    const int dst_w,
    const int dst_chw,
    const int dst_hw)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= dst_w || y >= dst_h || z >= dst_n) {
        return;
    }

    __global const uchar *srcptr = _srcptr + z * src_chw;
    __global T *dstptr = _dstptr + z * dst_chw;

    __global const uchar *ysrc = srcptr + mad24(y << 1, src_step, (x << 1));
    __global const uchar *usrc = srcptr + mad24(src_row + y, src_step, (x << 1));

    T3 r = (T3)(ysrc[1], usrc[0], usrc[1]);
    T3 scale = (T3)0.003921;
    r *= scale;

    int index = mad24(y, dst_w, x);
    dstptr[index] = r.s0;
    dstptr[index + dst_hw] = r.s1;
    dstptr[index + (dst_hw << 1)] = r.s2;
}
