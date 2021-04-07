// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void
#if defined(USE_CONSTANT)
padding_constant
#elif defined(USE_EDGE)
padding_edge
#elif defined(USE_REFLECT)
padding_reflect
#elif defined(USE_SYMMETRIC)
padding_symmetric
#endif
    (const int iw_str,
        const int ih_str,
        const int iw_off,
        const int ih_off,
        const int ow_str,
        const int oh_str,
        const int ow_off,
        const int oh_off,
        const int iw,
        const int ih,
        const int ow,
        const int oh,
        const int pt,
        const int pb,
        const int pl,
        const int pr,
        const int in_offset,
        const int out_offset,
        const int bx,
        const int by,
        const __global const T *in,
        __global T *out)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 val = 0;
    int in_y = idy - pt;
    int in_x = idx - pl;
#if defined(USE_CONSTANT)
    if (in_y >= 0 && in_y < ih && in_x >= 0 && in_x < iw) {
        int in_off = (idz * iw_str + in_y + iw_off) * ih_str + in_x + ih_off;
        val = vload4(in_off, in + in_offset);
    }
#else
#if defined(USE_EDGE)
    int ht = 0;
    int hb = ih - 1;
    int wl = 0;
    int wr = iw - 1;
#elif defined(USE_REFLECT)
    int ht = -in_y;
    int hb = ih * 2 - 2 - in_y;
    int wl = -in_x;
    int wr = iw * 2 - 2 - in_x;
#elif defined(USE_SYMMETRIC)
    int ht = -in_y - 1;
    int hb = ih * 2 - 1 - in_y;
    int wl = -in_x - 1;
    int wr = iw * 2 - 1 - in_x;
#endif
    int pdy = clamp(in_y, ht, hb);
    int pdx = clamp(in_x, wl, wr);
    int in_off = (idz * iw_str + pdy + iw_off) * ih_str + pdx + ih_off;
    val = vload4(in_off, in + in_offset);
#endif
    int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;
    vstore4(val, out_off, out + out_offset);
}
