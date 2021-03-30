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
padding_nchw_constant
#elif defined(USE_EDGE)
padding_nchw_edge
#elif defined(USE_REFLECT)
padding_nchw_reflect
#elif defined(USE_SYMMETRIC)
padding_nchw_symmetric
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
    idx = idx << 2;
    int in_y = idy - pt;
    int in_x_be = idx - pl;
    int in_x_end = in_x_be + 4;
#if defined(USE_CONSTANT)
    if (in_y >= 0 && in_y < ih) {
        int in_off = (idz * ih_str + in_y + ih_off) * iw_str + in_x_be + iw_off + in_offset;
        if (in_x_be >= 0 && in_x_end <= iw) {
            val = vload4(0, in + in_off);
        } else {
            if (in_x_be >= 0 && in_x_be < iw) {
                val.x = in[in_off];
            }
            if (in_x_be + 1 >= 0 && in_x_be + 1 < iw) {
                val.y = in[in_off + 1];
            }
            if (in_x_be + 2 >= 0 && in_x_be + 2 < iw) {
                val.z = in[in_off + 2];
            }
            if (in_x_be + 3 >= 0 && in_x_be + 3 < iw) {
                val.w = in[in_off + 3];
            }
        }
    }
#else
#if defined(USE_EDGE)
    int pdy = clamp(in_y, 0, ih - 1);
    int in_off = (idz * ih_str + pdy + ih_off) * iw_str + iw_off + in_offset;
    if (in_x_be >= 0 && in_x_end <= iw) {
        val = vload4(0, in + in_off + in_x_be);
    } else {
        int4 pdx;
        pdx.x = clamp(in_x_be, 0, iw - 1);
        pdx.y = clamp(in_x_be + 1, 0, iw - 1);
        pdx.z = clamp(in_x_be + 2, 0, iw - 1);
        pdx.w = clamp(in_x_be + 3, 0, iw - 1);
        val.x = in[in_off + pdx.x];
        val.y = in[in_off + pdx.y];
        val.z = in[in_off + pdx.z];
        val.w = in[in_off + pdx.w];
    }
#else
#if defined(USE_REFLECT)
    int ht = -in_y;
    int hb = ih * 2 - 2 - in_y;
    int wl = -in_x_be;
    int wr = iw * 2 - 2 - in_x_be;
#elif defined(USE_SYMMETRIC)
    int ht = -in_y - 1;
    int hb = ih * 2 - 1 - in_y;
    int wl = -in_x_be - 1;
    int wr = iw * 2 - 1 - in_x_be;
#endif
    int pdy = clamp(in_y, ht, hb);
    int in_off = (idz * ih_str + pdy + ih_off) * iw_str + iw_off + in_offset;
    if (in_x_be >= 0 && in_x_end <= iw) {
        val = vload4(0, in + in_off + in_x_be);
    } else {
        int4 pdx;
        pdx.x = clamp(in_x_be, wl, wr);
        pdx.y = clamp(in_x_be + 1, wl - 1, wr - 1);
        pdx.z = clamp(in_x_be + 2, wl - 2, wr - 2);
        pdx.w = clamp(in_x_be + 3, wl - 3, wr - 3);
        val.x = in[in_off + pdx.x];
        val.y = in[in_off + pdx.y];
        val.z = in[in_off + pdx.z];
        val.w = in[in_off + pdx.w];
    }
#endif
#endif
    int out_off = (idz * oh_str + idy + oh_off) * ow_str + idx + ow_off + out_offset;
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
