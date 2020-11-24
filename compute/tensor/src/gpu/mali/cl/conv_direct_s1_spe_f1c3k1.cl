// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, EW) base##EW
#define MANGLE_NAME(base, EW) MANGLE_NAME_IMPL(base, EW)

__kernel void MANGLE_NAME(conv_direct_s1_spe_f1c3k1_, EW)(const int iw_str,
    const int ow_str,
    const int ow_off,
    const int oh_off,
    const int ow_d2,
    const int bx,
    const int by,
    __global const T *in,
    __global const T *flt,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 flt_val;
    T8 in_val;
    T2 out_val;
    flt_val = vload4(0, flt);
    out_val.x = flt_val.w;
    out_val.y = flt_val.w;
    int in_off = (idy * iw_str + (idx << 1)) * 3;

    in_val = vload8(0, in + in_off);
    out_val.x += in_val.s0 * flt_val.x + in_val.s1 * flt_val.y + in_val.s2 * flt_val.z;
    out_val.y += in_val.s3 * flt_val.x + in_val.s4 * flt_val.y + in_val.s5 * flt_val.z;

    int out_off = (idy + oh_off) * ow_str + (idx << 1) + ow_off;
#if (EW == 0)
    vstore2(out_val, 0, out + out_off);
#elif (EW == 1)
    if (idx < ow_d2) {
        vstore2(out_val, 0, out + out_off);
    } else {
        out[out_off] = out_val.x;
    }
#endif
}
