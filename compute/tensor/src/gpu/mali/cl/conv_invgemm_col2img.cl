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
#define MANGLE_NAME_IMPL(base, IOM, AM) base##IOM##AM
#define MANGLE_NAME(base, IOM, AM) MANGLE_NAME_IMPL(base, IOM, AM)

__kernel void MANGLE_NAME(conv_invgemm_col2img_, IOM, AM)(const int iw,
    const int ih,
    const int fw,
    const int fh,
    const int pw,
    const int ph,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int oc,
    const int bx,
    const int by,
    __global const T *in,
    __read_only image1d_t bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const ushort c_pitch = (oc + 3) >> 2;
    const int idc = idz % c_pitch;
    if (idx >= bx || idy >= by) {
        return;
    }

    const int pidx = idx + pw;
    const int pidy = idy + ph;

    int in_hx = (pidy < ih) ? pidy : (ih - 1);
    int in_hy = (pidy < ih) ? 0 : (pidy - ih + 1);
    int in_hl = fh - in_hy;
    if (in_hl > in_hx + 1) {
        in_hl = in_hx + 1;
    }
    if (pidy < 0) {
        in_hl = 0;
    }

    int in_wx = (pidx < iw) ? pidx : (iw - 1);
    int in_wy = (pidx < iw) ? 0 : (pidx - iw + 1);
    int in_wl = fw - in_wy;
    if (in_wl > in_wx + 1) {
        in_wl = in_wx + 1;
    }
    if (pidx < 0) {
        in_wl = 0;
    }

    int in_off_h = iw * (in_hx + ih * fw * (in_hy + idz * fh));
    int in_str_h = iw * (ih * fw - 1);
    int in_off_w = in_wx + in_wy * ih * iw;
    int in_str_w = ih * iw - 1;
    T4 sum = READ_IMAGE(bias, sampler, idc);

    for (int i = 0; i < in_hl; i++) {
        for (int j = 0; j < in_wl; j++) {
            sum += vload4(in_off_h + in_off_w + j * in_str_w, in);
        }
        in_off_h += in_str_h;
    }
    ACTIVATION_V4(sum);
    STORE_MEM_V4_COMMON(sum, idx, idy, idz, ow_str, oh_str, o_off, out);
}
