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
#define MANGLE_NAME_IMPL(base, IOM) base##IOM
#define MANGLE_NAME(base, IOM) MANGLE_NAME_IMPL(base, IOM)

__kernel void MANGLE_NAME(col2im_, IOM)(const int iw,
    const int ih,
    const int fw,
    const int fh,
    const int pw,
    const int ph,
    const int sw,
    const int sh,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int ow,
    const int oh,
    const int bx,
    const int by,
    __global const T *in,
    __read_only image1d_t bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    const int pidx = idx + pw;
    const int pidy = idy + ph;

    int sidh_i = pidy / sh;
    int sidh_j = pidy % sh;
    int in_hx = (sidh_i < ih) ? sidh_i : (ih - 1);
    int in_hy = (sidh_i < ih) ? sidh_j : ((sidh_i - ih + 1) * sh + sidh_j);
    int in_hl = (fh - in_hy + sh - 1) / sh;
    if (in_hl > in_hx + 1) {
        in_hl = in_hx + 1;
    }

    int sidw_i = pidx / sw;
    int sidw_j = pidx % sw;
    int in_wx = (sidw_i < iw) ? sidw_i : (iw - 1);
    int in_wy = (sidw_i < iw) ? sidw_j : ((sidw_i - iw + 1) * sw + sidw_j);
    int in_wl = (fw - in_wy + sw - 1) / sw;
    if (in_wl > in_wx + 1) {
        in_wl = in_wx + 1;
    }

    int in_off_h = iw * (in_hx + ih * fw * (in_hy + idz * fh));
    int in_str_h = iw * (ih * fw * sw - 1);
    int in_off_w = in_wx + in_wy * ih * iw;
    int in_str_w = ih * iw * sw - 1;
    T4 sum = READ_IMAGE(bias, sampler, idz);

    for (int i = 0; i < in_hl; i++) {
        for (int j = 0; j < in_wl; j++) {
            sum += vload4(in_off_h + in_off_w + j * in_str_w, in);
        }
        in_off_h += in_str_h;
    }
    STORE_MEM_V4_COMMON(sum, idx, idy, idz, ow_str, oh_str, o_off, out);
}
