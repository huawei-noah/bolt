// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void col2im(const int ih,
    const int iw,
    const int ic,
    const int kw,
    const int kh,
    const int pw,
    const int ph,
    const int sw,
    const int sh,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const int oh,
    const int ow,
    const int bx,
    __global const T *bias,
    __global const T *in,
    __global T *out)
{
    const int index = get_global_id(0);
    if (index >= bx) {
        return;
    }
    const int idx = index % oh;
    const int idy = (index % (ow * oh)) / oh;
    const int idz = index / (ow * oh);

    const int pidx = idx + ph;
    const int pidy = idy + pw;

    int sidw_i = pidy / sw;
    int sidw_j = pidy % sw;
    int in_wx = (sidw_i < iw) ? sidw_i : (iw - 1);
    int in_wy = (sidw_i < iw) ? sidw_j : ((sidw_i - iw + 1) * sw + sidw_j);
    int in_wl = (kw - in_wy + sw - 1) / sw;
    if (in_wl > in_wx + 1) {
        in_wl = in_wx + 1;
    }

    int sidh_i = pidx / sh;
    int sidh_j = pidx % sh;
    int in_hx = (sidh_i < ih) ? sidh_i : (ih - 1);
    int in_hy = (sidh_i < ih) ? sidh_j : ((sidh_i - ih + 1) * sh + sidh_j);
    int in_hl = (kh - in_hy + sh - 1) / sh;
    if (in_hl > in_hx + 1) {
        in_hl = in_hx + 1;
    }

    int in_off_w = ih * (in_wx + iw * kh * (in_wy + idz * kw));
    int in_str_w = ih * (iw * kh * sh - 1);
    int in_off_h = in_hx + in_hy * ih * iw;
    int in_str_h = ih * iw * sh - 1;
    T4 sum = vload4(idz, bias);

    for (int i = 0; i < in_wl; i++) {
        for (int j = 0; j < in_hl; j++) {
            sum += vload4(in_off_w + in_off_h + j * in_str_h, in);
        }
        in_off_w += in_str_w;
    }

    int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;
    vstore4(sum, out_off, out);
}
