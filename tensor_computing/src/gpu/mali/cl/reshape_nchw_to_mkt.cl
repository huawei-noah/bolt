// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




__kernel void reshape_nchw_to_mkt(const int iw_str, ih_str, const int iw_off, const int ih_off, const int ih, const int k, const int oh_str, const int ow_str, const int oh_off, 
    const int ow_off, const int bx, const int by, __global const T* in, __global T* out) {
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if(idx >= bx || idy >= by) return;
    T4 val = 0;
    int idk = (idy << 2);
    int ix = idx;
    int4 iy;
    int4 iz;
    iy.s0 =  idk % ih;
    iy.s1 = (idk + 1) % ih;
    iy.s2 = (idk + 2) % ih;
    iy.s3 = (idk + 3) % ih;
    iz.s0 =  idk / ih;
    iz.s1 = (idk + 1) / ih;
    iz.s2 = (idk + 2) / ih;
    iz.s3 = (idk + 3) / ih;
                    val.x = in[(iz.s0 * ih_str + iy.s0 + ih_off) * iw_str + ix + iw_off];
    if(idk + 1 < k) val.y = in[(iz.s1 * ih_str + iy.s1 + ih_off) * iw_str + ix + iw_off];
    if(idk + 2 < k) val.z = in[(iz.s2 * ih_str + iy.s2 + ih_off) * iw_str + ix + iw_off];
    if(idk + 3 < k) val.w = in[(iz.s3 * ih_str + iy.s3 + ih_off) * iw_str + ix + iw_off];
    const int out_off = (idy * ow_str + ow_off) * oh_str + idx + oh_off;
    vstore4(val, out_off, out);
}
