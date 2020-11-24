// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void resize_bilinear_nchw(const int ih,
    const int ih_str,
    const int ih_off,
    const int iw,
    const int iw_str,
    const int iw_off,
    const int oh,
    const int oh_str,
    const int oh_off,
    const int ow,
    const int ow_str,
    const int ow_off,
    const float ratioh,
    const float ratiow,
    __global T *input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);

    if (idx >= ow || idy >= oh) {
        return;
    }

    float2 posi;
    float2 ratio;
    ratio.x = ratiow;
    ratio.y = ratioh;

    posi.x = (float)idx * ratio.x;
    posi.y = (float)idy * ratio.y;

    int4 tblr;
    tblr.x = max(0, (int)floor(posi.x));  // L
    tblr.y = min(tblr.x + 1, iw - 1);     // R
    tblr.z = max(0, (int)floor(posi.y));  // T
    tblr.w = min(tblr.z + 1, ih - 1);     // B

    int4 in_off;
    in_off.x = (idz * ih_str + tblr.z + ih_off) * iw_str + tblr.x + iw_off;  // TL_off
    in_off.y = (idz * ih_str + tblr.z + ih_off) * iw_str + tblr.y + iw_off;  // TR_off
    in_off.z = (idz * ih_str + tblr.w + ih_off) * iw_str + tblr.x + iw_off;  // BL_off
    in_off.w = (idz * ih_str + tblr.w + ih_off) * iw_str + tblr.y + iw_off;  // BR_off

    T val_TL, val_TR, val_BL, val_BR;
    val_TL = input[in_off.x];
    val_TR = input[in_off.y];
    val_BL = input[in_off.z];
    val_BR = input[in_off.w];
    float dif1 = posi.x - (float)tblr.x;  // C-L
    float dif2 = posi.y - (float)tblr.z;  // C-T

    float top = mad((float)(val_TR - val_TL), dif1, (float)val_TL);
    float bottom = mad((float)(val_BR - val_BL), dif1, (float)val_BL);
    T out = mad((bottom - top), dif2, top);
    int out_off = (idz * oh_str + idy + oh_off) * ow_str + idx + ow_off;
    output[out_off] = out;
}
