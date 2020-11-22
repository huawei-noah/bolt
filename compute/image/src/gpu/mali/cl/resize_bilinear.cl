// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void resize_bilinear(const int ih,
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

    if (idx >= oh || idy >= ow) {
        return;
    }

    float2 posi;
    float2 ratio;
    ratio.x = ratioh;
    ratio.y = ratiow;

    posi.x = (float)idx * ratio.x;
    posi.y = (float)idy * ratio.y;

    int4 tblr;
    tblr.x = max(0, (int)floor(posi.y));  // T
    tblr.y = min(tblr.x + 1, iw - 1);     // B
    tblr.z = max(0, (int)floor(posi.x));  // L
    tblr.w = min(tblr.z + 1, ih - 1);     // R

    int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;
    int4 in_off;
    in_off.x = (idz * iw_str + tblr.x + iw_off) * ih_str + tblr.z + ih_off;  // TL_off
    in_off.y = (idz * iw_str + tblr.x + iw_off) * ih_str + tblr.w + ih_off;  // TR_off
    in_off.z = (idz * iw_str + tblr.y + iw_off) * ih_str + tblr.z + ih_off;  // BL_off
    in_off.w = (idz * iw_str + tblr.y + iw_off) * ih_str + tblr.w + ih_off;  // BR_off

    T4 val_TL, val_TR, val_BL, val_BR;
    val_TL = vload4(0, input + (in_off.x << 2));
    val_TR = vload4(0, input + (in_off.y << 2));
    val_BL = vload4(0, input + (in_off.z << 2));
    val_BR = vload4(0, input + (in_off.w << 2));
    float dif1 = posi.x - (float)tblr.z;  // C-L
    float dif2 = posi.y - (float)tblr.x;  // C-T

    T4 top = mad((val_TR - val_TL), dif1, val_TL);
    T4 bottom = mad((val_BR - val_BL), dif1, val_BL);
    T4 out = mad((bottom - top), dif2, top);
    vstore4(out, 0, output + (out_off << 2));
}
