// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void softmax_nchw_c(const int c,
    const int iw_str,
    const int ihw_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int ohw_str,
    const int ow_off,
    const int oh_off,
    const int ow,
    const int bx,
    const int by,
    __global T *in,
    __global T *out)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    int ew = ((idx << 2) + 4 <= ow) ? 4 : (ow & 3);

    float4 maxval = (float4)(-FLT_MAX);
    float4 tmp;
    T4 val;
    int index = (idy + ih_off) * iw_str + (idx << 2) + iw_off;
    for (int i = 0; i < c; i++) {
        val = vload4(0, in + index + i * ihw_str);
        tmp.x = (float)val.x;
        tmp.y = (float)val.y;
        tmp.z = (float)val.z;
        tmp.w = (float)val.w;
        maxval = fmax(maxval, tmp);
    }

    float4 sumexp = 0;
    for (int i = 0; i < c; i++) {
        val = vload4(0, in + index + i * ihw_str);
        sumexp.x += exp((float)val.x - maxval.x);
        sumexp.y += exp((float)val.y - maxval.y);
        sumexp.z += exp((float)val.z - maxval.z);
        sumexp.w += exp((float)val.w - maxval.w);
    }

    sumexp.x = 1.0 / sumexp.x;
    sumexp.y = 1.0 / sumexp.y;
    sumexp.z = 1.0 / sumexp.z;
    sumexp.w = 1.0 / sumexp.w;
    int out_off = (idy + oh_off) * ow_str + (idx << 2) + ow_off;
    if (ew == 4) {
        for (int i = 0; i < c; i++) {
            val = vload4(0, in + index + i * ihw_str);
            val.x = exp((float)val.x - maxval.x) * sumexp.x;
            val.y = exp((float)val.y - maxval.y) * sumexp.y;
            val.z = exp((float)val.z - maxval.z) * sumexp.z;
            val.w = exp((float)val.w - maxval.w) * sumexp.w;
            vstore4(val, 0, out + out_off + i * ohw_str);
        }
    } else {
        for (int i = 0; i < c; i++) {
            val = vload4(0, in + index + i * ihw_str);
            val.x = exp((float)val.x - maxval.x) * sumexp.x;
            val.y = exp((float)val.y - maxval.y) * sumexp.y;
            val.z = exp((float)val.z - maxval.z) * sumexp.z;
            if (ew < 2) {
                val.y = 0;
            }
            if (ew < 3) {
                val.z = 0;
            }
            val.w = 0;
            vstore4(val, 0, out + out_off + i * ohw_str);
        }
    }
}
