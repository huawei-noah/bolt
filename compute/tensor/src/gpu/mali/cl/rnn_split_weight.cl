// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void rnn_split_weight_(const int xDim,
    const int hDim,
    const int bx,
    const int by,
    __global const T *weight,
    __global T *gemmWeight,
    __global T *gemvWeight)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    int xb = (xDim + 3) >> 2;
    int hb = (hDim + 3) >> 2;
    T4 val = 0;
    int off = idy * (xDim + hDim);
    if (idx < xb) {
        char xr = ((idx << 2) + 4 <= xDim) ? 4 : (xDim & 3);
        int gemmOff = idy * xDim;
        if (xr == 4) {
            val = vload4(idx, weight + off);
            vstore4(val, idx, gemmWeight + gemmOff);
        } else if (xr == 3) {
            val.s012 = vload3(0, weight + (idx << 2) + off);
            vstore3(val.s012, 0, gemmWeight + (idx << 2) + gemmOff);
        } else if (xr == 2) {
            val.s01 = vload2(0, weight + (idx << 2) + off);
            vstore2(val.s01, 0, gemmWeight + (idx << 2) + gemmOff);
        } else if (xr == 1) {
            val.s0 = weight[(idx << 2) + off];
            gemmWeight[(idx << 2) + gemmOff] = val.s0;
        }
    } else if (idx < xb + hb) {
        idx = idx - xb;
        char hr = ((idx << 2) + 4 <= hDim) ? 4 : (hDim & 3);
        off += xDim;
        int gemvOff = idy * hDim;
        if (hr == 4) {
            val = vload4(idx, weight + off);
            vstore4(val, idx, gemvWeight + gemvOff);
        } else if (hr == 3) {
            val.s012 = vload3(0, weight + (idx << 2) + off);
            vstore3(val.s012, 0, gemvWeight + (idx << 2) + gemvOff);
        } else if (hr == 2) {
            val.s01 = vload2(0, weight + (idx << 2) + off);
            vstore2(val.s01, 0, gemvWeight + (idx << 2) + gemvOff);
        } else if (hr == 1) {
            val.s0 = weight[(idx << 2) + off];
            gemvWeight[(idx << 2) + gemvOff] = val.s0;
        }
    }
}
