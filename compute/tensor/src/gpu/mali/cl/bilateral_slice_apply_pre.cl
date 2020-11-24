// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void bilateral_slice_apply_pre(const int gh,
    const int gc,
    const int gcw,
    const int bx,
    const int bw,
    const float scale_y,
    global const T *grid,
    global T *gridTran)
{
    const int idx = get_global_id(0);  // dep * coe / 4
    const int idw = get_global_id(1);  // gw
    const int idh = get_global_id(2);  // H
    if (idx >= bx || idw >= bw) {
        return;
    }
    char j = 1;

    T2 wy;
    T gy = (idh + (T)0.5) * (T)scale_y;
    char fy = floor(gy - (T)0.5);
    char y_ = fy;
    if (fy < 0) {
        y_ = 0;
        j = 0;
    }
    if (fy == gh - 1) {
        j = 0;
    }
    wy.x = (T)1 - fabs(fy + (T)0.5 - gy);
    wy.y = (T)1 - fabs(fy + (T)1.5 - gy);

    int grid_off = y_ * gcw + idw * gc + (idx << 2);
    T4 val0;
    T4 val1;
    T4 res;
    val0 = vload4(0, grid + grid_off);
    val1 = (j == 0) ? val0 : vload4(0, grid + grid_off + gcw);
    res = wy.x * val0 + wy.y * val1;

    int gridTran_off = idh * gcw + idw * gc + (idx << 2);
    vstore4(res, 0, gridTran + gridTran_off);
}
