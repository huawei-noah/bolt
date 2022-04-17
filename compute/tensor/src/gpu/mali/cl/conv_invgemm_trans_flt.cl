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
#define MANGLE_NAME_IMPL(base, K) base##K
#define MANGLE_NAME(base, K) MANGLE_NAME_IMPL(base, K)

__kernel void MANGLE_NAME(conv_invgemm_trans_flt_, K)(const int fw,
    const int fh,
    const int fwh,
    const int fc,
    const int fn,
    __global const T *fltdata,
    __global T *flt)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    int iy = idy << 2;
    const int flt_off = (idz * fc + iy) * fwh + idx;
    T4 val = 0;
    val.x = fltdata[flt_off];
    if (iy + 1 < fc) {
        val.y = fltdata[flt_off + fwh];
    }
    if (iy + 2 < fc) {
        val.z = fltdata[flt_off + fwh * 2];
    }
    if (iy + 3 < fc) {
        val.w = fltdata[flt_off + fwh * 3];
    }
    const int bc = (fc + 3) >> 2;
    int ox = idz & 3;
    int oy = idy;
    int oz = (idz >> 2) * fwh + fwh - 1 - idx;
    int K_pitch = K >> 2;
    ox = ox + (oz % K_pitch) * 4;
    oz = oz / K_pitch;

    int out_off = (oz * bc + oy) * K + ox;
    vstore4(val, out_off, flt);
}
