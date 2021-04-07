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
#define MANGLE_NAME_IMPL(base, FM, AM) base##FM##AM
#define MANGLE_NAME(base, FM, AM) MANGLE_NAME_IMPL(base, FM, AM)
#define FM
#if defined(USE_NCHW)
#define FM nchw_
#endif

__kernel void MANGLE_NAME(activation_, FM, AM)(const int w,
    const int h,
    const int c,
    const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int bx,
    const int by,
    __global T *input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    T4 val = 0;
#if defined(USE_NCHW)    
    const int in_off = (idz * ih_str + idy) * iw_str + (idx << 2) + i_off;
    const int out_off = (idz * oh_str + idy) * ow_str + (idx << 2) + o_off;
    char ew = (((idx << 2) + 4) <= w) ? 4 : (w & 3);
    if (ew == 4) {
        val = vload4(0, input + in_off);
    } else {
        if (ew == 1) {
            val.x = input[in_off];
        }
        if (ew == 2) {
            val.xy = vload2(0, input + in_off);
        }
        if (ew == 3) {
            val.xyz = vload3(0, input + in_off);
        }
    }
    ACTIVATION_V4(val);
    if (ew == 4) {
        vstore4(val, 0, output + out_off);
    } else {
        if (ew == 1) {
            output[out_off] = val.x;
        }
        if (ew == 2) {
            vstore2(val.xy, 0, output + out_off);
        }
        if (ew == 3) {
            vstore3(val.xyz, 0, output + out_off);
        }
    }
#else
    const int in_off = (idz * iw_str + idy) * ih_str + idx + i_off;
    const int out_off = (idz * ow_str + idy) * oh_str + idx + o_off;
    val = vload4(in_off, input);
    ACTIVATION_V4(val);
#if defined(USE_TANH) || defined(USE_SIGMOID) || defined(USE_HSIGMOID) || defined(USE_GELU)
    char ec = (((idz << 2) + 4) <= c) ? 4 : (c & 3);
    if (ec < 2) {
        val.y = 0;
    }
    if (ec < 3) {
        val.z = 0;
    }
    if (ec < 4) {
        val.w = 0;
    }
#endif
    vstore4(val, out_off, output);
#endif
}
