R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"

__kernel void selectForward(const int h,
    const int w,
    const int c,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const int bx,
    const int by,
    const int ih0_str,
    const int iw0_str,
    const int ih0_off,
    const int iw0_off,
    __global const T *condition,
    const int ih1_str,
    const int iw1_str,
    const int ih1_off,
    const int iw1_off,
    __global const T *in1,
    const int ih2_str,
    const int iw2_str,
    const int ih2_off,
    const int iw2_off,
    __global const T *in2,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    char ew = 0;
    ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);

    LOAD_VAL_T4(ew, idx, idy, idz, ih0_str, iw0_str, ih0_off, iw0_off, condition, cond);
    LOAD_VAL_T4(ew, idx, idy, idz, ih1_str, iw1_str, ih1_off, iw1_off, in1, x);
    LOAD_VAL_T4(ew, idx, idy, idz, ih2_str, iw2_str, ih2_off, iw2_off, in2, y);

	x = cond * x + (1.0f - cond) * y;

    STORE_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, out, x);
}

__kernel void selectBackward(const int h,
    const int w,
    const int c,
    const int index,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const int bx,
    const int by,
    const int ih0_str,
    const int iw0_str,
    const int ih0_off,
    const int iw0_off,
    __global const T *condition,
    const int ih1_str,
    const int iw1_str,
    const int ih1_off,
    const int iw1_off,
    __global const T *deltas,
    __global T *prevLayerDelta)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    char ew = 0;
    ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
    LOAD_VAL_T4(ew, idx, idy, idz, ih0_str, iw0_str, ih0_off, iw0_off, condition, cond);
    LOAD_VAL_T4(ew, idx, idy, idz, ih1_str, iw1_str, ih1_off, iw1_off, deltas, del);
    LOAD_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, prevLayerDelta, prev);
    
	prev += (index == 0) ? cond * del : (1.0f - cond) * del;

    STORE_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, prevLayerDelta, prev);
}
)"