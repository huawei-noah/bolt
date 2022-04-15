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

__kernel void klDivLossForward(const int h,
    const int w,
    const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    __global T *input,
    __global T *target,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= ((w + 3) >> 2) || idy >= h) {
        return;
    }

    char ew = 0;
    ew = ((idx << 2) + 4 <= iw_str) ? 4 : (iw_str & 3);
    LOAD_VAL_T4(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, input, data);
    LOAD_VAL_T4(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, target, labels);
    data.x = (labels.x > 0.0f) * labels.x * (log(labels.x) - data.x);
	data.y = (labels.y > 0.0f) * labels.y * (log(labels.y) - data.y);
	data.z = (labels.z > 0.0f) * labels.z * (log(labels.z) - data.z);
	data.w = (labels.w > 0.0f) * labels.w * (log(labels.w) - data.w);
    STORE_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, output, data);
}

__kernel void klDivLossBackward(const int h,
    const int w,
    const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    __global T *target,
    __global T *deltas,
    __global T *prevLayerDelta)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= ((w + 3) >> 2) || idy >= h) {
        return;
    }

    char ew = 0;
    ew = ((idx << 2) + 4 <= iw_str) ? 4 : (iw_str & 3);
    LOAD_VAL_T4(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, deltas, del);
    LOAD_VAL_T4(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, target, labels);
    labels = -labels * del;
    LOAD_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, prevLayerDelta, prev);
    prev += labels;
    STORE_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, prevLayerDelta, prev);
}
)"