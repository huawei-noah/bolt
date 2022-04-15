R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void reduction_default_backward(const int bx,
    const int by,
    const int od_str,
    const int oh_str,
    const int ow_str,
    const int out_off,
    const T divisor,
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
    ew = ((idx << 2) + 4 <= ow_str) ? 4 : (ow_str & 3);
    int off = (idz * oh_str + idy) * ow_str + (idx << 2) + out_off;
    T4 val = vload4(0, prevLayerDelta + off);
    val.x += deltas[0] / divisor;
    val.y += deltas[0] / divisor;
    val.z += deltas[0] / divisor;
    val.w += deltas[0] / divisor;
    if (ew == 4) {
        vstore4(val, 0, prevLayerDelta + off);
    }
    else {
        if (ew == 1) {
            prevLayerDelta[off] = val.x;
        } if (ew == 2) {
            vstore2((T2)(val.x, val.y), 0, prevLayerDelta + off);
        } if (ew == 3) {
            vstore3((T3)(val.x, val.y, val.z), 0, prevLayerDelta + off);
        }
    }
}

)"