R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, TP, AXIS) base##TP##AXIS
#define MANGLE_NAME(base, TP, AXIS) MANGLE_NAME_IMPL(base, TP, AXIS)

__kernel void MANGLE_NAME(reduction_nchw_backward_, TP, AXIS)(const int bx,
    const int by,
    const int ih_str,
    const int iw_str,
    const int oh_str,
    const int ow_str,
    const int inputOff,
    const int outputOff,
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

    int out_off = (idz * oh_str + idy) * ow_str + (idx << 2) + outputOff;
    T4 res = vload4(0, prevLayerDelta + out_off);
    int in_off = inputOff;
#if (AXIS == 0)
    in_off += (idz * ih_str + idy) * iw_str;
    res.x += deltas[in_off] / divisor;
    res.y += deltas[in_off] / divisor;
    res.z += deltas[in_off] / divisor;
    res.w += deltas[in_off] / divisor;
#else
#if (AXIS == 1)
    in_off += (idz * ih_str) * iw_str + (idx << 2);
#elif (AXIS == 2)
    in_off += idy * iw_str + (idx << 2);
#elif (AXIS == 3)
    in_off = (idz * ih_str + idy) * iw_str + (idx << 2);
#endif
    T4 tmp = vload4(0, deltas + in_off);
    res.x += tmp.x / divisor;
    res.y += tmp.y / divisor;
    res.z += tmp.z / divisor;
    res.w += tmp.w / divisor;
#endif

    char ew = ((idx << 2) + 4 <= ow_str) ? 4 : (ow_str & 3);

    if (ew == 4) {
        vstore4(res, 0, prevLayerDelta + out_off);
    } else {
        if (ew == 3) {
            vstore3(res.xyz, 0, prevLayerDelta + out_off);
        }
        if (ew == 2) {
            vstore2(res.xy, 0, prevLayerDelta + out_off);
        }
        if (ew == 1) {
            prevLayerDelta[out_off] = res.x;
        }
    }
}
)"