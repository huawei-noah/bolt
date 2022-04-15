R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, AC) base##AC
#define MANGLE_NAME(base, AC) MANGLE_NAME_IMPL(base, AC)

#define AC
#if defined(CALCULATE_LOC_GRAD)
#define AC loc
#endif
#if defined(CALCULATE_SCALE_GRAD)
#define AC scale
#endif

#define RAUL_PI 3.14159265358979323846

__kernel void gaussian_upsampling_distribution_forward(const int b,
    const int h,
    const int ih_str,
    const int iw_str,
    const int oh_str,
    const int ow_str,
    __global const T *values,
    __global const T *loc,
    __global const T* scale,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idy >= h || idz >= b) {
        return;
    }

    T4 l;
    T4 s;
    int in_off = (idx << 2) + idz * ih_str;
    l = vload4(0, loc + in_off);
    s = vload4(0, scale + in_off);
    l.x = exp(log(exp(-0.5f * (values[idy] - l.x) * (values[idy] - l.x) / s.x / s.x) / sqrt(2.0f * RAUL_PI * s.x * s.x)));
    l.y = exp(log(exp(-0.5f * (values[idy] - l.y) * (values[idy] - l.y) / s.y / s.y) / sqrt(2.0f * RAUL_PI * s.y * s.y)));
    l.z = exp(log(exp(-0.5f * (values[idy] - l.z) * (values[idy] - l.z) / s.z / s.z) / sqrt(2.0f * RAUL_PI * s.z * s.z)));
    l.w = exp(log(exp(-0.5f * (values[idy] - l.w) * (values[idy] - l.w) / s.w / s.w) / sqrt(2.0f * RAUL_PI * s.w * s.w)));
    int out_off = (idx << 2) + ow_str * idy + idz * (ow_str * oh_str);
    char ew = ((idx << 2) + 4 <= ow_str) ? 4 : (ow_str & 3);
    if (ew == 4) {
        vstore4(l, 0, output + out_off);
    } else {
        if (ew == 1)
            output[out_off] = l.x;
        if (ew == 2) {
            vstore2((T2)(l.x, l.y), 0, output + out_off);
        }
        if (ew == 3) {
            vstore3((T3)(l.x, l.y, l.z), 0, output + out_off);
        }
    }
}

__kernel void MANGLE_NAME(gaussian_upsampling_distribution_backward_, AC)(const int w,
    const int b,
    const int w_str,
    const int vLen,
    __global const T *values,
    __global const T *loc,
    __global const T* scale,
    __global const T* deltas,
    __global T *prevLayerDelta)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= w || idy >= b) {
        return;
    }

    T4 l;
    T4 s;
    T4 grad;
    T4 del;
    int del_off = 0;
    int in_off = (idx << 2) + idy * w_str;
    char ew = ((idx << 2) + 4 <= w_str) ? 4 : (w_str & 3);
    l = vload4(0, loc + in_off);
    s = vload4(0, scale + in_off);
    grad = vload4(0, prevLayerDelta + in_off);
    for (int i = 0; i < vLen; ++i)
    {
        del_off = (idx << 2) + w_str * i + idy * (w_str * vLen);
        del = vload4(0, deltas + del_off);
#if defined(CALCULATE_LOC_GRAD)
        grad.x += del.x * 0.398942f * (values[i] - l.x) * exp(-0.5f * (values[i] - l.x) * (values[i] - l.x) / s.x / s.x) / s.x / s.x / s.x;
        grad.y += del.y * 0.398942f * (values[i] - l.y) * exp(-0.5f * (values[i] - l.y) * (values[i] - l.y) / s.y / s.y) / s.y / s.y / s.y;
        grad.z += del.z * 0.398942f * (values[i] - l.z) * exp(-0.5f * (values[i] - l.z) * (values[i] - l.z) / s.z / s.z) / s.z / s.z / s.z;
        grad.w += del.w * 0.398942f * (values[i] - l.w) * exp(-0.5f * (values[i] - l.w) * (values[i] - l.w) / s.w / s.w) / s.w / s.w / s.w;
#else
        grad.x += del.x * (0.398942f * (values[i] - l.x) * (values[i] - l.x) / s.x / s.x - 1.0f / sqrt(2.0f * RAUL_PI)) * exp(-0.5f * (values[i] - l.x) * (values[i] - l.x) / s.x / s.x) / s.x / s.x;
        grad.y += del.y * (0.398942f * (values[i] - l.y) * (values[i] - l.y) / s.y / s.y - 1.0f / sqrt(2.0f * RAUL_PI)) * exp(-0.5f * (values[i] - l.y) * (values[i] - l.y) / s.y / s.y) / s.y / s.y;
        grad.z += del.z * (0.398942f * (values[i] - l.z) * (values[i] - l.z) / s.z / s.z - 1.0f / sqrt(2.0f * RAUL_PI)) * exp(-0.5f * (values[i] - l.z) * (values[i] - l.z) / s.z / s.z) / s.z / s.z;
        grad.w += del.w * (0.398942f * (values[i] - l.w) * (values[i] - l.w) / s.w / s.w - 1.0f / sqrt(2.0f * RAUL_PI)) * exp(-0.5f * (values[i] - l.w) * (values[i] - l.w) / s.w / s.w) / s.w / s.w;
#endif
    }
    if (ew == 4) {
        vstore4(grad, 0, prevLayerDelta + in_off);
    } else {
        if (ew == 1)
            prevLayerDelta[in_off] = grad.x;
        if (ew == 2) {
            vstore2((T2)(grad.x, grad.y), 0, prevLayerDelta + in_off);
        }
        if (ew == 3) {
            vstore3((T3)(grad.x, grad.y, grad.z), 0, prevLayerDelta + in_off);
        }
    }
}

)"