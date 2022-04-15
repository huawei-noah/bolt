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
#define MANGLE_NAME_IMPL(base, IOM) base##IOM
#define MANGLE_NAME(base, IOM) MANGLE_NAME_IMPL(base, IOM)

__kernel void MANGLE_NAME(generate_proposals_apply_deltas_, IOM)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int iw,
    const int ih,
    const int bx,
    const int by,
    __global const T *anchor,
    __global const T *imgInfo,
    READ_ONLY_KERNEL_MEM input,
    KERNEL_MEM output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    T4 av = vload4(idz, anchor);
    av.x = av.z - av.x;
    av.y = av.w - av.y;
    int img_w = (int)imgInfo[0];
    int img_h = (int)imgInfo[1];
    int sw = img_w / iw;
    int sh = img_h / ih;

    T4 tmp;
    LOAD_MEM_V4_COMMON(tmp, idx, idy, idz, iw_str, ih_str, i_off, input);
    float4 val;
    val.x = tmp.x;
    val.y = tmp.y;
    val.z = tmp.z;
    val.w = tmp.w;

    float ctrX = idx * sw;
    float ctrY = idy * sh;
    val.x = val.x * (float)av.x + ctrX;
    val.y = val.y * (float)av.y + ctrY;
    val.z = exp(val.z) * (float)av.x;
    val.w = exp(val.w) * (float)av.y;

    T4 res;
    res.x = val.x - (float)0.5 * val.z;
    res.y = val.y - (float)0.5 * val.w;
    res.z = val.x + (float)0.5 * val.z;
    res.w = val.y + (float)0.5 * val.w;
    res.x = clamp(res.x, (T)0, (T)img_w);
    res.y = clamp(res.y, (T)0, (T)img_h);
    res.z = clamp(res.z, (T)0, (T)img_w);
    res.w = clamp(res.w, (T)0, (T)img_h);
    STORE_MEM_V4_COMMON(res, idx, idy, idz, ow_str, oh_str, o_off, output);
}
