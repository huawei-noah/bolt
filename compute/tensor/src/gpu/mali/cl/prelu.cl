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
#define MANGLE_NAME_IMPL(base, IOM, FM, AXIS, MD) base##IOM##FM##AXIS##MD
#define MANGLE_NAME(base, IOM, FM, AXIS, MD) MANGLE_NAME_IMPL(base, IOM, FM, AXIS, MD)
#define FM
#if defined(USE_NCHW)
#define FM nchw_
#endif

#define AXIS
#if defined(RELU_ON_AXIS_W)
#define AXIS w_
#elif defined(RELU_ON_AXIS_H)
#define AXIS h_
#endif

#define MD
#if defined(USE_PROPAGATE_DOWN)
#define MD propagate_down
#endif

__kernel void MANGLE_NAME(prelu_, IOM, FM, MD, AXIS)(const int iw_str,
    const int ih_str,
    const int i_off,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int w,
    const int bx,
    const int by,
    __global const T *weight,
    READ_ONLY_KERNEL_MEM input,
    KERNEL_MEM output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 wei = 0;
#if defined(USE_PROPAGATE_DOWN)
    wei.x = weight[0];
    wei.y = wei.x;
    wei.z = wei.x;
    wei.w = wei.x;
#else
#if defined(USE_NCHW)
#if defined(RELU_ON_AXIS_W)
    char rw = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
    if (rw == 4) {
        wei = vload4(idx, weight);
    } else {
        if (rw == 3) {
            wei.xyz = vload3(0, weight + (idx << 2));
        } else if (rw == 2) {
            wei.xy = vload2(0, weight + (idx << 2));
        } else if (rw == 1) {
            wei.x = weight[idx << 2];
        }
    }
#elif defined(RELU_ON_AXIS_H)
    wei.x = weight[idy];
    wei.y = wei.x;
    wei.z = wei.x;
    wei.w = wei.x;
#else
    wei.x = weight[idz];
    wei.y = wei.x;
    wei.z = wei.x;
    wei.w = wei.x;
#endif
#else
    wei = vload4(idz, weight);
#endif
#endif

    T4 val;
#if defined(USE_NCHW)
    LOAD_MEM_V4_C1_COMMON(val, idx, idy, idz, iw_str, ih_str, i_off, w, input);
#else
    LOAD_MEM_V4_COMMON(val, idx, idy, idz, iw_str, ih_str, i_off, input);
#endif
    val.s0 = (val.s0 > 0) ? val.s0 : wei.x * val.s0;
    val.s1 = (val.s1 > 0) ? val.s1 : wei.y * val.s1;
    val.s2 = (val.s2 > 0) ? val.s2 : wei.z * val.s2;
    val.s3 = (val.s3 > 0) ? val.s3 : wei.w * val.s3;
#if defined(USE_NCHW)
    STORE_MEM_V4_C1_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, w, output);
#else
    STORE_MEM_V4_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, output);
#endif
}
