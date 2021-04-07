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
#define MANGLE_NAME_IMPL(base, IT, OT, FM) base##IT##OT##FM
#define MANGLE_NAME(base, IT, OT, FM) MANGLE_NAME_IMPL(base, IT, OT, FM)

#if defined(INPUT_F16)
#define IT f16_to_
#elif defined(INPUT_I32)
#define IT i32_to_
#endif

#if defined(OUTPUT_F16)
#define OT f16
#elif defined(OUTPUT_I32)
#define OT i32
#endif

#define FM
#if defined(USE_NCHW)
#define FM _nchw
#endif

__kernel void MANGLE_NAME(cast_, IT, OT, FM)(const int w,
    const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
    const int bx,
    const int by,
#if defined(INPUT_F16)
    __global T *in,
#elif defined(INPUT_I32)
    __global int *in,
#endif
#if defined(OUTPUT_F16)
    __global T *out
#elif defined(OUTPUT_I32)
    __global int *out
#endif
)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
#if defined(INPUT_F16)
    T4 iv = 0;
#elif defined(INPUT_I32)
    int4 iv = 0;
#endif

#if defined(OUTPUT_F16)
    T4 ov = 0;
#elif defined(OUTPUT_I32)
    int4 ov = 0;
#endif

#if defined(USE_NCHW)
    int in_off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off;
    char ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
    if (ew == 4) {
        iv = vload4(0, in + in_off);
    } else {
        if (ew == 3) {
            iv.xyz = vload3(0, in + in_off);
        }
        if (ew == 2) {
            iv.xy = vload2(0, in + in_off);
        }
        if (ew == 1) {
            iv.x = in[in_off];
        }
    }
#else
    int in_off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;
    iv = vload4(in_off, in);
#endif
    ov.x = iv.x;
    ov.y = iv.y;
    ov.z = iv.z;
    ov.w = iv.w;
#if defined(USE_NCHW)
    int out_off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off;
    if (ew == 4) {
        vstore4(ov, 0, out + out_off);
    } else {
        if (ew == 3) {
            vstore3(ov.xyz, 0, out + out_off);
        }
        if (ew == 2) {
            vstore2(ov.xy, 0, out + out_off);
        }
        if (ew == 1) {
            out[out_off] = ov.x;
        }
    }
#else
    int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;
    vstore4(ov, out_off, out);
#endif
}
