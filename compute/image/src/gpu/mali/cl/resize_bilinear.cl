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

#if defined(USE_HALF_PIXEL)
#define CALCOORD(ret, x, iw, ow, r0, r1)               \
    {                                                  \
        ret = max(0.f, ((float)x + 0.5f) * r0 - 0.5f); \
    }
#elif defined(USE_PYTORCH_HALF_PIXEL)
#define CALCOORD(ret, x, iw, ow, r0, r1)                                \
    {                                                                   \
        ret = (ow > 1) ? max(0.f, (((float)x + 0.5f) * r0 - 0.5f)) : 0; \
    }
#elif defined(USE_ALIGN_CORNERS)
#define CALCOORD(ret, x, iw, ow, r0, r1) \
    {                                    \
        ret = x * r1;                    \
    }
#elif defined(USE_ASYMMETRIC)
#define CALCOORD(ret, x, iw, ow, r0, r1) \
    {                                    \
        ret = x * r0;                    \
    }
#endif

#if defined(OUTPUT_UCHAR)
#define func convert_uchar_sat_rte
#else
#define func
#endif

__kernel void KERNEL_NAME(const int iw_str,
    const int ih_str,
    const int i_off,
    const int iw,
    const int ih,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int ow,
    const int oh,
    const float r0_w,
    const float r0_h,
    const float r1_w,
    const float r1_h,
#if defined(USE_NCHW) || defined(USE_NHWC)
    __global const IT *input,
    __global OT *output
#else
    READ_ONLY_KERNEL_MEM input,
    KERNEL_MEM output
#endif
)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= ow || idy >= oh) {
        return;
    }

    float ix, iy;
    CALCOORD(ix, idx, iw, ow, r0_w, r1_w);
    CALCOORD(iy, idy, ih, oh, r0_h, r1_h);

    int4 tblr;
    tblr.x = max(0, (int)floor(ix));   // L
    tblr.y = min(tblr.x + 1, iw - 1);  // R
    tblr.z = max(0, (int)floor(iy));   // T
    tblr.w = min(tblr.z + 1, ih - 1);  // B
    T dif1 = ix - (float)tblr.x;              // C-L
    T dif2 = iy - (float)tblr.z;              // C-T

#if defined(USE_NCHW) || defined(USE_NHWC)
    int x = (idz * ih_str + tblr.z) * iw_str + tblr.x + i_off;  // TL_off
    int y = (idz * ih_str + tblr.z) * iw_str + tblr.y + i_off;  // TR_off
    int z = (idz * ih_str + tblr.w) * iw_str + tblr.x + i_off;  // BL_off
    int w = (idz * ih_str + tblr.w) * iw_str + tblr.y + i_off;  // BR_off
    int out_off = (idz * oh_str + idy) * ow_str + idx + o_off;

#if defined(USE_NCHW)
    T val_TL = input[x];
    T val_TR = input[y];
    T val_BL = input[z];
    T val_BR = input[w];
    T top = mad(val_TR - val_TL, dif1, val_TL);
    T bottom = mad(val_BR - val_BL, dif1, val_BL);
    T v = mad(bottom - top, dif2, top);
    output[out_off] = func(v);
#else
    T3 val_TL, val_TR, val_BL, val_BR;
    x *= 3;
    y *= 3;
    z *= 3;
    w *= 3;
    out_off *= 3;
    val_TL.x = input[x];
    val_TL.y = input[x + 1];
    val_TL.z = input[x + 2];
    val_TR.x = input[y];
    val_TR.y = input[y + 1];
    val_TR.z = input[y + 2];
    val_BL.x = input[z];
    val_BL.y = input[z + 1];
    val_BL.z = input[z + 2];
    val_BR.x = input[w];
    val_BR.y = input[w + 1];
    val_BR.z = input[w + 2];
    T3 top = mad(val_TR - val_TL, dif1, val_TL);
    T3 bottom = mad(val_BR - val_BL, dif1, val_BL);
    T3 v = mad(bottom - top, dif2, top);
    output[out_off] = func(v.x);
    output[out_off + 1] = func(v.y);
    output[out_off + 2] = func(v.z);
#endif
#else
    T4 val_TL, val_TR, val_BL, val_BR;
    LOAD_MEM_V4_COMMON(val_TL, tblr.x, tblr.z, idz, iw_str, ih_str, i_off, input);
    LOAD_MEM_V4_COMMON(val_TR, tblr.y, tblr.z, idz, iw_str, ih_str, i_off, input);
    LOAD_MEM_V4_COMMON(val_BL, tblr.x, tblr.w, idz, iw_str, ih_str, i_off, input);
    LOAD_MEM_V4_COMMON(val_BR, tblr.y, tblr.w, idz, iw_str, ih_str, i_off, input);

    T4 top = mad((val_TR - val_TL), dif1, val_TL);
    T4 bottom = mad((val_BR - val_BL), dif1, val_BL);
    T4 out = mad((bottom - top), dif2, top);
    STORE_MEM_V4_COMMON(out, idx, idy, idz, ow_str, oh_str, o_off, output);
#endif
}
