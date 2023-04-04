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
    int ix, iy;
    CALCOORD(ix, idx, iw, ow, r0_w, r1_w);
    CALCOORD(iy, idy, ih, oh, r0_h, r1_h);

#if defined(USE_NCHW) || defined(USE_NHWC)
    int in_off = (idz * ih_str + iy) * iw_str + ix + i_off;
    int out_off = (idz * oh_str + idy) * ow_str + idx + o_off;
#if defined(USE_NCHW)
    output[out_off] = input[in_off];
#else
    in_off *= 3;
    out_off *= 3;
    output[out_off] = input[in_off];
    output[out_off + 1] = input[in_off + 1];
    output[out_off + 2] = input[in_off + 2];
#endif
#else
    IT4 val = 0;
    LOAD_MEM_V4_COMMON(val, ix, iy, idz, iw_str, ih_str, i_off, input);
    STORE_MEM_V4_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, output);
#endif
}
