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
#define MANGLE_NAME_IMPL(base, IOM, FM, MODE) base##IOM##FM##MODE
#define MANGLE_NAME(base, IOM, FM, MODE) MANGLE_NAME_IMPL(base, IOM, FM, MODE)

#define FM
#if defined(USE_NCHW)
#define FM nchw
#endif
#if defined(USE_HALF_PIXEL)
#define MODE _half_pixel
#elif defined(USE_PYTORCH_HALF_PIXEL)
#define MODE _pytorch_half_pixel
#elif defined(USE_ALIGN_CORNERS)
#define MODE _align_corners
#elif defined(USE_ASYMMETRIC)
#define MODE _asymmetric
#endif
#if defined(USE_HALF_PIXEL) || defined(USE_PYTORCH_HALF_PIXEL)
#define CALCOORD(id, od, rat)                             \
    {                                                     \
        id = ((float)od + (float)0.5) * rat - (float)0.5; \
        id = max(id, 0);                                  \
    }
#elif defined(USE_ALIGN_CORNERS) || defined(USE_ASYMMETRIC)
#define CALCOORD(id, od, rat) \
    {                         \
        id = od * rat;        \
    }
#endif

__kernel void MANGLE_NAME(resize_nearest_, IOM, FM, MODE)(const int iw_str,
    const int ih_str,
    const int i_off,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int ow,
    const int oh,
    const int bx,
    const int by,
    const float ratiow,
    const float ratioh,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
#if defined(USE_NCHW)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T val;
    int ix;
    int iy;
    CALCOORD(ix, idx, ratiow);
    CALCOORD(iy, idy, ratioh);

#if defined(USE_PYTORCH_HALF_PIXEL)
    if (ow == 1) {
        ix = 0;
    }
    if (oh == 1) {
        iy = 0;
    }
#endif
    int in_off = (idz * ih_str + iy) * iw_str + ix + i_off;
    val = in[in_off];
    int out_off = (idz * oh_str + idy) * ow_str + idx + o_off;
    out[out_off] = val;
}
#else
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 val = 0;
    int ix;
    int iy;
    CALCOORD(ix, idx, ratiow);
    CALCOORD(iy, idy, ratioh);

#if defined(USE_PYTORCH_HALF_PIXEL)
    if (ow == 1) {
        ix = 0;
    }
    if (oh == 1) {
        iy = 0;
    }
#endif
    LOAD_MEM_V4_COMMON(val, ix, iy, idz, iw_str, ih_str, i_off, in);
    STORE_MEM_V4_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, out);
}
#endif
