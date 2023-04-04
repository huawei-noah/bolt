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

__kernel void KERNEL_NAME(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int w,
    const int bx,
    const int by,
    const float alp,
    const float bet,
    float power,
#if defined(USE_INPUT_IMG)
    __read_only image3d_t input,
#else
    __global const IT1 *input,
#endif
#if defined(USE_OUTPUT_IMG)
    __write_only image3d_t output
#else
    __global OT1 *output
#endif
)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    IT4 iv = 0;
    OT4 ov;
#if defined(USE_NCHW)
    LOAD_MEM_V4_C1_COMMON(iv, idx, idy, idz, iw_str, ih_str, i_off, w, input);
#else
    LOAD_MEM_V4_COMMON(iv, idx, idy, idz, iw_str, ih_str, i_off, input);
#endif

    ov.x = iv.x;
    ov.y = iv.y;
    ov.z = iv.z;
    ov.w = iv.w;
    ov = FMA(ov, alp, bet);
#if defined(HAS_POW)
    ov = pow(ov, power);
#endif

#if defined(USE_NCHW)
    STORE_MEM_V4_C1_COMMON(ov, idx, idy, idz, ow_str, oh_str, o_off, w, output);
#else
    STORE_MEM_V4_COMMON(ov, idx, idy, idz, ow_str, oh_str, o_off, output);
#endif
}
