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
#define MANGLE_NAME_IMPL(base, IOM, FM, DT) base##IOM##FM##DT
#define MANGLE_NAME(base, IOM, FM, DT) MANGLE_NAME_IMPL(base, IOM, FM, DT)

#define FM
#define DT
#if defined(USE_I32)
#define DT _i32
#endif
#if defined(USE_NCHW)
#define FM nchw
#endif

__kernel void MANGLE_NAME(power_, IOM, FM, DT)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int w,
    const int bx,
    const int by,
    const int has_power,
    const float alp,
    const float bet,
    float power,
    READ_ONLY_KERNEL_MEM input,
    KERNEL_MEM output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 val = 0;
#if defined(USE_NCHW)
    LOAD_MEM_V4_C1_COMMON(val, idx, idy, idz, iw_str, ih_str, i_off, w, input);
#else
    LOAD_MEM_V4_COMMON(val, idx, idy, idz, iw_str, ih_str, i_off, input);
#endif
    val.x = (T)(((float)val.x) * alp + bet);
    val.y = (T)(((float)val.y) * alp + bet);
    val.z = (T)(((float)val.z) * alp + bet);
    val.w = (T)(((float)val.w) * alp + bet);
    if (has_power) {
        val.x = pow((float)val.x, power);
        val.y = pow((float)val.y, power);
        val.z = pow((float)val.z, power);
        val.w = pow((float)val.w, power);
    }

#if defined(USE_NCHW)
    STORE_MEM_V4_C1_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, w, output);
#else
    STORE_MEM_V4_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, output);
#endif
}
