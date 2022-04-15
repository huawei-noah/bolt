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

#define sumvec4(x, y)        \
    {                        \
        x.s0 += (float)y.s0; \
        x.s1 += (float)y.s1; \
        x.s2 += (float)y.s2; \
        x.s3 += (float)y.s3; \
    }

#if defined(USE_INPUT_IMG)
#define LOAD_INPUT LOAD_MEM_V4(val, (int4)(idx, i, idy, 0), in);
#else
#define LOAD_INPUT LOAD_MEM_V4(val, in_off + iw_str * i, in);
#endif

__kernel void MANGLE_NAME(pooling_global_mean_h_, IOM)(const int iw_str,
    const int ihw_str,
    const int i_off,
    const int iw,
    const int ih,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }

#if !defined(USE_INPUT_IMG)
    int in_off = idy * ihw_str + idx + i_off;
#endif

    T4 val;
    float4 sum = 0;
    for (int i = 0; i < ih; ++i) {
        LOAD_INPUT;
        sumvec4(sum, val);
    }
    sum = sum / (float)(ih);
    int out_off = (idy * iw) + idx;
    vstore4((T4)(sum.x, sum.y, sum.z, sum.w), out_off, out);
}
