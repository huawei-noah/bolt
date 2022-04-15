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
#if defined(USE_OUTPUT_IMG)
#define STORE_OUT                                                                  \
    {                                                                              \
        STORE_MEM_V4((T4)(sum.x, sum.y, sum.z, sum.w), (int4)(0, 0, idx, 0), out); \
    }
#else
#define STORE_OUT                                                     \
    {                                                                 \
        int out_off = idx * ohw_str + o_off;                          \
        STORE_MEM_V4((T4)(sum.x, sum.y, sum.z, sum.w), out_off, out); \
    }
#endif

__kernel void MANGLE_NAME(pooling_global_mean_w_, IOM)(const int iw,
    const int ow_str,
    const int ohw_str,
    const int o_off,
    const int bx,
    __global const T *in,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    const int in_off = idx * iw;

    T4 val;
    float4 sum = 0;
    for (int i = 0; i < iw; ++i) {
        val = vload4(in_off + i, in);
        sumvec4(sum, val);
    }
    sum = sum / ((float)(iw));
    STORE_OUT;
}
