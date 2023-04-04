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

#if defined(USE_POOLING_MAX)
#define CALC_0 res = UNI_F16_MIN;
#define CALC_1(x, y)                        \
    {                                       \
        x.s0 = (x.s0 > y.s0) ? x.s0 : y.s0; \
        x.s1 = (x.s1 > y.s1) ? x.s1 : y.s1; \
        x.s2 = (x.s2 > y.s2) ? x.s2 : y.s2; \
        x.s3 = (x.s3 > y.s3) ? x.s3 : y.s3; \
    }
#define CALC_2
#elif defined(USE_POOLING_MEAN)
#define CALC_0 res = 0;
#define CALC_1(x, y) \
    {                \
        x += y;      \
    }
#define CALC_2                                                             \
    {                                                                      \
        T psize = count_include_pad ? (kh * kw) : ((eh - bh) * (ew - bw)); \
        res = res / psize;                                                 \
    }
#else
error not support this pooling mode.
#endif

#if defined(USE_INPUT_IMG)
#define LOAD_INPUT LOAD_MEM_V4(val, (int4)(j, i, in_off_z, 0), in);
#define ADD_IN_OFF
#else
#define LOAD_INPUT LOAD_VECTOR(val, in_off + j, in);
#define ADD_IN_OFF in_off += iw_str;
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUT                                                                      \
    {                                                                                  \
        STORE_MEM_V4((T4)(res.x, res.y, res.z, res.w), (int4)(idx, idy, idz, 0), out); \
    }
#else
#define STORE_OUT                                                  \
    {                                                              \
        int out_off = (idz * oh_str + idy) * ow_str + o_off + idx; \
        STORE_VECTOR(res, out_off, out);                           \
    }
#endif

__kernel void KERNEL_NAME(const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int iw,
    const int ih,
    const int ow,
    const int oh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int kw,
    const int kh,
    const int count_include_pad,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= ow || idy >= oh) {
        return;
    }

    int bw = idx * sw - pw;
    int bh = idy * sh - ph;
    int ew = bw + kw;
    int eh = bh + kh;
    bw = (bw < 0) ? 0 : bw;
    bh = (bh < 0) ? 0 : bh;
    ew = (ew < iw) ? ew : iw;
    eh = (eh < ih) ? eh : ih;
    bw += iw_off;
    ew += iw_off;
    bh += ih_off;
    eh += ih_off;

#if defined(USE_INPUT_IMG)
    int in_off_z = idz;
#else
    int in_off = (idz * ih_str + bh) * iw_str;
#endif

    VECTOR res, val;

    CALC_0
    for (int i = bh; i < eh; ++i) {
        for (int j = bw; j < ew; ++j) {
            LOAD_INPUT;
            CALC_1(res, val);
        }
        ADD_IN_OFF
    }
    CALC_2

    STORE_OUT;
}
