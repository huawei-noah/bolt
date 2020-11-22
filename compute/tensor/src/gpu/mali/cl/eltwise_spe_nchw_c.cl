// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, TP) base##TP
#define MANGLE_NAME(base, TP) MANGLE_NAME_IMPL(base, TP)

#if defined(USE_SUM)
#define calCore(v, res) \
    {                   \
        res.s0 += v.s0; \
        res.s1 += v.s1; \
        res.s2 += v.s2; \
        res.s3 += v.s3; \
    }
#endif

#if defined(USE_MAX)
#define calCore(v, res)     \
    {                       \
        res = fmax(res, v); \
    }
#endif

#if defined(USE_PROD)
#define calCore(v, res) \
    {                   \
        res.s0 *= v.s0; \
        res.s1 *= v.s1; \
        res.s2 *= v.s2; \
        res.s3 *= v.s3; \
    }
#endif

__kernel void MANGLE_NAME(eltwise_spe_nchw_c_, TP)(const int h,
    const int w,
    const int c,
    const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    __global const T *in,
    __global const T *ine,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= h || idy >= w) {
        return;
    }

    T4 val;
    T4 res;
    const int in_off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;
    res = vload4(in_off, in);
    val = vload4(idz, ine);
    calCore(val, res);
    const int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;
    vstore4(res, out_off, out);
}
