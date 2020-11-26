// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, TP, B_AXIS) base##TP##B_AXIS
#define MANGLE_NAME(base, TP, B_AXIS) MANGLE_NAME_IMPL(base, TP, B_AXIS)

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

#if defined(USE_NCHW)
__kernel void MANGLE_NAME(eltwise_broadcast_nchw_, TP, B_AXIS)
#else
__kernel void MANGLE_NAME(eltwise_broadcast_, TP, B_AXIS)
#endif
    (const int c,
        const int ih_str,
        const int iw_str,
        const int ih_off,
        const int iw_off,
        const int oh_str,
        const int ow_str,
        const int oh_off,
        const int ow_off,
        __global const T *in0,
        __global const T *in1,
        __global T *out)
{
    char ec = 4;
#if defined(USE_NCHW)
    ec = c & 3;
    if (ec == 0) {
        ec = 4;
    }
#endif
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if (idx >= h || idy >= w) {
        return;
    }

    int idn = 0;
    int idz, in_off_res;
    if (ec == 4) {
        idz = get_global_id(2);
        in_off_res = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;
    } else {
        int cd4 = (c + 3) >> 2;
        idz = get_global_id(2) % cd4;
        idn = get_gloabl_id(2) / cd4;
        in_off_res = ((idn * cd4 + idz) * iw_str + idy + iw_off) * ih_str + idx + ih_off;
        if ((idz << 2) + 4 <= c) {
            ec = 4;
        }
    }
    T4 val = 0;
    T4 res;

#if defined(BROAD_XYZ)
    const int in_off_val = 0;
#if defined(USE_NCHW)
    val.x = in1[in_off_val];
    val.y = val.x;
    val.z = val.x;
    val.w = val.x;
#endif

#elif defined(BROAD_XY)
    const int in_off_val = idz;
#if defined(USE_NCHW)
    if (ec == 4) {
        val = vload4(in_off_val, in1);
    } else {
        in_off_val = idn * ic + (idz << 2);
        if (ec == 1) {
            val.x = in1[in_off_val];
        }
        if (ec == 2) {
            val.xy = vload2(0, in1 + in_off_val)
        }
        if (ec == 3) {
            val.xyz = vload3(0, in1 + in_off_val)
        }
    }
#endif

#elif defined(BROAD_Y)
    const int in_off_val = idz * iw_str + idy + iw_off;
#if defined(USE_NCHW)
    in_off_val = (idn * ic + (idz << 2)) * iw_str + idy + iw_off;
    val.x = in1[in_off_val];
    if (ec > 1)
        val.y = in1[in_off_val + iw_str];
    if (ec > 2)
        val.z = in1[in_off_val + iw_str * 2];
    if (ec > 3)
        val.w = in1[in_off_val + iw_str * 3];
#endif

#elif defined(BROAD_X)
    const int in_off_val = idz * ih_str + idx + ih_off;
#if defined(USE_NCHW)
    in_off_val = (idn * ic + (idz << 2)) * ih_str + idx + ih_off;
    val.x = in1[in_off_val];
    if (ec > 1)
        val.y = in1[in_off_val + ih_str];
    if (ec > 2)
        val.z = in1[in_off_val + ih_str * 2];
    if (ec > 3)
        val.w = in1[in_off_val + ih_str * 3];
#endif
#endif

#if !defined(USE_NCHW)
    val = vload4(in_off_val, in1);
#endif
    res = vload4(in_off_res, in0);
    calCore(val, res);
    const int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;
    vstore4(res, out_off, out);
}
