// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, TP, AXIS) base##TP##AXIS
#define MANGLE_NAME(base, TP, AXIS) MANGLE_NAME_IMPL(base, TP, AXIS)

#define SUM(v, res)     \
    {                   \
        res.s0 += v.s0; \
        res.s1 += v.s1; \
        res.s2 += v.s2; \
        res.s3 += v.s3; \
    }

#define VAR(mean, v, res)          \
    {                              \
        T4 tmp;                    \
        tmp.s0 = v.s0 - mean.x;    \
        tmp.s1 = v.s1 - mean.y;    \
        tmp.s2 = v.s2 - mean.z;    \
        tmp.s3 = v.s3 - mean.w;    \
        res.s0 += tmp.s0 * tmp.s0; \
        res.s1 += tmp.s1 * tmp.s1; \
        res.s2 += tmp.s2 * tmp.s2; \
        res.s3 += tmp.s3 * tmp.s3; \
    }

__kernel void MANGLE_NAME(reduction_nchw_, TP, AXIS)(const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const int ih,
    const int iw,
    const int ic,
    const int ow,
    const int oh,
    const int keep_dim,
    const int od,
    const int bx,
    const int by,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    int in_off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off;
#if (AXIS == 0)
    int in_str = 4;
    int loop = (iw + 3) >> 2;
#elif (AXIS == 1)
    int in_str = iw_str;
    int loop = ih;
#elif (AXIS == 2)
    int in_str = ih_str * iw_str;
    int loop = ic;
#endif
    T4 val = 0;
    float4 mean = 0;
    float4 res = 0;
    for (int i = 0; i < loop; ++i) {
        val = vload4(0, in + in_off + in_str * i);
#if defined(USE_SUM) || defined(USE_MEAN) || defined(USE_STD_DEVIATION)
        SUM(val, res);
#elif defined(USE_SCALAR_PRODUCT)
        VAR(mean, val, res);
#endif
    }

    int avgNum = loop;
#if (AXIS == 0)
    res.x = res.x + res.y + res.z + res.w;
    avgNum = iw;
#endif

#if defined(USE_MEAN) || defined(USE_STD_DEVIATION)
    res.x = res.x / avgNum;
#if (AXIS == 0)
    res.y = res.x;
    res.z = res.x;
    res.w = res.x;
#else
    res.y = res.y / avgNum;
    res.z = res.z / avgNum;
    res.w = res.w / avgNum;
#endif
#endif

#if defined(USE_STD_DEVIATION)
    mean = res;
    res = 0;
    for (int i = 0; i < loop; ++i) {
        val = vload4(in, in_off + in_str * i);
        VAR(mean, val, res);
    }
#if (AXIS == 0)
    res.x = res.x + res.y + res.z + res.w;
    res.x = sqrt(res.x);
#else
    res.x = sqrt(res.x);
    res.y = sqrt(res.y);
    res.z = sqrt(res.z);
    res.w = sqrt(res.w);
#endif
#endif

    int out_off;
#if (AXIS == 0)
    if (keep_dim) {
        if (od == 4) {
            out_off = (idz * oh_str + idy + oh_off) * ow_str + ow_off;
        } else if (od == 3) {
            out_off = (idy + oh_off) * ow_str + ow_off;
        } else {
            out_off = idy + ow_off;
        }
    } else {
        if (od == 3) {
            out_off = (idz + oh_off) * ow_str + idy + ow_off;
        } else {
            out_off = idy + ow_off;
        }
    }
    out[out_off] = (T)res.x;
#elif (AXIS == 1)
    if (keep_dim) {
        if (od == 4) {
            out_off = (idz * oh_str + oh_off) * ow_str + (idx << 2) + ow_off;
        } else {
            out_off = (idx << 2) + ow_off;
        }
    } else {
        if (od == 3) {
            out_off = (idz + oh_off) * ow_str + (idx << 2) + ow_off;
        } else {
            out_off = (idx << 2) + ow_off;
        }
    }
#elif (AXIS == 2)
    out_off = (idy + oh_off) * ow_str + (idx << 2) + ow_off;
#endif

#if (AXIS != 0)
    char ew = ((idx << 2) + 4 <= ow) ? 4 : (ow & 3);
    T4 res_t;
    res_t.x = (T)res.x;
    res_t.y = (T)res.y;
    res_t.z = (T)res.z;
    res_t.w = (T)res.w;

    if (ew == 4) {
        vstore4(res_t, 0, out + out_off);
    } else {
        if (ew == 3) {
            vstore3(res_t.xyz, 0, out + out_off);
        }
        if (ew == 2) {
            vstore2(res_t.xy, 0, out + out_off);
        }
        if (ew == 1) {
            out[out_off] = res_t.x;
        }
    }
#endif
}
