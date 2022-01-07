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

#define BILINEAR_CAL_POS_RAT(w, h, x, y, pos, rat) \
    {                                              \
        if (x < 0) {                               \
            x = 0;                                 \
        }                                          \
        if (y < 0) {                               \
            y = 0;                                 \
        }                                          \
        pos.s0 = x;                                \
        pos.s1 = y;                                \
        if (pos.s0 >= w - 1) {                     \
            pos.s0 = w - 1;                        \
            pos.s2 = pos.s0;                       \
            x = pos.s0;                            \
        } else {                                   \
            pos.s2 = pos.s0 + 1;                   \
        }                                          \
        if (pos.s1 >= h - 1) {                     \
            pos.s1 = h - 1;                        \
            pos.s3 = pos.s1;                       \
            y = pos.s1;                            \
        } else {                                   \
            pos.s3 = pos.s1 + 1;                   \
        }                                          \
        rat.s0 = x - pos.s0;                       \
        rat.s1 = y - pos.s1;                       \
    }

#if defined(USE_MAX)
#define MODE max
#define CALCORE(res, v)     \
    {                       \
        res = fmax(res, v); \
    }
#else
#define MODE avg
#define CALCORE(res, v) \
    {                   \
        res += v;       \
    }
#endif

#if defined(USE_NCHW)
#define FM nchw_
#define BILINEAR_INTERPOLATE(w, h, x, y, in, rv)                 \
    {                                                            \
        if (y < -1.0 || y > h || x < -1.0 || x > w) {            \
            continue;                                            \
        }                                                        \
        int4 pos;                                                \
        float2 rat;                                              \
        BILINEAR_CAL_POS_RAT(w, h, x, y, pos, rat);              \
        T4 val;                                                  \
        val.s0 = in[in_off + pos.s1 * iw_str + pos.s0];          \
        val.s1 = in[in_off + pos.s1 * iw_str + pos.s2];          \
        val.s2 = in[in_off + pos.s3 * iw_str + pos.s0];          \
        val.s3 = in[in_off + pos.s3 * iw_str + pos.s2];          \
        CALCORE(rv, (float)val.s0 *(1 - rat.s0) * (1 - rat.s1)); \
        CALCORE(rv, (float)val.s1 *rat.s0 *(1 - rat.s1));        \
        CALCORE(rv, (float)val.s2 *(1 - rat.s0) * rat.s1);       \
        CALCORE(rv, (float)val.s3 *rat.s0 *rat.s1);              \
    }
#else
#define FM
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(x, y, v)                          \
    {                                                \
        LOAD_MEM_V4(v, (int4)(x, y, idc, 0), input); \
    \ 
}
#else
#define LOAD_INPUT(x, y, v)                             \
    {                                                   \
        LOAD_MEM_V4(v, in_off + y * iw_str + x, input); \
    }
#endif
#define PROCESS_VAL(tv, v, r0, r1) \
    {                              \
        v.x = tv.x * r0 * r1;      \
        v.y = tv.y * r0 * r1;      \
        v.z = tv.z * r0 * r1;      \
        v.w = tv.w * r0 * r1;      \
    }
#define LOAD_AND_CAL(x, y, r0, r1, rv) \
    {                                  \
        T4 tv;                         \
        float4 fv;                     \
        LOAD_INPUT(x, y, tv);          \
        PROCESS_VAL(tv, fv, r0, r1);   \
        CALCORE(rv, fv);               \
    }
#define BILINEAR_INTERPOLATE(w, h, x, y, in, rv)                      \
    {                                                                 \
        if (y < -1.0 || y > h || x < -1.0 || x > w) {                 \
            continue;                                                 \
        }                                                             \
        int4 pos;                                                     \
        float2 rat;                                                   \
        BILINEAR_CAL_POS_RAT(w, h, x, y, pos, rat);                   \
        LOAD_AND_CAL(pos.s0, pos.s1, (1 - rat.s0), (1 - rat.s1), rv); \
        LOAD_AND_CAL(pos.s2, pos.s1, rat.s0, (1 - rat.s1), rv);       \
        LOAD_AND_CAL(pos.s0, pos.s3, (1 - rat.s0), rat.s1, rv);       \
        LOAD_AND_CAL(pos.s2, pos.s3, rat.s0, rat.s1, rv);             \
    }
#endif

__kernel void MANGLE_NAME(roialign_, IOM, FM, MODE)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int roi_off,
    const int iw,
    const int ih,
    const int ow,
    const int oh,
    const int oc,
    const int bx,
    const int by,
    const int sampling_ratio,
    float spatial_scale,
    __global const T *rois,
    READ_ONLY_KERNEL_MEM input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    int idc = idz % oc;
    int idn = idz / oc;
    if (idx >= bx || idy >= by) {
        return;
    }

    T4 rv = vload4(idn, rois + roi_off);
    float rw = (rv.z - rv.x) * spatial_scale;
    float rh = (rv.w - rv.y) * spatial_scale;
    float be_x = rv.x * spatial_scale;
    float be_y = rv.y * spatial_scale;
    if (rw < 1) {
        rw = 1;
    }
    if (rh < 1) {
        rh = 1;
    }
    float bin_w = rw / ow;
    float bin_h = rh / oh;
    int bin_gw = (sampling_ratio > 0) ? sampling_ratio : ceil(bin_w);
    int bin_gh = (sampling_ratio > 0) ? sampling_ratio : ceil(bin_h);
    be_x += idx * bin_w;
    be_y += idy * bin_h;
    bin_w = bin_w / bin_gw;
    bin_h = bin_h / bin_gh;

#if defined(USE_NCHW)
#if defined(USE_MAX)
    float res = -FLT_MAX;
#else
    float res = 0;
#endif
#else
#if defined(USE_MAX)
    float4 res = (float4)-FLT_MAX;
#else
    float4 res = (float4)0;
#endif
#endif

#if !defined(USE_INPUT_IMG)
    int in_off = i_off + idc * iw_str * ih_str;
#endif
    for (int by = 0; by < bin_gh; by++) {
        float y = be_y + (by + 0.5) * bin_h;
        for (int bx = 0; bx < bin_gw; bx++) {
            float x = be_x + (bx + 0.5) * bin_w;
            BILINEAR_INTERPOLATE(iw, ih, x, y, input, res);
        }
    }

#if defined(USE_NCHW)
    res = res / (float)(bin_gw * bin_gh);
    output[((idn * oc + idc) * oh_str + idy) * ow_str + idx + o_off] = res;
#else
    res.x = res.x / (float)(bin_gw * bin_gh);
    res.y = res.y / (float)(bin_gw * bin_gh);
    res.z = res.z / (float)(bin_gw * bin_gh);
    res.w = res.w / (float)(bin_gw * bin_gh);
    vstore4((T4)(res.x, res.y, res.z, res.w),
        ((idn * oc + idc) * oh_str + idy) * ow_str + idx + o_off, output);
#endif
}
