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
#define MANGLE_NAME_IMPL(base, IOM, FM, AXIS) base##IOM##FM##AXIS
#define MANGLE_NAME(base, IOM, FM, AXIS) MANGLE_NAME_IMPL(base, IOM, FM, AXIS)
#define FM
#if defined(USE_NCHW)
#define FM nchw_
#if (AXIS == 0)
#define LOAD_VEC_ON_AXIS
#endif
#else
#if (AXIS == 2)
#define LOAD_VEC_ON_AXIS
#endif
#endif

#define UPDATE_MAX_VEC(res, val)                          \
    {                                                     \
        float4 tv = (float4)(val.x, val.y, val.z, val.w); \
        res = fmax(res, tv);                              \
    }
#define UPDATE_SUM_VEC(res, val)               \
    {                                          \
        res.x += exp((float)val.x - maxval.x); \
        res.y += exp((float)val.y - maxval.y); \
        res.z += exp((float)val.z - maxval.z); \
        res.w += exp((float)val.w - maxval.w); \
    }
#define UPDATE_RES_VEC(res)                              \
    {                                                    \
        res.x = exp((float)res.x - maxval.x) * sumexp.x; \
        res.y = exp((float)res.y - maxval.y) * sumexp.y; \
        res.z = exp((float)res.z - maxval.z) * sumexp.z; \
        res.w = exp((float)res.w - maxval.w) * sumexp.w; \
    }

#if defined(USE_NCHW)
#if (AXIS == 0)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(i, idx, idy, 0), in)
#else
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, i, in + in_off)
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(i, idx, idy, 0), out)
#else
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, i, out + out_off)
#endif

#elif (AXIS == 1)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(idx, i, idy, 0), in)
#else
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, 0, in + in_off + i * iw_str)
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(idx, i, idy, 0), out)
#else
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, 0, out + out_off + i * ow_str)
#endif

#elif (AXIS == 2)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(idx, idy, i, 0), in)
#else
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, 0, in + in_off + i * iw_str * ih_str)
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(idx, idy, i, 0), out)
#else
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, 0, out + out_off + i * ow_str * oh_str)
#endif
#endif

#else
#if (AXIS == 0)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(i, idx, idy, 0), in)
#else
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, in_off + i, in)
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(i, idx, idy, 0), out)
#else
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, out_off + i, out)
#endif

#elif (AXIS == 1)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(idx, i, idy, 0), in)
#else
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, in_off + i * iw_str, in)
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(idx, i, idy, 0), out)
#else
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, out_off + i * ow_str, out)
#endif

#elif (AXIS == 2)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(idx, idy, i, 0), in)
#else
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, in_off + i * iw_str * ih_str, in)
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(idx, idy, i, 0), out)
#else
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, out_off + i * ow_str * oh_str, out)
#endif
#endif
#endif

__kernel void MANGLE_NAME(softmax_, IOM, FM, AXIS)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int w,
    const int h,
    const int c,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
#if defined(LOAD_VEC_ON_AXIS)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    float4 maxval = (float4)(-FLT_MAX);
    float4 sumexp = 0;
    T4 val, rval;
    int xd4;
    char xe4;
#if (AXIS == 0)
#if !defined(USE_INPUT_IMG)
    int in_off = (idy * ih_str + idx) * iw_str + i_off;
#endif
#if !defined(USE_OUTPUT_IMG)
    int out_off = (idy * oh_str + idx) * ow_str + o_off;
#endif
    xd4 = (w + 3) >> 2;
    xe4 = ((w & 3) == 0) ? 4 : (w & 3);
#elif (AXIS == 2)
#if !defined(USE_INPUT_IMG)
    int in_off = idy * iw_str + idx + i_off;
#endif
#if !defined(USE_OUTPUT_IMG)
    int out_off = idy * ow_str + idx + o_off;
#endif
    xd4 = (c + 3) >> 2;
    xe4 = ((c & 3) == 0) ? 4 : (c & 3);
#endif
    for (int i = 0; i < xd4 - 1; i++) {
        LOAD_INPUT(i, val);
        UPDATE_MAX_VEC(maxval, val);
    }
    LOAD_INPUT((xd4 - 1), rval);
    maxval.x = fmax(maxval.x, maxval.y);
    maxval.x = fmax(maxval.x, maxval.z);
    maxval.x = fmax(maxval.x, maxval.w);
    maxval.x = fmax(maxval.x, (float)rval.x);
    if (xe4 > 1) {
        maxval.x = fmax(maxval.x, (float)rval.y);
    }
    if (xe4 > 2) {
        maxval.x = fmax(maxval.x, (float)rval.z);
    }
    if (xe4 > 3) {
        maxval.x = fmax(maxval.x, (float)rval.w);
    }
    maxval.y = maxval.x;
    maxval.z = maxval.x;
    maxval.w = maxval.x;
    for (int i = 0; i < xd4 - 1; i++) {
        LOAD_INPUT(i, val);
        UPDATE_SUM_VEC(sumexp, val);
    }
    sumexp.x += (sumexp.y + sumexp.z + sumexp.w);
    sumexp.x += exp((float)rval.x - maxval.x);
    if (xe4 > 1) {
        sumexp.x += exp((float)rval.y - maxval.y);
    }
    if (xe4 > 2) {
        sumexp.x += exp((float)rval.z - maxval.z);
    }
    if (xe4 > 3) {
        sumexp.x += exp((float)rval.w - maxval.w);
    }
    sumexp.x = 1.0 / sumexp.x;
    sumexp.y = sumexp.x;
    sumexp.z = sumexp.x;
    sumexp.w = sumexp.x;
    for (int i = 0; i < xd4 - 1; i++) {
        LOAD_INPUT(i, val);
        UPDATE_RES_VEC(val);
        STORE_OUTPUT(i, val);
    }
    rval.x = exp((float)rval.x - maxval.x) * sumexp.x;
    if (xe4 > 1) {
        rval.y = exp((float)rval.y - maxval.y) * sumexp.y;
    }
    if (xe4 > 2) {
        rval.z = exp((float)rval.z - maxval.z) * sumexp.z;
    }
    if (xe4 > 3) {
        rval.w = exp((float)rval.w - maxval.w) * sumexp.w;
    }
    STORE_OUTPUT((xd4 - 1), rval);
}
#else
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    float4 maxval = (float4)(-FLT_MAX);
    float4 sumexp = 0;
    T4 val;
#if defined(USE_NCHW)
    char ex = (((idx << 2) + 4) <= w) ? 4 : (w & 3);
#if (AXIS == 1)
    int loop = h;
#elif (AXIS == 2)
    int loop = c;
#endif

#if !defined(USE_INPUT_IMG)
#if (AXIS == 1)
    int in_off = idy * ih_str * iw_str + (idx << 2) + i_off;
#elif (AXIS == 2)
    int in_off = idy * iw_str + (idx << 2) + i_off;
#endif
#endif

#if !defined(USE_OUTPUT_IMG)
#if (AXIS == 1)
    int out_off = idy * oh_str * ow_str + (idx << 2) + o_off;
#elif (AXIS == 2)
    int out_off = idy * ow_str + (idx << 2) + o_off;
#endif
#endif
#else
    char ex = (((idy << 2) + 4) <= c) ? 4 : (c & 3);
#if (AXIS == 0)
    int loop = w;
#elif (AXIS == 1)
    int loop = h;
#endif

#if !defined(USE_INPUT_IMG)
#if (AXIS == 0)
    int in_off = (idy * ih_str + idx) * iw_str + i_off;
#elif (AXIS == 1)
    int in_off = idy * ih_str * iw_str + idx + i_off;
#endif
#endif

#if !defined(USE_OUTPUT_IMG)
#if (AXIS == 0)
    int out_off = (idy * oh_str + idx) * ow_str + o_off;
#elif (AXIS == 1)
    int out_off = idy * oh_str * ow_str + idx + o_off;
#endif
#endif
#endif
    for (int i = 0; i < loop; i++) {
        LOAD_INPUT(i, val);
        UPDATE_MAX_VEC(maxval, val);
    }
    for (int i = 0; i < loop; i++) {
        LOAD_INPUT(i, val);
        UPDATE_SUM_VEC(sumexp, val);
    }
    sumexp.x = 1.0 / sumexp.x;
    sumexp.y = 1.0 / sumexp.y;
    sumexp.z = 1.0 / sumexp.z;
    sumexp.w = 1.0 / sumexp.w;
    if (ex == 4) {
        for (int i = 0; i < loop; i++) {
            LOAD_INPUT(i, val);
            UPDATE_RES_VEC(val);
            STORE_OUTPUT(i, val);
        }
    } else {
        for (int i = 0; i < loop; i++) {
            LOAD_INPUT(i, val);
            val.x = exp((float)val.x - maxval.x) * sumexp.x;
            if (ex > 1) {
                val.y = exp((float)val.y - maxval.y) * sumexp.y;
            }
            if (ex > 2) {
                val.z = exp((float)val.z - maxval.z) * sumexp.z;
            }
            STORE_OUTPUT(i, val);
        }
    }
}
#endif
