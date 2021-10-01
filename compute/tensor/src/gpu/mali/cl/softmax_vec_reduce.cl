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
#define MANGLE_NAME_IMPL(base, IOM, FM, IA, OA) base##IOM##FM##IA##OA
#define MANGLE_NAME(base, IOM, FM, IA, OA) MANGLE_NAME_IMPL(base, IOM, FM, IA, OA)
#define FM
#if defined(USE_NCHW)
#define FM nchw_
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

#if defined(USE_INPUT_IMG)
#if defined(INPUT_IMG_AXIS_X)
#define IA ix_
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(i, 0, 0, 0), in)
#elif defined(INPUT_IMG_AXIS_Y)
#define IA iy_
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(0, i, 0, 0), in)
#elif defined(INPUT_IMG_AXIS_Z)
#define IA iz_
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, (int4)(0, 0, i, 0), in)
#endif
#else
#define IA
#define LOAD_INPUT(i, val) LOAD_MEM_V4(val, i, in)
#endif

#if defined(USE_OUTPUT_IMG)
#if defined(OUTPUT_IMG_AXIS_X)
#define OA ox_
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(i, 0, 0, 0), out)
#elif defined(OUTPUT_IMG_AXIS_Y)
#define OA oy_
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(0, i, 0, 0), out)
#elif defined(OUTPUT_IMG_AXIS_Z)
#define OA oz_
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, (int4)(0, 0, i, 0), out)
#endif
#else
#define OA
#define STORE_OUTPUT(i, val) STORE_MEM_V4(val, i, out)
#endif

__kernel void MANGLE_NAME(softmax_vec_reduce_, IOM, FM, IA, OA)(
    const int d4, const int e4, const int bx, READ_ONLY_KERNEL_MEM in, __global T *tmp, KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    float4 maxval = (float4)(-FLT_MAX);
    float4 sumexp = 0;
    T4 val, rval;
    for (int i = idx; i < d4 - 1; i += bx) {
        LOAD_INPUT(i, val);
        UPDATE_MAX_VEC(maxval, val);
    }
    if (idx == bx - 1) {
        LOAD_INPUT(d4 - 1, rval);
        maxval.x = fmax(maxval.x, (float)rval.x);
        if (e4 > 1) {
            maxval.y = fmax(maxval.y, (float)rval.y);
        }
        if (e4 > 2) {
            maxval.z = fmax(maxval.z, (float)rval.z);
        }
        if (e4 > 3) {
            maxval.w = fmax(maxval.w, (float)rval.w);
        }
    }
    maxval.x = fmax(maxval.x, maxval.y);
    maxval.x = fmax(maxval.x, maxval.z);
    maxval.x = fmax(maxval.x, maxval.w);
    tmp[idx] = (T)maxval.x;

    int rt = bx;
    while (rt > 16) {
        rt = rt >> 1;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        if (idx < rt) {
            maxval.y = (float)tmp[idx + rt];
            maxval.x = fmax(maxval.x, maxval.y);
            tmp[idx] = (T)maxval.x;
        }
    }
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    if (idx == 0) {
        T16 tv = vload16(0, tmp);
        tv.s01234567 = fmax(tv.s01234567, tv.s89abcdef);
        tv.s0123 = fmax(tv.s0123, tv.s4567);
        tv.s01 = fmax(tv.s01, tv.s23);
        tv.s0 = fmax(tv.s0, tv.s1);
        tmp[bx] = tv.s0;
    }
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    maxval.x = (float)tmp[bx];
    maxval.y = maxval.x;
    maxval.z = maxval.x;
    maxval.w = maxval.x;

    for (int i = idx; i < d4 - 1; i += bx) {
        LOAD_INPUT(i, val);
        UPDATE_SUM_VEC(sumexp, val);
    }
    if (idx == bx - 1) {
        sumexp.x += exp((float)rval.x - maxval.x);
        if (e4 > 1) {
            sumexp.x += exp((float)rval.y - maxval.y);
        }
        if (e4 > 2) {
            sumexp.x += exp((float)rval.z - maxval.z);
        }
        if (e4 > 3) {
            sumexp.x += exp((float)rval.w - maxval.w);
        }
    }
    sumexp.x += (sumexp.y + sumexp.z + sumexp.w);
    tmp[idx] = (T)sumexp.x;

    rt = bx;
    while (rt > 16) {
        rt = rt >> 1;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        if (idx < rt) {
            sumexp.y = (float)tmp[idx + rt];
            sumexp.x += sumexp.y;
            tmp[idx] = (T)sumexp.x;
        }
    }
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    if (idx == 0) {
        T16 tv = vload16(0, tmp);
        sumexp.x = (float)tv.s0 + (float)tv.s1 + (float)tv.s2 + (float)tv.s3;
        sumexp.y = (float)tv.s4 + (float)tv.s5 + (float)tv.s6 + (float)tv.s7;
        sumexp.z = (float)tv.s8 + (float)tv.s9 + (float)tv.sa + (float)tv.sb;
        sumexp.w = (float)tv.sc + (float)tv.sd + (float)tv.se + (float)tv.sf;
        sumexp.x += (sumexp.y + sumexp.z + sumexp.w);
        tmp[bx + 1] = (T)(1.0 / sumexp.x);
    }
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    sumexp.x = (float)tmp[bx + 1];
    sumexp.y = sumexp.x;
    sumexp.z = sumexp.x;
    sumexp.w = sumexp.x;
    for (int i = idx; i < d4 - 1; i += bx) {
        LOAD_INPUT(i, val);
        UPDATE_RES_VEC(val);
        STORE_OUTPUT(i, val);
    }
    if (idx == bx - 1) {
        rval.x = exp((float)rval.x - maxval.x) * sumexp.x;
        if (e4 > 1) {
            rval.y = exp((float)rval.y - maxval.y) * sumexp.y;
        }
        if (e4 > 2) {
            rval.z = exp((float)rval.z - maxval.z) * sumexp.z;
        }
        if (e4 > 3) {
            rval.w = exp((float)rval.w - maxval.w) * sumexp.w;
        }
        STORE_OUTPUT(d4 - 1, rval);
    }
}
