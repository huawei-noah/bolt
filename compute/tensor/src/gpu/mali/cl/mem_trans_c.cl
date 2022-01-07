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
#define MANGLE_NAME_IMPL(base, IOM, TF) base##IOM##TF
#define MANGLE_NAME(base, IOM, TF) MANGLE_NAME_IMPL(base, IOM, TF)

#if defined(TRANS_C1_TO_C4)
#define TF c1_to_c4
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(v)                      \
    {                                      \
        LOAD_MEM_V4_C1(v, in_off, ex, in); \
        in_off.z += 1;                     \
    }
#else
#define LOAD_INPUT(v)                      \
    {                                      \
        LOAD_MEM_V4_C1(v, in_off, ex, in); \
        in_off += ixy_str;                 \
    }
#endif
#if defined(USE_OUTPUT_IMG)
#define STORE_OUT(v0, v1, v2, v3)                         \
    {                                                     \
        STORE_MEM_V4((T4)(v0, v1, v2, v3), out_off, out); \
        out_off.x += 1;                                   \
    }
#else
#define STORE_OUT(v0, v1, v2, v3)                                      \
    {                                                                  \
        STORE_MEM_V4((T4)(v0, v1, v2, v3), out_off, out + offset_out); \
        out_off += 1;                                                  \
    }
#endif
#endif

#if defined(TRANS_C4_TO_C1)
#define TF c4_to_c1
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(v)               \
    {                               \
        LOAD_MEM_V4(v, in_off, in); \
        in_off.x += 1;              \
    }
#else
#define LOAD_INPUT(v)                           \
    {                                           \
        LOAD_MEM_V4(v, in_off, in + offset_in); \
        in_off += 1;                            \
    }
#endif
#if defined(USE_OUTPUT_IMG)
#define STORE_OUT(v0, v1, v2, v3)                                  \
    {                                                              \
        STORE_MEM_V4_C1(((T4)(v0, v1, v2, v3)), out_off, ex, out); \
        out_off.z += 1;                                            \
    }
#else
#define STORE_OUT(v0, v1, v2, v3)                                  \
    {                                                              \
        STORE_MEM_V4_C1(((T4)(v0, v1, v2, v3)), out_off, ex, out); \
        out_off += oxy_str;                                        \
    }
#endif
#endif

__kernel void MANGLE_NAME(mem_trans_c_, IOM, TF)(const int ix_str,
    const int iy_str,
    const int ix_off,
    const int iy_off,
    const int ox_str,
    const int oy_str,
    const int ox_off,
    const int oy_off,
    const int x,
    const int y,
    const int c,
    const int offset_in,
    const int offset_out,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
#if defined(TRANS_C1_TO_C4)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if ((idx << 2) >= x || idy >= y) {
        return;
    }
    const int idc = idz % ((c + 3) >> 2);
    const int idn = idz / ((c + 3) >> 2);
#if defined(USE_INPUT_IMG)
    int4 in_off = (int4)(idx, idy, idn * c + (idc << 2), 0);
#else
    int ixy_str = ix_str * iy_str;
    const int iz_off = (idn * c + (idc << 2)) * ixy_str;
    int in_off = iz_off + (idy + iy_off) * ix_str + (idx << 2) + ix_off + offset_in;
#endif

    char ec = ((idc << 2) + 4 <= c) ? 4 : (c & 3);
    char ex = ((idx << 2) + 4 <= x) ? 4 : (x & 3);
    T4 val[4];
    val[0] = 0;
    val[1] = 0;
    val[2] = 0;
    val[3] = 0;
    LOAD_INPUT(val[0]);
    if (ec > 1) {
        LOAD_INPUT(val[1]);
    }
    if (ec > 2) {
        LOAD_INPUT(val[2]);
    }
    if (ec > 3) {
        LOAD_INPUT(val[3]);
    }

#if defined(USE_OUTPUT_IMG)
    int4 out_off = (int4)((idx << 2), idy, idz, 0);
#else
    int out_off = (idz * oy_str + idy + oy_off) * ox_str + (idx << 2) + ox_off;
#endif

    STORE_OUT(val[0].x, val[1].x, val[2].x, val[3].x);
    if (ex > 1) {
        STORE_OUT(val[0].y, val[1].y, val[2].y, val[3].y);
    }
    if (ex > 2) {
        STORE_OUT(val[0].z, val[1].z, val[2].z, val[3].z);
    }
    if (ex > 3) {
        STORE_OUT(val[0].w, val[1].w, val[2].w, val[3].w);
    }
}
#endif

#if defined(TRANS_C4_TO_C1)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if ((idx << 2) >= x || idy >= y) {
        return;
    }
    const int idc = idz % ((c + 3) >> 2);
    const int idn = idz / ((c + 3) >> 2);
#if defined(USE_INPUT_IMG)
    int4 in_off = (int4)((idx << 2), idy, idz, 0);
#else
    int in_off = (idz * iy_str + idy + iy_off) * ix_str + (idx << 2) + ix_off;
#endif
    char ec = ((idc << 2) + 4 <= c) ? 4 : (c & 3);
    char ex = ((idx << 2) + 4 <= x) ? 4 : (x & 3);
    T4 val[4];
    val[0] = 0;
    val[1] = 0;
    val[2] = 0;
    val[3] = 0;
    LOAD_INPUT(val[0]);
    if (ex > 1) {
        LOAD_INPUT(val[1]);
    }
    if (ex > 2) {
        LOAD_INPUT(val[2]);
    }
    if (ex > 3) {
        LOAD_INPUT(val[3]);
    }

#if defined(USE_INPUT_IMG)
    int4 in_off = (int4)(idx, idy, idn * c + (idc << 2), 0);
#else
    const int iz_off = (idn * c + (idc << 2)) * ixy_str;
    int in_off = iz_off + (idy + iy_off) * ix_str + (idx << 2) + ix_off + offset_in;
#endif

#if defined(USE_OUTPUT_IMG)
    int4 out_off = (int4)(idx, idy, idn * c + (idc << 2), 0);
#else
    int oxy_str = ox_str * oy_str;
    const int oz_off = (idn * c + (idc << 2)) * oxy_str;
    int out_off = oz_off + (idy + oy_off) * ox_str + (idx << 2) + ox_off + offset_out;
#endif

    STORE_OUT(val[0].x, val[1].x, val[2].x, val[3].x);
    if (ec > 1) {
        STORE_OUT(val[0].y, val[1].y, val[2].y, val[3].y);
    }
    if (ec > 2) {
        STORE_OUT(val[0].z, val[1].z, val[2].z, val[3].z);
    }
    if (ec > 3) {
        STORE_OUT(val[0].w, val[1].w, val[2].w, val[3].w);
    }
}
#endif
