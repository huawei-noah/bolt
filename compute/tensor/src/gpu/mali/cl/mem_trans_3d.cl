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
#define MANGLE_NAME_IMPL(base, IOM, IFM, OFM) base##IOM##IFM##OFM
#define MANGLE_NAME(base, IOM, IFM, OFM) MANGLE_NAME_IMPL(base, IOM, IFM, OFM)

#if defined(USE_INPUT_NCHW)
#define IFM nchw_
#elif defined(USE_INPUT_NCHWC4)
#define IFM nchwc4_
#endif

#if defined(USE_OUTPUT_NCHW)
#define OFM to_nchw
#elif defined(USE_OUTPUT_NCHWC4)
#define OFM to_nchwc4
#endif

#if defined(USE_INPUT_NCHW) && defined(USE_OUTPUT_NCHWC4)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(v)                      \
    {                                      \
        LOAD_MEM_V4_C1(v, in_off, ew, in); \
        in_off.z += t;                     \
    }
#else
#define LOAD_INPUT(v)          \
    {                          \
        v = in[in_off];        \
        in_off += iwh_str * t; \
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

#if defined(USE_INPUT_NCHWC4) && defined(USE_OUTPUT_NCHW)
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
        STORE_MEM_V4_C1(((T4)(v0, v1, v2, v3)), out_off, ew, out); \
        out_off.z += t;                                            \
    }
#else
#define STORE_OUT(v0, v1, v2, v3)                                  \
    {                                                              \
        STORE_MEM_V4_C1(((T4)(v0, v1, v2, v3)), out_off, ew, out); \
        out_off += owh_str * t;                                    \
    }
#endif
#endif

__kernel void MANGLE_NAME(mem_trans_3d_, IOM, IFM, OFM)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int w,
    const int h,
    const int c,
    const int t,
    const int offset_in,
    const int offset_out,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
#if defined(USE_INPUT_NCHW) && defined(USE_OUTPUT_NCHW)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if ((idx << 2) >= w || idy >= h) {
        return;
    }
    char ew = (((idx << 2) + 4) <= w) ? 4 : (w & 3);

    T4 val = 0;
#if defined(USE_INPUT_IMG)
    int4 in_off = (int4)(idx, idy, idz, 0);
#else
    const int in_off = (idz * ih_str + idy) * iw_str + (idx << 2) + i_off + offset_in;
#endif
    LOAD_MEM_V4_C1(val, in_off, ew, in);

#if defined(USE_OUTPUT_IMG)
    int4 out_off = (int4)(idx, idy, idz, 0);
#else
    const int out_off = (idz * oh_str + idy) * ow_str + (idx << 2) + o_off + offset_out;
#endif
    STORE_MEM_V4_C1(val, out_off, ew, out);
}
#endif

#if defined(USE_INPUT_NCHW) && defined(USE_OUTPUT_NCHWC4)
#if defined(USE_INPUT_IMG)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int idt = idz % t;
    const int idc = idz / t;
    if ((idx << 2) >= w || idy >= h || idt >= t) {
        return;
    }
    int4 in_off = (int4)(idx, idy, idt + (idc << 2) * t, 0);
    char ec = ((idc << 2) + 4 <= c) ? 4 : (c & 3);
    char ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
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
    int out_off = (idz * oh_str + idy) * ow_str + (idx << 2) + o_off;
#endif
    STORE_OUT(val[0].x, val[1].x, val[2].x, val[3].x);
    if (ew > 1) {
        STORE_OUT(val[0].y, val[1].y, val[2].y, val[3].y);
    }
    if (ew > 2) {
        STORE_OUT(val[0].z, val[1].z, val[2].z, val[3].z);
    }
    if (ew > 3) {
        STORE_OUT(val[0].w, val[1].w, val[2].w, val[3].w);
    }
}
#else
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int idt = idz % t;
    const int idc = idz / t;
    if (idx >= w || idy >= h || idt >= t) {
        return;
    }

    T4 val = 0;
    char ec = ((idc << 2) + 4 <= c) ? 4 : (c & 3);
    int iwh_str = iw_str * ih_str;
    const int iz_off = (idc << 2) * iwh_str * t;
    int in_off = iz_off + (idt * ih_str + idy) * iw_str + idx + i_off + offset_in;
    LOAD_INPUT(val.x);
    if (ec > 1) {
        LOAD_INPUT(val.y);
    }
    if (ec > 2) {
        LOAD_INPUT(val.z);
    }
    if (ec > 3) {
        LOAD_INPUT(val.w);
    }
#if defined(USE_OUTPUT_IMG)
    int4 out_off = (int4)(idx, idy, idz, 0);
#else
    int out_off = (idz * oh_str + idy) * ow_str + idx + o_off;
#endif
    STORE_OUT(val.x, val.y, val.z, val.w);
}
#endif
#endif

#if defined(USE_INPUT_NCHWC4) && defined(USE_OUTPUT_NCHW)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int idt = idz % t;
    const int idc = idz / t;

    if ((idx << 2) >= w || idy >= h || idt >= t) {
        return;
    }
#if defined(USE_INPUT_IMG)
    int4 in_off = (int4)(idx << 2, idy, idz, 0);
#else
    int in_off = (idz * ih_str + idy) * iw_str + (idx << 2) + i_off;
#endif
    char ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
    char ec = ((idc << 2) + 4 <= c) ? 4 : (c & 3);
    T4 val[4];
    val[0] = 0;
    val[1] = 0;
    val[2] = 0;
    val[3] = 0;
    LOAD_INPUT(val[0]);
    if (ew > 1) {
        LOAD_INPUT(val[1]);
    }
    if (ew > 2) {
        LOAD_INPUT(val[2]);
    }
    if (ew > 3) {
        LOAD_INPUT(val[3]);
    }
#if defined(USE_OUTPUT_IMG)
    int4 out_off = (int4)(idx, idy, (idc << 2) * t + idt, 0);
#else
    int owh_str = ow_str * oh_str;
    int oz_off = ((idc << 2) * t + idt) * owh_str;
    int out_off = oz_off + idy * ow_str + (idx << 2) + o_off + offset_out;
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

#if defined(USE_INPUT_NCHWC4) && defined(USE_OUTPUT_NCHWC4)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= w || idy >= h) {
        return;
    }
    T4 val = 0;
#if defined(USE_INPUT_IMG)
    int4 in_off = (int4)(idx, idy, idz, 0);
    LOAD_MEM_V4(val, in_off, in);
#else
    int in_off = (idz * ih_str + idy) * iw_str + idx + i_off;
    LOAD_MEM_V4(val, in_off, in + offset_in);
#endif
#if defined(USE_OUTPUT_IMG)
    int4 out_off = (int4)(idx, idy, idz, 0);
    STORE_MEM_V4(val, out_off, out);
#else
    const int out_off = (idz * oh_str + idy) * ow_str + idx + o_off;
    STORE_MEM_V4(val, out_off, out + offset_out);
#endif
}
#endif
