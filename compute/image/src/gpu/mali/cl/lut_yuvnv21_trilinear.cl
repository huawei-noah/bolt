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

inline int4 get_id1(int x, int3 y)
{
    int4 v;
    v.s0 = x + y.s2;
    v.s1 = v.s0 + y.s0;
    v.s2 = v.s0 + y.s1;
    v.s3 = v.s2 + y.s0;
    return v;
}

inline T8 get_lut(__global const T *lut, int4 id, int offset)
{
    int4 tmp_id = id + offset;
    T8 v;
    v.s01 = vload2(0, lut + tmp_id.s0);
    v.s23 = vload2(0, lut + tmp_id.s1);
    v.s45 = vload2(0, lut + tmp_id.s2);
    v.s67 = vload2(0, lut + tmp_id.s3);
    return v;
}

inline T4 get_ww1(float g_d, float b_d)
{
    T4 ww;
    T2 tmp_gd = (T2)(1 - g_d, g_d);
    ww.s01 = tmp_gd * (T2)(1 - b_d);
    ww.s23 = tmp_gd * (T2)b_d;
    return ww;
}

inline T8 get_ww2(float r_d, T4 ww1)
{
    T8 ww;
    ww.s0246 = (T4)(1 - r_d) * ww1;
    ww.s1357 = (T4)r_d * ww1;
    return ww;
}

__kernel void KERNEL_NAME(__global const uchar *_image,
    __global const T *lut,
    __global uchar *_output,
    const int dim,
    const int shift,
    const float binsize,
    const int src_step,
    const int rows,
    const int src_n,
    const int src_chw,
    const int dst_chw)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int x_2 = x << 1;
    int y_2 = y << 1;
    if (x_2 >= src_step || y_2 >= rows || z >= src_n) {
        return;
    }
    __global const uchar *image = _image + z * src_chw;
    __global uchar *output = _output + z * dst_chw;

    int index_y = mad24(y_2, src_step, x_2);
    int index_u = mad24(rows + y, src_step, x_2);
    __global const uchar *ysrc = image + index_y;
    __global const uchar *usrc = image + index_u;
    __global uchar *ydst = output + index_y;
    __global uchar *udst = output + index_u;

    float4 ydata;
    float2 uvdata;
    ydata.s01 = (float2)(ysrc[0], ysrc[1]) / 255.0f;
    ydata.s23 = (float2)(ysrc[src_step], ysrc[src_step + 1]) / 255.0f;
    uvdata = (float2)(usrc[0], usrc[1]) / 255.0f;

    float4 y_gid = ydata / binsize;
    float2 uv_gid = uvdata / binsize;

    float4 y_fid = floor(y_gid);
    float2 uv_fid = floor(uv_gid);

    float4 y_dd = y_gid - y_fid;
    float2 uv_dd = uv_gid - uv_fid;

    int3 tmp_id;
    tmp_id.s0 = dim;
    tmp_id.s1 = dim * dim;
    tmp_id.s2 = tmp_id.s0 * uv_fid.s0 + tmp_id.s1 * uv_fid.s1;

    int4 id1 = get_id1(y_fid.s0, tmp_id);
    int4 id2 = get_id1(y_fid.s1, tmp_id);
    int4 id3 = get_id1(y_fid.s2, tmp_id);
    int4 id4 = get_id1(y_fid.s3, tmp_id);

    T4 ww1 = get_ww1(uv_dd.s0, uv_dd.s1);
    T8 tmp_ww1 = get_ww2(y_dd.s0, ww1);
    T8 tmp_ww2 = get_ww2(y_dd.s1, ww1);
    T8 tmp_ww3 = get_ww2(y_dd.s2, ww1);
    T8 tmp_ww4 = get_ww2(y_dd.s3, ww1);

    T8 y_lut1 = get_lut(lut, id1, 0);
    T8 y_lut2 = get_lut(lut, id2, 0);
    T8 y_lut3 = get_lut(lut, id3, 0);
    T8 y_lut4 = get_lut(lut, id4, 0);
    T8 u_lut = get_lut(lut, id1, shift);
    T8 v_lut = get_lut(lut, id1, shift * 2);

    float4 tmp_y;
    float2 tmp_uv;
    tmp_y.s0 = dot(tmp_ww1.s0123, y_lut1.s0123) + dot(tmp_ww1.s4567, y_lut1.s4567);
    tmp_y.s1 = dot(tmp_ww2.s0123, y_lut2.s0123) + dot(tmp_ww2.s4567, y_lut2.s4567);
    tmp_y.s2 = dot(tmp_ww3.s0123, y_lut3.s0123) + dot(tmp_ww3.s4567, y_lut3.s4567);
    tmp_y.s3 = dot(tmp_ww4.s0123, y_lut4.s0123) + dot(tmp_ww4.s4567, y_lut4.s4567);

    tmp_uv.s0 = dot(tmp_ww1.s0123, u_lut.s0123) + dot(tmp_ww1.s4567, u_lut.s4567);
    tmp_uv.s1 = dot(tmp_ww1.s0123, v_lut.s0123) + dot(tmp_ww1.s4567, v_lut.s4567);

    tmp_y *= 255.0f;
    tmp_uv *= 255.0f;

    vstore2((uchar2)(tmp_y.s0, tmp_y.s1), 0, ydst);
    vstore2((uchar2)(tmp_y.s2, tmp_y.s3), 0, ydst + src_step);
    vstore2((uchar2)(tmp_uv.s0, tmp_uv.s1), 0, udst);
}
