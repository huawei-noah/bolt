// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#if defined(USE_HALF)
#define READ_IMAGE(image, sampler, coord) read_imageh(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data) write_imageh(image, coord, data)
#else
#define READ_IMAGE(image, sampler, coord) read_imagef(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data) write_imagef(image, coord, data)
#endif
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

/*these parameters are belong to matrix mult/add and conv*/
/*they are extract from HDR model*/
/*they may be changful for different model*/
#define guide_cal(v, g)                                                                    \
    {                                                                                      \
        T3 tmp;                                                                            \
        tmp.x = v.x * (T)0.900616 - v.y * (T)0.1006 - v.z * (T)0.058384 + (T)0.072721;     \
        tmp.y = -v.x * (T)0.079311 + v.y * (T)0.91976 - v.z * (T)0.037624 + (T)0.124359;   \
        tmp.z = -v.x * (T)0.068347 - v.y * (T)0.069032 + v.z * (T)0.975032 + (T)0.129721;  \
        tmp.x = (tmp.x < 0) ? 0 : tmp.x;                                                   \
        tmp.y = (tmp.y < 0) ? 0 : tmp.y;                                                   \
        tmp.z = (tmp.z < 0) ? 0 : tmp.z;                                                   \
        tmp.x = tmp.x * (T)0.003211 * 16;                                                  \
        tmp.y = tmp.y * (T)0.007948 * 16;                                                  \
        tmp.z = tmp.z * (T)0.046259 * 16;                                                  \
        g = tmp.x * (T)0.249512 + tmp.y * (T)0.274577 + tmp.z * (T)0.324276 + (T)0.078941; \
    }

#if defined(CONV)
#if defined(UCHAR)
__kernel void bilateral_slice_apply_c12_conv_uchar
#else
__kernel void bilateral_slice_apply_c12_conv
#endif
#else
#if defined(UCHAR)
__kernel void bilateral_slice_apply_c12_uchar
#else
__kernel void bilateral_slice_apply_c12
#endif
#endif
    (const int w,
        const int wh,
        const int gc,
        const int gw,
        const int gh,
        const int gcw,
        const int gd,
        const int coe,
        const int bx,
        const int by,
        const float scale_x,
        const float scale_y,
        global const T *guide,
        global const T *grid,
#if defined(UCHAR)
        global const uchar *input,
        global uchar *out)
{
#else
        global const T *input,
        global T *out)
{
#endif

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= bx || y >= by) {
        return;
    }
    int in_off = y * w + x;
    T3 in_val;
#if defined(UCHAR)
    uchar3 tmp = vload3(0, input + in_off * 3);
    in_val.x = tmp.x / 256.0;
    in_val.y = tmp.y / 256.0;
    in_val.z = tmp.z / 256.0;
#else
    in_val = vload3(0, input + in_off * 3);
#endif

    T gx = (x + (T)0.5) * (T)scale_x;
    T gz;
#if defined(CONV)
    guide_cal(in_val, gz);
#else
    gz = guide[in_off];
#endif
    gz = gz * gd;
    char fx = (char)floor(gx - (T)0.5);
    char fz = (char)floor(gz - (T)0.5);

    char i = 0;
    char k = 0;
    char x_ = fx;
    char z_ = fz;
    if (fx < 0) {
        x_ = 0;
        i = 1;
    }
    if (fz < 0) {
        z_ = 0;
        k = 1;
    }
    if (fx == gw - 1) {
        i = 1;
    }
    if (fz == gd - 1) {
        k = 1;
    }

    T8 g_val[3];
    T4 p;
    T4 sum[3];
    T2 wx, wz;
    sum[0] = (T4)0;
    sum[1] = (T4)0;
    sum[2] = (T4)0;

    wx.s0 = (T)1 - fabs(fx + (T)0.5 - gx);
    wx.s1 = (T)1 - fabs(fx + (T)1.5 - gx);
    wz.s0 = (T)1 - fabs(fz + (T)0.5 - gz);
    wz.s1 = (T)1 - fabs(fz + (T)1.5 - gz);

    if (wx.s0 < 0) {
        wx.s0 = 0;
    }
    if (wx.s1 < 0) {
        wx.s0 = 0;
    }
    if (wz.s0 < 0) {
        wz.s0 = 0;
    }
    if (wz.s1 < 0) {
        wz.s0 = 0;
    }

    p.xy = wx.s0 * wz;
    p.zw = wx.s1 * wz;

    int grid_off = y * gcw + x_ * gc + z_ * coe;
    g_val[0] = vload8(0, grid + grid_off);
    g_val[1] = vload8(0, grid + grid_off + 8);
    p.x = p.x + (T)k * p.y + (T)i * (p.z + (T)k * p.w);
    sum[0] += g_val[0].s0123 * p.x;
    sum[1] += g_val[0].s4567 * p.x;
    sum[2] += g_val[1].s0123 * p.x;
    if (k == 0) {
        p.y = p.y + (T)i * p.w;
        g_val[2] = vload8(0, grid + grid_off + 16);
        sum[0] += g_val[1].s4567 * p.y;
        sum[1] += g_val[2].s0123 * p.y;
        sum[2] += g_val[2].s4567 * p.y;
    }

    if (i == 0) {
        grid_off += gc;
        p.z = p.z + (T)k * p.w;
        g_val[0] = vload8(0, grid + grid_off);
        g_val[1] = vload8(0, grid + grid_off + 8);
        sum[0] += g_val[0].s0123 * p.z;
        sum[1] += g_val[0].s4567 * p.z;
        sum[2] += g_val[1].s0123 * p.z;
        if (k == 0) {
            g_val[2] = vload8(0, grid + grid_off + 16);
            sum[0] += g_val[1].s4567 * p.w;
            sum[1] += g_val[2].s0123 * p.w;
            sum[2] += g_val[2].s4567 * p.w;
        }
    }

    sum[0].x = sum[0].x * in_val.x + sum[0].y * in_val.y + sum[0].z * in_val.z + sum[0].w;
    sum[1].x = sum[1].x * in_val.x + sum[1].y * in_val.y + sum[1].z * in_val.z + sum[1].w;
    sum[2].x = sum[2].x * in_val.x + sum[2].y * in_val.y + sum[2].z * in_val.z + sum[2].w;
#if defined(UCHAR)
    tmp.x = (uchar)(sum[0].x * 256.0);
    tmp.y = (uchar)(sum[1].x * 256.0);
    tmp.z = (uchar)(sum[2].x * 256.0);
    vstore3(tmp, 0, out + in_off * 3);
#else
    vstore3((T3)(sum[0].x, sum[1].x, sum[2].x), 0, out + in_off * 3);
#endif
}
