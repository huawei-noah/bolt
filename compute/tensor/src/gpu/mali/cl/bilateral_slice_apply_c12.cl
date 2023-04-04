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

/*these parameters are belong to matrix mult/add and conv*/
/*they are extract from HDR model*/
/*they may be changful for different model*/
inline T guide_cal0(T3 v)
{
    T3 tmp;
    tmp.x = v.x * (T)0.900616 - v.y * (T)0.1006 - v.z * (T)0.058384 + (T)0.072721;
    tmp.y = -v.x * (T)0.079311 + v.y * (T)0.91976 - v.z * (T)0.037624 + (T)0.124359;
    tmp.z = -v.x * (T)0.068347 - v.y * (T)0.069032 + v.z * (T)0.975032 + (T)0.129721;
    tmp = max(tmp, (T)0);
    tmp.x = tmp.x * (T)0.003211 * 16;
    tmp.y = tmp.y * (T)0.007948 * 16;
    tmp.z = tmp.z * (T)0.046259 * 16;
    T g = tmp.x * (T)0.249512 + tmp.y * (T)0.274577 + tmp.z * (T)0.324276 + (T)0.078941;
    return g;
}

inline T guide_cal1(T3 v)
{
    T4 a = {v.x, v.y, v.z, 1};

    T4 wx = {0.9266905188560486, -0.07651382684707642, -0.11796596646308899, 0.03732128441333771};
    T4 wy = {0.016965966671705246, 1.0332931280136108, 0.09558156877756119, 0.049296945333480835};
    T4 wz = {-0.060142070055007935, -0.0184615608304739, 0.9641872048377991, 0.03588166460394859};

    T x = dot(a, wx);
    T y = dot(a, wy);
    T z = dot(a, wz);

    T16 sx = {-0.04031608998775482, 0.203898087143898, 0.21509018540382385, 0.2156994342803955,
        0.22189579904079437, 0.2710961699485779, 0.33060845732688904, 0.3510134816169739,
        0.3799624741077423, 0.4165642559528351, 0.5429311394691467, 0.6519719958305359,
        0.7579551339149475, 0.8117461800575256, 0.8115477561950684, 0.811525821685791};

    T16 sy = {-0.04493796080350876, 0.2501078248023987, 0.24961410462856293, 0.24829524755477905,
        0.25029096007347107, 0.25275537371635437, 0.2535839378833771, 0.25915712118148804,
        0.992545485496521, 0.869307279586792, 0.8143411874771118, 0.8268355131149292,
        0.849763810634613, 0.8641695380210876, 0.8749480843544006, 0.9124495387077332};

    T16 sz = {-0.0450710691511631, 0.17914339900016785, 0.20727036893367767, 0.21128158271312714,
        0.785589873790741, 0.40014126896858215, 0.39716723561286926, 0.4003089666366577,
        0.5749346613883972, 0.6277766227722168, 0.7884474992752075, 0.788446307182312,
        0.789533257484436, 0.7905913591384888, 0.7964500188827515, 0.7964839339256287};

    T16 rx = max(x - sx, (T)0);
    T16 ry = max(x - sy, (T)0);
    T16 rz = max(z - sz, (T)0);

    T16 mx = {0.9483454823493958, -0.02504969760775566, -0.0731356292963028, -0.08960649371147156,
        -0.0989985391497612, -0.0911787822842598, -0.07849951088428497, -0.07431424409151077,
        -0.05982533469796181, -0.027073463425040245, 0.09377846121788025, 0.07562971860170364,
        -0.05076618492603302, 0.2615104913711548, 0.42631882429122925, 0.6887183785438538};

    T16 my = {0.9732255339622498, -0.03841959312558174, -0.07476486265659332, -0.08849595487117767,
        -0.10008298605680466, -0.10915014147758484, -0.1108635663986206, -0.09364574402570724,
        -0.04355158284306526, -0.015994733199477196, -0.025348246097564697, -0.051913388073444366,
        -0.07183714956045151, -0.0823502317070961, -0.09460879862308502, -0.13453315198421478};

    T16 mz = {0.951180636882782, -0.014929438941180706, -0.022745108231902122, -0.042111292481422424,
        0.061638616025447845, -0.04308458790183067, -0.050973013043403625, -0.045611534267663956,
        0.037990815937519073, 0.04962018504738808, 0.15617141127586365, 0.13662904500961304,
        0.16109246015548706, 0.160025492310524, 0.12079561501741409, 0.15001150965690613};

    x = dot(rx.s0123, mx.s0123) + dot(rx.s4567, mx.s4567) + dot(rx.s89ab, mx.s89ab) +
        dot(rx.scdef, mx.scdef);
    y = dot(ry.s0123, my.s0123) + dot(ry.s4567, my.s4567) + dot(ry.s89ab, my.s89ab) +
        dot(ry.scdef, my.scdef);
    z = dot(rz.s0123, mz.s0123) + dot(rz.s4567, mz.s4567) + dot(rz.s89ab, mz.s89ab) +
        dot(rz.scdef, mz.scdef);

    T4 t1 = {x, y, z, 1};
    T4 w = {0.28540247678756714, 0.31782254576683044, 0.28381019830703735, 0.06326253712177277};
    T g = dot(t1, w);

    g = min(max((T)0., g), (T)1.);

    return g;
}

__kernel void KERNEL_NAME 
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
    in_val.x = tmp.x / 255.0;
    in_val.y = tmp.y / 255.0;
    in_val.z = tmp.z / 255.0;
#else
    in_val = vload3(0, input + in_off * 3);
#endif

    T gx = (x + (T)0.5) * (T)scale_x;
#if defined(CONV)
    T gz = guide_cal1(in_val);
#else
    T gz = guide[in_off];
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
    tmp.x = (uchar)(sum[0].x * 255.0);
    tmp.y = (uchar)(sum[1].x * 255.0);
    tmp.z = (uchar)(sum[2].x * 255.0);
    vstore3(tmp, 0, out + in_off * 3);
#else
    vstore3((T3)(sum[0].x, sum[1].x, sum[2].x), 0, out + in_off * 3);
#endif
}
