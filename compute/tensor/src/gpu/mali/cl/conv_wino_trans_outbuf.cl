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
#define MANGLE_NAME_IMPL(base, IOM, AM, AN) base##IOM##AM##AN
#define MANGLE_NAME(base, IOM, AM, AN) MANGLE_NAME_IMPL(base, IOM, AM, AN)
#define AN
#if defined(USE_ALIGN)
#define AN align
#endif

#define loadR(val, str, off, in)    \
    {                               \
        val[0] = in[off];           \
        val[1] = in[off + str];     \
        val[2] = in[off + str * 2]; \
        val[3] = in[off + str * 3]; \
        val[4] = in[off + str * 4]; \
        val[5] = in[off + str * 5]; \
    }

#define calCore(t, s_f2, s_f3, s_f4, s) \
    {                                   \
        t = FMA(s_f2, s[2], s[1]);      \
        t = FMA(s_f3, s[3], t);      \
        t = FMA(s_f4, s[4], t);      \
        t.x += s[0];                    \
        t.w += s[5];                    \
    }

#if defined(USE_OUTPUT_IMG)
#if defined(USE_ALIGN)
#define STORE_OUT(v0, v1, v2, v3, out)                                                    \
    {                                                                                     \
        WRITE_IMAGE(out, (int4)(x_off, y_off, idz, 0), (T4)(v0.x, v1.x, v2.x, v3.x));     \
        WRITE_IMAGE(out, (int4)(x_off + 1, y_off, idz, 0), (T4)(v0.y, v1.y, v2.y, v3.y)); \
        WRITE_IMAGE(out, (int4)(x_off + 2, y_off, idz, 0), (T4)(v0.z, v1.z, v2.z, v3.z)); \
        WRITE_IMAGE(out, (int4)(x_off + 3, y_off, idz, 0), (T4)(v0.w, v1.w, v2.w, v3.w)); \
        y_off += 1;                                                                       \
    }
#else
#define STORE_OUT(v0, v1, v2, v3, x_off, y_off, ow, out)                                      \
    {                                                                                         \
        WRITE_IMAGE(out, (int4)(x_off, y_off, idz, 0), (T4)(v0.x, v1.x, v2.x, v3.x));         \
        if (x_off + 1 < ow) {                                                                 \
            WRITE_IMAGE(out, (int4)(x_off + 1, y_off, idz, 0), (T4)(v0.y, v1.y, v2.y, v3.y)); \
        }                                                                                     \
        if (x_off + 2 < ow) {                                                                 \
            WRITE_IMAGE(out, (int4)(x_off + 2, y_off, idz, 0), (T4)(v0.z, v1.z, v2.z, v3.z)); \
        }                                                                                     \
        if (x_off + 3 < ow) {                                                                 \
            WRITE_IMAGE(out, (int4)(x_off + 3, y_off, idz, 0), (T4)(v0.w, v1.w, v2.w, v3.w)); \
        }                                                                                     \
    }
#endif
#else
#if defined(USE_ALIGN)
#define STORE_OUT(v0, v1, v2, v3, out)                                                  \
    {                                                                                   \
        vstore16((T16)(v0.x, v1.x, v2.x, v3.x,                    \ 
            v0.y,                                                                       \
                     v1.y, v2.y, v3.y, v0.z, v1.z, v2.z, v3.z, v0.w, v1.w, v2.w, v3.w), \
            0, out + out_off);                                                          \
        out_off += (ow_str << 2);                                                       \
    }
#else
#define STORE_OUT(v0, v1, v2, v3, x_off, y_off, ow, out)                  \
    {                                                                     \
        vstore4((T4)(v0.x, v1.x, v2.x, v3.x), 0, out + out_off);          \
        if (x_off + 1 < ow) {                                             \
            vstore4((T4)(v0.y, v1.y, v2.y, v3.y), 0, out + out_off + 4);  \
        }                                                                 \
        if (x_off + 2 < ow) {                                             \
            vstore4((T4)(v0.z, v1.z, v2.z, v3.z), 0, out + out_off + 8);  \
        }                                                                 \
        if (x_off + 3 < ow) {                                             \
            vstore4((T4)(v0.w, v1.w, v2.w, v3.w), 0, out + out_off + 12); \
        }                                                                 \
        out_off += (ow_str << 2);                                         \
    }
#endif
#endif

__kernel void MANGLE_NAME(conv_wino_trans_outbuf_, IOM, AM, AN)(const int wino_w,
    const int wino_h,
    const int pw_str,
    const int pwh_str,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int ow,
    const int oh,
    const int oc,
    __read_only image1d_t bias,
    __global const T *in,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= wino_w || idy >= wino_h) {
        return;
    }

    T4 r0, r1, r2, r3;  //channel 0   r0~3 -> row 0~3 for 16 res
    T4 r4, r5, r6, r7;  //        1
    T4 r8, r9, ra, rb;  //        2
    T4 rc, rd, re, rf;  //        3
    T4 bias_v4 = READ_IMAGE(bias, sampler, idz);

    int iz = idz << 2;
    int in_off = iz * pw_str + idy * wino_w + idx;
    T s[6];
    T4 t;
    T bias_val;
    T4 s_f2 = {1, -1, 1, -1};
    T4 s_f3 = {1, 2, 4, 8};
    T4 s_f4 = {1, -2, 4, -8};
    for (uchar ii = 0; ii < 4; ii++) {
        r0 = r4;
        r1 = r5;
        r2 = r6;
        r3 = r7;

        r4 = r8;
        r5 = r9;
        r6 = ra;
        r7 = rb;

        r8 = rc;
        r9 = rd;
        ra = re;
        rb = rf;
        if (iz + ii >= oc) {
            rc = 0;
            rd = 0;
            re = 0;
            rf = 0;
            continue;
        }

        if (ii == 0) {
            bias_val = bias_v4.x;
        }
        if (ii == 1) {
            bias_val = bias_v4.y;
        }
        if (ii == 2) {
            bias_val = bias_v4.z;
        }
        if (ii == 3) {
            bias_val = bias_v4.w;
        }

        rd = (T4)bias_val;
        re = (T4)bias_val;

        loadR(s, pwh_str, in_off, in);
        calCore(t, s_f2, s_f3, s_f4, s);
        rc = FMA((T)0.11111, t, bias_val);
        loadR(s, pwh_str, in_off + 30 * pwh_str, in);
        calCore(t, s_f2, s_f3, s_f4, s);
        rf = FMA((T)0.11111, t, bias_val);

        loadR(s, pwh_str, in_off + 6 * pwh_str, in);
        calCore(t, s_f2, s_f3, s_f4, s);
        t = (T)0.11111 * t;
        rc += t;
        rd += t;
        re += t;
        rf += t;
        loadR(s, pwh_str, in_off + 12 * pwh_str, in);
        calCore(t, s_f2, s_f3, s_f4, s);
        t = (T)0.11111 * t;
        rc += t;
        rd -= t;
        re += t;
        rf -= t;
        loadR(s, pwh_str, in_off + 18 * pwh_str, in);
        calCore(t, s_f2, s_f3, s_f4, s);
        t = (T)0.11111 * t;
        rc += t;
        rd += 2 * t;
        re += 4 * t;
        rf += 8 * t;
        loadR(s, pwh_str, in_off + 24 * pwh_str, in);
        calCore(t, s_f2, s_f3, s_f4, s);
        t = (T)0.11111 * t;
        rc += t;
        rd -= 2 * t;
        re += 4 * t;
        rf -= 8 * t;

        ACTIVATION_V4(rc);
        ACTIVATION_V4(rd);
        ACTIVATION_V4(re);
        ACTIVATION_V4(rf);
        in_off += pw_str;
    }

    int x_off = idx << 2;
    int y_off = idy << 2;
#if !defined(USE_OUTPUT_IMG)
    int out_off = (idz * oh_str + y_off) * (ow_str << 2) + (x_off << 2) + (o_off << 2);
#endif
#if defined(USE_ALIGN)
    STORE_OUT(r0, r4, r8, rc, out);
    STORE_OUT(r1, r5, r9, rd, out);
    STORE_OUT(r2, r6, ra, re, out);
    STORE_OUT(r3, r7, rb, rf, out);
#else
    STORE_OUT(r0, r4, r8, rc, x_off, y_off, ow, out);
    if (y_off + 1 < oh) {
        STORE_OUT(r1, r5, r9, rd, x_off, y_off + 1, ow, out);
    }
    if (y_off + 2 < oh) {
        STORE_OUT(r2, r6, ra, re, x_off, y_off + 2, ow, out);
    }
    if (y_off + 3 < oh) {
        STORE_OUT(r3, r7, rb, rf, x_off, y_off + 3, ow, out);
    }
#endif
}
