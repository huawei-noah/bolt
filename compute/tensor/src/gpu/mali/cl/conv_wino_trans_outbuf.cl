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
#define loadR(val, str, off, in)    \
    {                               \
        val[0] = in[off];           \
        val[1] = in[off + str];     \
        val[2] = in[off + str * 2]; \
        val[3] = in[off + str * 3]; \
        val[4] = in[off + str * 4]; \
        val[5] = in[off + str * 5]; \
    }

#define calCore(s, t, tmp)                    \
    {                                         \
        t.x = s[1] + s[2];                    \
        t.y = s[3] + s[4];                    \
        t.z = s[1] - s[2];                    \
        t.w = s[3] - s[4];                    \
        tmp[0] = s[0] + t.x + t.y;            \
        tmp[1] = t.z + (T)(2.0) * t.w;        \
        tmp[2] = t.x + (T)(4.0) * t.y;        \
        tmp[3] = t.z + (T)(8.0) * t.w + s[5]; \
    }

#if defined(ALIGN)
#if defined(USE_RELU)
__kernel void conv_wino_trans_outbuf_relu_align
#else
__kernel void conv_wino_trans_outbuf_align
#endif
#else
#if defined(USE_RELU)
__kernel void conv_wino_trans_outbuf_relu
#else
__kernel void conv_wino_trans_outbuf
#endif
#endif
    (const int wino_h,
        const int wino_w,
        const int pw_str,
        const int pwh_str,
        const int oh_str,
        const int ow_str,
        const int oh_off,
        const int ow_off,
        const int oh,
        const int ow,
        __read_only image1d_t bias,
        __global const T *in,
        __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= wino_h || idy >= wino_w) {
        return;
    }

    T4 r0, r1, r2, r3;
    T4 r4, r5, r6, r7;
    T4 r8, r9, ra, rb;
    T4 rc, rd, re, rf;
    T4 bias_v4 = READ_IMAGE(bias, sampler, idz);

    int in_off = (idz << 2) * pw_str + idy * wino_h + idx;
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

        T s[6];
        T4 t;
        T bias_val;
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
        for (uchar i = 0; i < 2; ++i) {
            rc.x = rf.x;
            rc.y = rf.y;
            rc.z = rf.z;
            rc.w = rf.w;
            loadR(s, pwh_str, in_off + i * 30 * pwh_str, in);
            for (uchar j = 0; j < 4; ++j) {
                rf.x = rf.y;
                rf.y = rf.z;
                rf.z = rf.w;
                rf.w = bias_val;
                if (j == 0) {
                    rf.w += s[0] + s[1] + s[2] + s[3] + s[4];
                }
                if (j == 1) {
                    rf.w += s[1] - s[2] + (T)2 * (s[3] - s[4]);
                }
                if (j == 2) {
                    rf.w += s[1] + s[2] + (T)4 * (s[3] + s[4]);
                }
                if (j == 3) {
                    rf.w += s[1] - s[2] + (T)8 * (s[3] - s[4]) + s[5];
                }
            }
        }

        for (uchar i = 0; i < 4; ++i) {
            loadR(s, pwh_str, in_off + (i + 1) * 6 * pwh_str, in);
            for (uchar j = 0; j < 4; ++j) {
                t.x = t.y;
                t.y = t.z;
                t.z = t.w;
                if (j == 0) {
                    t.w = s[0] + s[1] + s[2] + s[3] + s[4];
                }
                if (j == 1) {
                    t.w = s[1] - s[2] + (T)2 * (s[3] - s[4]);
                }
                if (j == 2) {
                    t.w = s[1] + s[2] + (T)4 * (s[3] + s[4]);
                }
                if (j == 3) {
                    t.w = s[1] - s[2] + (T)8 * (s[3] - s[4]) + s[5];
                }
            }
            if (i == 0) {
                rc += t;
                rd += t;
                re += t;
                rf += t;
            }
            if (i == 1) {
                rc += t;
                rd -= t;
                re += t;
                rf -= t;
            }
            if (i == 2) {
                rc += t;
                rd += (T)2 * t;
                re += (T)4 * t;
                rf += (T)8 * t;
            }
            if (i == 3) {
                rc += t;
                rd -= (T)2 * t;
                re += (T)4 * t;
                rf -= (T)8 * t;
            }
        }
        ACTIVATION_V4(rc);
        ACTIVATION_V4(rd);
        ACTIVATION_V4(re);
        ACTIVATION_V4(rf);
        in_off += pw_str;
    }

    const int x_off = idx << 2;
    const int y_off = idy << 2;
    int out_off = (idz * ow_str + y_off + ow_off) * (oh_str << 2) + (x_off << 2) + (oh_off << 2);
#if defined(ALIGN)
    vstore16((T16)(r0.x, r4.x, r8.x, rc.x, r1.x, r5.x, r9.x, rd.x, r2.x, r6.x, ra.x, re.x, r3.x,
                 r7.x, rb.x, rf.x),
        0, out + out_off);
    out_off += (oh_str << 2);
    vstore16((T16)(r0.y, r4.y, r8.y, rc.y, r1.y, r5.y, r9.y, rd.y, r2.y, r6.y, ra.y, re.y, r3.y,
                 r7.y, rb.y, rf.y),
        0, out + out_off);
    out_off += (oh_str << 2);
    vstore16((T16)(r0.z, r4.z, r8.z, rc.z, r1.z, r5.z, r9.z, rd.z, r2.z, r6.z, ra.z, re.z, r3.z,
                 r7.z, rb.z, rf.z),
        0, out + out_off);
    out_off += (oh_str << 2);
    vstore16((T16)(r0.w, r4.w, r8.w, rc.w, r1.w, r5.w, r9.w, rd.w, r2.w, r6.w, ra.w, re.w, r3.w,
                 r7.w, rb.w, rf.w),
        0, out + out_off);
#else
    vstore4((T4)(r0.x, r4.x, r8.x, rc.x), 0, out + out_off);
    if (x_off + 1 < oh) {
        vstore4((T4)(r1.x, r5.x, r9.x, rd.x), 0, out + out_off + 4);
    }
    if (x_off + 2 < oh) {
        vstore4((T4)(r2.x, r6.x, ra.x, re.x), 0, out + out_off + 8);
    }
    if (x_off + 3 < oh) {
        vstore4((T4)(r3.x, r7.x, rb.x, rf.x), 0, out + out_off + 12);
    }

    if (y_off + 1 < ow) {
        out_off += (oh_str << 2);
        vstore4((T4)(r0.y, r4.y, r8.y, rc.y), 0, out + out_off);
        if (x_off + 1 < oh) {
            vstore4((T4)(r1.y, r5.y, r9.y, rd.y), 0, out + out_off + 4);
        }
        if (x_off + 2 < oh) {
            vstore4((T4)(r2.y, r6.y, ra.y, re.y), 0, out + out_off + 8);
        }
        if (x_off + 3 < oh) {
            vstore4((T4)(r3.y, r7.y, rb.y, rf.y), 0, out + out_off + 12);
        }
    }

    if (y_off + 2 < ow) {
        out_off += (oh_str << 2);
        vstore4((T4)(r0.z, r4.z, r8.z, rc.z), 0, out + out_off);
        if (x_off + 1 < oh) {
            vstore4((T4)(r1.z, r5.z, r9.z, rd.z), 0, out + out_off + 4);
        }
        if (x_off + 2 < oh) {
            vstore4((T4)(r2.z, r6.z, ra.z, re.z), 0, out + out_off + 8);
        }
        if (x_off + 3 < oh) {
            vstore4((T4)(r3.z, r7.z, rb.z, rf.z), 0, out + out_off + 12);
        }
    }

    if (y_off + 3 < ow) {
        out_off += (oh_str << 2);
        vstore4((T4)(r0.w, r4.w, r8.w, rc.w), 0, out + out_off);
        if (x_off + 1 < oh) {
            vstore4((T4)(r1.w, r5.w, r9.w, rd.w), 0, out + out_off + 4);
        }
        if (x_off + 2 < oh) {
            vstore4((T4)(r2.w, r6.w, ra.w, re.w), 0, out + out_off + 8);
        }
        if (x_off + 3 < oh) {
            vstore4((T4)(r3.w, r7.w, rb.w, rf.w), 0, out + out_off + 12);
        }
    }
#endif
}
