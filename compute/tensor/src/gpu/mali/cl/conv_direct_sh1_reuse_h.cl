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
#define MANGLE_NAME_IMPL(base, AM, FW, FH, ON, KN) base##AM##FW##FH##ON##KN
#define MANGLE_NAME(base, AM, FW, FH, ON, KN) MANGLE_NAME_IMPL(base, AM, FW, FH, ON, KN)

#if (ON == 2)
#define SET_BIAS_VAL(bv, ov) \
    {                        \
        ov.s0 = bv.x;        \
        ov.s1 = bv.y;        \
        ov.s2 = bv.z;        \
        ov.s3 = bv.w;        \
        ov.s4 = bv.x;        \
        ov.s5 = bv.y;        \
        ov.s6 = bv.z;        \
        ov.s7 = bv.w;        \
    }

#define calCore(iv, fv, ov)                                                     \
    {                                                                           \
        ov.s0 += iv.s0 * fv.s0 + iv.s1 * fv.s1 + iv.s2 * fv.s2 + iv.s3 * fv.s3; \
        ov.s1 += iv.s0 * fv.s4 + iv.s1 * fv.s5 + iv.s2 * fv.s6 + iv.s3 * fv.s7; \
        ov.s2 += iv.s0 * fv.s8 + iv.s1 * fv.s9 + iv.s2 * fv.sa + iv.s3 * fv.sb; \
        ov.s3 += iv.s0 * fv.sc + iv.s1 * fv.sd + iv.s2 * fv.se + iv.s3 * fv.sf; \
        ov.s4 += iv.s4 * fv.s0 + iv.s5 * fv.s1 + iv.s6 * fv.s2 + iv.s7 * fv.s3; \
        ov.s5 += iv.s4 * fv.s4 + iv.s5 * fv.s5 + iv.s6 * fv.s6 + iv.s7 * fv.s7; \
        ov.s6 += iv.s4 * fv.s8 + iv.s5 * fv.s9 + iv.s6 * fv.sa + iv.s7 * fv.sb; \
        ov.s7 += iv.s4 * fv.sc + iv.s5 * fv.sd + iv.s6 * fv.se + iv.s7 * fv.sf; \
    }

#define VLOAD_VEC(off, buf) vload8(0, buf + off);
#define VSTORE_VEC(v, off, buf)   \
    {                             \
        ACTIVATION_V8(v);         \
        vstore8(v, 0, buf + off); \
    }
#elif (ON == 4)
#define SET_BIAS_VAL(bv, ov) \
    {                        \
        ov.s0 = bv.x;        \
        ov.s1 = bv.y;        \
        ov.s2 = bv.z;        \
        ov.s3 = bv.w;        \
        ov.s4 = bv.x;        \
        ov.s5 = bv.y;        \
        ov.s6 = bv.z;        \
        ov.s7 = bv.w;        \
        ov.s8 = bv.x;        \
        ov.s9 = bv.y;        \
        ov.sa = bv.z;        \
        ov.sb = bv.w;        \
        ov.sc = bv.x;        \
        ov.sd = bv.y;        \
        ov.se = bv.z;        \
        ov.sf = bv.w;        \
    }
#define calCore(iv, fv, ov)                                                     \
    {                                                                           \
        ov.s0 += iv.s0 * fv.s0 + iv.s1 * fv.s1 + iv.s2 * fv.s2 + iv.s3 * fv.s3; \
        ov.s1 += iv.s0 * fv.s4 + iv.s1 * fv.s5 + iv.s2 * fv.s6 + iv.s3 * fv.s7; \
        ov.s2 += iv.s0 * fv.s8 + iv.s1 * fv.s9 + iv.s2 * fv.sa + iv.s3 * fv.sb; \
        ov.s3 += iv.s0 * fv.sc + iv.s1 * fv.sd + iv.s2 * fv.se + iv.s3 * fv.sf; \
        ov.s4 += iv.s4 * fv.s0 + iv.s5 * fv.s1 + iv.s6 * fv.s2 + iv.s7 * fv.s3; \
        ov.s5 += iv.s4 * fv.s4 + iv.s5 * fv.s5 + iv.s6 * fv.s6 + iv.s7 * fv.s7; \
        ov.s6 += iv.s4 * fv.s8 + iv.s5 * fv.s9 + iv.s6 * fv.sa + iv.s7 * fv.sb; \
        ov.s7 += iv.s4 * fv.sc + iv.s5 * fv.sd + iv.s6 * fv.se + iv.s7 * fv.sf; \
        ov.s8 += iv.s8 * fv.s0 + iv.s9 * fv.s1 + iv.sa * fv.s2 + iv.sb * fv.s3; \
        ov.s9 += iv.s8 * fv.s4 + iv.s9 * fv.s5 + iv.sa * fv.s6 + iv.sb * fv.s7; \
        ov.sa += iv.s8 * fv.s8 + iv.s9 * fv.s9 + iv.sa * fv.sa + iv.sb * fv.sb; \
        ov.sb += iv.s8 * fv.sc + iv.s9 * fv.sd + iv.sa * fv.se + iv.sb * fv.sf; \
        ov.sc += iv.sc * fv.s0 + iv.sd * fv.s1 + iv.se * fv.s2 + iv.sf * fv.s3; \
        ov.sd += iv.sc * fv.s4 + iv.sd * fv.s5 + iv.se * fv.s6 + iv.sf * fv.s7; \
        ov.se += iv.sc * fv.s8 + iv.sd * fv.s9 + iv.se * fv.sa + iv.sf * fv.sb; \
        ov.sf += iv.sc * fv.sc + iv.sd * fv.sd + iv.se * fv.se + iv.sf * fv.sf; \
    }

#define VLOAD_VEC(off, buf) vload16(0, buf + off);
#define VSTORE_VEC(v, off, buf)    \
    {                              \
        ACTIVATION_V16(v);         \
        vstore16(v, 0, buf + off); \
    }
#endif
__kernel void MANGLE_NAME(conv_direct_sh1_reuse_h_, AM, FW, FH, ON, KN)(const int ih_str,
    const int ihw_str,
    const int ic_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ohw_str,
    const int oh_off,
    const int ow_off,
    const int oh,
    const int oc,
    const int sw,
    const int in_str,
    const int on_str,
    const int bx,
    const int by,
    __global const T *in,
    __global const T *flt,
    __read_only image1d_t bias,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2) % (((oc + 3) >> 2) / KN);
    const int idn = get_global_id(2) / (((oc + 3) >> 2) / KN);
    if (idx >= bx || idy >= by) {
        return;
    }
#if (ON == 2)
    T8 in_val;
    T8 out_val[KN];
#elif (ON == 4)
    T16 in_val;
    T16 out_val[KN];
#endif
    T16 flt_val;
    T4 bias_val = read_imageh(bias, sampler, idz * KN);
    SET_BIAS_VAL(bias_val, out_val[0]);
#if (KN > 1)
    bias_val = read_imageh(bias, sampler, idz * KN + 1);
    SET_BIAS_VAL(bias_val, out_val[1]);
#endif
#if (KN > 2)
    bias_val = read_imageh(bias, sampler, idz * KN + 2);
    SET_BIAS_VAL(bias_val, out_val[2]);
    bias_val = read_imageh(bias, sampler, idz * KN + 3);
    SET_BIAS_VAL(bias_val, out_val[3]);
#endif
    int in_off = (idn * in_str + (idy + iw_off) * ih_str + idx * ON + ih_off) << 2;
    int flt_off = idz * ic_str * KN;
    int ihw_str_4 = ihw_str << 2;

    for (int i = 0; i < ic_str; ++i) {
        in_val = VLOAD_VEC(in_off, in);
#if (KN == 1)
        flt_val = vload16(flt_off, flt);
        calCore(in_val, flt_val, out_val[0]);
#elif (KN == 2)
        flt_val = vload16(flt_off, flt);
        calCore(in_val, flt_val, out_val[0]);
        flt_val = vload16(flt_off + 1, flt);
        calCore(in_val, flt_val, out_val[1]);
#elif (KN == 4)
        for (uchar j = 0; j < KN; ++j) {
            flt_val = vload16(flt_off + j, flt);
            if (j == 0) {
                calCore(in_val, flt_val, out_val[0]);
            }
            if (j == 1) {
                calCore(in_val, flt_val, out_val[1]);
            }
            if (j == 2) {
                calCore(in_val, flt_val, out_val[2]);
            }
            if (j == 3) {
                calCore(in_val, flt_val, out_val[3]);
            }
        }
#endif
        flt_off += KN;
        in_off += ihw_str_4;
    }
    int out_off = idn * on_str + idz * KN * ohw_str + (idy + ow_off) * oh_str + idx * ON + oh_off;
    out_off = out_off << 2;
    VSTORE_VEC(out_val[0], out_off, out);

#if (KN > 1)
    int ohw_str_4 = ohw_str << 2;
    out_off += ohw_str_4;
    VSTORE_VEC(out_val[1], out_off, out);
#endif

#if (KN > 2)
    out_off += ohw_str_4;
    VSTORE_VEC(out_val[2], out_off, out);
    out_off += ohw_str_4;
    VSTORE_VEC(out_val[3], out_off, out);
#endif
}
