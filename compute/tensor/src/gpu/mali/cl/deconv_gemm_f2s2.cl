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
#define MANGLE_NAME_IMPL(base, ON, KN) base##ON##KN
#define MANGLE_NAME(base, ON, KN) MANGLE_NAME_IMPL(base, ON, KN)

#if defined(REUSE_H)
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
#define VSTORE_VEC(v0, v1, off, buf)                                                         \
    {                                                                                        \
        ACTIVATION_V8(v0);                                                                   \
        ACTIVATION_V8(v1);                                                                   \
        vstore16((T16)(v0.s0, v0.s1, v0.s2, v0.s3, v1.s0, v1.s1, v1.s2, v1.s3, v0.s4, v0.s5, \
                     v0.s6, v0.s7, v1.s4, v1.s5, v1.s6, v1.s7),                              \
            0, buf + off);                                                                   \
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
#define VSTORE_VEC(v0, v1, off, buf)                                                         \
    {                                                                                        \
        ACTIVATION_V16(v0);                                                                  \
        ACTIVATION_V16(v1);                                                                  \
        vstore16((T16)(v0.s0, v0.s1, v0.s2, v0.s3, v1.s0, v1.s1, v1.s2, v1.s3, v0.s4, v0.s5, \
                     v0.s6, v0.s7, v1.s4, v1.s5, v1.s6, v1.s7),                              \
            0, buf + off);                                                                   \
        vstore16((T16)(v0.s8, v0.s9, v0.sa, v0.sb, v1.s8, v1.s9, v1.sa, v1.sb, v0.sc, v0.sd, \
                     v0.se, v0.sf, v1.sc, v1.sd, v1.se, v1.sf),                              \
            0, buf + off + 16);                                                              \
    }
#endif

#if defined(USE_RELU)
__kernel void MANGLE_NAME(deconv_gemm_f2s2_h_relu_, ON, KN)
#else
__kernel void MANGLE_NAME(deconv_gemm_f2s2_h_, ON, KN)
#endif
    (const int ih_str,
        int ihw_str,
        const int ic_str,
        const int ih_off,
        const int iw_off,
        const int oh_str,
        int ohw_str,
        const int oh_off,
        const int ow_off,
        const int oh,
        const int bx,
        const int by,
        __global const T *in,
        __global const T *flt,
        __read_only image1d_t bias,
        __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
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
    T4 bias_val;

#if (KN == 2)
    bias_val = read_imageh(bias, sampler, (idz >> 1));
    SET_BIAS_VAL(bias_val, out_val[0]);
    SET_BIAS_VAL(bias_val, out_val[1]);
#elif (KN == 4)
    bias_val = read_imageh(bias, sampler, idz);
    SET_BIAS_VAL(bias_val, out_val[0]);
    SET_BIAS_VAL(bias_val, out_val[1]);
    SET_BIAS_VAL(bias_val, out_val[2]);
    SET_BIAS_VAL(bias_val, out_val[3]);
#endif

    int in_off = ((idy + iw_off) * ih_str + idx * ON + ih_off) << 2;
    int flt_off = idz * ic_str * KN;
    ihw_str = ihw_str << 2;

    for (int i = 0; i < ic_str; ++i) {
        in_val = VLOAD_VEC(in_off, in);
#if (KN == 2)
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
        in_off += ihw_str;
    }

#if (KN == 2)
    int out_off = (idx << 1) * ON + oh_off;
    out_off += ((idy << 1) + ow_off + (idz & 1)) * oh_str;
    out_off += (idz >> 1) * ohw_str;
    out_off = (out_off << 2);
    VSTORE_VEC(out_val[0], out_val[1], out_off, out);
#elif (KN == 4)
    int out_off = (idx << 1) * ON + oh_off;
    out_off += ((idy << 1) + ow_off) * oh_str;
    out_off += idz * ohw_str;
    out_off = (out_off << 2);
    VSTORE_VEC(out_val[0], out_val[1], out_off, out);
    VSTORE_VEC(out_val[2], out_val[3], out_off + oh_str * 4, out);
#endif
}

// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // /
#else

#define VSTORE_VEC(v0, v1, off, buf)                                                         \
    {                                                                                        \
        ACTIVATION_V4(v0);                                                                   \
        ACTIVATION_V4(v1);                                                                   \
        vstore8((T8)(v0.s0, v0.s1, v0.s2, v0.s3, v1.s0, v1.s1, v1.s2, v1.s3), 0, buf + off); \
    }

#if defined(USE_RELU)
__kernel void MANGLE_NAME(deconv_gemm_f2s2_relu_, ON, KN)
#else
__kernel void MANGLE_NAME(deconv_gemm_f2s2_, ON, KN)
#endif
    (const int ih_str,
        const int ihw_str,
        const int ic_str,
        const int ih_off,
        const int iw_off,
        const int oh_str,
        const int ohw_str,
        const int oh_off,
        const int ow_off,
        const int ow,
        const int bx,
        const int by,
        __global const T *in,
        __global const T *flt,
        __read_only image1d_t bias,
        __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 in_val[IN];
    T16 flt_val;
    T4 out_val[KN][ON];
    T4 bias_val;

#if (KN == 2)
    bias_val = read_imageh(bias, sampler, (idz >> 1));
    SET_REG_ARRAY(bias_val, out_val[0]);
    SET_REG_ARRAY(bias_val, out_val[1]);
#elif (KN == 4)
    bias_val = read_imageh(bias, sampler, idz);
    SET_REG_ARRAY(bias_val, out_val[0]);
    SET_REG_ARRAY(bias_val, out_val[1]);
    SET_REG_ARRAY(bias_val, out_val[2]);
    SET_REG_ARRAY(bias_val, out_val[3]);
#endif

    int in_off = (idy * ON + iw_off) * ih_str + idx + ih_off;
    int flt_off = idz * ic_str * KN;

    for (int i = 0; i < ic_str; ++i) {
        LOAD_INPUT_BUF_ARRAY_V4(in_val, in_off, ih_str, in);
#if (KN == 2)
        flt_val = vload16(flt_off, flt);
        DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[0]);
        flt_val = vload16(flt_off + 1, flt);
        DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[1]);
#elif (KN == 4)
        for (uchar j = 0; j < KN; ++j) {
            flt_val = vload16(flt_off + j, flt);
            if (j == 0) {
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[0]);
            }
            if (j == 1) {
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[1]);
            }
            if (j == 2) {
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[2]);
            }
            if (j == 3) {
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[3]);
            }
        }
#endif
        flt_off += KN;
        in_off += ihw_str;
    }
#if (KN == 2)
    int index_y = (idy << 1) * ON + (idz & 1);
    int out_off = (idx << 1) + oh_off;
    out_off += (index_y + ow_off) * oh_str;
    out_off += (idz >> 1) * ohw_str;
    out_off = (out_off << 2);
    VSTORE_VEC(out_val[0][0], out_val[1][0], out_off, out);
#if (ON > 1)
    if (index_y + 2 < ow) {
        VSTORE_VEC(out_val[0][1], out_val[1][1], out_off + oh_str * 8, out);
    }
#endif
#if (ON > 2)
    if (index_y + 4 < ow) {
        VSTORE_VEC(out_val[0][2], out_val[1][2], out_off + oh_str * 16, out);
    }
#endif
#if (ON > 3)
    if (index_y + 6 < ow) {
        VSTORE_VEC(out_val[0][3], out_val[1][3], out_off + oh_str * 24, out);
    }
#endif
#elif (KN == 4)
    int index_y = (idy << 1) * ON;
    int out_off = (idx << 1) + oh_off;
    out_off += (index_y + ow_off) * oh_str;
    out_off += idz * ohw_str;
    out_off = (out_off << 2);
    VSTORE_VEC(out_val[0][0], out_val[1][0], out_off, out);
    if (index_y + 1 < ow) {
        VSTORE_VEC(out_val[2][0], out_val[3][0], out_off + oh_str * 4, out);
    }
#if (ON > 1)
    if (index_y + 2 < ow) {
        VSTORE_VEC(out_val[0][1], out_val[1][1], out_off + oh_str * 8, out);
    }
    if (index_y + 3 < ow) {
        VSTORE_VEC(out_val[2][1], out_val[3][1], out_off + oh_str * 12, out);
    }
#endif
#if (ON > 2)
    if (index_y + 4 < ow) {
        VSTORE_VEC(out_val[0][2], out_val[1][2], out_off + oh_str * 16, out);
    }
    if (index_y + 5 < ow) {
        VSTORE_VEC(out_val[2][2], out_val[3][2], out_off + oh_str * 20, out);
    }
#endif
#endif
}
#endif
