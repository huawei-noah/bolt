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
#define MANGLE_NAME_IMPL(base, FWH, FT, ON) base##FWH##FT##ON
#define MANGLE_NAME(base, FWH, FT, ON) MANGLE_NAME_IMPL(base, FWH, FT, ON)

#if (FWH == 1)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s2 * B; \
        C[2] += A.s4 * B; \
        C[3] += A.s6 * B; \
        C[4] += A.s8 * B; \
        C[5] += A.sa * B; \
        C[6] += A.sc * B; \
        C[7] += A.se * B; \
    }
#elif (FWH == 3)
#define calCore(a0, a1, a2, a3, a4, a5, a6, B, C) \
    {                                             \
        C[0] += a0 * B;                           \
        C[1] += a1 * B;                           \
        C[2] += a2 * B;                           \
        C[3] += a3 * B;                           \
        C[4] += a4 * B;                           \
        C[5] += a5 * B;                           \
        C[6] += a6 * B;                           \
    }
#define calCore0(A, B, C) calCore(A.s0, A.s2, A.s4, A.s6, A.s8, A.sa, A.sc, B, C)
#define calCore1(A, B, C) calCore(A.s1, A.s3, A.s5, A.s7, A.s9, A.sb, A.sd, B, C)
#define calCore2(A, B, C) calCore(A.s2, A.s4, A.s6, A.s8, A.sa, A.sc, A.se, B, C)
#elif (FWH == 5)
#define calCore(a0, a1, a2, a3, a4, a5, B, C) \
    {                                         \
        C[0] += a0 * B;                       \
        C[1] += a1 * B;                       \
        C[2] += a2 * B;                       \
        C[3] += a3 * B;                       \
        C[4] += a4 * B;                       \
        C[5] += a5 * B;                       \
    }
#define calCore0(A, B, C) calCore(A.s0, A.s2, A.s4, A.s6, A.s8, A.sa, B, C)
#define calCore1(A, B, C) calCore(A.s1, A.s3, A.s5, A.s7, A.s9, A.sb, B, C)
#define calCore2(A, B, C) calCore(A.s2, A.s4, A.s6, A.s8, A.sa, A.sc, B, C)
#define calCore3(A, B, C) calCore(A.s3, A.s5, A.s7, A.s9, A.sb, A.sd, B, C)
#define calCore4(A, B, C) calCore(A.s4, A.s6, A.s8, A.sa, A.sc, A.se, B, C)
#elif (FWH == 7)
#define calCore(a0, a1, a2, a3, a4, B, C) \
    {                                     \
        C[0] += a0 * B;                   \
        C[1] += a1 * B;                   \
        C[2] += a2 * B;                   \
        C[3] += a3 * B;                   \
        C[4] += a4 * B;                   \
    }
#define calCore0(A, B, C) calCore(A.s0, A.s2, A.s4, A.s6, A.s8, B, C)
#define calCore1(A, B, C) calCore(A.s1, A.s3, A.s5, A.s7, A.s9, B, C)
#define calCore2(A, B, C) calCore(A.s2, A.s4, A.s6, A.s8, A.sa, B, C)
#define calCore3(A, B, C) calCore(A.s3, A.s5, A.s7, A.s9, A.sb, B, C)
#define calCore4(A, B, C) calCore(A.s4, A.s6, A.s8, A.sa, A.sc, B, C)
#define calCore5(A, B, C) calCore(A.s5, A.s7, A.s9, A.sb, A.sd, B, C)
#define calCore6(A, B, C) calCore(A.s6, A.s8, A.sa, A.sc, A.se, B, C)
#endif

#if defined(USE_RELU)
__kernel void MANGLE_NAME(conv_direct_3d_sw2_nchw_to_ncwhc4_relu_, FWH, FT, ON)
#elif defined(USE_RELU6)
__kernel void MANGLE_NAME(conv_direct_3d_sw2_nchw_to_ncwhc4_relu6_, FWH, FT, ON)
#else
__kernel void MANGLE_NAME(conv_direct_3d_sw2_nchw_to_ncwhc4_, FWH, FT, ON)
#endif
    (const int iw_str,
        const int iwh_str,
        const int ic_str,
        const int iw_off,
        const int ih_off,
        const int oh_str,
        const int ow_str,
        const int oh_off,
        const int ow_off,
        const int ow,
        const int ot,
        const int it,
        const int pt,
        const int sh,
        const int st,
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
    const int idt = idz % ot;
    const int idk = idz / ot;

    if (idx >= bx || idy >= by) {
        return;
    }

    T16 in_val;
    T4 flt_val;
    T4 out_val[ON];

    LOADBIAS_IMAGE_ARRAY_V4(out_val, idk, bias);
    int in_off = (idy * sh + ih_off) * iw_str + (idx << 1) * ON + iw_off;
    int flt_off = idk * ic_str * FWHT;

    int t_be = idt * st - pt;
    int t_end = t_be + FT;
    if (t_be < 0) {
        t_be = 0;
        flt_off += pt * FWH * FWH;
    }
    if (t_end > it) {
        t_end = it;
    }

    for (int i = 0; i < ic_str; ++i) {
        for (int tt = t_be; tt < t_end; ++tt) {
#if (FWH == 1)
            flt_val = vload4(flt_off, flt);
            in_val = vload16(0, in + in_off + tt * iwh_str);
            calCore(in_val, flt_val, out_val);
            flt_off++;
#else
            for (uchar j = 0; j < FWH; ++j) {
                in_val = vload16(0, in + in_off + tt * iwh_str + j * iw_str);
                for (uchar k = 0; k < FWH; ++k) {
                    flt_val = vload4(flt_off + k, flt);
                    if (k == 0) {
                        calCore0(in_val, flt_val, out_val);
                    } else if (k == 1) {
                        calCore1(in_val, flt_val, out_val);
                    } else if (k == 2) {
                        calCore2(in_val, flt_val, out_val);
#if (FWH >= 5)
                    } else if (k == 3) {
                        calCore3(in_val, flt_val, out_val);
                    } else if (k == 4) {
                        calCore4(in_val, flt_val, out_val);
#endif
#if (FWH == 7)
                    } else if (k == 5) {
                        calCore5(in_val, flt_val, out_val);
                    } else if (k == 6) {
                        calCore6(in_val, flt_val, out_val);
#endif
                    }
                }
                flt_off += FWH;
            }
#endif
        }
        in_off += iwh_str * it;
    }

    int xn = idx * ON;
    int out_off = (idz * ow_str + xn + ow_off) * oh_str + idy + oh_off;
    STORE_OUTPUT_BUF_ARRAY_V4(out_val, out_off, oh_str, xn, ow, out);
}
