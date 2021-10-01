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
#define MANGLE_NAME_IMPL(base, IOM, AM, FW, FH, ON) base##IOM##AM##FW##FH##ON
#define MANGLE_NAME(base, IOM, AM, FW, FH, ON) MANGLE_NAME_IMPL(base, IOM, AM, FW, FH, ON)

#if (IN <= 4)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(v)                                     \
    {                                                     \
        LOAD_MEM_V4(v, (int4)(ix, iy + j, iz + i, 0), in);\
    }
#else
#define LOAD_INPUT(v)                                \
    {                                                \
        in_val = vload4(0, in_off + j * iw_str + in);\
    }
#endif
#define UPDATE_REG(v) {\
    v.s0 = v.s1;\
    v.s1 = v.s2;\
    v.s2 = v.s3;\
}
#elif (IN <= 8)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(v)                                               \
    {                                                               \
        LOAD_MEM_V4(v.s0123, (int4)(ix, iy + j, iz + i, 0), in);    \
        LOAD_MEM_V4(v.s4567, (int4)(ix + 1, iy + j, iz + i, 0), in);\
    }
#else
#define LOAD_INPUT(v)                                \
    {                                                \
        in_val = vload8(0, in_off + j * iw_str + in);\
    }
#endif
#define UPDATE_REG(v) {\
    v.s0 = v.s1;\
    v.s1 = v.s2;\
    v.s2 = v.s3;\
    v.s3 = v.s4;\
    v.s4 = v.s5;\
    v.s5 = v.s6;\
    v.s6 = v.s7;\
}
#elif (IN <= 16)
#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(v)                                               \
    {                                                               \
        LOAD_MEM_V4(v.s0123, (int4)(ix, iy + j, iz + i, 0), in);    \
        LOAD_MEM_V4(v.s4567, (int4)(ix + 1, iy + j, iz + i, 0), in);\
        LOAD_MEM_V4(v.s89ab, (int4)(ix + 2, iy + j, iz + i, 0), in);\
        LOAD_MEM_V4(v.scdef, (int4)(ix + 3, iy + j, iz + i, 0), in);\
    }
#else
#define LOAD_INPUT(v)                                 \
    {                                                 \
        in_val = vload16(0, in_off + j * iw_str + in);\
    }
#endif
#define UPDATE_REG(v) {\
    v.s0 = v.s1;\
    v.s1 = v.s2;\
    v.s2 = v.s3;\
    v.s3 = v.s4;\
    v.s4 = v.s5;\
    v.s5 = v.s6;\
    v.s6 = v.s7;\
    v.s7 = v.s8;\
    v.s8 = v.s9;\
    v.s9 = v.sa;\
    v.sa = v.sb;\
    v.sb = v.sc;\
    v.sc = v.sd;\
    v.sd = v.se;\
    v.se = v.sf;\
}
#endif
#if defined(USE_INPUT_IMG)
#define ADD_IN_OFF
#else
#define ADD_IN_OFF in_off += iwh_str;
#endif

#if (ON == 2)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
    }
#elif (ON == 3)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
        C[2] += A.s2 * B; \
    }
#elif (ON == 4)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
        C[2] += A.s2 * B; \
        C[3] += A.s3 * B; \
    }
#elif (ON == 5)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
        C[2] += A.s2 * B; \
        C[3] += A.s3 * B; \
        C[4] += A.s4 * B; \
    }
#elif (ON == 6)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
        C[2] += A.s2 * B; \
        C[3] += A.s3 * B; \
        C[4] += A.s4 * B; \
        C[5] += A.s5 * B; \
    }
#elif (ON == 7)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
        C[2] += A.s2 * B; \
        C[3] += A.s3 * B; \
        C[4] += A.s4 * B; \
        C[5] += A.s5 * B; \
        C[6] += A.s6 * B; \
    }
#elif (ON == 8)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
        C[2] += A.s2 * B; \
        C[3] += A.s3 * B; \
        C[4] += A.s4 * B; \
        C[5] += A.s5 * B; \
        C[6] += A.s6 * B; \
        C[7] += A.s7 * B; \
    }
#elif (ON == 9)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
        C[2] += A.s2 * B; \
        C[3] += A.s3 * B; \
        C[4] += A.s4 * B; \
        C[5] += A.s5 * B; \
        C[6] += A.s6 * B; \
        C[7] += A.s7 * B; \
        C[8] += A.s8 * B; \
    }
#elif (ON == 10)
#define calCore(A, B, C)  \
    {                     \
        C[0] += A.s0 * B; \
        C[1] += A.s1 * B; \
        C[2] += A.s2 * B; \
        C[3] += A.s3 * B; \
        C[4] += A.s4 * B; \
        C[5] += A.s5 * B; \
        C[6] += A.s6 * B; \
        C[7] += A.s7 * B; \
        C[8] += A.s8 * B; \
        C[9] += A.s9 * B; \
    }
#elif (ON == 11)
#define calCore(A, B, C)   \
    {                      \
        C[0] += A.s0 * B;  \
        C[1] += A.s1 * B;  \
        C[2] += A.s2 * B;  \
        C[3] += A.s3 * B;  \
        C[4] += A.s4 * B;  \
        C[5] += A.s5 * B;  \
        C[6] += A.s6 * B;  \
        C[7] += A.s7 * B;  \
        C[8] += A.s8 * B;  \
        C[9] += A.s9 * B;  \
        C[10] += A.sa * B; \
    }
#elif (ON == 12)
#define calCore(A, B, C)   \
    {                      \
        C[0] += A.s0 * B;  \
        C[1] += A.s1 * B;  \
        C[2] += A.s2 * B;  \
        C[3] += A.s3 * B;  \
        C[4] += A.s4 * B;  \
        C[5] += A.s5 * B;  \
        C[6] += A.s6 * B;  \
        C[7] += A.s7 * B;  \
        C[8] += A.s8 * B;  \
        C[9] += A.s9 * B;  \
        C[10] += A.sa * B; \
        C[11] += A.sb * B; \
    }
#elif (ON == 13)
#define calCore(A, B, C)   \
    {                      \
        C[0] += A.s0 * B;  \
        C[1] += A.s1 * B;  \
        C[2] += A.s2 * B;  \
        C[3] += A.s3 * B;  \
        C[4] += A.s4 * B;  \
        C[5] += A.s5 * B;  \
        C[6] += A.s6 * B;  \
        C[7] += A.s7 * B;  \
        C[8] += A.s8 * B;  \
        C[9] += A.s9 * B;  \
        C[10] += A.sa * B; \
        C[11] += A.sb * B; \
        C[12] += A.sc * B; \
    }
#elif (ON == 14)
#define calCore(A, B, C)   \
    {                      \
        C[0] += A.s0 * B;  \
        C[1] += A.s1 * B;  \
        C[2] += A.s2 * B;  \
        C[3] += A.s3 * B;  \
        C[4] += A.s4 * B;  \
        C[5] += A.s5 * B;  \
        C[6] += A.s6 * B;  \
        C[7] += A.s7 * B;  \
        C[8] += A.s8 * B;  \
        C[9] += A.s9 * B;  \
        C[10] += A.sa * B; \
        C[11] += A.sb * B; \
        C[12] += A.sc * B; \
        C[13] += A.sd * B; \
    }
#elif (ON == 15)
#define calCore(A, B, C)   \
    {                      \
        C[0] += A.s0 * B;  \
        C[1] += A.s1 * B;  \
        C[2] += A.s2 * B;  \
        C[3] += A.s3 * B;  \
        C[4] += A.s4 * B;  \
        C[5] += A.s5 * B;  \
        C[6] += A.s6 * B;  \
        C[7] += A.s7 * B;  \
        C[8] += A.s8 * B;  \
        C[9] += A.s9 * B;  \
        C[10] += A.sa * B; \
        C[11] += A.sb * B; \
        C[12] += A.sc * B; \
        C[13] += A.sd * B; \
        C[14] += A.se * B; \
    }
#elif (ON == 16)
#define calCore(A, B, C)   \
    {                      \
        C[0] += A.s0 * B;  \
        C[1] += A.s1 * B;  \
        C[2] += A.s2 * B;  \
        C[3] += A.s3 * B;  \
        C[4] += A.s4 * B;  \
        C[5] += A.s5 * B;  \
        C[6] += A.s6 * B;  \
        C[7] += A.s7 * B;  \
        C[8] += A.s8 * B;  \
        C[9] += A.s9 * B;  \
        C[10] += A.sa * B; \
        C[11] += A.sb * B; \
        C[12] += A.sc * B; \
        C[13] += A.sd * B; \
        C[14] += A.se * B; \
        C[15] += A.sf * B; \
    }
#endif

#define CALCORE_FUNC(FUNC, iv, fv, ov) \
    {                                  \
        fv = vload4(flt_off, flt);     \
        FUNC(iv, fv, ov);              \
        UPDATE_REG(iv);                \
        flt_off++;                     \
    }

__kernel void MANGLE_NAME(conv_direct_sw1_nchw_to_nchwc4_qc_, IOM, AM, FW, FH, ON)(const int iw_str,
    const int iwh_str,
    const int ic_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int ow,
    const int oc,
    const int sh,
    const int in_str,
    const int on_str,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    __global const T *flt,
    __read_only image1d_t bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    ushort pitch = (oc + 3) >> 2;
    const int id = get_global_id(2);
    const int idz = id % pitch;
    const int idn = id / pitch;
#if (IN <= 4)
    T4 in_val;
#elif (IN <= 8)
    T8 in_val;
#elif (IN <= 16)
    T16 in_val;
#endif
    T4 flt_val;
    T4 out_val[ON];

    out_val[0] = READ_IMAGE(bias, sampler, idz);
    for (uchar i = 0; i < ON; i++) {
        out_val[i] = out_val[0];
    }

#if defined(USE_INPUT_IMG)
    int ix = idx * (ON >> 2) + iw_off;
    int iy = idy * sh + ih_off;
    int iz = idn * ic_str;
#else
    int in_off = idn * in_str + (idy * sh + ih_off) * iw_str + idx * ON + iw_off;
#endif
    int flt_off = idz * ic_str * FWH;

    for (int i = 0; i < ic_str; ++i) {
        for (uchar j = 0; j < FH; ++j) {
            LOAD_INPUT(in_val);
            CALCORE_FUNC(calCore, in_val, flt_val, out_val);
#if (FW > 1)
            CALCORE_FUNC(calCore, in_val, flt_val, out_val);
#endif
#if (FW > 2)
            CALCORE_FUNC(calCore, in_val, flt_val, out_val);
#endif
#if (FW > 3)
            CALCORE_FUNC(calCore, in_val, flt_val, out_val);
#endif
#if (FW > 4)
            CALCORE_FUNC(calCore, in_val, flt_val, out_val);
#endif
#if (FW > 5)
            CALCORE_FUNC(calCore, in_val, flt_val, out_val);
#endif
#if (FW > 6)
            CALCORE_FUNC(calCore, in_val, flt_val, out_val);
#endif
        }
        ADD_IN_OFF
    }

    int xn = idx * ON;
#if defined(USE_OUTPUT_IMG)
    int img_z = idz + ((oc + 3) >> 2) * idn;
    for (uchar i = 0; i < ON; i++) {
        if (xn + i < ow) {
            ACTIVATION_V4(out_val[i]);
            STORE_MEM_V4(out_val[i], (int4)(xn + i, idy, img_z, 0), out);
        }
    }
#else
    int out_off = idn * on_str + (idz * oh_str + idy) * ow_str + xn + o_off;
    for (uchar i = 0; i < ON; i++) {
        if (xn + i < ow) {
            ACTIVATION_V4(out_val[i]);
            STORE_MEM_V4(out_val[i], out_off + i, out);
        }
    }
#endif
}
