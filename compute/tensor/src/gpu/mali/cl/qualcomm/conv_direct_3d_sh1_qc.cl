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
#define MANGLE_NAME_IMPL(base, IOM, AM, BM, FW, FH, FT, ON, KN) \
    base##IOM##AM##BM##FW##FH##FT##ON##KN
#define MANGLE_NAME(base, IOM, AM, BM, FW, FH, FT, ON, KN) \
    MANGLE_NAME_IMPL(base, IOM, AM, BM, FW, FH, FT, ON, KN)
#if defined(NO_BIAS)
#define SET_BIAS(ov)                     \
    {                                    \
        for (uchar i = 0; i < KN; i++) { \
            ov[i][0] = (T4)0;            \
        }                                \
    }
#define BM nobias_
#else
#define SET_BIAS(ov)                                            \
    {                                                           \
        for (uchar i = 0; i < KN; i++) {                        \
            ov[i][0] = READ_IMAGE(bias, sampler, idk * KN + i); \
        }                                                       \
    }
#define BM
#endif

#define CALCORE(iv, fv, ov) DIRECT_CONV_CAL_CORE_S1(iv, fv, ov)

#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(iv, mem)                                                  \
    {                                                                        \
        for (uchar k = 0; k < IN; k++) {                                     \
            LOAD_MEM_V4(iv[k], (int4)(ix + j, iy + k, i * it + tt, 0), mem); \
        }                                                                    \
    }
#define UPDATE_INPUT(iv, mem, k)                                               \
    {                                                                          \
        UPDATE_REG(iv);                                                        \
        LOAD_MEM_V4(iv[LN], (int4)(ix + j, iy + LN + k, i * it + tt, 0), mem); \
    }
#define ADD_IN_OFF
#else
#define LOAD_INPUT(iv, mem)                                                  \
    {                                                                        \
        for (uchar k = 0; k < IN; k++) {                                     \
            LOAD_MEM_V4(iv[k], in_off + j + k * iw_str + tt * ihw_str, mem); \
        }                                                                    \
    }
#define UPDATE_INPUT(iv, mem, k)                                                 \
    {                                                                            \
        UPDATE_REG(iv);                                                          \
        LOAD_MEM_V4(iv[LN], in_off + j + (LN + k) * iw_str + tt * ihw_str, mem); \
    }
#define ADD_IN_OFF in_off += ihw_str * it
#endif

#define UPDATE_FLT(fv, off, mem) \
    {                            \
        fv = vload16(off, mem);  \
        off++;                   \
    }

#if (KN == 1)
#define CALCORE_FUNC(FUNC, iv, fv, ov, flt_off, flt_mem) \
    {                                                    \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[0]);                             \
    }
#elif (KN == 2)
#define CALCORE_FUNC(FUNC, iv, fv, ov, flt_off, flt_mem) \
    {                                                    \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[0]);                             \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[1]);                             \
    }
#elif (KN == 4)
#define CALCORE_FUNC(FUNC, iv, fv, ov, flt_off, flt_mem) \
    {                                                    \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[0]);                             \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[1]);                             \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[2]);                             \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[3]);                             \
    }
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT(ov, mem)                                                         \
    {                                                                                 \
        int oy = idy * ON;                                                            \
        int oz = idk * KN * ot + idt;                                                 \
        for (uchar i = 0; i < KN; i++) {                                              \
            for (uchar j = 0; j < ON; j++) {                                          \
                if (oy + j < oh) {                                                    \
                    ACTIVATION_V4(ov[i][j]);                                          \
                    STORE_MEM_V4(ov[i][j], (int4)(idx, oy + j, oz + i * ot, 0), mem); \
                }                                                                     \
            }                                                                         \
        }                                                                             \
    }
#else
#define STORE_OUTPUT(ov, mem)                                                                \
    {                                                                                        \
        int othw_str = ohw_str * ot;                                                         \
        int out_off = idk * KN * othw_str + idt * ohw_str + idy * ON * ow_str + idx + o_off; \
        for (uchar i = 0; i < KN; i++) {                                                     \
            for (uchar j = 0; j < ON; j++) {                                                 \
                if (idy * ON + j < oh) {                                                     \
                    ACTIVATION_V4(ov[i][j]);                                                 \
                    STORE_MEM_V4(ov[i][j], out_off + j * ow_str, mem);                       \
                }                                                                            \
            }                                                                                \
            out_off += othw_str;                                                             \
        }                                                                                    \
    }
#endif

__kernel void MANGLE_NAME(conv_direct_3d_sh1_qc_, IOM, AM, BM, FW, FH, FT, ON, KN)(const int iw_str,
    const int ihw_str,
    const int ic_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int ohw_str,
    const int o_off,
    const int oh,
    const int oc,
    const int ot,
    const int it,
    const int pt,
    const int sw,
    const int st,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    __global T *flt,
    __read_only image1d_t bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int idt = idz % ot;
    const int idk = idz / ot;

    if (idx >= bx || idy >= by) {
        return;
    }

    T4 in_val[IN];
    T16 flt_val;
    T4 out_val[KN][ON];

    SET_BIAS(out_val);
    for (uchar i = 0; i < KN; i++) {
        for (uchar j = 1; j < ON; j++) {
            out_val[i][j] = out_val[i][0];
        }
    }

#if defined(USE_INPUT_IMG)
    int ix = idx * sw + iw_off;
    int iy = idy * ON + ih_off;
#else
    int in_off = (idy * ON + ih_off) * iw_str + idx * sw + iw_off;
#endif
    int flt_off = idk * ic_str * FWHT * KN;

    int t_be = idt * st - pt;
    int t_end = t_be + FT;
    int flt_be = 0;
    int flt_end = 0;
    if (t_be < 0) {
        flt_be -= t_be * FW * FH * KN;
        t_be = 0;
    }
    if (t_end > it) {
        flt_end = (t_end - it) * FW * FH * KN;
        t_end = it;
    }

    for (ushort i = 0; i < ic_str; i++) {
        flt_off += flt_be;
        for (int tt = t_be; tt < t_end; ++tt) {
            for (uchar j = 0; j < FW; j++) {
                LOAD_INPUT(in_val, in);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);

#if (FH > 1)
                UPDATE_INPUT(in_val, in, 1);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 2)
                UPDATE_INPUT(in_val, in, 2);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 3)
                UPDATE_INPUT(in_val, in, 3);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 4)
                UPDATE_INPUT(in_val, in, 4);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 5)
                UPDATE_INPUT(in_val, in, 5);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 6)
                UPDATE_INPUT(in_val, in, 6);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 7)
                UPDATE_INPUT(in_val, in, 7);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 8)
                UPDATE_INPUT(in_val, in, 8);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 9)
                UPDATE_INPUT(in_val, in, 9);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 10)
                UPDATE_INPUT(in_val, in, 10);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif

#if (FH > 11)
                UPDATE_INPUT(in_val, in, 11);
                CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
#endif
            }
        }
        ADD_IN_OFF;
        flt_off += flt_end;
    }
    STORE_OUTPUT(out_val, out);
}
