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
#define SET_BIAS(val, off, bias) SET_REG_ARRAY(0, val);
#define BM nobias_
#else
#define SET_BIAS(val, off, bias) LOADBIAS_IMAGE_ARRAY_V4(val, off, bias);
#define BM
#endif

#if (FH == 1)
#define LOAD_FILTER(index, val)              \
    {                                        \
        val = vload16(flt_off + index, flt); \
    }
#else
#define LOAD_FILTER(index, val)                       \
    {                                                 \
        val = vload16(flt_off + k * KN + index, flt); \
    }
#endif

#define STORE_OUTPUT_ARRAY(val)                                               \
    {                                                                         \
        STORE_OUTPUT_MEM_ARRAY_V4(val, out_off, ow_str, 0, idy *ON, oh, out); \
        out_off += othw_str;                                                  \
    }

__kernel void MANGLE_NAME(conv_direct_3d_sh1_, IOM, AM, BM, FW, FH, FT, ON, KN)(const int iw_str,
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
    SET_BIAS(out_val[0], idk * KN, bias);
#if (KN > 1)
    SET_BIAS(out_val[1], idk * KN + 1, bias);
#endif
#if (KN > 2)
    SET_BIAS(out_val[2], idk * KN + 2, bias);
    SET_BIAS(out_val[3], idk * KN + 3, bias);
#endif

    int in_off = (idy * ON + ih_off) * iw_str + idx * sw + iw_off;
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

    for (int i = 0; i < ic_str; ++i) {
        flt_off += flt_be;
        for (int tt = t_be; tt < t_end; ++tt) {
#if (FH == 1)
            for (uchar j = 0; j < FW; ++j) {
                LOAD_INPUT_MEM_ARRAY_V4(in_val, in_off + tt * ihw_str + j, iw_str, 0, 1, in);
                LOAD_FILTER(0, flt_val);
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[0]);
#if (KN > 1)
                LOAD_FILTER(1, flt_val);
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[1]);
#endif
#if (KN > 2)
                LOAD_FILTER(2, flt_val);
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[2]);
                LOAD_FILTER(3, flt_val);
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[3]);
#endif
                flt_off += KN;
            }
#else
            for (uchar j = 0; j < FW; ++j) {
                LOAD_INPUT_MEM_ARRAY_V4(in_val, in_off + tt * ihw_str + j, iw_str, 0, 1, in);
                for (uchar k = 0; k < FH; ++k) {
#if defined(BASIC_REG)
                    LOAD_MEM_V4(in_val[LN], in_off + tt * ihw_str + j + (LN + k) * iw_str, in);
#endif
                    LOAD_FILTER(0, flt_val);
                    DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[0]);
#if (KN > 1)
                    LOAD_FILTER(1, flt_val);
                    DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[1]);
#endif
#if (KN > 2)
                    LOAD_FILTER(2, flt_val);
                    DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[2]);
                    LOAD_FILTER(3, flt_val);
                    DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[3]);
#endif
                    UPDATE_REG(in_val);
                }
                flt_off += KN * FH;
            }
#endif
        }
        in_off += ihw_str * it;
        flt_off += flt_end;
    }

    int othw_str = ohw_str * ot;
    int out_off = idk * KN * othw_str + idt * ohw_str + idy * ON * ow_str + idx + o_off;

    STORE_OUTPUT_ARRAY(out_val[0]);
#if (KN > 1)
    STORE_OUTPUT_ARRAY(out_val[1]);
#endif
#if (KN > 2)
    STORE_OUTPUT_ARRAY(out_val[2]);
    STORE_OUTPUT_ARRAY(out_val[3]);
#endif
}
