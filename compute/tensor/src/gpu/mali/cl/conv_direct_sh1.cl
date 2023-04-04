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
#define MANGLE_NAME_IMPL(base, IOM, AM, BM, FW, FH, ON, KN) base##IOM##AM##BM##FW##FH##ON##KN
#define MANGLE_NAME(base, IOM, AM, BM, FW, FH, ON, KN) \
    MANGLE_NAME_IMPL(base, IOM, AM, BM, FW, FH, ON, KN)
#if defined(NO_BIAS)
#define SET_BIAS(val, off, bias) SET_REG_ARRAY(0, val);
#define BM nobias_
#else
#define SET_BIAS(val, off, bias) LOADBIAS_IMAGE_ARRAY_V4(val, off, bias);
#define BM
#endif

#if defined(USE_INPUT_IMG)
#define LOAD_INPUT_ARRAY \
    LOAD_INPUT_MEM_ARRAY_V4(in_val, in_off_x + j, in_off_y, in_off_z + i, 1, in);
#define LOAD_REST_MEM \
    LOAD_MEM_V4(in_val[LN], (int4)(in_off_x + j, in_off_y + LN + k, in_off_z + i, 0), in);
#define ADD_IN_OFF
#else
#define LOAD_INPUT_ARRAY LOAD_INPUT_MEM_ARRAY_V4(in_val, in_off + j, iw_str, 0, 1, in);
#define LOAD_REST_MEM LOAD_MEM_V4(in_val[LN], in_off + j + (LN + k) * iw_str, in);
#define ADD_IN_OFF in_off += ihw_str;
#endif

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT_ARRAY(val)                                                    \
    {                                                                              \
        STORE_OUTPUT_MEM_ARRAY_V4(val, idx, idy *ON, out_off_z, idy *ON, oh, out); \
        out_off_z += 1;                                                            \
    }
#else
#define STORE_OUTPUT_ARRAY(val)                                               \
    {                                                                         \
        STORE_OUTPUT_MEM_ARRAY_V4(val, out_off, ow_str, 0, idy *ON, oh, out); \
        out_off += ohw_str;                                                   \
    }
#endif

__kernel void MANGLE_NAME(conv_direct_sh1_, IOM, AM, BM, FW, FH, ON, KN)(const int iw_str,
    const int ihw_str,
    const int ic_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int ohw_str,
    const int o_off,
    const int oh,
    const int oc,
    const int sw,
    const int in_str,
    const int on_str,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    __global const T16 *flt,
    __read_only image1d_t bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2) % (((oc + 3) >> 2) / KN);
    const int idn = get_global_id(2) / (((oc + 3) >> 2) / KN);

    if (idx >= bx || idy >= by) {
        return;
    }
    T4 in_val[IN];
    T16 flt_val;
    T4 out_val[KN][ON];
    SET_BIAS(out_val[0], idz * KN, bias);
#if (KN > 1)
    SET_BIAS(out_val[1], idz * KN + 1, bias);
#endif
#if (KN > 2)
    SET_BIAS(out_val[2], idz * KN + 2, bias);
    SET_BIAS(out_val[3], idz * KN + 3, bias);
#endif

#if defined(USE_INPUT_IMG)
    int in_off_x = idx * sw + iw_off;
    int in_off_y = idy * ON + ih_off;
    int in_off_z = idn * ic_str;
#else
    int in_off = idn * in_str + (idy * ON + ih_off) * iw_str + idx * sw + iw_off;
#endif
    int flt_off = idz * ic_str * FWH * KN;

    for (int i = 0; i < ic_str; ++i) {
#if (FH == 1)
        for (uchar j = 0; j < FW; ++j) {
            LOAD_INPUT_ARRAY;
            flt_val = flt[flt_off];
            DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[0]);
#if (KN > 1)
            flt_val = flt[flt_off + 1];
            DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[1]);
#endif
#if (KN > 2)
            flt_val = flt[flt_off + 2];
            DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[2]);
            flt_val = flt[flt_off + 3];
            DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[3]);
#endif
            flt_off += KN;
        }
#else
        for (uchar j = 0; j < FW; ++j) {
            LOAD_INPUT_ARRAY;
            for (uchar k = 0; k < FH; ++k) {
#if defined(BASIC_REG)
                LOAD_REST_MEM;
#endif
                flt_val = flt[flt_off + k * KN];
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[0]);
#if (KN > 1)
                flt_val = flt[flt_off + k * KN + 1];
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[1]);
#endif
#if (KN > 2)
                flt_val = flt[flt_off + k * KN + 2];
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[2]);
                flt_val = flt[flt_off + k * KN + 3];
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[3]);
#endif
                UPDATE_REG(in_val);
            }
            flt_off += KN * FH;
        }
#endif
        ADD_IN_OFF;
    }

#if defined(USE_OUTPUT_IMG)
    int out_off_z = idz * KN + ((oc + 3) >> 2) * idn;
#else
    int out_off = idn * on_str + idz * KN * ohw_str + idy * ON * ow_str + idx + o_off;
#endif

    STORE_OUTPUT_ARRAY(out_val[0]);
#if (KN > 1)
    STORE_OUTPUT_ARRAY(out_val[1]);
#endif
#if (KN > 2)
    STORE_OUTPUT_ARRAY(out_val[2]);
    STORE_OUTPUT_ARRAY(out_val[3]);
#endif
}
