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

__kernel void MANGLE_NAME(conv_direct_sw1_, AM, FW, FH, ON, KN)(const int ih_str,
    const int ihw_str,
    const int ic_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ohw_str,
    const int oh_off,
    const int ow_off,
    const int ow,
    const int oc,
    const int sh,
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
    T4 in_val[IN];
    T16 flt_val;
    T4 out_val[KN][ON];
    LOADBIAS_IMAGE_ARRAY_V4(out_val[0], idz * KN, bias);
#if (KN > 1)
    LOADBIAS_IMAGE_ARRAY_V4(out_val[1], idz * KN + 1, bias);
#endif
#if (KN > 2)
    LOADBIAS_IMAGE_ARRAY_V4(out_val[2], idz * KN + 2, bias);
    LOADBIAS_IMAGE_ARRAY_V4(out_val[3], idz * KN + 3, bias);
#endif

    int in_off = idn * in_str + (idy * ON + iw_off) * ih_str + idx * sh + ih_off;
    int flt_off = idz * ic_str * FWH * KN;

    for (int i = 0; i < ic_str; ++i) {
#if (FW == 1)
        for (uchar j = 0; j < FH; ++j) {
            LOAD_INPUT_BUF_ARRAY_V4(in_val, in_off + j, ih_str, in);
#if (KN == 1)
            flt_val = vload16(flt_off, flt);
            DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[0]);
#elif (KN == 2)
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
        }
#else
        for (uchar j = 0; j < FH; ++j) {
            LOAD_INPUT_BUF_ARRAY_V4(in_val, in_off + j, ih_str, in);
            for (uchar k = 0; k < FW; ++k) {
#if defined(BASIC_REG)
                in_val[LN] = vload4(in_off + j + (LN + k) * ih_str, in);
#endif
                flt_val = vload16(flt_off + k * KN, flt);
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[0]);
#if (KN > 1)
                flt_val = vload16(flt_off + k * KN + 1, flt);
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[1]);
#endif
#if (KN > 2)
                flt_val = vload16(flt_off + k * KN + 2, flt);
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[2]);
                flt_val = vload16(flt_off + k * KN + 3, flt);
                DIRECT_CONV_CAL_CORE_S1(in_val, flt_val, out_val[3]);
#endif
                UPDATE_REG(in_val);
            }
            flt_off += FW * KN;
        }
#endif
        in_off += ihw_str;
    }

    int out_off = idn * on_str + idz * KN * ohw_str + (idy * ON + ow_off) * oh_str + idx + oh_off;
    STORE_OUTPUT_BUF_ARRAY_V4(out_val[0], out_off, oh_str, idy * ON, ow, out);
#if (KN > 1)
    out_off += ohw_str;
    STORE_OUTPUT_BUF_ARRAY_V4(out_val[1], out_off, oh_str, idy * ON, ow, out);
#endif
#if (KN > 2)
    out_off += ohw_str;
    STORE_OUTPUT_BUF_ARRAY_V4(out_val[2], out_off, oh_str, idy * ON, ow, out);
    out_off += ohw_str;
    STORE_OUTPUT_BUF_ARRAY_V4(out_val[3], out_off, oh_str, idy * ON, ow, out);
#endif
}
