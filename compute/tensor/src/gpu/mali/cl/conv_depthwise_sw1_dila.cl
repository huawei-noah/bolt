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
#define MANGLE_NAME_IMPL(base, DN, AM, FM, FW, FH, ON) base##DN##AM##FM##FW##FH##ON
#define MANGLE_NAME(base, DN, AM, FM, FW, FH, ON) MANGLE_NAME_IMPL(base, DN, AM, FM, FW, FH, ON)

#define FM
#define DN x_

#if defined(USE_NCWH)
#define FM ncwh_
#endif

#if defined(DILATION2)
#define DN 2_
#endif

__kernel void MANGLE_NAME(conv_depthwise_sw1_dila, DN, AM, FM, FW, FH, ON)(const int ih_str,
    const int ihw_str,
    const int ic_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int ohw_str,
    const int oh_off,
    const int ow_off,
    const int ow,
    const int sh,
    const int dw,
    const int dh,
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
    T4 flt_val;
    T4 out_val[ON];

    LOADBIAS_IMAGE_ARRAY_V4(out_val, idz, bias);
    int in_off = idz * ihw_str + (idy * ON + iw_off) * ih_str + idx * sh + ih_off;
    int flt_off = idz * FWH;

#if defined(DILATION2)
    for (uchar i = 0; i < FH; ++i) {
        LOAD_INPUT_BUF_ARRAY_V4(in_val, in_off + i * dh, ih_str, in);
        for (uchar j = 0; j < FW; ++j) {
            LOAD_INPUT_EXCESS_DILATION2(in_val, in_off + j * dw * ih_str + i * dh, ih_str, LN, in);
            flt_val = vload4(flt_off + j, flt);
            DEPTHWISE_CAL_CORE_S1(in_val, flt_val, out_val);
            UPDATE_REG_DILATION2(in_val);
        }
        flt_off += FW;
    }
#else
    for (uchar i = 0; i < FH; ++i) {
        for (uchar j = 0; j < FW; ++j) {
            LOAD_INPUT_BUF_ARRAY_V4(in_val, in_off + j * dw * ih_str + i * dh, ih_str, in);
            flt_val = vload4(flt_off + j, flt);
            DEPTHWISE_CAL_CORE_S1(in_val, flt_val, out_val);
        }
        flt_off += FW;
    }
#endif
#if defined(USE_NCWH)
    int out_off = (idz << 2) * ohw_str + (idy * ON + ow_off) * oh_str + idx + oh_off;
    STORE_OUTPUT_BUF_ARRAY_V4_NCWH(out_val, out_off, oh_str, ohw_str, idy * ON, ow, out);
#else
    int out_off = (idz * ow_str + idy * ON + ow_off) * oh_str + idx + oh_off;
    STORE_OUTPUT_BUF_ARRAY_V4(out_val, out_off, oh_str, idy * ON, ow, out);
#endif
}
