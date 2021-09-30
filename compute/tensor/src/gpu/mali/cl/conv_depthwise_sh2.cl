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
#define MANGLE_NAME_IMPL(base, IOM, AM, FM, FW, FH, ON) base##IOM##AM##FM##FW##FH##ON
#define MANGLE_NAME(base, IOM, AM, FM, FW, FH, ON) MANGLE_NAME_IMPL(base, IOM, AM, FM, FW, FH, ON)

#define FM
#if defined(USE_NCHW)
#define FM nchw_
#endif

#if defined(USE_INPUT_IMG)
#define LOAD_INPUT_ARRAY_STEP(y_base, y_step)                                                   \
    {                                                                                           \
        LOAD_INPUT_MEM_ARRAY_V4(in_val, in_off_x + i, in_off_y + y_base, in_off_z, y_step, in); \
    }
#define LOAD_REST_MEM \
    LOAD_MEM_V4(in_val[LN], (int4)(in_off_x + i, in_off_y + (LN << 1) + j, in_off_z, 0), in);
#else
#define LOAD_INPUT_ARRAY_STEP(y_base, y_step)                                                 \
    {                                                                                         \
        LOAD_INPUT_MEM_ARRAY_V4(in_val, in_off + i + y_base * iw_str, iw_str, 0, y_step, in); \
    }
#define LOAD_REST_MEM LOAD_MEM_V4(in_val[LN], in_off + i + ((LN << 1) + j) * iw_str, in);
#endif
#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT_ARRAY(val)                                                    \
    {                                                                              \
        STORE_OUTPUT_MEM_ARRAY_V4(val, idx, idy *ON, out_off_z, idy *ON, oh, out); \
    }
#else
#define STORE_OUTPUT_ARRAY(val)                                               \
    {                                                                         \
        STORE_OUTPUT_MEM_ARRAY_V4(val, out_off, ow_str, 0, idy *ON, oh, out); \
    }
#endif

__kernel void MANGLE_NAME(conv_depthwise_sh2_, IOM, AM, FM, FW, FH, ON)(const int iw_str,
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
    __global const T *flt,
    __read_only image1d_t bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2) % ((oc + 3) >> 2);
    const int idn = get_global_id(2) / ((oc + 3) >> 2);
    if (idx >= bx || idy >= by) {
        return;
    }

    T4 in_val[IN];
    T4 flt_val;
    T4 out_val[ON];

    LOADBIAS_IMAGE_ARRAY_V4(out_val, idz, bias);
#if defined(USE_INPUT_IMG)
    int in_off_x = idx * sw + iw_off;
    int in_off_y = (idy << 1) * ON + ih_off;
    int in_off_z = idn * ic_str + idz;
#else
    int in_off =
        idn * in_str + idz * ihw_str + ((idy << 1) * ON + ih_off) * iw_str + idx * sw + iw_off;
#endif
    int flt_off = idz * FWH;

    for (uchar i = 0; i < FW; ++i) {
#if defined(BASIC_REG)
        LOAD_INPUT_ARRAY_STEP(0, 2);
        for (uchar j = 0; j < FH; j += 2) {
            LOAD_REST_MEM;
            flt_val = vload4(flt_off + j, flt);
            DEPTHWISE_CAL_CORE_S1(in_val, flt_val, out_val);
            UPDATE_REG(in_val);
        }
        LOAD_INPUT_ARRAY_STEP(1, 2);
        for (uchar j = 1; j < FH; j += 2) {
            LOAD_REST_MEM;
            flt_val = vload4(flt_off + j, flt);
            DEPTHWISE_CAL_CORE_S1(in_val, flt_val, out_val)
            UPDATE_REG(in_val);
        }
#else
        LOAD_INPUT_ARRAY_STEP(0, 1);
        for (uchar j = 0; j < FH; ++j) {
            flt_val = vload4(flt_off + j, flt);
            DEPTHWISE_CAL_CORE_S2(in_val, flt_val, out_val);
            UPDATE_REG(in_val);
        }
#endif
        flt_off += FH;
    }
#if defined(USE_NCHW)
    int out_off = idn * on_str + (idz << 2) * ohw_str + idy * ON * ow_str + idx + o_off;
    STORE_OUTPUT_BUF_ARRAY_V4_NCHW(out_val, out_off, ow_str, ohw_str, idy * ON, oh, out);
#else
#if defined(USE_OUTPUT_IMG)
    int out_off_z = idz + ((oc + 3) >> 2) * idn;
#else
    int out_off = idn * on_str + idz * ohw_str + idy * ON * ow_str + idx + o_off;
#endif
    STORE_OUTPUT_ARRAY(out_val);
#endif
}
