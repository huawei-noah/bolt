// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND N4INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTI4 OF C4TRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN C4NECTI4 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"
__kernel void conv_wino_trans_picbuf_right(const int ih_str4,
    const int iw_str,
    const int ih_off4,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int ohwc_str,
    const int oh_off4,
    const int bx,
    const int by,
    __global const T *in,
    __global T *out)
{
    int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    idx = idx - oh_off4;
    if ((idy << 2) >= ow_str) {
        return;
    }
    T out_val0[4];
    T out_val1[4];
    T out_val2[4];
    T out_val3[4];
    T out_val4[4];
    T out_val5[4];
    T p[2];
    SET_REG_ARRAY(0, out_val0);
    SET_REG_ARRAY(0, out_val1);
    SET_REG_ARRAY(0, out_val2);
    SET_REG_ARRAY(0, out_val3);
    SET_REG_ARRAY(0, out_val4);
    SET_REG_ARRAY(0, out_val5);

    if (idx >= 0 && idx < ih_str4) {
        int in_off = (idz * iw_str + (idy << 4) + iw_off) * ih_str4 + idx + ih_off4;
        T in_val[6];
        in_val[0] = in[in_off];
        in_val[1] = in[in_off + ih_str4];
        for (uchar i = 0; i < 4; ++i) {
            in_val[2] = in[in_off + 2 * ih_str4];
            in_val[3] = in[in_off + 3 * ih_str4];
            in_val[4] = in[in_off + 4 * ih_str4];
            in_val[5] = in[in_off + 5 * ih_str4];

            UPDATE_REG(out_val0);
            UPDATE_REG(out_val1);
            UPDATE_REG(out_val2);
            UPDATE_REG(out_val3);
            UPDATE_REG(out_val4);
            UPDATE_REG(out_val5);

            p[0] = -4;
            p[1] = 1;
            for (uchar j = 0; j < 2; ++j) {
                out_val1[UN] = out_val2[UN];
                out_val2[UN] = p[0] * in_val[1] - (T)(4) * in_val[2] + p[1] * in_val[3] + in_val[4];
                p[0] = -p[0];
                p[1] = -p[1];
            }

            p[0] = -2;
            p[1] = 2;
            for (uchar j = 0; j < 2; ++j) {
                out_val3[UN] = out_val4[UN];
                out_val4[UN] = p[0] * in_val[1] - in_val[2] + p[1] * in_val[3] + in_val[4];
                p[0] = -p[0];
                p[1] = -p[1];
            }

            p[0] = 4;
            p[1] = -5;
            for (uchar j = 0; j < 2; ++j) {
                out_val0[UN] = out_val5[UN];
                out_val5[UN] = p[0] * in_val[0] + p[1] * in_val[2] + in_val[4];
                in_val[0] = in_val[1];
                in_val[2] = in_val[3];
                in_val[3] = in_val[4];
                in_val[4] = in_val[5];
            }

            in_val[0] = in_val[2];
            in_val[1] = in_val[3];
            in_off += (ih_str4 << 2);
        }
    }

    idx += oh_off4;
    int out_off = (((idz << 2) + (idx & 3)) * ow_str + (idy << 2)) * oh_str + (idx >> 2);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val0, out_off, oh_str, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val1, out_off + ohwc_str, oh_str, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val2, out_off + 2 * ohwc_str, oh_str, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val3, out_off + 3 * ohwc_str, oh_str, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val4, out_off + 4 * ohwc_str, oh_str, out);
    STORE_OUTPUT_BUF_ARRAY_ALIGN(out_val5, out_off + 5 * ohwc_str, oh_str, out);
}
