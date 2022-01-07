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
#define MANGLE_NAME_IMPL(base, IOM) base##IOM
#define MANGLE_NAME(base, IOM) MANGLE_NAME_IMPL(base, IOM)
#if defined(USE_INPUT_IMG)
#define loadH(val, y_off, pic)    \
    {                           \
        T4 tmp0 = READ_IMAGE(pic, sampler, (int4)(idx, iy + y_off, idz, 0));\
        T4 tmp1 = READ_IMAGE(pic, sampler, (int4)(idx + 1, iy + y_off, idz, 0));\
        val[0] = tmp0.x;\
        val[1] = tmp0.y;\
        val[2] = tmp0.z;\
        val[3] = tmp0.w;\
        val[4] = tmp1.x;\
        val[5] = tmp1.y;\
    }
#else
#define loadH(val, y_off, pic)    \
    {                           \
        T4 tmp4 = vload4(0, pic + in_off + y_off * iw_str);\
        T2 tmp2 = vload2(0, pic + in_off + y_off * iw_str + 4);\
        val[0] = tmp4.x;\
        val[1] = tmp4.y;\
        val[2] = tmp4.z;\
        val[3] = tmp4.w;\
        val[4] = tmp2.x;\
        val[5] = tmp2.y;\
    }
#endif

__kernel void MANGLE_NAME(conv_wino_trans_picbuf_nchw_, IOM)(const int iw_str,
    const int ih_str,
    const int i_off,
    const int pw_str,
    const int pwh_str,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    __global T *pictran)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    int iy = idy << 2; 
#if !defined(USE_INPUT_IMG)
    const int in_off = (idz * ih_str + iy) * iw_str + (idx << 2) + i_off;
#endif
    const int pictran_off = idz * pw_str + idy * bx + idx;
    T tmp[16];
    T h0[6], h1[6], h2[6], h3[6], h4[6], h5[6];

    loadH(h0, 0, in);
    loadH(h1, 1, in);
    loadH(h2, 2, in);
    loadH(h3, 3, in);
    loadH(h4, 4, in);
    loadH(h5, 5, in);

    h0[1] = (T)(4.0) * h0[1] - (T)(5.0) * h2[1] + h4[1];
    h0[2] = (T)(4.0) * h0[2] - (T)(5.0) * h2[2] + h4[2];
    h0[3] = (T)(4.0) * h0[3] - (T)(5.0) * h2[3] + h4[3];
    h0[4] = (T)(4.0) * h0[4] - (T)(5.0) * h2[4] + h4[4];

    tmp[0] = (T)(-4.0) * (h1[1] + h2[1]) + h3[1] + h4[1];
    tmp[1] = (T)(-4.0) * (h1[2] + h2[2]) + h3[2] + h4[2];
    tmp[2] = (T)(-4.0) * (h1[3] + h2[3]) + h3[3] + h4[3];
    tmp[3] = (T)(-4.0) * (h1[4] + h2[4]) + h3[4] + h4[4];

    tmp[4] = (T)(4.0) * (h1[1] - h2[1]) - h3[1] + h4[1];
    tmp[5] = (T)(4.0) * (h1[2] - h2[2]) - h3[2] + h4[2];
    tmp[6] = (T)(4.0) * (h1[3] - h2[3]) - h3[3] + h4[3];
    tmp[7] = (T)(4.0) * (h1[4] - h2[4]) - h3[4] + h4[4];

    tmp[8] = (T)(2.0) *  (h3[1] - h1[1]) - h2[1] + h4[1];
    tmp[9] = (T)(2.0) *  (h3[2] - h1[2]) - h2[2] + h4[2];
    tmp[10] = (T)(2.0) * (h3[3] - h1[3]) - h2[3] + h4[3];
    tmp[11] = (T)(2.0) * (h3[4] - h1[4]) - h2[4] + h4[4];

    tmp[12] = (T)(2.0) * (h1[1] - h3[1]) - h2[1] + h4[1];
    tmp[13] = (T)(2.0) * (h1[2] - h3[2]) - h2[2] + h4[2];
    tmp[14] = (T)(2.0) * (h1[3] - h3[3]) - h2[3] + h4[3];
    tmp[15] = (T)(2.0) * (h1[4] - h3[4]) - h2[4] + h4[4];

    h5[1] = (T)(4.0) * h1[1] - (T)(5.0) * h3[1] + h5[1];
    h5[2] = (T)(4.0) * h1[2] - (T)(5.0) * h3[2] + h5[2];
    h5[3] = (T)(4.0) * h1[3] - (T)(5.0) * h3[3] + h5[3];
    h5[4] = (T)(4.0) * h1[4] - (T)(5.0) * h3[4] + h5[4];

    pictran[pictran_off] =
        (T)(16.0) * h0[0] - (T)(20.0) * h2[0] + (T)(4.0) * h4[0] - (T)(5.0) * h0[2] + h0[4];
    pictran[pictran_off + pwh_str] = (T)(-4.0) *    (h0[1] + h0[2]) + h0[3] + h0[4];
    pictran[pictran_off + pwh_str * 2] = (T)(4.0) * (h0[1] - h0[2]) - h0[3] + h0[4];
    pictran[pictran_off + pwh_str * 3] = (T)(2.0) * (h0[3] - h0[1]) - h0[2] + h0[4];
    pictran[pictran_off + pwh_str * 4] = (T)(2.0) * (h0[1] - h0[3]) - h0[2] + h0[4];
    pictran[pictran_off + pwh_str * 5] =
        (T)(4.0) * (h0[1] + h0[5]) - (T)(5.0) * (h0[3] + h2[5]) + h4[5];

    pictran[pictran_off + pwh_str * 6] =
        (T)(-16.0) * (h1[0] + h2[0]) + (T)(4.0) * (h3[0] + h4[0]) - (T)(5.0) * tmp[1] + tmp[3];
    pictran[pictran_off + pwh_str * 7] = (T)(-4.0) * (tmp[0] + tmp[1]) + tmp[2] + tmp[3];
    pictran[pictran_off + pwh_str * 8] = (T)(4.0) * (tmp[0] - tmp[1]) - tmp[2] + tmp[3];
    pictran[pictran_off + pwh_str * 9] = (T)(2.0) * (tmp[2] - tmp[0]) - tmp[1] + tmp[3];
    pictran[pictran_off + pwh_str * 10] = (T)(2.0) * (tmp[0] - tmp[2]) - tmp[1] + tmp[3];
    pictran[pictran_off + pwh_str * 11] =
        (T)(4.0) * (tmp[0] - h1[5] - h2[5]) - (T)(5.0) * tmp[2] + h3[5] + h4[5];

    pictran[pictran_off + pwh_str * 12] =
        (T)(16.0) * (h1[0] - h2[0]) + (T)(4.0) * (h4[0] - h3[0]) - (T)(5.0) * tmp[5] + tmp[7];
    pictran[pictran_off + pwh_str * 13] = (T)(-4.0) * (tmp[4] + tmp[5]) + tmp[6] + tmp[7];
    pictran[pictran_off + pwh_str * 14] = (T)(4.0) * (tmp[4] - tmp[5]) - tmp[6] + tmp[7];
    pictran[pictran_off + pwh_str * 15] = (T)(2.0) * (tmp[6] - tmp[4]) - tmp[5] + tmp[7];
    pictran[pictran_off + pwh_str * 16] = (T)(2.0) * (tmp[4] - tmp[6]) - tmp[5] + tmp[7];
    pictran[pictran_off + pwh_str * 17] =
        (T)(4.0) * (tmp[4] + h1[5] - h2[5]) - (T)(5.0) * tmp[6] - h3[5] + h4[5];

    pictran[pictran_off + pwh_str * 18] =
        (T)(8.0) * (h3[0] - h1[0]) + (T)(4.0) * (h4[0] - h2[0]) - (T)(5.0) * tmp[9] + tmp[11];
    pictran[pictran_off + pwh_str * 19] = (T)(-4.0) * (tmp[8] + tmp[9]) + tmp[10] + tmp[11];
    pictran[pictran_off + pwh_str * 20] = (T)(4.0) * (tmp[8] - tmp[9]) - tmp[10] + tmp[11];
    pictran[pictran_off + pwh_str * 21] = (T)(2.0) * (tmp[10] - tmp[8]) - tmp[9] + tmp[11];
    pictran[pictran_off + pwh_str * 22] = (T)(2.0) * (tmp[8] - tmp[10]) - tmp[9] + tmp[11];
    pictran[pictran_off + pwh_str * 23] =
        (T)(4.0) * tmp[8] + (T)(2.0) * (h3[5] - h1[5]) - h2[5] - (T)(5.0) * tmp[10] + h4[5];

    pictran[pictran_off + pwh_str * 24] =
        (T)(8.0) * (h1[0] - h3[0]) + (T)(4.0) * (h4[0] - h2[0]) - (T)(5.0) * tmp[13] + tmp[15];
    pictran[pictran_off + pwh_str * 25] = (T)(-4.0) * (tmp[12] + tmp[13]) + tmp[14] + tmp[15];
    pictran[pictran_off + pwh_str * 26] = (T)(4.0) * (tmp[12] - tmp[13]) - tmp[14] + tmp[15];
    pictran[pictran_off + pwh_str * 27] = (T)(2.0) * (tmp[14] - tmp[12]) - tmp[13] + tmp[15];
    pictran[pictran_off + pwh_str * 28] = (T)(2.0) * (tmp[12] - tmp[14]) - tmp[13] + tmp[15];
    pictran[pictran_off + pwh_str * 29] =
        (T)(4.0) * tmp[12] + (T)(2.0) * (h1[5] - h3[5]) - h2[5] - (T)(5.0) * tmp[14] + h4[5];

    pictran[pictran_off + pwh_str * 30] =
        (T)(16.0) * h1[0] - (T)(20.0) * h3[0] + (T)(4.0) * h5[0] - (T)(5.0) * h5[2] + h5[4];
    pictran[pictran_off + pwh_str * 31] = (T)(-4.0) * (h5[1] + h5[2]) + h5[3] + h5[4];
    pictran[pictran_off + pwh_str * 32] = (T)(4.0) *  (h5[1] - h5[2]) - h5[3] + h5[4];
    pictran[pictran_off + pwh_str * 33] = (T)(2.0) *  (h5[3] - h5[1]) - h5[2] + h5[4];
    pictran[pictran_off + pwh_str * 34] = (T)(2.0) *  (h5[1] - h5[3]) - h5[2] + h5[4];
    pictran[pictran_off + pwh_str * 35] =
        (T)(4.0) * (h5[1] + h1[5]) - (T)(5.0) * (h5[3] + h3[5]) + h5[5];
}
