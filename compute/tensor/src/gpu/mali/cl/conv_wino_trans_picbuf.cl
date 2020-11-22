// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define loadH(val, off, pic)    \
    {                           \
        val[0] = pic[off];      \
        val[1] = pic[off + 4];  \
        val[2] = pic[off + 8];  \
        val[3] = pic[off + 12]; \
        val[4] = pic[off + 16]; \
        val[5] = pic[off + 20]; \
    }

__kernel void conv_wino_trans_picbuf(const int ih_str4,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh4,
    const int pw_str,
    const int pwh_str,
    __global const T *in,
    __global T *pictran)
{
    const int id = get_global_id(0);
    const int idhc = id % oh4;
    const int idx = idhc >> 2;
    const int idc = idhc & 3;
    const int idy = id / oh4;
    const int idz = get_global_id(1);

    const int in_off =
        (idz * iw_str + (idy << 2) + iw_off) * ih_str4 + (idx << 4) + idc + (ih_off << 2);
    const int pictran_off = ((idz << 2) + idc) * pw_str + (id >> 2);
    T tmp[16];
    T h0[6], h1[6], h2[6], h3[6], h4[6], h5[6];

    loadH(h0, in_off, in);
    loadH(h1, in_off + ih_str4, in);
    loadH(h2, in_off + ih_str4 * 2, in);
    loadH(h3, in_off + ih_str4 * 3, in);
    loadH(h4, in_off + ih_str4 * 4, in);
    loadH(h5, in_off + ih_str4 * 5, in);

    h1[0] = (T)(4.0) * h1[0] - (T)(5.0) * h1[2] + h1[4];
    h2[0] = (T)(4.0) * h2[0] - (T)(5.0) * h2[2] + h2[4];
    h3[0] = (T)(4.0) * h3[0] - (T)(5.0) * h3[2] + h3[4];
    h4[0] = (T)(4.0) * h4[0] - (T)(5.0) * h4[2] + h4[4];

    tmp[0] = (T)(-4.0) * (h1[1] + h1[2]) + h1[3] + h1[4];
    tmp[1] = (T)(-4.0) * (h2[1] + h2[2]) + h2[3] + h2[4];
    tmp[2] = (T)(-4.0) * (h3[1] + h3[2]) + h3[3] + h3[4];
    tmp[3] = (T)(-4.0) * (h4[1] + h4[2]) + h4[3] + h4[4];

    tmp[4] = (T)(4.0) * (h1[1] - h1[2]) - h1[3] + h1[4];
    tmp[5] = (T)(4.0) * (h2[1] - h2[2]) - h2[3] + h2[4];
    tmp[6] = (T)(4.0) * (h3[1] - h3[2]) - h3[3] + h3[4];
    tmp[7] = (T)(4.0) * (h4[1] - h4[2]) - h4[3] + h4[4];

    tmp[8] = (T)(2.0) * (h1[3] - h1[1]) - h1[2] + h1[4];
    tmp[9] = (T)(2.0) * (h2[3] - h2[1]) - h2[2] + h2[4];
    tmp[10] = (T)(2.0) * (h3[3] - h3[1]) - h3[2] + h3[4];
    tmp[11] = (T)(2.0) * (h4[3] - h4[1]) - h4[2] + h4[4];

    tmp[12] = (T)(2.0) * (h1[1] - h1[3]) - h1[2] + h1[4];
    tmp[13] = (T)(2.0) * (h2[1] - h2[3]) - h2[2] + h2[4];
    tmp[14] = (T)(2.0) * (h3[1] - h3[3]) - h3[2] + h3[4];
    tmp[15] = (T)(2.0) * (h4[1] - h4[3]) - h4[2] + h4[4];

    h1[5] = (T)(4.0) * h1[1] - (T)(5.0) * h1[3] + h1[5];
    h2[5] = (T)(4.0) * h2[1] - (T)(5.0) * h2[3] + h2[5];
    h3[5] = (T)(4.0) * h3[1] - (T)(5.0) * h3[3] + h3[5];
    h4[5] = (T)(4.0) * h4[1] - (T)(5.0) * h4[3] + h4[5];

    pictran[pictran_off] =
        (T)(16.0) * h0[0] - (T)(20.0) * h0[2] + (T)(4.0) * h0[4] - (T)(5.0) * h2[0] + h4[0];
    pictran[pictran_off + pwh_str] = (T)(-4.0) * (h1[0] + h2[0]) + h3[0] + h4[0];
    pictran[pictran_off + pwh_str * 2] = (T)(4.0) * (h1[0] - h2[0]) - h3[0] + h4[0];
    pictran[pictran_off + pwh_str * 3] = (T)(2.0) * (h3[0] - h1[0]) - h2[0] + h4[0];
    pictran[pictran_off + pwh_str * 4] = (T)(2.0) * (h1[0] - h3[0]) - h2[0] + h4[0];
    pictran[pictran_off + pwh_str * 5] =
        (T)(4.0) * (h1[0] + h5[0]) - (T)(5.0) * (h3[0] + h5[2]) + h5[4];

    pictran[pictran_off + pwh_str * 6] =
        (T)(-16.0) * (h0[1] + h0[2]) + (T)(4.0) * (h0[3] + h0[4]) - (T)(5.0) * tmp[1] + tmp[3];
    pictran[pictran_off + pwh_str * 7] = (T)(-4.0) * (tmp[0] + tmp[1]) + tmp[2] + tmp[3];
    pictran[pictran_off + pwh_str * 8] = (T)(4.0) * (tmp[0] - tmp[1]) - tmp[2] + tmp[3];
    pictran[pictran_off + pwh_str * 9] = (T)(2.0) * (tmp[2] - tmp[0]) - tmp[1] + tmp[3];
    pictran[pictran_off + pwh_str * 10] = (T)(2.0) * (tmp[0] - tmp[2]) - tmp[1] + tmp[3];
    pictran[pictran_off + pwh_str * 11] =
        (T)(4.0) * (tmp[0] - h5[1] - h5[2]) - (T)(5.0) * tmp[2] + h5[3] + h5[4];

    pictran[pictran_off + pwh_str * 12] =
        (T)(16.0) * (h0[1] - h0[2]) + (T)(4.0) * (h0[4] - h0[3]) - (T)(5.0) * tmp[5] + tmp[7];
    pictran[pictran_off + pwh_str * 13] = (T)(-4.0) * (tmp[4] + tmp[5]) + tmp[6] + tmp[7];
    pictran[pictran_off + pwh_str * 14] = (T)(4.0) * (tmp[4] - tmp[5]) - tmp[6] + tmp[7];
    pictran[pictran_off + pwh_str * 15] = (T)(2.0) * (tmp[6] - tmp[4]) - tmp[5] + tmp[7];
    pictran[pictran_off + pwh_str * 16] = (T)(2.0) * (tmp[4] - tmp[6]) - tmp[5] + tmp[7];
    pictran[pictran_off + pwh_str * 17] =
        (T)(4.0) * (tmp[4] + h5[1] - h5[2]) - (T)(5.0) * tmp[6] - h5[3] + h5[4];

    pictran[pictran_off + pwh_str * 18] =
        (T)(8.0) * (h0[3] - h0[1]) + (T)(4.0) * (h0[4] - h0[2]) - (T)(5.0) * tmp[9] + tmp[11];
    pictran[pictran_off + pwh_str * 19] = (T)(-4.0) * (tmp[8] + tmp[9]) + tmp[10] + tmp[11];
    pictran[pictran_off + pwh_str * 20] = (T)(4.0) * (tmp[8] - tmp[9]) - tmp[10] + tmp[11];
    pictran[pictran_off + pwh_str * 21] = (T)(2.0) * (tmp[10] - tmp[8]) - tmp[9] + tmp[11];
    pictran[pictran_off + pwh_str * 22] = (T)(2.0) * (tmp[8] - tmp[10]) - tmp[9] + tmp[11];
    pictran[pictran_off + pwh_str * 23] =
        (T)(4.0) * tmp[8] + (T)(2.0) * (h5[3] - h5[1]) - h5[2] - (T)(5.0) * tmp[10] + h5[4];

    pictran[pictran_off + pwh_str * 24] =
        (T)(8.0) * (h0[1] - h0[3]) + (T)(4.0) * (h0[4] - h0[2]) - (T)(5.0) * tmp[13] + tmp[15];
    pictran[pictran_off + pwh_str * 25] = (T)(-4.0) * (tmp[12] + tmp[13]) + tmp[14] + tmp[15];
    pictran[pictran_off + pwh_str * 26] = (T)(4.0) * (tmp[12] - tmp[13]) - tmp[14] + tmp[15];
    pictran[pictran_off + pwh_str * 27] = (T)(2.0) * (tmp[14] - tmp[12]) - tmp[13] + tmp[15];
    pictran[pictran_off + pwh_str * 28] = (T)(2.0) * (tmp[12] - tmp[14]) - tmp[13] + tmp[15];
    pictran[pictran_off + pwh_str * 29] =
        (T)(4.0) * tmp[12] + (T)(2.0) * (h5[1] - h5[3]) - h5[2] - (T)(5.0) * tmp[14] + h5[4];

    pictran[pictran_off + pwh_str * 30] =
        (T)(16.0) * h0[1] - (T)(20.0) * h0[3] + (T)(4.0) * h0[5] - (T)(5.0) * h2[5] + h4[5];
    pictran[pictran_off + pwh_str * 31] = (T)(-4.0) * (h1[5] + h2[5]) + h3[5] + h4[5];
    pictran[pictran_off + pwh_str * 32] = (T)(4.0) * (h1[5] - h2[5]) - h3[5] + h4[5];
    pictran[pictran_off + pwh_str * 33] = (T)(2.0) * (h3[5] - h1[5]) - h2[5] + h4[5];
    pictran[pictran_off + pwh_str * 34] = (T)(2.0) * (h1[5] - h3[5]) - h2[5] + h4[5];
    pictran[pictran_off + pwh_str * 35] =
        (T)(4.0) * (h1[5] + h5[1]) - (T)(5.0) * (h3[5] + h5[3]) + h5[5];
}
