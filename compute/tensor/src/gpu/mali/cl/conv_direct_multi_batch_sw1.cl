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
#define MANGLE_NAME_IMPL(base, AM, FW, FH, ON, KN, BN) base##AM##FW##FH##ON##KN##BN
#define MANGLE_NAME(base, AM, FW, FH, ON, KN, BN) MANGLE_NAME_IMPL(base, AM, FW, FH, ON, KN, BN)

#if (KN == 1)
#define LOAD_BIAS(val, idz, bias)                   \
    {                                               \
        LOADBIAS_IMAGE_ARRAY_V4(val[0], idz, bias); \
    }
#define COPY_BIAS(src, dst)               \
    {                                     \
        SET_REG_ARRAY(src[0][0], dst[0]); \
    }
#define STORE_BUF(ov, off, hw_str, h_str, id, ow, out)             \
    {                                                              \
        STORE_OUTPUT_BUF_ARRAY_V4(ov[0], off, h_str, id, ow, out); \
    }
#elif (KN == 2)
#define LOAD_BIAS(val, idz, bias)                           \
    {                                                       \
        LOADBIAS_IMAGE_ARRAY_V4(val[0], idz * 2, bias);     \
        LOADBIAS_IMAGE_ARRAY_V4(val[1], idz * 2 + 1, bias); \
    }
#define COPY_BIAS(src, dst)               \
    {                                     \
        SET_REG_ARRAY(src[0][0], dst[0]); \
        SET_REG_ARRAY(src[1][0], dst[1]); \
    }
#define STORE_BUF(ov, off, hw_str, h_str, id, ow, out)                      \
    {                                                                       \
        STORE_OUTPUT_BUF_ARRAY_V4(ov[0], off, h_str, id, ow, out);          \
        STORE_OUTPUT_BUF_ARRAY_V4(ov[1], off + hw_str, h_str, id, ow, out); \
    }
#elif (KN == 4)
#define LOAD_BIAS(val, idz, bias)                           \
    {                                                       \
        LOADBIAS_IMAGE_ARRAY_V4(val[0], idz * 4, bias);     \
        LOADBIAS_IMAGE_ARRAY_V4(val[1], idz * 4 + 1, bias); \
        LOADBIAS_IMAGE_ARRAY_V4(val[2], idz * 4 + 2, bias); \
        LOADBIAS_IMAGE_ARRAY_V4(val[3], idz * 4 + 3, bias); \
    }
#define COPY_BIAS(src, dst)               \
    {                                     \
        SET_REG_ARRAY(src[0][0], dst[0]); \
        SET_REG_ARRAY(src[1][0], dst[1]); \
        SET_REG_ARRAY(src[2][0], dst[2]); \
        SET_REG_ARRAY(src[3][0], dst[3]); \
    }
#define STORE_BUF(ov, off, hw_str, h_str, id, ow, out)                          \
    {                                                                           \
        STORE_OUTPUT_BUF_ARRAY_V4(ov[0], off, h_str, id, ow, out);              \
        STORE_OUTPUT_BUF_ARRAY_V4(ov[1], off + hw_str, h_str, id, ow, out);     \
        STORE_OUTPUT_BUF_ARRAY_V4(ov[2], off + hw_str * 2, h_str, id, ow, out); \
        STORE_OUTPUT_BUF_ARRAY_V4(ov[3], off + hw_str * 3, h_str, id, ow, out); \
    }
#endif

#if (BN == 1)
#define LOAD_INPUT(iv, off, h_str, n_str, buf)           \
    {                                                    \
        LOAD_INPUT_BUF_ARRAY_V4(iv[0], off, h_str, buf); \
    }
#define UPDATE_INPUT_FROM_BUF(iv, off, n_str, buf) \
    {                                              \
        iv[0][LN] = vload4(off, buf);              \
    }
#define UPDATE_INPUT_REG(iv) \
    {                        \
        UPDATE_REG(iv[0]);   \
    }
#define CALCORE_BN(iv, fv, ov0)                       \
    {                                                 \
        DIRECT_CONV_CAL_CORE_S1(iv[0], flt_val, ov0); \
    }
#define CALCORE0 CALCORE_BN(in_val, flt_val, out_val0[0]);
#if (KN > 1)
#define CALCORE1 CALCORE_BN(in_val, flt_val, out_val0[1]);
#endif
#if (KN > 2)
#define CALCORE2 CALCORE_BN(in_val, flt_val, out_val0[2]);
#define CALCORE3 CALCORE_BN(in_val, flt_val, out_val0[3]);
#endif
#endif

#if (BN == 2)
#define LOAD_INPUT(iv, off, h_str, n_str, buf)                   \
    {                                                            \
        LOAD_INPUT_BUF_ARRAY_V4(iv[0], off, h_str, buf);         \
        LOAD_INPUT_BUF_ARRAY_V4(iv[1], off + n_str, h_str, buf); \
    }
#define UPDATE_INPUT_FROM_BUF(iv, off, n_str, buf) \
    {                                              \
        iv[0][LN] = vload4(off, buf);              \
        iv[1][LN] = vload4(off + n_str, buf);      \
    }
#define UPDATE_INPUT_REG(iv) \
    {                        \
        UPDATE_REG(iv[0]);   \
        UPDATE_REG(iv[1]);   \
    }
#define CALCORE_BN(iv, fv, ov0, ov1)                  \
    {                                                 \
        DIRECT_CONV_CAL_CORE_S1(iv[0], flt_val, ov0); \
        DIRECT_CONV_CAL_CORE_S1(iv[1], flt_val, ov1); \
    }
#define CALCORE0 CALCORE_BN(in_val, flt_val, out_val0[0], out_val1[0]);
#if (KN > 1)
#define CALCORE1 CALCORE_BN(in_val, flt_val, out_val0[1], out_val1[1]);
#endif
#if (KN > 2)
#define CALCORE2 CALCORE_BN(in_val, flt_val, out_val0[2], out_val1[2]);
#define CALCORE3 CALCORE_BN(in_val, flt_val, out_val0[3], out_val1[3]);
#endif
#endif

#if (BN == 3)
#define LOAD_INPUT(iv, off, h_str, n_str, buf)                          \
    {                                                                   \
        LOAD_INPUT_BUF_ARRAY_V4(iv[0], off, h_str, buf);                \
        LOAD_INPUT_BUF_ARRAY_V4(iv[1], off + n_str, h_str, buf);        \
        LOAD_INPUT_BUF_ARRAY_V4(iv[2], off + (n_str << 1), h_str, buf); \
    }
#define UPDATE_INPUT_FROM_BUF(iv, off, n_str, buf)   \
    {                                                \
        iv[0][LN] = vload4(off, buf);                \
        iv[1][LN] = vload4(off + n_str, buf);        \
        iv[2][LN] = vload4(off + (n_str << 1), buf); \
    }
#define UPDATE_INPUT_REG(iv) \
    {                        \
        UPDATE_REG(iv[0]);   \
        UPDATE_REG(iv[1]);   \
        UPDATE_REG(iv[2]);   \
    }
#define CALCORE_BN(iv, fv, ov0, ov1, ov2)             \
    {                                                 \
        DIRECT_CONV_CAL_CORE_S1(iv[0], flt_val, ov0); \
        DIRECT_CONV_CAL_CORE_S1(iv[1], flt_val, ov1); \
        DIRECT_CONV_CAL_CORE_S1(iv[2], flt_val, ov2); \
    }
#define CALCORE0 CALCORE_BN(in_val, flt_val, out_val0[0], out_val1[0], out_val2[0]);
#if (KN > 1)
#define CALCORE1 CALCORE_BN(in_val, flt_val, out_val0[1], out_val1[1], out_val2[1]);
#endif
#if (KN > 2)
#define CALCORE2 CALCORE_BN(in_val, flt_val, out_val0[2], out_val1[2], out_val2[2]);
#define CALCORE3 CALCORE_BN(in_val, flt_val, out_val0[3], out_val1[3], out_val2[3]);
#endif
#endif
#if (BN == 4)
#define LOAD_INPUT(iv, off, h_str, n_str, buf)                          \
    {                                                                   \
        LOAD_INPUT_BUF_ARRAY_V4(iv[0], off, h_str, buf);                \
        LOAD_INPUT_BUF_ARRAY_V4(iv[1], off + n_str, h_str, buf);        \
        LOAD_INPUT_BUF_ARRAY_V4(iv[2], off + (n_str << 1), h_str, buf); \
        LOAD_INPUT_BUF_ARRAY_V4(iv[3], off + n_str * 3, h_str, buf);    \
    }
#define UPDATE_INPUT_FROM_BUF(iv, off, n_str, buf)   \
    {                                                \
        iv[0][LN] = vload4(off, buf);                \
        iv[1][LN] = vload4(off + n_str, buf);        \
        iv[2][LN] = vload4(off + (n_str << 1), buf); \
        iv[3][LN] = vload4(off + n_str * 3, buf);    \
    }
#define UPDATE_INPUT_REG(iv) \
    {                        \
        UPDATE_REG(iv[0]);   \
        UPDATE_REG(iv[1]);   \
        UPDATE_REG(iv[2]);   \
        UPDATE_REG(iv[3]);   \
    }
#define CALCORE_BN(iv, fv, ov0, ov1, ov2, ov3)        \
    {                                                 \
        DIRECT_CONV_CAL_CORE_S1(iv[0], flt_val, ov0); \
        DIRECT_CONV_CAL_CORE_S1(iv[1], flt_val, ov1); \
        DIRECT_CONV_CAL_CORE_S1(iv[2], flt_val, ov2); \
        DIRECT_CONV_CAL_CORE_S1(iv[3], flt_val, ov3); \
    }
#define CALCORE0 CALCORE_BN(in_val, flt_val, out_val0[0], out_val1[0], out_val2[0], out_val3[0]);
#if (KN > 1)
#define CALCORE1 CALCORE_BN(in_val, flt_val, out_val0[1], out_val1[1], out_val2[1], out_val3[1]);
#endif
#if (KN > 2)
#define CALCORE2 CALCORE_BN(in_val, flt_val, out_val0[2], out_val1[2], out_val2[2], out_val3[2]);
#define CALCORE3 CALCORE_BN(in_val, flt_val, out_val0[3], out_val1[3], out_val2[3], out_val3[3]);
#endif
#endif

__kernel void MANGLE_NAME(conv_direct_multi_batch_sw1_, AM, FW, FH, ON, KN, BN)(const int ih_str,
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
    const int on,
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
    T4 in_val[BN][IN];
    T16 flt_val;
    T4 out_val0[KN][ON];

    char en = ((idn * BN + BN) <= on) ? BN : (on % BN);
    LOAD_BIAS(out_val0, idz, bias);
#if (BN > 1)
    T4 out_val1[KN][ON];
    if (en > 1) {
        COPY_BIAS(out_val0, out_val1);
    }
#endif
#if (BN > 2)
    T4 out_val2[KN][ON];
    if (en > 2) {
        COPY_BIAS(out_val0, out_val2);
    }
#endif
#if (BN > 3)
    T4 out_val3[KN][ON];
    if (en > 3) {
        COPY_BIAS(out_val0, out_val3);
    }
#endif

    int in_off = idn * BN * in_str + (idy * ON + iw_off) * ih_str + idx * sh + ih_off;
    int flt_off = idz * ic_str * FWH * KN;

    for (int i = 0; i < ic_str; ++i) {
#if (FW == 1)
        for (uchar j = 0; j < FH; ++j) {
            LOAD_INPUT(in_val, in_off + j, ih_str, in_str, in);
            flt_val = vload16(flt_off, flt);
            CALCORE0;
#if (KN > 1)
            flt_val = vload16(flt_off + 1, flt);
            CALCORE1;
#endif
#if (KN > 2)
            flt_val = vload16(flt_off + 2, flt);
            CALCORE2;
            flt_val = vload16(flt_off + 3, flt);
            CALCORE3;
#endif
            flt_off += KN;
        }
#else
        for (uchar j = 0; j < FH; ++j) {
            LOAD_INPUT(in_val, in_off + j, ih_str, in_str, in);
            for (uchar k = 0; k < FW; ++k) {
#if defined(BASIC_REG)
                UPDATE_INPUT_FROM_BUF(in_val, in_off + j + (LN + k) * ih_str, in_str, in);
#endif
                flt_val = vload16(flt_off + k * KN, flt);
                CALCORE0;
#if (KN > 1)
                flt_val = vload16(flt_off + k * KN + 1, flt);
                CALCORE1;
#endif
#if (KN > 2)
                flt_val = vload16(flt_off + k * KN + 2, flt);
                CALCORE2;
                flt_val = vload16(flt_off + k * KN + 3, flt);
                CALCORE3;
#endif
                UPDATE_INPUT_REG(in_val);
            }
            flt_off += FW * KN;
        }
#endif
        in_off += ihw_str;
    }

    int out_off =
        idn * BN * on_str + idz * KN * ohw_str + (idy * ON + ow_off) * oh_str + idx + oh_off;
    STORE_BUF(out_val0, out_off, ohw_str, oh_str, idy * ON, ow, out);
#if (BN > 1)
    if (en > 1) {
        STORE_BUF(out_val1, out_off + on_str, ohw_str, oh_str, idy * ON, ow, out);
    }
#endif
#if (BN > 2)
    if (en > 2) {
        STORE_BUF(out_val2, out_off + on_str * 2, ohw_str, oh_str, idy * ON, ow, out);
    }
#endif
#if (BN > 3)
    if (en > 3) {
        STORE_BUF(out_val3, out_off + on_str * 3, ohw_str, oh_str, idy * ON, ow, out);
    }
#endif
}
