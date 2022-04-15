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
#define SET_BIAS(ov)                            \
    {                                           \
        ov[0] = READ_IMAGE(bias, sampler, idz); \
    }

#define CALCORE(iv, fv, ov) DEPTHWISE_CAL_CORE_S1(iv, fv, ov)

#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(iv, mem, base)                                                 \
    {                                                                             \
        for (uchar k = 0; k < IN; k++) {                                          \
            LOAD_MEM_V4(iv[k], (int4)(ix + j, iy + (k << 1) + base, iz, 0), mem); \
        }                                                                         \
    }
#define UPDATE_INPUT(iv, mem, k)                                             \
    {                                                                        \
        UPDATE_REG(iv);                                                      \
        LOAD_MEM_V4(iv[LN], (int4)(ix + j, iy + (LN << 1) + k, iz, 0), mem); \
    }
#else
#define LOAD_INPUT(iv, mem, base)                                             \
    {                                                                         \
        for (uchar k = 0; k < IN; k++) {                                      \
            LOAD_MEM_V4(iv[k], in_off + j + ((k << 1) + base) * iw_str, mem); \
        }                                                                     \
    }
#define UPDATE_INPUT(iv, mem, k)                                         \
    {                                                                    \
        UPDATE_REG(iv);                                                  \
        LOAD_MEM_V4(iv[LN], in_off + j + ((LN << 1) + k) * iw_str, mem); \
    }
#endif

#define UPDATE_FLT(fv, off, mem) \
    {                            \
        fv = vload4(off, mem);   \
    }

#define CALCORE_FUNC(FUNC, iv, fv, ov, flt_off, flt_mem) \
    {                                                    \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov);                                \
    }

#if defined(USE_OUTPUT_IMG)
#define STORE_OUTPUT(ov, mem)                                         \
    {                                                                 \
        int oy = idy * ON;                                            \
        int oz = idz + ((oc + 3) >> 2) * idn;                         \
        for (uchar i = 0; i < ON; i++) {                              \
            if (oy + i < oh) {                                        \
                ACTIVATION_V4(ov[i]);                                 \
                STORE_MEM_V4(ov[i], (int4)(idx, oy + i, oz, 0), mem); \
            }                                                         \
        }                                                             \
    }
#else
#define STORE_OUTPUT(ov, mem)                                                         \
    {                                                                                 \
        int out_off = idn * on_str + idz * ohw_str + idy * ON * ow_str + idx + o_off; \
        for (uchar i = 0; i < ON; i++) {                                              \
            if (idy * ON + i < oh) {                                                  \
                ACTIVATION_V4(ov[i]);                                                 \
                STORE_MEM_V4(ov[i], out_off + i * ow_str, mem);                       \
            }                                                                         \
        }                                                                             \
    }
#endif

__kernel void MANGLE_NAME(conv_depthwise_sh2_qc_, IOM, AM, FM, FW, FH, ON)(const int iw_str,
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
    __global T *flt,
    __read_only image1d_t bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const ushort c_pitch = (oc + 3) >> 2;

    const int idz = get_global_id(2) % c_pitch;
    const int idn = get_global_id(2) / c_pitch;

    if (idx >= bx || idy >= by) {
        return;
    }

    T4 in_val[IN];
    T4 flt_val;
    T4 out_val[ON];

    SET_BIAS(out_val);
    for (uchar i = 1; i < ON; i++) {
        out_val[i] = out_val[0];
    }

#if defined(USE_INPUT_IMG)
    int ix = idx * sw + iw_off;
    int iy = idy * (ON << 1) + ih_off;
    int iz = idn * ic_str + idz;
#else
    int in_off =
        idn * in_str + +idz * ihw_str + ((idy << 1) * ON + ih_off) * iw_str + idx * sw + iw_off;
#endif
    int flt_off = idz * FWH;

    for (uchar j = 0; j < FW; j++) {
        LOAD_INPUT(in_val, in, 0);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);

#if (FH > 2)
        UPDATE_INPUT(in_val, in, 2);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 2, flt);
#endif

#if (FH > 4)
        UPDATE_INPUT(in_val, in, 4);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 4, flt);
#endif

#if (FH > 6)
        UPDATE_INPUT(in_val, in, 6);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 6, flt);
#endif

#if (FH > 8)
        UPDATE_INPUT(in_val, in, 8);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 8, flt);
#endif

#if (FH > 10)
        UPDATE_INPUT(in_val, in, 10);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 10, flt);
#endif

#if (FH > 1)
        LOAD_INPUT(in_val, in, 1);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 1, flt);
#endif

#if (FH > 3)
        UPDATE_INPUT(in_val, in, 3);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 3, flt);
#endif

#if (FH > 5)
        UPDATE_INPUT(in_val, in, 5);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 5, flt);
#endif

#if (FH > 7)
        LOAD_INPUT(in_val, in, 7);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 7, flt);
#endif

#if (FH > 9)
        UPDATE_INPUT(in_val, in, 9);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 9, flt);
#endif

#if (FH > 11)
        UPDATE_INPUT(in_val, in, 11);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off + 11, flt);
#endif
        flt_off += FH;
    }
#if defined(USE_NCHW)
    int out_off = idn * on_str + (idz << 2) * ohw_str + idy * ON * ow_str + idx + o_off;
    STORE_OUTPUT_BUF_ARRAY_V4_NCHW(out_val, out_off, ow_str, ohw_str, idy * ON, oh, out);
#else
    STORE_OUTPUT(out_val, out);
#endif
}
