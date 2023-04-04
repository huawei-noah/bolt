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
#define MANGLE_NAME_IMPL(base, IOM, AM, ON, KN) base##IOM##AM##ON##KN
#define MANGLE_NAME(base, IOM, AM, ON, KN) MANGLE_NAME_IMPL(base, IOM, AM, ON, KN)

#define CALCORE(iv, fv, ov) DIRECT_CONV_CAL_CORE_S1(iv, fv, ov)

#if defined(USE_INPUT_IMG)
#define LOAD_INPUT(iv, mem)                                    \
    {                                                          \
        for (uchar k = 0; k < IN; k++) {                       \
            LOAD_MEM_V4(iv[k], (int4)(ix, iy + k, i, 0), mem); \
        }                                                      \
    }
#define ADD_IN_OFF
#else
#define LOAD_INPUT(iv, mem)                               \
    {                                                     \
        for (uchar k = 0; k < IN; k++) {                  \
            LOAD_MEM_V4(iv[k], in_off + k * iw_str, mem); \
        }                                                 \
    }
#define ADD_IN_OFF in_off += ihw_str
#endif

#define UPDATE_FLT(fv, off, mem) \
    {                            \
        fv = vload16(off, mem);  \
        off++;                   \
    }

#if (KN == 2)
#define CALCORE_FUNC(FUNC, iv, fv, ov, flt_off, flt_mem) \
    {                                                    \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[0]);                             \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[1]);                             \
    }
#elif (KN == 4)
#define CALCORE_FUNC(FUNC, iv, fv, ov, flt_off, flt_mem) \
    {                                                    \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[0]);                             \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[1]);                             \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[2]);                             \
        UPDATE_FLT(fv, flt_off, flt_mem);                \
        FUNC(iv, fv, ov[3]);                             \
    }
#endif

#if defined(USE_OUTPUT_IMG)
#define VSTORE_VEC(v0, v1, ix, iy, iz, img)               \
    {                                                     \
        ACTIVATION_V4(v0);                                \
        ACTIVATION_V4(v1);                                \
        STORE_MEM_V4(v0, (int4)(ix, iy, iz, 0), img);     \
        STORE_MEM_V4(v1, (int4)(ix + 1, iy, iz, 0), img); \
    }
#if (KN == 2)
#define STORE_OUTPUT(ov, mem)                                                         \
    {                                                                                 \
        int ix = idx << 1;                                                            \
        int iy = (idy << 1) * ON + (idz & 1);                                         \
        int iz = idz >> 1;                                                            \
        for (uchar i = 0; i < ON; i++) {                                              \
            if (iy + (i << 1) < oh) {                                                 \
                VSTORE_VEC(out_val[0][i], out_val[1][i], ix, iy + (i << 1), iz, out); \
            }                                                                         \
        }                                                                             \
    }
#elif (KN == 4)
#define STORE_OUTPUT(ov, mem)                                                          \
    {                                                                                  \
        int ix = idx << 1;                                                             \
        int iy = (idy << 1) * ON;                                                      \
        int iz = idz;                                                                  \
        uchar k = 0;                                                                   \
        for (uchar i = 0; i < ON; ++i) {                                               \
            for (uchar j = 0; j < 4; j += 2) {                                         \
                if (iy + k < oh) {                                                     \
                    VSTORE_VEC(out_val[j][i], out_val[j + 1][i], ix, iy + k, iz, out); \
                    k++;                                                               \
                }                                                                      \
            }                                                                          \
        }                                                                              \
    }
#endif

#else
#define VSTORE_VEC(v0, v1, off, buf)                                                         \
    {                                                                                        \
        ACTIVATION_V4(v0);                                                                   \
        ACTIVATION_V4(v1);                                                                   \
        vstore8((T8)(v0.s0, v0.s1, v0.s2, v0.s3, v1.s0, v1.s1, v1.s2, v1.s3), 0, buf + off); \
    }
#if (KN == 2)
#define STORE_OUTPUT(ov, mem)                                                               \
    {                                                                                       \
        int index_y = (idy << 1) * ON + (idz & 1);                                          \
        int out_off = (idx << 1) + o_off;                                                   \
        out_off += index_y * ow_str;                                                        \
        out_off += (idz >> 1) * ohw_str;                                                    \
        out_off = (out_off << 2);                                                           \
        for (uchar i = 0; i < ON; i++) {                                                    \
            if (index_y + (i << 1) < oh) {                                                  \
                VSTORE_VEC(out_val[0][i], out_val[1][i], out_off + ow_str * (i << 3), out); \
            }                                                                               \
        }                                                                                   \
    }
#elif (KN == 4)
#define STORE_OUTPUT(ov, mem)                                                                       \
    {                                                                                               \
        int index_y = (idy << 1) * ON;                                                              \
        int out_off = (idx << 1) + o_off;                                                           \
        out_off += index_y * ow_str;                                                                \
        out_off += idz * ohw_str;                                                                   \
        out_off = (out_off << 2);                                                                   \
        uchar k = 0;                                                                                \
        for (uchar i = 0; i < ON; ++i) {                                                            \
            for (uchar j = 0; j < 4; j += 2) {                                                      \
                if (index_y + k < oh) {                                                             \
                    VSTORE_VEC(out_val[j][i], out_val[j + 1][i], out_off + ow_str * (k << 2), out); \
                    k++;                                                                            \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
    }
#endif
#endif

__kernel void MANGLE_NAME(deconv_gemm_f2s2_qc_, IOM, AM, ON, KN)(const int iw_str,
    const int ihw_str,
    const int ic_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int ohw_str,
    const int o_off,
    const int oh,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    __global const T *flt,
    __read_only image1d_t bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 in_val[IN];
    T16 flt_val;
    T4 out_val[KN][ON];

#if (KN == 2)
    out_val[0][0] = READ_IMAGE(bias, sampler, (idz >> 1));
#elif (KN == 4)
    out_val[0][0] = READ_IMAGE(bias, sampler, idz);
#endif
    for (uchar i = 0; i < KN; i++) {
        for (uchar j = 0; j < ON; j++) {
            out_val[i][j] = out_val[0][0];
        }
    }

#if defined(USE_INPUT_IMG)
    int ix = idx + iw_off;
    int iy = idy * ON + ih_off;
#else
    int in_off = (idy * ON + ih_off) * iw_str + idx + iw_off;
#endif
    int flt_off = idz * ic_str * KN;

    for (int i = 0; i < ic_str; ++i) {
        LOAD_INPUT(in_val, in);
        CALCORE_FUNC(CALCORE, in_val, flt_val, out_val, flt_off, flt);
        ADD_IN_OFF;
    }
    STORE_OUTPUT(out_val, out);
}
