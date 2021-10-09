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
#if defined(USE_NCHW)
#define FM nchw_
#endif

#if defined(USE_INPUT_IMG)
#define LOAD_INPUT_ARRAY(j)                                                           \
    {                                                                                 \
        LOAD_INPUT_MEM_ARRAY_V4(in_val, in_off_x + j, in_off_y, in_off_z + i, 1, in); \
    }
#define ADD_IN_OFF
#else
#define LOAD_INPUT_ARRAY(j)                                            \
    {                                                                  \
        LOAD_INPUT_MEM_ARRAY_V4(in_val, in_off + j, iw_str, 0, 1, in); \
    }
#define ADD_IN_OFF in_off += ihw_str;
#endif

#if (ON == 3)
#define LOAD_BIAS(ov, bias) \
    {                       \
        ov[0] = bias[0];    \
        ov[1] = ov[0];      \
        ov[2] = ov[0];      \
    }
#define CALCORE(iv, fv, ov)                                                         \
    {                                                                               \
        ov[0] += iv[0].x * fv.x + iv[0].y * fv.y + iv[0].z * fv.z + iv[0].w * fv.w; \
        ov[1] += iv[1].x * fv.x + iv[1].y * fv.y + iv[1].z * fv.z + iv[1].w * fv.w; \
        ov[2] += iv[2].x * fv.x + iv[2].y * fv.y + iv[2].z * fv.z + iv[2].w * fv.w; \
    }
#define STORE_OUT(ov, off, str, id, oh, buf) \
    {                                        \
        ACTIVATION_ARRAY3(ov);               \
        buf[off] = ov[0];                    \
        if (id + 1 < oh) {                   \
            buf[off + str] = ov[1];          \
        }                                    \
        if (id + 2 < oh) {                   \
            buf[off + str * 2] = ov[2];      \
        }                                    \
    }
#endif

#if (ON == 4)
#define LOAD_BIAS(ov, bias) \
    {                       \
        ov[0] = bias[0];    \
        ov[1] = ov[0];      \
        ov[2] = ov[0];      \
        ov[3] = ov[0];      \
    }
#define CALCORE(iv, fv, ov)                                                         \
    {                                                                               \
        ov[0] += iv[0].x * fv.x + iv[0].y * fv.y + iv[0].z * fv.z + iv[0].w * fv.w; \
        ov[1] += iv[1].x * fv.x + iv[1].y * fv.y + iv[1].z * fv.z + iv[1].w * fv.w; \
        ov[2] += iv[2].x * fv.x + iv[2].y * fv.y + iv[2].z * fv.z + iv[2].w * fv.w; \
        ov[3] += iv[3].x * fv.x + iv[3].y * fv.y + iv[3].z * fv.z + iv[3].w * fv.w; \
    }
#define STORE_OUT(ov, off, str, id, oh, buf) \
    {                                        \
        ACTIVATION_ARRAY4(ov);               \
        buf[off] = ov[0];                    \
        if (id + 1 < oh) {                   \
            buf[off + str] = ov[1];          \
        }                                    \
        if (id + 2 < oh) {                   \
            buf[off + str * 2] = ov[2];      \
        }                                    \
        if (id + 3 < oh) {                   \
            buf[off + str * 3] = ov[3];      \
        }                                    \
    }
#endif

#if (ON == 5)
#define LOAD_BIAS(ov, bias) \
    {                       \
        ov[0] = bias[0];    \
        ov[1] = ov[0];      \
        ov[2] = ov[0];      \
        ov[3] = ov[0];      \
        ov[4] = ov[0];      \
    }
#define CALCORE(iv, fv, ov)                                                         \
    {                                                                               \
        ov[0] += iv[0].x * fv.x + iv[0].y * fv.y + iv[0].z * fv.z + iv[0].w * fv.w; \
        ov[1] += iv[1].x * fv.x + iv[1].y * fv.y + iv[1].z * fv.z + iv[1].w * fv.w; \
        ov[2] += iv[2].x * fv.x + iv[2].y * fv.y + iv[2].z * fv.z + iv[2].w * fv.w; \
        ov[3] += iv[3].x * fv.x + iv[3].y * fv.y + iv[3].z * fv.z + iv[3].w * fv.w; \
        ov[4] += iv[4].x * fv.x + iv[4].y * fv.y + iv[4].z * fv.z + iv[4].w * fv.w; \
    }
#define STORE_OUT(ov, off, str, id, oh, buf) \
    {                                        \
        ACTIVATION_ARRAY5(ov);               \
        buf[off] = ov[0];                    \
        if (id + 1 < oh) {                   \
            buf[off + str] = ov[1];          \
        }                                    \
        if (id + 2 < oh) {                   \
            buf[off + str * 2] = ov[2];      \
        }                                    \
        if (id + 3 < oh) {                   \
            buf[off + str * 3] = ov[3];      \
        }                                    \
        if (id + 4 < oh) {                   \
            buf[off + str * 4] = ov[4];      \
        }                                    \
    }
#endif

#if (ON == 6)
#define LOAD_BIAS(ov, bias) \
    {                       \
        ov[0] = bias[0];    \
        ov[1] = ov[0];      \
        ov[2] = ov[0];      \
        ov[3] = ov[0];      \
        ov[4] = ov[0];      \
        ov[5] = ov[0];      \
    }
#define CALCORE(iv, fv, ov)                                                         \
    {                                                                               \
        ov[0] += iv[0].x * fv.x + iv[0].y * fv.y + iv[0].z * fv.z + iv[0].w * fv.w; \
        ov[1] += iv[1].x * fv.x + iv[1].y * fv.y + iv[1].z * fv.z + iv[1].w * fv.w; \
        ov[2] += iv[2].x * fv.x + iv[2].y * fv.y + iv[2].z * fv.z + iv[2].w * fv.w; \
        ov[3] += iv[3].x * fv.x + iv[3].y * fv.y + iv[3].z * fv.z + iv[3].w * fv.w; \
        ov[4] += iv[4].x * fv.x + iv[4].y * fv.y + iv[4].z * fv.z + iv[4].w * fv.w; \
        ov[5] += iv[5].x * fv.x + iv[5].y * fv.y + iv[5].z * fv.z + iv[5].w * fv.w; \
    }
#define STORE_OUT(ov, off, str, id, oh, buf) \
    {                                        \
        ACTIVATION_ARRAY6(ov);               \
        buf[off] = ov[0];                    \
        if (id + 1 < oh) {                   \
            buf[off + str] = ov[1];          \
        }                                    \
        if (id + 2 < oh) {                   \
            buf[off + str * 2] = ov[2];      \
        }                                    \
        if (id + 3 < oh) {                   \
            buf[off + str * 3] = ov[3];      \
        }                                    \
        if (id + 4 < oh) {                   \
            buf[off + str * 4] = ov[4];      \
        }                                    \
        if (id + 5 < oh) {                   \
            buf[off + str * 5] = ov[5];      \
        }                                    \
    }
#endif

#if (ON == 7)
#define LOAD_BIAS(ov, bias) \
    {                       \
        ov[0] = bias[0];    \
        ov[1] = ov[0];      \
        ov[2] = ov[0];      \
        ov[3] = ov[0];      \
        ov[4] = ov[0];      \
        ov[5] = ov[0];      \
        ov[6] = ov[0];      \
    }
#define CALCORE(iv, fv, ov)                                                         \
    {                                                                               \
        ov[0] += iv[0].x * fv.x + iv[0].y * fv.y + iv[0].z * fv.z + iv[0].w * fv.w; \
        ov[1] += iv[1].x * fv.x + iv[1].y * fv.y + iv[1].z * fv.z + iv[1].w * fv.w; \
        ov[2] += iv[2].x * fv.x + iv[2].y * fv.y + iv[2].z * fv.z + iv[2].w * fv.w; \
        ov[3] += iv[3].x * fv.x + iv[3].y * fv.y + iv[3].z * fv.z + iv[3].w * fv.w; \
        ov[4] += iv[4].x * fv.x + iv[4].y * fv.y + iv[4].z * fv.z + iv[4].w * fv.w; \
        ov[5] += iv[5].x * fv.x + iv[5].y * fv.y + iv[5].z * fv.z + iv[5].w * fv.w; \
        ov[6] += iv[6].x * fv.x + iv[6].y * fv.y + iv[6].z * fv.z + iv[6].w * fv.w; \
    }
#define STORE_OUT(ov, off, str, id, oh, buf) \
    {                                        \
        ACTIVATION_ARRAY7(ov);               \
        buf[off] = ov[0];                    \
        if (id + 1 < oh) {                   \
            buf[off + str] = ov[1];          \
        }                                    \
        if (id + 2 < oh) {                   \
            buf[off + str * 2] = ov[2];      \
        }                                    \
        if (id + 3 < oh) {                   \
            buf[off + str * 3] = ov[3];      \
        }                                    \
        if (id + 4 < oh) {                   \
            buf[off + str * 4] = ov[4];      \
        }                                    \
        if (id + 5 < oh) {                   \
            buf[off + str * 5] = ov[5];      \
        }                                    \
        if (id + 6 < oh) {                   \
            buf[off + str * 6] = ov[6];      \
        }                                    \
    }
#endif

#if (ON == 8)
#define LOAD_BIAS(ov, bias) \
    {                       \
        ov[0] = bias[0];    \
        ov[1] = ov[0];      \
        ov[2] = ov[0];      \
        ov[3] = ov[0];      \
        ov[4] = ov[0];      \
        ov[5] = ov[0];      \
        ov[6] = ov[0];      \
        ov[7] = ov[0];      \
    }
#define CALCORE(iv, fv, ov)                                                         \
    {                                                                               \
        ov[0] += iv[0].x * fv.x + iv[0].y * fv.y + iv[0].z * fv.z + iv[0].w * fv.w; \
        ov[1] += iv[1].x * fv.x + iv[1].y * fv.y + iv[1].z * fv.z + iv[1].w * fv.w; \
        ov[2] += iv[2].x * fv.x + iv[2].y * fv.y + iv[2].z * fv.z + iv[2].w * fv.w; \
        ov[3] += iv[3].x * fv.x + iv[3].y * fv.y + iv[3].z * fv.z + iv[3].w * fv.w; \
        ov[4] += iv[4].x * fv.x + iv[4].y * fv.y + iv[4].z * fv.z + iv[4].w * fv.w; \
        ov[5] += iv[5].x * fv.x + iv[5].y * fv.y + iv[5].z * fv.z + iv[5].w * fv.w; \
        ov[6] += iv[6].x * fv.x + iv[6].y * fv.y + iv[6].z * fv.z + iv[6].w * fv.w; \
        ov[7] += iv[7].x * fv.x + iv[7].y * fv.y + iv[7].z * fv.z + iv[7].w * fv.w; \
    }
#define STORE_OUT(ov, off, str, id, oh, buf) \
    {                                        \
        ACTIVATION_ARRAY8(ov);               \
        buf[off] = ov[0];                    \
        if (id + 1 < oh) {                   \
            buf[off + str] = ov[1];          \
        }                                    \
        if (id + 2 < oh) {                   \
            buf[off + str * 2] = ov[2];      \
        }                                    \
        if (id + 3 < oh) {                   \
            buf[off + str * 3] = ov[3];      \
        }                                    \
        if (id + 4 < oh) {                   \
            buf[off + str * 4] = ov[4];      \
        }                                    \
        if (id + 5 < oh) {                   \
            buf[off + str * 5] = ov[5];      \
        }                                    \
        if (id + 6 < oh) {                   \
            buf[off + str * 6] = ov[6];      \
        }                                    \
        if (id + 7 < oh) {                   \
            buf[off + str * 7] = ov[7];      \
        }                                    \
    }
#endif

__kernel void MANGLE_NAME(conv_direct_sh1_fn_spe_, IOM, AM, FM, FW, FH, ON)(const int iw_str,
    const int ihw_str,
    const int ic_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int o_off,
    const int oh,
    const int sw,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    __global const T *flt,
    __global const T *bias,
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 flt_val;
    T4 in_val[IN];
    T out_val[ON];

    LOAD_BIAS(out_val, bias);
#if defined(USE_INPUT_IMG)
    int in_off_x = idx * sw + iw_off;
    int in_off_y = idy * ON + ih_off;
    int in_off_z = 0;
#else
    int in_off = (idy * ON + ih_off) * iw_str + idx * sw + iw_off;
#endif
    int flt_off = 0;

#if (FW == 1 && FH == 1)
    for (int i = 0; i < ic_str; ++i) {
        LOAD_INPUT_ARRAY(0);
        flt_val = vload4(flt_off, flt);
        CALCORE(in_val, flt_val, out_val);
        flt_off += 1;
        ADD_IN_OFF;
    }
#else
    for (int i = 0; i < ic_str; ++i) {
        for (uchar j = 0; j < FW; j++) {
            LOAD_INPUT_ARRAY(j);
            for (uchar k = 0; k < FH; k++) {
                flt_val = vload4(flt_off + k, flt);
                CALCORE(in_val, flt_val, out_val);
                UPDATE_REG(in_val);
            }
            flt_off += FH;
        }
        ADD_IN_OFF;
    }
#endif
    int out_off = idy * ON * ow_str + idx + o_off;
    STORE_OUT(out_val, out_off, ow_str, idy * ON, oh, out);
}
