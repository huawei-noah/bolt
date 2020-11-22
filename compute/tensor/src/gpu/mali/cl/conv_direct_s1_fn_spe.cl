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
#define MANGLE_NAME_IMPL(base, F, ON) base##F##ON
#define MANGLE_NAME(base, F, ON) MANGLE_NAME_IMPL(base, F, ON)

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
#endif

#if defined(USE_NCHW)
#if (ON == 3)
#define STORE_OUT(ov, off, id, ow, buf)     \
    {                                       \
        ACTIVATION_ARRAY3(ov);              \
        if (id + 3 < ow) {                  \
            STORE_BUF_ARRAY3(ov, off, buf); \
        } else {                            \
            buf[off] = ov[0];               \
            if (id + 1 < ow)                \
                buf[off + 1] = ov[1];       \
            if (id + 2 < ow)                \
                buf[off + 2] = ov[2];       \
        }                                   \
    }
#endif
#if (ON == 4)
#define STORE_OUT(ov, off, id, ow, buf)     \
    {                                       \
        ACTIVATION_ARRAY4(ov);              \
        if (id + 4 < ow) {                  \
            STORE_BUF_ARRAY4(ov, off, buf); \
        } else {                            \
            buf[off] = ov[0];               \
            if (id + 1 < ow)                \
                buf[off + 1] = ov[1];       \
            if (id + 2 < ow)                \
                buf[off + 2] = ov[2];       \
            if (id + 3 < ow)                \
                buf[off + 3] = ov[3];       \
        }                                   \
    }
#endif
#if (ON == 5)
#define STORE_OUT(ov, off, id, ow, buf)     \
    {                                       \
        ACTIVATION_ARRAY5(ov);              \
        if (id + 5 < ow) {                  \
            STORE_BUF_ARRAY5(ov, off, buf); \
        } else {                            \
            buf[off] = ov[0];               \
            if (id + 1 < ow)                \
                buf[off + 1] = ov[1];       \
            if (id + 2 < ow)                \
                buf[off + 2] = ov[2];       \
            if (id + 3 < ow)                \
                buf[off + 3] = ov[3];       \
            if (id + 4 < ow)                \
                buf[off + 4] = ov[4];       \
        }                                   \
    }
#endif
#if (ON == 6)
#define STORE_OUT(ov, off, id, ow, buf)     \
    {                                       \
        ACTIVATION_ARRAY6(ov);              \
        if (id + 6 < ow) {                  \
            STORE_BUF_ARRAY6(ov, off, buf); \
        } else {                            \
            buf[off] = ov[0];               \
            if (id + 1 < ow)                \
                buf[off + 1] = ov[1];       \
            if (id + 2 < ow)                \
                buf[off + 2] = ov[2];       \
            if (id + 3 < ow)                \
                buf[off + 3] = ov[3];       \
            if (id + 4 < ow)                \
                buf[off + 4] = ov[4];       \
            if (id + 5 < ow)                \
                buf[off + 5] = ov[5];       \
        }                                   \
    }
#endif
#if (ON == 7)
#define STORE_OUT(ov, off, id, ow, buf)     \
    {                                       \
        ACTIVATION_ARRAY7(ov);              \
        if (id + 7 < ow) {                  \
            STORE_BUF_ARRAY7(ov, off, buf); \
        } else {                            \
            buf[off] = ov[0];               \
            if (id + 1 < ow)                \
                buf[off + 1] = ov[1];       \
            if (id + 2 < ow)                \
                buf[off + 2] = ov[2];       \
            if (id + 3 < ow)                \
                buf[off + 3] = ov[3];       \
            if (id + 4 < ow)                \
                buf[off + 4] = ov[4];       \
            if (id + 5 < ow)                \
                buf[off + 5] = ov[5];       \
            if (id + 6 < ow)                \
                buf[off + 6] = ov[6];       \
        }                                   \
    }
#endif
#if (ON == 8)
#define STORE_OUT(ov, off, id, ow, buf)     \
    {                                       \
        ACTIVATION_ARRAY8(ov);              \
        if (id + 8 < ow) {                  \
            STORE_BUF_ARRAY8(ov, off, buf); \
        } else {                            \
            buf[off] = ov[0];               \
            if (id + 1 < ow)                \
                buf[off + 1] = ov[1];       \
            if (id + 2 < ow)                \
                buf[off + 2] = ov[2];       \
            if (id + 3 < ow)                \
                buf[off + 3] = ov[3];       \
            if (id + 4 < ow)                \
                buf[off + 4] = ov[4];       \
            if (id + 5 < ow)                \
                buf[off + 5] = ov[5];       \
            if (id + 6 < ow)                \
                buf[off + 6] = ov[6];       \
            if (id + 7 < ow)                \
                buf[off + 7] = ov[7];       \
        }                                   \
    }
#endif
#else
#if (ON == 3)
#define STORE_OUT(ov, off, str, id, ow, buf)  \
    {                                         \
        ACTIVATION_ARRAY3(ov);                \
        T4 tmp = 0;                           \
        tmp.x = ov[0];                        \
        vstore4(tmp, off, buf);               \
        if (id + 1 < ow) {                    \
            tmp.x = ov[1];                    \
            vstore4(tmp, off + str, buf);     \
        }                                     \
        if (id + 2 < ow) {                    \
            tmp.x = ov[2];                    \
            vstore4(tmp, off + 2 * str, buf); \
        }                                     \
    }
#endif
#if (ON == 4)
#define STORE_OUT(ov, off, str, id, ow, buf)  \
    {                                         \
        ACTIVATION_ARRAY4(ov);                \
        T4 tmp = 0;                           \
        tmp.x = ov[0];                        \
        vstore4(tmp, off, buf);               \
        if (id + 1 < ow) {                    \
            tmp.x = ov[1];                    \
            vstore4(tmp, off + str, buf);     \
        }                                     \
        if (id + 2 < ow) {                    \
            tmp.x = ov[2];                    \
            vstore4(tmp, off + 2 * str, buf); \
        }                                     \
        if (id + 3 < ow) {                    \
            tmp.x = ov[3];                    \
            vstore4(tmp, off + 3 * str, buf); \
        }                                     \
    }
#endif
#if (ON == 5)
#define STORE_OUT(ov, off, str, id, ow, buf)  \
    {                                         \
        ACTIVATION_ARRAY5(ov);                \
        T4 tmp = 0;                           \
        tmp.x = ov[0];                        \
        vstore4(tmp, off, buf);               \
        if (id + 1 < ow) {                    \
            tmp.x = ov[1];                    \
            vstore4(tmp, off + str, buf);     \
        }                                     \
        if (id + 2 < ow) {                    \
            tmp.x = ov[2];                    \
            vstore4(tmp, off + 2 * str, buf); \
        }                                     \
        if (id + 3 < ow) {                    \
            tmp.x = ov[3];                    \
            vstore4(tmp, off + 3 * str, buf); \
        }                                     \
        if (id + 4 < ow) {                    \
            tmp.x = ov[4];                    \
            vstore4(tmp, off + 4 * str, buf); \
        }                                     \
    }
#endif
#if (ON == 6)
#define STORE_OUT(ov, off, str, id, ow, buf)  \
    {                                         \
        ACTIVATION_ARRAY6(ov);                \
        T4 tmp = 0;                           \
        tmp.x = ov[0];                        \
        vstore4(tmp, off, buf);               \
        if (id + 1 < ow) {                    \
            tmp.x = ov[1];                    \
            vstore4(tmp, off + str, buf);     \
        }                                     \
        if (id + 2 < ow) {                    \
            tmp.x = ov[2];                    \
            vstore4(tmp, off + 2 * str, buf); \
        }                                     \
        if (id + 3 < ow) {                    \
            tmp.x = ov[3];                    \
            vstore4(tmp, off + 3 * str, buf); \
        }                                     \
        if (id + 4 < ow) {                    \
            tmp.x = ov[4];                    \
            vstore4(tmp, off + 4 * str, buf); \
        }                                     \
        if (id + 5 < ow) {                    \
            tmp.x = ov[5];                    \
            vstore4(tmp, off + 5 * str, buf); \
        }                                     \
    }
#endif
#if (ON == 7)
#define STORE_OUT(ov, off, str, id, ow, buf)  \
    {                                         \
        ACTIVATION_ARRAY7(ov);                \
        T4 tmp = 0;                           \
        tmp.x = ov[0];                        \
        vstore4(tmp, off, buf);               \
        if (id + 1 < ow) {                    \
            tmp.x = ov[1];                    \
            vstore4(tmp, off + str, buf);     \
        }                                     \
        if (id + 2 < ow) {                    \
            tmp.x = ov[2];                    \
            vstore4(tmp, off + 2 * str, buf); \
        }                                     \
        if (id + 3 < ow) {                    \
            tmp.x = ov[3];                    \
            vstore4(tmp, off + 3 * str, buf); \
        }                                     \
        if (id + 4 < ow) {                    \
            tmp.x = ov[4];                    \
            vstore4(tmp, off + 4 * str, buf); \
        }                                     \
        if (id + 5 < ow) {                    \
            tmp.x = ov[5];                    \
            vstore4(tmp, off + 5 * str, buf); \
        }                                     \
        if (id + 6 < ow) {                    \
            tmp.x = ov[6];                    \
            vstore4(tmp, off + 6 * str, buf); \
        }                                     \
    }
#endif
#if (ON == 8)
#define STORE_OUT(ov, off, str, id, ow, buf)  \
    {                                         \
        ACTIVATION_ARRAY8(ov);                \
        T4 tmp = 0;                           \
        tmp.x = ov[0];                        \
        vstore4(tmp, off, buf);               \
        if (id + 1 < ow) {                    \
            tmp.x = ov[1];                    \
            vstore4(tmp, off + str, buf);     \
        }                                     \
        if (id + 2 < ow) {                    \
            tmp.x = ov[2];                    \
            vstore4(tmp, off + 2 * str, buf); \
        }                                     \
        if (id + 3 < ow) {                    \
            tmp.x = ov[3];                    \
            vstore4(tmp, off + 3 * str, buf); \
        }                                     \
        if (id + 4 < ow) {                    \
            tmp.x = ov[4];                    \
            vstore4(tmp, off + 4 * str, buf); \
        }                                     \
        if (id + 5 < ow) {                    \
            tmp.x = ov[5];                    \
            vstore4(tmp, off + 5 * str, buf); \
        }                                     \
        if (id + 6 < ow) {                    \
            tmp.x = ov[6];                    \
            vstore4(tmp, off + 6 * str, buf); \
        }                                     \
        if (id + 7 < ow) {                    \
            tmp.x = ov[7];                    \
            vstore4(tmp, off + 7 * str, buf); \
        }                                     \
    }
#endif
#endif

#if defined(USE_NCHW)
#if defined(USE_RELU)
__kernel void MANGLE_NAME(conv_direct_s1_fn_spe_relu_nchw_, F, ON)
#elif defined(USE_RELU6)
__kernel void MANGLE_NAME(conv_direct_s1_fn_spe_relu6_nchw_, F, ON)
#else
__kernel void MANGLE_NAME(conv_direct_s1_fn_spe_nchw_, F, ON)
#endif
#else
#if defined(USE_RELU)
__kernel void MANGLE_NAME(conv_direct_s1_fn_spe_relu_, F, ON)
#elif defined(USE_RELU6)
__kernel void MANGLE_NAME(conv_direct_s1_fn_spe_relu6_, F, ON)
#else
__kernel void MANGLE_NAME(conv_direct_s1_fn_spe_, F, ON)
#endif
#endif
    (const int ih_str,
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
        const int bx,
        const int by,
        __global const T *in,
        __global const T *flt,
        __global const T *bias,
        __global T *out)
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
    int flt_off = 0;
    int in_off = (idy * ON + iw_off) * ih_str + idx * sh + ih_off;

#if (F == 1)
    for (int i = 0; i < ic_str; ++i) {
        LOAD_INPUT_BUF_ARRAY_V4(in_val, in_off, ih_str, in);
        flt_val = vload4(flt_off, flt);
        CALCORE(in_val, flt_val, out_val);
        flt_off += 1;
        in_off += ihw_str;
    }
#else
    for (int i = 0; i < ic_str; ++i) {
        for (uchar j = 0; j < F; j++) {
            LOAD_INPUT_BUF_ARRAY_V4(in_val, in_off + j, ih_str, in);
            for (uchar k = 0; k < F; k++) {
                flt_val = vload4(flt_off + k, flt);
                CALCORE(in_val, flt_val, out_val);
                UPDATE_REG(in_val);
            }
            flt_off += F;
        }
        in_off += ihw_str;
    }
#endif

#if defined(USE_NCHW)
    int out_off = (idx + oh_off) * ow_str + idy * ON + ow_off;
    STORE_OUT(out_val, out_off, idy * ON, ow, out);
#else
    int out_off = (idy * ON + ow_off) * oh_str + idx + oh_off;
    STORE_OUT(out_val, out_off, oh_str, idy * ON, ow, out);
#endif
}
