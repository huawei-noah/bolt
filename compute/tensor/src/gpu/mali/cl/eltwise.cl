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
#define MANGLE_NAME_IMPL(base, IOM, AM, EM, FM, N) base##IOM##AM##EM##FM##N
#define MANGLE_NAME(base, IOM, AM, EM, FM, N) MANGLE_NAME_IMPL(base, IOM, AM, EM, FM, N)
#define FM
#if defined(USE_NCHW)
#define FM nchw_
#endif

#define READ_ONLY_KERNEL_MEM_1 __global const T *
#define READ_ONLY_KERNEL_MEM_2 __global const T *
#define READ_ONLY_KERNEL_MEM_3 __global const T *
#if defined(USE_INPUT_IMG1)
#define READ_ONLY_KERNEL_MEM_1 __read_only image3d_t
#endif
#if defined(USE_INPUT_IMG2)
#define READ_ONLY_KERNEL_MEM_2 __read_only image3d_t
#endif
#if defined(USE_INPUT_IMG3)
#define READ_ONLY_KERNEL_MEM_3 __read_only image3d_t
#endif

#if defined(USE_NCHW)
#define LOAD_VAL(ew, idx, idy, idz, iw_str, ih_str, i_off, buf, val)  \
    {                                                                 \
        int off = (idz * ih_str + idy) * iw_str + (idx << 2) + i_off; \
        val = 0;                                                      \
        if (ew == 4) {                                                \
            val = vload4(0, buf + off);                               \
        } else {                                                      \
            if (ew == 1)                                              \
                val.x = buf[off];                                     \
            if (ew == 2) {                                            \
                T2 tmp = vload2(0, buf + off);                        \
                val.x = tmp.x;                                        \
                val.y = tmp.y;                                        \
            }                                                         \
            if (ew == 3) {                                            \
                T3 tmp = vload3(0, buf + off);                        \
                val.x = tmp.x;                                        \
                val.y = tmp.y;                                        \
                val.z = tmp.z;                                        \
            }                                                         \
        }                                                             \
    }
#else
#define LOAD_VAL(ew, idx, idy, idz, iw_str, ih_str, i_off, buf, val) \
    {                                                                \
        int off = (idz * ih_str + idy) * iw_str + idx + i_off;       \
        val = vload4(off, buf);                                      \
    }
#endif
#define LOAD_VAL_IMG(idx, idy, idz, img, val)                     \
    {                                                             \
        val = READ_IMAGE(img, sampler, (int4)(idx, idy, idz, 0)); \
    }

__kernel void MANGLE_NAME(eltwise_, IOM, AM, EM, FM, N)(const int w,
    const int h,
    const int c,
    const int ow_str,
    const int oh_str,
    const int o_off,
    const int bx,
    const int by,
    const int iw0_str,
    const int ih0_str,
    const int i0_off,
    READ_ONLY_KERNEL_MEM in0,
#if (N > 1)
    const int iw1_str,
    const int ih1_str,
    const int i1_off,
    READ_ONLY_KERNEL_MEM_1 in1,
#endif
#if (N > 2)
    const int iw2_str,
    const int ih2_str,
    const int i2_off,
    READ_ONLY_KERNEL_MEM_2 in2,
#endif
#if (N > 3)
    const int iw3_str,
    const int ih3_str,
    const int i3_off,
    READ_ONLY_KERNEL_MEM_3 in3,
#endif
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    char ew = 0;
#if defined(USE_NCHW)
    ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
#endif

    T4 val;
    T4 res;
#if defined(USE_INPUT_IMG)
    LOAD_VAL_IMG(idx, idy, idz, in0, res);
#else
    LOAD_VAL(ew, idx, idy, idz, iw0_str, ih0_str, i0_off, in0, res);
#endif
#if (N > 1)
#if defined(USE_INPUT_IMG1)
    LOAD_VAL_IMG(idx, idy, idz, in1, val);
#else
    LOAD_VAL(ew, idx, idy, idz, iw1_str, ih1_str, i1_off, in1, val);
#endif
    ELTWISE_V4(val, res);
#endif
#if (N > 2)
#if defined(USE_INPUT_IMG2)
    LOAD_VAL_IMG(idx, idy, idz, in2, val);
#else
    LOAD_VAL(ew, idx, idy, idz, iw2_str, ih2_str, i2_off, in2, val);
#endif
    ELTWISE_V4(val, res);
#endif
#if (N > 3)
#if defined(USE_INPUT_IMG3)
    LOAD_VAL_IMG(idx, idy, idz, in3, val);
#else
    LOAD_VAL(ew, idx, idy, idz, iw3_str, ih3_str, i3_off, in3, val);
#endif
    ELTWISE_V4(val, res);
#endif
    ACTIVATION_V4(res);

#if defined(USE_NCHW)
    STORE_MEM_V4_C1_COMMON(res, idx, idy, idz, ow_str, oh_str, o_off, w, out);
#else
    STORE_MEM_V4_COMMON(res, idx, idy, idz, ow_str, oh_str, o_off, out);
#endif
}
