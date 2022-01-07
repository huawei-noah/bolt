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
#define MANGLE_NAME_IMPL(base, IOM, AXIS, N) base##IOM##AXIS##N
#define MANGLE_NAME(base, IOM, AXIS, N) MANGLE_NAME_IMPL(base, IOM, AXIS, N)

#if defined(AXIS_W)
#define AXIS w_
#elif defined(AXIS_H)
#define AXIS h_
#elif defined(AXIS_C)
#define AXIS c_
#elif defined(NON_ALIGN_AXIS_W)
#define AXIS non_align_w_
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

#define LOAD_VAL(idx, idy, idz, h_str, w_str, wh_off, val, buf)      \
    {                                                                \
        int off = (idz * h_str + idy) * w_str + (idx << 2) + wh_off; \
        val = vload4(0, buf + off);                                  \
    }

#define LOAD_VAL_IMG(idx, idy, idz, val, img)                     \
    {                                                             \
        val = READ_IMAGE(img, sampler, (int4)(idx, idy, idz, 0)); \
    }

__kernel void MANGLE_NAME(concat_nchw_, IOM, AXIS, N)(const int ow_str,
    const int ohw_str,
    const int o_off,
    const int axis_max,
    const int nmax,
    const int out_size,
    const int oc,
    const int bx,
    const int by,
    const int iw_str0,
    const int ih_str0,
    const int i_off0,
    const int iw0,
    const int axis_len_0,
    READ_ONLY_KERNEL_MEM in0,
#if (N > 1)
    const int iw_str1,
    const int ih_str1,
    const int i_off1,
    const int iw1,
    const int axis_len_1,
    READ_ONLY_KERNEL_MEM_1 in1,
#endif
#if (N > 2)
    const int iw_str2,
    const int ih_str2,
    const int i_off2,
    const int iw2,
    const int axis_len_2,
    READ_ONLY_KERNEL_MEM_2 in2,
#endif
#if (N > 3)
    const int iw_str3,
    const int ih_str3,
    const int i_off3,
    const int iw3,
    const int axis_len_3,
    READ_ONLY_KERNEL_MEM_3 in3,
#endif
    KERNEL_MEM out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
#if defined(AXIS_W) || defined(NON_ALIGN_AXIS_W)
    int id_axis = idx - axis_max;
#elif defined(AXIS_H)
    int id_axis = idy - axis_max;
#elif defined(AXIS_C)
    int pitch = idz / axis_max;
    idz = idz % axis_max;
    int id_axis = idz - axis_max;
#endif
    int idn = nmax;
#if (N > 3)
    if (id_axis < 0) {
        id_axis += axis_len_3;
        idn = 3;
    }
#endif
#if (N > 2)
    if (id_axis < 0) {
        id_axis += axis_len_2;
        idn = 2;
    }
#endif
#if (N > 1)
    if (id_axis < 0) {
        id_axis += axis_len_1;
        idn = 1;
    }
#endif
    if (id_axis < 0) {
        id_axis += axis_len_0;
        idn = 0;
    }

    T4 val;
    int in_idx = idx;
    int in_idy = idy;
    int in_idz = idz;

#if defined(AXIS_W) || defined(NON_ALIGN_AXIS_W)
    in_idx = id_axis;
#elif defined(AXIS_H)
    in_idy = id_axis;
#elif defined(AXIS_C)
    in_idz = id_axis;
    if (idn == 0) {
        in_idz += pitch * axis_len_0;
#if (N > 1)
    } else if (idn == 1) {
        in_idz += pitch * axis_len_1;
#endif
#if (N > 2)
    } else if (idn == 2) {
        in_idz += pitch * axis_len_2;
#endif
#if (N > 3)
    } else if (idn == 3) {
        in_idz += pitch * axis_len_3;
#endif
    }
#endif

#if defined(NON_ALIGN_AXIS_W)
    char ew = 4;
    int out_off = idz * ohw_str + idy * ow_str + (id_axis << 2) + o_off;
#endif

    if (idn == 0) {
#if defined(USE_INPUT_IMG)
        LOAD_VAL_IMG(in_idx, in_idy, in_idz, val, in0);
#else
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str0, iw_str0, i_off0, val, in0);
#endif
#if defined(NON_ALIGN_AXIS_W)
        if (id_axis * 4 + 4 > iw0) {
            ew = iw0 & 3;
        }
#endif
    }
#if (N > 1)
    if (idn == 1) {
#if defined(USE_INPUT_IMG1)
        LOAD_VAL_IMG(in_idx, in_idy, in_idz, val, in1);
#else
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str1, iw_str1, i_off1, val, in1);
#endif
#if defined(NON_ALIGN_AXIS_W)
        out_off += iw0;
        if (id_axis * 4 + 4 > iw1) {
            ew = iw1 & 3;
        }
#endif
    }
#endif
#if (N > 2)
    if (idn == 2) {
#if defined(USE_INPUT_IMG2)
        LOAD_VAL_IMG(in_idx, in_idy, in_idz, val, in2);
#else
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str2, iw_str2, i_off2, val, in2);
#endif
#if defined(NON_ALIGN_AXIS_W)
        out_off += (iw0 + iw1);
        if (id_axis * 4 + 4 > iw2) {
            ew = iw2 & 3;
        }
#endif
    }
#endif
#if (N > 3)
    if (idn == 3) {
#if defined(USE_INPUT_IMG3)
        LOAD_VAL_IMG(in_idx, in_idy, in_idz, val, in3);
#else
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str3, iw_str3, i_off3, val, in3);
#endif
#if defined(NON_ALIGN_AXIS_W)
        out_off += (iw0 + iw1 + iw2);
        if (id_axis * 4 + 4 > iw3) {
            ew = iw3 & 3;
        }
#endif
    }
#endif

#if defined(NON_ALIGN_AXIS_W)
    STORE_MEM_V4_C1(val, out_size + out_off, ew, out);
#else
#if defined(USE_OUTPUT_IMG)
    int4 out_off = (int4)(idx, idy, idz, 0);
#if defined(AXIS_W)
    out_off.x += out_size;
#elif defined(AXIS_H)
    out_off.y += out_size;
#elif defined(AXIS_C)
    out_off.z += (out_size + pitch * oc);
#endif
    WRITE_IMAGE(out, out_off, val);
#else
    int out_off = idz * ohw_str + idy * ow_str + (idx << 2) + o_off;
#if defined(AXIS_C)
    out_off += pitch * oc * ohw_str;
#endif
    STORE_MEM_V4_C1(val, out_size + out_off, 4, out);
#endif
#endif
}
