// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, AXIS, N) base##AXIS##N
#define MANGLE_NAME(base, AXIS, N) MANGLE_NAME_IMPL(base, AXIS, N)

#if defined(AXIS_W)
#define AXIS w_
#elif defined(AXIS_H)
#define AXIS h_
#elif defined(AXIS_C)
#define AXIS c_
#elif defined(NON_ALIGN_AXIS_C)
#define AXIS non_align_c_
#endif

#define LOAD_VAL(idx, idy, idz, h_str, w_str, h_off, w_off, val, buf) \
    {                                                                 \
        int off = (idz * w_str + idy + w_off) * h_str + idx + h_off;  \
        val = vload4(off, buf);                                       \
    }

__kernel void MANGLE_NAME(concat_, AXIS, N)
        (const int oh_str,
            const int ohw_str,
            const int oh_off,
            const int ow_off,
            const int axis_max,
            const int nmax,
            const int out_size,
            const int bx,
            const int by,
            const int ih_str0,
            const int iw_str0,
            const int ih_off0,
            const int iw_off0,
            const int ic0,
            __global const T *in0,
#if (N > 1)
            const int ih_str1,
            const int iw_str1,
            const int ih_off1,
            const int iw_off1,
            const int ic1,
            const int axis_len_0,
            __global const T *in1,
#endif
#if (N > 2)
            const int ih_str2,
            const int iw_str2,
            const int ih_off2,
            const int iw_off2,
            const int ic2,
            const int axis_len_1,
            __global const T *in2,
#endif
#if (N > 3)
            const int ih_str3,
            const int iw_str3,
            const int ih_off3,
            const int iw_off3,
            const int ic3,
            const int axis_len_2,
            __global const T *in3,
#endif
            __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
#if defined(AXIS_W)
    int id_axis = idy - axis_max;
#elif defined(AXIS_H)
    int id_axis = idx - axis_max;
#elif defined(AXIS_C) || defined(NON_ALIGN_AXIS_C)
    int id_axis = idz - axis_max;
#endif
    int idn = nmax;
#if (N > 3)
    if (id_axis < 0) {
        id_axis += axis_len_2;
        idn = 2;
    }
#endif
#if (N > 2)
    if (id_axis < 0) {
        id_axis += axis_len_1;
        idn = 1;
    }
#endif
#if (N > 1)
    if (id_axis < 0) {
        id_axis += axis_len_0;
        idn = 0;
    }
#endif
    T4 val;
    int in_idx = idx;
    int in_idy = idy;
    int in_idz = idz;

#if defined(AXIS_W)
    in_idy = id_axis;
#elif defined(AXIS_H)
    in_idx = id_axis;
#elif defined(AXIS_C) || defined(NON_ALIGN_AXIS_C)
    in_idz = id_axis;
#endif

#if defined(NON_ALIGN_AXIS_C)
    char ec = 4;
    int out_off = id_axis * ohw_str * 4 + idy * oh_str + idx;
#else
    int out_off = idz * ohw_str + (idy + ow_off) * oh_str + idx + oh_off;
#endif
    if (idn == 0) {
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str0, iw_str0, ih_off0, iw_off0, val, in0);
#if defined(NON_ALIGN_AXIS_C)
        if (id_axis * 4 + 4 > ic0) {
            ec = ic0 & 3;
        }
#endif
    }
#if (N > 1)
    if (idn == 1) {
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str1, iw_str1, ih_off1, iw_off1, val, in1);
#if defined(NON_ALIGN_AXIS_C)
        out_off += ic0 * ohw_str;
        if (id_axis * 4 + 4 > ic1) {
            ec = ic1 & 3;
        }
#endif
    }
#endif
#if (N > 2)
    if (idn == 2) {
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str2, iw_str2, ih_off2, iw_off2, val, in2);
#if defined(NON_ALIGN_AXIS_C)
        out_off += (ic0 + ic1) * ohw_str;
        if (id_axis * 4 + 4 > ic2) {
            ec = ic2 & 3;
        }
#endif
    }
#endif
#if (N > 3)
    if (idn == 3) {
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str3, iw_str3, ih_off3, iw_off3, val, in3);
#if defined(NON_ALIGN_AXIS_C)
        out_off += (ic0 + ic1 + ic2) * ohw_str;
        if (id_axis * 4 + 4 > ic3) {
            ec = ic3 & 3;
        }
#endif
    }
#endif

#if defined(NON_ALIGN_AXIS_C)
    out[out_size + out_off] = val.x;
    if (ec > 1) {
        out[out_size + out_off + ohw_str] = val.y;
    }
    if (ec > 2) {
        out[out_size + out_off + ohw_str * 2] = val.z;
    }
    if (ec > 3) {
        out[out_size + out_off + ohw_str * 3] = val.w;
    }
#else
    vstore4(val, out_off, out + out_size);
#endif
}
