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
#endif

#define LOAD_VAL(idx, idy, idz, h_str, w_str, h_off, w_off, val, buf)       \
    {                                                                       \
        int off = (idz * h_str + idy + h_off) * w_str + (idx << 2) + w_off; \
        val = vload4(0, buf + off);                                         \
    }

__kernel void MANGLE_NAME(concat_nchw_, AXIS, N)
        (const int ow_str,
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
            const int iw0,
            __global const T *in0,
#if (N > 1)
            const int ih_str1,
            const int iw_str1,
            const int ih_off1,
            const int iw_off1,
            const int iw1,
            const int axis_len_0,
            __global const T *in1,
#endif
#if (N > 2)
            const int ih_str2,
            const int iw_str2,
            const int ih_off2,
            const int iw_off2,
            const int iw2,
            const int axis_len_1,
            __global const T *in2,
#endif
#if (N > 3)
            const int ih_str3,
            const int iw_str3,
            const int ih_off3,
            const int iw_off3,
            const int iw3,
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
    int id_axis = idx - axis_max;
#elif defined(AXIS_H)
    int id_axis = idy - axis_max;
#elif defined(AXIS_C)
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
    in_idx = id_axis;
#elif defined(AXIS_H)
    in_idy = id_axis;
#elif defined(AXIS_C)
    in_idz = id_axis;
#endif

#if defined(AXIS_W)
    char ew = 4;
    int out_off = idz * ohw_str + (idy + oh_off) * ow_str + (id_axis << 2);
#else
    char ew = ((idx * 4 + 4) <= iw0) ? 4 : (iw0 & 3);
    int out_off = idz * ohw_str + (idy + oh_off) * ow_str + (idx << 2);
#endif

    if (idn == 0) {
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str0, iw_str0, ih_off0, iw_off0, val, in0);
#if defined(AXIS_W)
        if (id_axis * 4 + 4 > iw0) {
            ew = iw0 & 3;
        }
#endif
    }
#if (N > 1)
    if (idn == 1) {
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str1, iw_str1, ih_off1, iw_off1, val, in1);
#if defined(AXIS_W)
        out_off += iw0;
        if (id_axis * 4 + 4 > iw1) {
            ew = iw1 & 3;
        }
#endif
    }
#endif
#if (N > 2)
    if (idn == 2) {
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str2, iw_str2, ih_off2, iw_off2, val, in2);
#if defined(AXIS_W)
        out_off += (iw0 + iw1);
        if (id_axis * 4 + 4 > iw2) {
            ew = iw2 & 3;
        }
#endif
    }
#endif
#if (N > 3)
    if (idn == 3) {
        LOAD_VAL(in_idx, in_idy, in_idz, ih_str3, iw_str3, ih_off3, iw_off3, val, in3);
#if defined(AXIS_W)
        out_off += (iw0 + iw1 + iw2);
        if (id_axis * 4 + 4 > iw3) {
            ew = iw3 & 3;
        }
#endif
    }
#endif

    if (ew == 4) {
        vstore4(val, 0, out + out_size + out_off);
    } else {
        if (ew == 3) {
            vstore3(val.xyz, 0, out + out_size + out_off);
        } else if (ew == 2) {
            vstore2(val.xy, 0, out + out_size + out_off);
        } else if (ew == 1) {
            out[out_size + out_off] = val.x;
        }
    }
}
