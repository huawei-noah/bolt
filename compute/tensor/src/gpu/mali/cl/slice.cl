// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, FM, AXIS_NUM, ON) base##FM##AXIS_NUM##ON
#define MANGLE_NAME(base, FM, AXIS_NUM, ON) MANGLE_NAME_IMPL(base, FM, AXIS_NUM, ON)

#define FM
#if defined(USE_NCHW)
#define FM nchw_
#define LOAD_VAL(off, ex, val, buf)             \
    {                                           \
        if (ex == 4) {                          \
            val = vload4(0, buf + off);         \
        } else {                                \
            if (ex == 3) {                      \
                val.xyz = vload3(0, buf + off); \
            } else if (ex == 2) {               \
                val.xy = vload2(0, buf + off);  \
            } else {                            \
                val.x = buf[off];               \
            }                                   \
        }                                       \
    }

#define STORE_VAL(off, ex, val, buf)            \
    {                                           \
        if (ex == 4) {                          \
            vstore4(val, 0, buf + off);         \
        } else {                                \
            if (ex == 3) {                      \
                vstore3(val.xyz, 0, buf + off); \
            } else if (ex == 2) {               \
                vstore2(val.xy, 0, buf + off);  \
            } else {                            \
                buf[off] = val.x;               \
            }                                   \
        }                                       \
    }

#define CALCORE_AXIS_0(ix, iy, iz, ow, ow_str, oh_str, ow_off, oh_off, in_off, in, out) \
    {                                                                                   \
        T4 val = 0;                                                                     \
        char ew = (((ix << 2) + 4) <= ow) ? 4 : (ow & 3);                               \
        int out_off = (iz * oh_str + iy + oh_off) * ow_str + (ix << 2) + ow_off;        \
        LOAD_VAL(in_off, ew, val, in);                                                  \
        STORE_VAL(out_off, ew, val, out);                                               \
    }
#endif

__kernel void MANGLE_NAME(slice_, FM, AXIS_NUM, ON)(const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int axis_max,
    const int nmax,
    const int in_size,
    const int bx,
    const int by,
    __global T *input,
    const int ow_str0,
    const int oh_str0,
    const int ow_off0,
    const int oh_off0,
    const int ow0,
    __global T *output0
#if (ON > 1)
    ,
    const int ow_str1,
    const int oh_str1,
    const int ow_off1,
    const int oh_off1,
    const int ow1,
    const int axis_len_0,
    __global T *output1
#endif
#if (ON > 2)
    ,
    const int ow_str2,
    const int oh_str2,
    const int ow_off2,
    const int oh_off2,
    const int ow2,
    const int axis_len_1,
    __global T *output2
#endif
#if (ON > 3)
    ,
    const int ow_str3,
    const int oh_str3,
    const int ow_off3,
    const int oh_off3,
    const int ow3,
    const int axis_len_2,
    __global T *output3
#endif
)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    // axis_len_xxx is the thread number used for write output xxx on target axis
    // axis_max = axis_len_0 + axis_len_1 + axis_len_2;
#if (AXIS_NUM == 0)
    int id_axis = idx - axis_max;
#elif (AXIS_NUM == 1)
    int id_axis = idy - axis_max;
#elif (AXIS_NUM == 2)
    int id_axis = idz - axis_max;
#endif
    int idn = nmax;
#if (ON > 3)
    if (id_axis < 0) {
        id_axis += axis_len_2;
        idn = 2;
    }
#endif
#if (ON > 2)
    if (id_axis < 0) {
        id_axis += axis_len_1;
        idn = 1;
    }
#endif
#if (ON > 1)
    if (id_axis < 0) {
        id_axis += axis_len_0;
        idn = 0;
    }
#endif
    int on_idx = idx;
    int on_idy = idy;
    int on_idz = idz;

#if (AXIS_NUM == 0)
    on_idx = id_axis;
#elif (AXIS_NUM == 1)
    on_idy = id_axis;
#elif (AXIS_NUM == 2)
    on_idz = id_axis;
#endif

#if defined(USE_NCHW)
#if (AXIS_NUM == 0)
    int in_off = (on_idz * ih_str + on_idy + ih_off) * iw_str + (on_idx << 2) + iw_off + in_size;
    if (idn == 0) {
        CALCORE_AXIS_0(on_idx, on_idy, on_idz, ow0, ow_str0, oh_str0, ow_off0, oh_off0, in_off,
            input, output0);
    }
#if (ON > 1)
    if (idn == 1) {
        in_off += ow0;
        CALCORE_AXIS_0(on_idx, on_idy, on_idz, ow1, ow_str1, oh_str1, ow_off1, oh_off1, in_off,
            input, output1);
    }
#endif
#if (ON > 2)
    if (idn == 2) {
        in_off += ow0 + ow1;
        CALCORE_AXIS_0(on_idx, on_idy, on_idz, ow2, ow_str2, oh_str2, ow_off2, oh_off2, in_off,
            input, output2);
    }
#endif
#if (ON > 3)
    if (idn == 3) {
        in_off += ow0 + ow1 + ow2;
        CALCORE_AXIS_0(on_idx, on_idy, on_idz, ow3, ow_str3, oh_str3, ow_off3, oh_off3, in_off,
            input, output3);
    }
#endif
#else
    T4 val;
    char ew = (((idx << 2) + 4) <= ow0) ? 4 : (ow0 & 3);
    int in_off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off + in_size;
    LOAD_VAL(in_off, ew, val, input);
    if (idn == 0) {
        int out_off = (on_idz * oh_str0 + on_idy + oh_off0) * ow_str0 + (on_idx << 2) + ow_off0;
        STORE_VAL(out_off, ew, val, output0);
    }
#if (ON > 1)
    if (idn == 1) {
        int out_off = (on_idz * oh_str1 + on_idy + oh_off1) * ow_str1 + (on_idx << 2) + ow_off1;
        STORE_VAL(out_off, ew, val, output1);
    }
#endif
#if (ON > 2)
    if (idn == 2) {
        int out_off = (on_idz * oh_str2 + on_idy + oh_off2) * ow_str2 + (on_idx << 2) + ow_off2;
        STORE_VAL(out_off, ew, val, output2);
    }
#endif
#if (ON > 3)
    if (idn == 3) {
        int out_off = (on_idz * oh_str3 + on_idy + oh_off3) * ow_str3 + (on_idx << 2) + ow_off3;
        STORE_VAL(out_off, ew, val, output3);
    }
#endif
#endif
#else
    //use format ncwhc4, except for slice on axis c and slice len not align to 4
    T4 val;
    int in_off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;
    val = vload4(in_off, input + in_size);
    if (idn == 0) {
        int out_off = (on_idz * ow_str0 + on_idy + ow_off0) * oh_str1 + on_idx + oh_off0;
        vstore4(val, out_off, output0);
    }
#if (ON > 1)
    if (idn == 1) {
        int out_off = (on_idz * ow_str1 + on_idy + ow_off1) * oh_str1 + on_idx + oh_off1;
        vstore4(val, out_off, output1);
    }
#endif
#if (ON > 2)
    if (idn == 2) {
        int out_off = (on_idz * ow_str2 + on_idy + ow_off2) * oh_str2 + on_idx + oh_off2;
        vstore4(val, out_off, output2);
    }
#endif
#if (ON > 3)
    if (idn == 3) {
        int out_off = (on_idz * ow_str3 + on_idy + ow_off3) * oh_str3 + on_idx + oh_off3;
        vstore4(val, out_off, output3);
    }
#endif
#endif
}
