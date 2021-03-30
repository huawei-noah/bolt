// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, FM, DT) base##FM##DT
#define MANGLE_NAME(base, FM, DT) MANGLE_NAME_IMPL(base, FM, DT)

#define FM
#define DT
#if defined(USE_I32)
#define DT _i32
#endif
#if defined(USE_NCHW)
#define FM _nchw
#endif

__kernel void MANGLE_NAME(power, FM, DT)(const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
    const int w,
    const int bx,
    const int by,
    const int has_power,
    const float alp,
    const float bet,
    float power,
    __global T *input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
#if defined(USE_NCHW)
    char ew = (((idx << 2) + 4) <= w) ? 4 : (w & 3);
    int in_off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off;
    int out_off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off;
#else
    char ew = 4;
    int in_off = ((idz * iw_str + idy + iw_off) * ih_str + idx + ih_off) << 2;
    int out_off = ((idz * ow_str + idy + ow_off) * oh_str + idx + oh_off) << 2;
#endif
    T4 val = 0;
    if (ew == 4) {
        val = vload4(0, input + in_off);
    } else {
        if (ew == 1) {
            val.x = input[in_off];
        }
        if (ew == 2) {
            val.xy = vload2(0, input + in_off);
        }
        if (ew == 3) {
            val.xyz = vload3(0, input + in_off);
        }
    }
    val.x = (T)(((float)val.x) * alp + bet);
    val.y = (T)(((float)val.y) * alp + bet);
    val.z = (T)(((float)val.z) * alp + bet);
    val.w = (T)(((float)val.w) * alp + bet);
    if (has_power) {
        val.x = pow((float)val.x, power);
        val.y = pow((float)val.y, power);
        val.z = pow((float)val.z, power);
        val.w = pow((float)val.w, power);
    }

    if (ew == 4) {
        vstore4(val, 0, output + out_off);
    } else {
        if (ew == 1) {
            output[out_off] = (T)val.x;
        }
        if (ew == 2) {
            vstore2(val.xy, 0, output + out_off);
        }
        if (ew == 3) {
            vstore3(val.xyz, 0, output + out_off);
        }
    }
}
