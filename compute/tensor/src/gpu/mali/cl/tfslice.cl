// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, FM) base##FM
#define MANGLE_NAME(base, FM) MANGLE_NAME_IMPL(base, FM)

#if defined(USE_NCHW)
    #define FM nchw
#else
    #define FM
#endif

__kernel void MANGLE_NAME(tfslice_, FM)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int ic,
    const int oc,
    const int w_be,
    const int h_be,
    const int c_be,
    const int n_be,
    const int sw,
    const int sh,
    const int sc,
    const int sn,
    const int bx,
    const int by,
    __global T *input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    int idc = idz % oc;
    int idn = idz / oc;
    if (idx >= bx || idy >= by) {
        return;
    }

    int idn_off = idn * sn + n_be;
    int idc_off = idc * sc + c_be;
    int idh_off = idy * sh + h_be;
    int idw_off = idx * sw + w_be + i_off;
    int in_off = ((idn_off * ic + idc_off) * ih_str + idh_off) * iw_str + idw_off;
    int out_off = (idz * oh_str + idy) * ow_str + idx + o_off;

#if defined(USE_NCHW)
    output[out_off] = input[in_off];
#else
    vstore4(vload4(in_off, input), out_off, output);
#endif
}
