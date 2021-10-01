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
__kernel void gather_(const int axisBeforeLen,
    const int inAxisLen,
    const int outAxisLen,
    const int index_w_str,
    const int index_h_str,
    const int index_off,
    const int index_w,
    const int index_h,
    const int bx,
    const int by,
    __global const T *in,
    __global const int *index,
    __global T *out)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    int index_x = idy % index_w;
    int index_y = idy / index_w;
    int index_z = idy / (index_w * index_h);
    int index_id_off = (index_z * index_h_str + index_y) * index_w_str + index_x + index_off;
    int index_val = index[index_id_off];
    T val;
    int in_off = (idz * inAxisLen + index_val) * axisBeforeLen + idx;
    int out_off = (idz * outAxisLen + idy) * axisBeforeLen + idx;
    val = in[in_off];
    out[out_off] = val;
}
