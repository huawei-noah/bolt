// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




#define MANGLE_NAME_IMPL(base, N) base ## N
#define MANGLE_NAME(base, N) MANGLE_NAME_IMPL(base, N)

__kernel void MANGLE_NAME(slice_h_, N)(const int ih_str, const int iw_str, const int ih_off, const int iw_off, const int bx, const int by, __global T* input,
    const int oh_str0, const int ow_str0, const int oh_off0, const int ow_off0, const int slice_end0, __global T* output0,
    const int oh_str1, const int ow_str1, const int oh_off1, const int ow_off1, const int slice_end1, __global T* output1
    ) {
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if(idx >= bx || idy >= by) return;

    T4 val;
    int in_off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;
    val = vload4(in_off, input);
    if(idx < slice_end0) {
        int out_off = (idz * ow_str0 + idy + ow_off0) * oh_str0 + idx + oh_off0;
        vstore4(val, out_off, output0);
        return;
    }
    if(idx < slice_end1) {
        int out_off = (idz * ow_str1 + idy + ow_off1) * oh_str1 + idx + oh_off1;
        vstore4(val, out_off, output1);
        return;
    }
}
