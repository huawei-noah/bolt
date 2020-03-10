// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





#define MANGLE_NAME_IMPL(base, MD) base ## MD
#define MANGLE_NAME(base, MD) MANGLE_NAME_IMPL(base, MD)


__kernel void MANGLE_NAME(scale_, MD)(const int h, const int ih_str, const int iw_str, const int ih_off, const int iw_off, 
    const int oh_str, const int ow_str, const int oh_off, const int ow_off, __global const T* alpha, __global const T* beta, __global T* input, __global T* output) {

    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);

    T4 alp = vload4(idz, alpha);
    T4 bet = 0;
#if defined(USE_BETA)
    bet = vload4(idz, beta);
#endif    
    T8 val;
    int in_off = (idz * iw_str + idy + iw_off) * ih_str + (idx << 1) + ih_off;
    val = vload8(0, input + (in_off << 2));
    val.s0 = val.s0 * alp.x + bet.x;
    val.s1 = val.s1 * alp.y + bet.y;
    val.s2 = val.s2 * alp.z + bet.z;
    val.s3 = val.s3 * alp.w + bet.w;
    val.s4 = val.s4 * alp.x + bet.x;
    val.s5 = val.s5 * alp.y + bet.y;
    val.s6 = val.s6 * alp.z + bet.z;
    val.s7 = val.s7 * alp.w + bet.w;

    int out_off = (idz * ow_str + idy + ow_off) * oh_str + (idx << 1) + oh_off;
    if((idx << 1) + 1 < h){
        vstore8(val, 0, output + (out_off << 2));
    } else {
        vstore4(val.s0123, 0, output + (out_off << 2));
    }
}
