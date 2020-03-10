// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.






#define MANGLE_NAME_IMPL(base, A, N) base ## A ## N
#define MANGLE_NAME(base, A, N) MANGLE_NAME_IMPL(base, A, N)

__kernel void MANGLE_NAME(concat_, A, N)(const int ih_str, const int iw_str, const int ih_off, const int iw_off, 
    const int oh_str, const int ow_str, const int oh_off, const int ow_off, const int cmax, const int nmax, const int out_size, 
    __global const T* in0,
#if (N > 1)
    const int c0,
    __global const T* in1,
#endif     
#if (N > 2)
    const int c1,
    __global const T* in2,
#endif     
#if (N > 3)
    const int c2,
    __global const T* in3,
#endif
#if (N > 4)
    const int c3,
    __global const T* in4,
#endif     
#if (N > 5 )
    const int c4,
    __global const T* in5,
#endif     
#if (N > 6)
    const int c5,
    __global const T* in6,
#endif     
#if (N > 7)
    const int c6,
    __global const T* in7,
#endif
    __global T* out) {    
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    int idc = idz - cmax;
    int idn = nmax;
    int out_c = cmax;
#if (N > 7)    
    if(idc < 0) {idc += c6; idn = 6; out_c -= c6;}
#endif
#if (N > 6)    
    if(idc < 0) {idc += c5; idn = 5; out_c -= c5;}
#endif
#if (N > 5)    
    if(idc < 0) {idc += c4; idn = 4; out_c -= c4;}
#endif
#if (N > 4)    
    if(idc < 0) {idc += c3; idn = 3; out_c -= c3;}
#endif
#if (N > 3)    
    if(idc < 0) {idc += c2; idn = 2; out_c -= c2;}
#endif
#if (N > 2)    
    if(idc < 0) {idc += c1; idn = 1; out_c -= c1;}
#endif
#if (N > 1)    
    if(idc < 0) {idc += c0; idn = 0; out_c -= c0;}
#endif
    T4 val;
    int in_off = (idc * iw_str + idy + iw_off) * ih_str + idx + ih_off;
    if(idn == 0) val = vload4(in_off, in0);
#if (N > 1)    
    if(idn == 1) val = vload4(in_off, in1);
#endif    
#if (N > 2)    
    if(idn == 2) val = vload4(in_off, in2);
#endif    
#if (N > 3)    
    if(idn == 3) val = vload4(in_off, in3);
#endif    
#if (N > 4)    
    if(idn == 4) val = vload4(in_off, in4);
#endif    
#if (N > 5)    
    if(idn == 5) val = vload4(in_off, in5);
#endif    
#if (N > 6)    
    if(idn == 6) val = vload4(in_off, in6);
#endif    
#if (N > 7)    
    if(idn == 7) val = vload4(in_off, in7);
#endif    
    int out_off = ((out_c + idc) * ow_str + idy + ow_off) * oh_str + idx + oh_off;
    vstore4(val, out_off, out + out_size);
}

