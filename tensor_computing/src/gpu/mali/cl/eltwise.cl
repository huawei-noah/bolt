// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.






#define MANGLE_NAME_IMPL(base, TP, N) base ## TP ## N
#define MANGLE_NAME(base, TP, N) MANGLE_NAME_IMPL(base, TP, N)

#if defined(USE_SUM)
#define calCore(in, off, v, res) {\
    v  = vload8(0, in + (off << 2));\
    res.s0 += v.s0;\
    res.s1 += v.s1;\
    res.s2 += v.s2;\
    res.s3 += v.s3;\
    res.s4 += v.s4;\
    res.s5 += v.s5;\
    res.s6 += v.s6;\
    res.s7 += v.s7;\
}
#endif

#if defined(USE_MAX)
#define calCore(in, off, v, res) {\
    v  = vload8(0, in + (off << 2));\
    res = fmax(res, v);\
}
#endif

#if defined(USE_PROD)
#define calCore(in, off, v, res) {\
    v  = vload8(0, in + (off << 2));\
    res.s0 *= v.s0;\
    res.s1 *= v.s1;\
    res.s2 *= v.s2;\
    res.s3 *= v.s3;\
    res.s4 *= v.s4;\
    res.s5 *= v.s5;\
    res.s6 *= v.s6;\
    res.s7 *= v.s7;\
}
#endif

__kernel void MANGLE_NAME(eltwise_, TP, N)(const int h, const int w, const int c, const int ih_str, const int iw_str, const int ih_off, const int iw_off, 
    const int oh_str, const int ow_str, const int oh_off, const int ow_off, 
    __global const T* in0,
#if (N > 1)
    __global const T* in1,
#endif     
#if (N > 2)
    __global const T* in2,
#endif     
#if (N > 3)
    __global const T* in3,
#endif
#if (N > 4)
    __global const T* in4,
#endif     
#if (N > 5 )
    __global const T* in5,
#endif     
#if (N > 6)
    __global const T* in6,
#endif     
#if (N > 7)
    __global const T* in7,
#endif
    __global T* out){    
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    T8 val;
    T8 res;
    const int in_off = (idz * iw_str + idy + iw_off) * ih_str + (idx << 1) + ih_off;
    res = vload8(0, in0 + (in_off << 2));
#if(N > 1)
    calCore(in1, in_off, val, res);
#endif    
#if(N > 2)
    calCore(in2, in_off, val, res);
#endif    
#if(N > 3)
    calCore(in3, in_off, val, res);
#endif    
#if(N > 4)
    calCore(in4, in_off, val, res);
#endif    
#if(N > 5)
    calCore(in5, in_off, val, res);
#endif    
#if(N > 6)
    calCore(in6, in_off, val, res);
#endif    
#if(N > 7)
    calCore(in7, in_off, val, res);
#endif    

    const int out_off = (idz * ow_str + idy + ow_off) * oh_str + (idx << 1) + oh_off;
    if((idx << 1) + 1 < h){
        vstore8(res, 0, out + (out_off << 2));
    } else {
        vstore4(res.s0123, 0, out + (out_off << 2));
    }    
}

