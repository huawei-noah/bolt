// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





#define MANGLE_NAME_IMPL(base, C, K) base ## C ## K
#define MANGLE_NAME(base, C, K) MANGLE_NAME_IMPL(base, C, K)
#if(C == 4)
#define loadFltval(off, str, flt, val){\
    val.x = flt[off];\
    val.y = flt[off + str];\
    val.z = flt[off + (str << 1)];\
    val.w = flt[off + str * 3];\
}

#define loadFltvalEdge(off, str, flt, val, edge){\
                 val.x = flt[off];\
    if(edge > 1) val.y = flt[off + str];\
    if(edge > 2) val.z = flt[off + (str << 1)];\
}
#endif

#if(C == 8)
#define loadFltval(off, str, flt, val){\
    val.s0 = flt[off];\
    val.s1 = flt[off + str];\
    val.s2 = flt[off +(str << 1)];\
    val.s3 = flt[off + str * 3];\
    val.s4 = flt[off +(str << 2)];\
    val.s5 = flt[off + str * 5];\
    val.s6 = flt[off + str * 6];\
    val.s7 = flt[off + str * 7];\
}
#define loadFltvalEdge(off, str, flt, val, edge){\
                 val.s0 = flt[off];\
    if(edge > 1) val.s1 = flt[off + str];\
    if(edge > 2) val.s2 = flt[off +(str << 1)];\
    if(edge > 3) val.s3 = flt[off + str * 3];\
    if(edge > 4) val.s4 = flt[off +(str << 2)];\
    if(edge > 5) val.s5 = flt[off + str * 5];\
    if(edge > 6) val.s6 = flt[off + str * 6];\
}
#endif

__kernel void MANGLE_NAME(fc_trans_fltbuf_, C, K)(const int fw, const int fh, const int fwh, const int fc, const int fn, 
    __global const T* fltdata, __global T* fltbuf){
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int bc  = (fc + C - 1) / C;
    const int idc = idz % bc;
    const int idn = idz / bc;
    short ec = ((idc + 1) * C <= fc) ? C : (fc % C);
 
    const int flt_off = ((idn * fc + idc * C) * fh + idy) * fw + idx;
#if (C == 4)
    T4 val = 0;
#elif (C == 8)
    T8 val = 0;
#endif
    if(idn < fn){
        if(ec == C){
            loadFltval(flt_off, fwh, fltdata, val);
        } else {
            loadFltvalEdge(flt_off, fwh, fltdata, val, ec);        
        }
    }
    const int out_off = ((idn / K * bc  + idc) * fh + idx) * fw * K + idy * K + (idn % K);
#if (C == 4)
    vstore4(val, out_off, fltbuf);
#elif (C == 8)
    vstore8(val, out_off, fltbuf);
#endif
}
