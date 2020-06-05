// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NOCINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTIOC OF COCTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN COCNECTIOC WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





#include "kernel_def.h"
#define MANGLE_NAME_IMPL(base, OC) base ## OC
#define MANGLE_NAME(base, OC) MANGLE_NAME_IMPL(base, OC)

#if(OC == 1)
    #define calCore(ov, i_off, f_off, in, flt) {\
        T iv = in[i_off];\
        T fv = flt[f_off];\
        ov += iv * fv;\
    }
#endif

#if(OC == 2)
    #define calCore(ov, i_off, f_off, in, flt) {\
        T2 iv = vload2(i_off, in);\
        T2 fv = vload2(f_off, flt);\
        ov += iv.x * fv.x + iv.y * fv.y;\
    }
#endif

#if(OC == 3)
    #define calCore(ov, i_off, f_off, in, flt) {\
        T3 iv = vload3(i_off, in);\
        T3 fv = vload3(f_off, flt);\
        ov += iv.x * fv.x + iv.y * fv.y + iv.z * fv.z;\
    }
#endif

#if(OC == 4)
    #define calCore(ov, i_off, f_off, in, flt) {\
        T4 iv = vload4(i_off, in);\
        T4 fv = vload4(f_off, flt);\
        DOT_A4B4C1(iv, fv, ov);\
    }
#endif

#if(OC == 8)
    #define calCore(ov, i_off, f_off, in, flt) {\
        T8 iv = vload8(i_off, in);\
        T8 fv = vload8(f_off, flt);\
        DOT_A4B4C1(iv.s0123, fv.s0123, ov);\
        DOT_A4B4C1(iv.s4567, fv.s4567, ov);\
    }
#endif

#if(OC == 16)
    #define calCore(ov, i_off, f_off, in, flt) {\
        T16 iv = vload16(i_off, in);\
        T16 fv = vload16(f_off, flt);\
        DOT_A4B4C1(iv.s0123, fv.s0123, ov);\
        DOT_A4B4C1(iv.s4567, fv.s4567, ov);\
        DOT_A4B4C1(iv.s89ab, fv.s89ab, ov);\
        DOT_A4B4C1(iv.scdef, fv.scdef, ov);\
    }
#endif

#if defined(USE_RELU)
__kernel MANGLE_NAME(void conv_direct_spe_fwhs1_relu_, OC)
#else
__kernel MANGLE_NAME(void conv_direct_spe_fwhs1_, OC)
#endif
(const int ih_str, const int ihw_str, const int ic_str, const int ih_off, const int iw_off, const int oh_str, const int ow_str, const int oh_off, const int ow_off, 
    const int flt_str, const int bx, const int by, __global const T* in, __global const T* flt, __global const T* bias, __global T* out)
{

    const int idx = get_global_id(0);
    if(idx >= bx) return;

    T out_val = bias[idx];
    int in_off  =  iw_off * ih_str + ih_off;
    int flt_off = idx;
    for(int i = 0; i < ic_str; ++i) {
        calCore(out_val, in_off, flt_off, in, flt);
        in_off  += ihw_str;
        flt_off += flt_str;
    }

    ACTIVATION_V1(out_val);
    const int ox = idx >> 2;
    const int oy = idx & 3;
    int out_off = (ox * ow_str + ow_off) * oh_str + oh_off;
    out[out_off * 4 + oy] = out_val;
}
