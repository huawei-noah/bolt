// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





#define MANGLE_NAME_IMPL(base, F, W) base ## F ## W
#define MANGLE_NAME(base, F, W) MANGLE_NAME_IMPL(base, F, W)

#if defined(USE_HALF)
#define READ_IMAGE(image, sampler, coord)    read_imageh(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data)      write_imageh(image, coord, data)
#else
#define READ_IMAGE(image, sampler, coord)    read_imagef(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data)      write_imagef(image, coord, data)
#endif
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define vec4Mul(a, b, c){\
    c.x += a.x * b.x;\
    c.y += a.y * b.y;\
    c.z += a.z * b.z;\
    c.w += a.w * b.w;\
}

#if (W == 1)
#define dirCalCore3(a0, b, c){\
    vec4Mul(a0, b, c[0]);\
}
#endif

#if (W == 2)
#define dirCalCore3(a0, a1, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
}
#endif

#if (W == 3)
#define dirCalCore3(a0, a1, a2, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
    vec4Mul(a2, b, c[2]);\
}
#endif

#if (W == 4)
#define dirCalCore4(a0, a1, a2, a3, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
    vec4Mul(a2, b, c[2]);\
    vec4Mul(a3, b, c[3]);\
}
#endif

#if (W == 5)
#define dirCalCore5(a0, a1, a2, a3, a4, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
    vec4Mul(a2, b, c[2]);\
    vec4Mul(a3, b, c[3]);\
    vec4Mul(a4, b, c[4]);\
}
#endif

#if (W == 6)
#define dirCalCore6(a0, a1, a2, a3, a4, a5, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
    vec4Mul(a2, b, c[2]);\
    vec4Mul(a3, b, c[3]);\
    vec4Mul(a4, b, c[4]);\
    vec4Mul(a5, b, c[5]);\
}
#endif

#if (W == 7)
#define dirCalCore7(a0, a1, a2, a3, a4, a5, a6, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
    vec4Mul(a2, b, c[2]);\
    vec4Mul(a3, b, c[3]);\
    vec4Mul(a4, b, c[4]);\
    vec4Mul(a5, b, c[5]);\
    vec4Mul(a6, b, c[6]);\
}
#endif

#if (W == 8)
#define dirCalCore8(a0, a1, a2, a3, a4, a5, a6, a7, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
    vec4Mul(a2, b, c[2]);\
    vec4Mul(a3, b, c[3]);\
    vec4Mul(a4, b, c[4]);\
    vec4Mul(a5, b, c[5]);\
    vec4Mul(a6, b, c[6]);\
    vec4Mul(a7, b, c[7]);\
}
#endif

#if (W == 9)
#define dirCalCore9(a0, a1, a2, a3, a4, a5, a6, a7, a8, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
    vec4Mul(a2, b, c[2]);\
    vec4Mul(a3, b, c[3]);\
    vec4Mul(a4, b, c[4]);\
    vec4Mul(a5, b, c[5]);\
    vec4Mul(a6, b, c[6]);\
    vec4Mul(a7, b, c[7]);\
    vec4Mul(a8, b, c[8]);\
}
#endif

#if (W == 10)
#define dirCalCore10(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, b, c){\
    vec4Mul(a0, b, c[0]);\
    vec4Mul(a1, b, c[1]);\
    vec4Mul(a2, b, c[2]);\
    vec4Mul(a3, b, c[3]);\
    vec4Mul(a4, b, c[4]);\
    vec4Mul(a5, b, c[5]);\
    vec4Mul(a6, b, c[6]);\
    vec4Mul(a7, b, c[7]);\
    vec4Mul(a8, b, c[8]);\
    vec4Mul(a9, b, c[9]);\
}
#endif

#if (W == 1)
#define calCore0 dirCalCore3(in_val[0], flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore3(in_val[1], flt_val[1], out_val);
#define calCore2 dirCalCore3(in_val[2], flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore3(in_val[3], flt_val[3], out_val);
#define calCore4 dirCalCore3(in_val[4], flt_val[4], out_val);
#endif
#endif

#if (W == 2)
#define calCore0 dirCalCore3(in_val[0], in_val[1], flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore3(in_val[1], in_val[2], flt_val[1], out_val);
#define calCore2 dirCalCore3(in_val[2], in_val[3], flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore3(in_val[3], in_val[4], flt_val[3], out_val);
#define calCore4 dirCalCore3(in_val[4], in_val[5], flt_val[4], out_val);
#endif
#endif

#if (W == 3)
#define calCore0 dirCalCore3(in_val[0], in_val[1], in_val[2], flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore3(in_val[1], in_val[2], in_val[3], flt_val[1], out_val);
#define calCore2 dirCalCore3(in_val[2], in_val[3], in_val[4], flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore3(in_val[3], in_val[4], in_val[5], flt_val[3], out_val);
#define calCore4 dirCalCore3(in_val[4], in_val[5], in_val[6], flt_val[4], out_val);
#endif
#endif

#if (W == 4)
#define calCore0 dirCalCore4(in_val[0], in_val[1], in_val[2], in_val[3], flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore4(in_val[1], in_val[2], in_val[3], in_val[4], flt_val[1], out_val);
#define calCore2 dirCalCore4(in_val[2], in_val[3], in_val[4], in_val[5], flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore4(in_val[3], in_val[4], in_val[5], in_val[6], flt_val[3], out_val);
#define calCore4 dirCalCore4(in_val[4], in_val[5], in_val[6], in_val[7], flt_val[4], out_val);
#endif
#endif

#if (W == 5)
#define calCore0 dirCalCore5(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore5(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], flt_val[1], out_val);
#define calCore2 dirCalCore5(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore5(in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], flt_val[3], out_val);
#define calCore4 dirCalCore5(in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], flt_val[4], out_val);
#endif
#endif

#if (W == 6)
#define calCore0 dirCalCore6(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore6(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], flt_val[1], out_val);
#define calCore2 dirCalCore6(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore6(in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], flt_val[3], out_val);
#define calCore4 dirCalCore6(in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9], flt_val[4], out_val);
#endif
#endif

#if (W == 7)
#define calCore0 dirCalCore7(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6],  flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore7(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7],  flt_val[1], out_val);
#define calCore2 dirCalCore7(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8],  flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore7(in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9],  flt_val[3], out_val);
#define calCore4 dirCalCore7(in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9], in_val[10], flt_val[4], out_val);
#endif
#endif

#if (W == 8)
#define calCore0 dirCalCore8(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7],    flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore8(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8],    flt_val[1], out_val);
#define calCore2 dirCalCore8(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9],    flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore8(in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8],  in_val[9],  in_val[10], flt_val[3], out_val);
#define calCore4 dirCalCore8(in_val[4], in_val[5], in_val[6], in_val[7], in_val[9], in_val[10], in_val[11], in_val[12], flt_val[4], out_val);
#endif
#endif

#if (W == 9)
#define calCore0 dirCalCore9(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8],    flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore9(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9],    flt_val[1], out_val);
#define calCore2 dirCalCore9(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9], in_val[10],   flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore9(in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9],  in_val[10], in_val[11], flt_val[3], out_val);
#define calCore4 dirCalCore9(in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9], in_val[10], in_val[11], in_val[12], flt_val[4], out_val);
#endif
#endif

#if (W == 10)
#define calCore0 dirCalCore10(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8],  in_val[9],    flt_val[0], out_val);
#if (F > 1)
#define calCore1 dirCalCore10(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9],  in_val[10],   flt_val[1], out_val);
#define calCore2 dirCalCore10(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9], in_val[10], in_val[11],   flt_val[2], out_val);
#endif
#if (F > 3)
#define calCore3 dirCalCore10(in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9],  in_val[10], in_val[11], in_val[12], flt_val[3], out_val);
#define calCore4 dirCalCore10(in_val[4], in_val[5], in_val[6], in_val[7], in_val[8], in_val[9], in_val[10], in_val[11], in_val[12], in_val[13], flt_val[4], out_val);
#endif
#endif

#if(N == 1)
#define loadInval(val, off, str, in){\
    val[0] = vload4(off, in);\
}
#endif
#if(N == 2)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off + str, in);\
}
#endif
#if(N == 3)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
}
#endif
#if(N == 4)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
}
#endif
#if(N == 5)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
}
#endif
#if(N == 6)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
    val[5] = vload4(off +  str * 5,   in);\
}
#endif
#if(N == 7)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
    val[5] = vload4(off +  str * 5,   in);\
    val[6] = vload4(off +  str * 6,   in);\
}
#endif
#if(N == 8)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
    val[5] = vload4(off +  str * 5,   in);\
    val[6] = vload4(off +  str * 6,   in);\
    val[7] = vload4(off +  str * 7,   in);\
}
#endif
#if(N == 9)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
    val[5] = vload4(off +  str * 5,   in);\
    val[6] = vload4(off +  str * 6,   in);\
    val[7] = vload4(off +  str * 7,   in);\
    val[8] = vload4(off + (str << 3), in);\
}
#endif
#if(N == 10)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
    val[5] = vload4(off +  str * 5,   in);\
    val[6] = vload4(off +  str * 6,   in);\
    val[7] = vload4(off +  str * 7,   in);\
    val[8] = vload4(off + (str << 3), in);\
    val[9] = vload4(off +  str * 9,   in);\
}
#endif

#if (F == 3)
#define calCore(flt, off, val){\
    val[0] = vload4(off,     flt);\
    val[1] = vload4(off + 1, flt);\
    val[2] = vload4(off + 2, flt);\
    calCore0;\
    calCore1;\
    calCore2;\
    off += 3;\
}
#endif

#if (F == 5)
#define calCore(flt, off, val){\
    val[0] = vload4(off,     flt);\
    val[1] = vload4(off + 1, flt);\
    val[2] = vload4(off + 2, flt);\
    val[3] = vload4(off + 3, flt);\
    val[4] = vload4(off + 4, flt);\
    calCore0;\
    calCore1;\
    calCore2;\
    calCore3;\
    calCore4;\
    off += 5;\
}
#endif

#if defined(USE_RELU)
#define reluVec4(val){\
    if(val.x < 0) val.x = (T)0;\
    if(val.y < 0) val.y = (T)0;\
    if(val.z < 0) val.z = (T)0;\
    if(val.w < 0) val.w = (T)0;\
}
#endif


#if defined(USE_RELU)
__kernel void MANGLE_NAME(conv_depthwise_s1_relu_, F, W)
#else
__kernel void MANGLE_NAME(conv_depthwise_s1_, F, W)
#endif
(const int ih_str, const int ihw_str, const int ic_str, const int ih_off, const int iw_off, const int oh_str, const int ow_str, const int oh_off, const int ow_off, const int ow, 
     __global const T* in, __global const T* flt, __read_only image1d_t bias, __global T* out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    T4  in_val[N];
    T4  flt_val[F];
    T4  out_val[W];        
    out_val[0] = READ_IMAGE(bias, sampler, idz);
#if(W > 1)
    out_val[1] = out_val[0];
#endif
#if(W > 2)
    out_val[2] = out_val[0];
#endif
#if(W > 3)
    out_val[3] = out_val[0];
#endif
#if(W > 4)
    out_val[4] = out_val[0];
#endif
#if(W > 5)
    out_val[5] = out_val[0];
#endif
#if(W > 6)
    out_val[6] = out_val[0];
#endif
#if(W > 7)
    out_val[7] = out_val[0];
#endif
    int in_off  = idz * ihw_str + (idy * W  + iw_off) * ih_str + idx + ih_off;
    int flt_off = idz * Fsq;

    loadInval(in_val, in_off, ih_str, in);
    calCore(flt, flt_off, flt_val);
    loadInval(in_val, in_off + 1, ih_str, in);
    calCore(flt, flt_off, flt_val);
    loadInval(in_val, in_off + 2, ih_str, in);
    calCore(flt, flt_off, flt_val);
#if(F > 3)
    loadInval(in_val, in_off + 3, ih_str, in);
    calCore(flt, flt_off, flt_val);
    loadInval(in_val, in_off + 4, ih_str, in);
    calCore(flt, flt_off, flt_val);
#endif 

    int out_off = (idz * ow_str + idy * W + ow_off) * oh_str + idx + oh_off;
    
#if defined (USE_RELU)
    reluVec4(out_val[0]);
#endif
    vstore4(out_val[0], out_off, out);

#if(W > 1)
#if defined (USE_RELU)
    reluVec4(out_val[1]);
#endif
    if(idy * W + 1 < ow) vstore4(out_val[1], out_off + oh_str,        out);
#endif

#if(W > 2)
#if defined (USE_RELU)
    reluVec4(out_val[2]);
#endif
    if(idy * W + 2 < ow) vstore4(out_val[2], out_off + (oh_str << 1), out);
#endif

#if(W > 3)
#if defined (USE_RELU)
    reluVec4(out_val[3]);
#endif
    if(idy * W + 3 < ow) vstore4(out_val[3], out_off + oh_str * 3,    out);
#endif

#if(W > 4)
#if defined (USE_RELU)
    reluVec4(out_val[4]);
#endif
    if(idy * W + 4 < ow) vstore4(out_val[4], out_off + (oh_str << 2), out);
#endif

#if(W > 5)
#if defined (USE_RELU)
    reluVec4(out_val[5]);
#endif
    if(idy * W + 5 < ow) vstore4(out_val[5], out_off + oh_str * 5,    out);
#endif

#if(W > 6)
#if defined (USE_RELU)
    reluVec4(out_val[6]);
#endif
    if(idy * W + 6 < ow) vstore4(out_val[6], out_off + oh_str * 6,    out);
#endif

#if(W > 7)
#if defined (USE_RELU)
    reluVec4(out_val[7]);
#endif
    if(idy * W + 7 < ow) vstore4(out_val[7], out_off + oh_str * 7,    out);
#endif    
}
