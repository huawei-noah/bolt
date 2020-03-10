// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





#define MANGLE_NAME_IMPL(base, AC, H) base ## AC ## H
#define MANGLE_NAME(base, AC, H) MANGLE_NAME_IMPL(base, AC, H)

#if(H == 1)
#if defined(USE_RELU)
#define calCore(val) {\
    if(val.s0 < 0) val.s0 = 0;\
    if(val.s1 < 0) val.s1 = 0;\
    if(val.s2 < 0) val.s2 = 0;\
    if(val.s3 < 0) val.s3 = 0;\
}
#elif defined(USE_RELU6)
#define calCore(val) {\
    val.s0 = clamp(val.s0, (T)0, (T)6.0);\
    val.s1 = clamp(val.s1, (T)0, (T)6.0);\
    val.s2 = clamp(val.s2, (T)0, (T)6.0);\
    val.s3 = clamp(val.s3, (T)0, (T)6.0);\
}
#elif defined(USE_HSIGMOID)
#define calCore(val) {\
    val.s0 = val.s0 + (T)3.0;\
    val.s1 = val.s1 + (T)3.0;\
    val.s2 = val.s2 + (T)3.0;\
    val.s3 = val.s3 + (T)3.0;\
    val.s0 = clamp(val.s0, (T)0, (T)6.0);\
    val.s1 = clamp(val.s1, (T)0, (T)6.0);\
    val.s2 = clamp(val.s2, (T)0, (T)6.0);\
    val.s3 = clamp(val.s3, (T)0, (T)6.0);\
    val.s0 = val.s0 * 0.166667;\
    val.s1 = val.s1 * 0.166667;\
    val.s2 = val.s2 * 0.166667;\
    val.s3 = val.s3 * 0.166667;\
}
#elif defined(USE_HSWISH)
#define calCore(val) {\
    T4 tmp = val;\
    val.s0 = val.s0 + (T)3.0;\
    val.s1 = val.s1 + (T)3.0;\
    val.s2 = val.s2 + (T)3.0;\
    val.s3 = val.s3 + (T)3.0;\
    val.s0 = clamp(val.s0, (T)0, (T)6.0);\
    val.s1 = clamp(val.s1, (T)0, (T)6.0);\
    val.s2 = clamp(val.s2, (T)0, (T)6.0);\
    val.s3 = clamp(val.s3, (T)0, (T)6.0);\
    val.s0 = tmp.s0 * (val.s0 * 0.166667);\
    val.s1 = tmp.s1 * (val.s1 * 0.166667);\
    val.s2 = tmp.s2 * (val.s2 * 0.166667);\
    val.s3 = tmp.s3 * (val.s3 * 0.166667);\
}
#elif defined(USE_GELU)
#define calCore(val) {\
    T4 tmp = val;\
    val.s0 = 0.797885 * (val.s0 + 0.044715 * pown(val.s0, 3));\
    val.s1 = 0.797885 * (val.s1 + 0.044715 * pown(val.s1, 3));\
    val.s2 = 0.797885 * (val.s2 + 0.044715 * pown(val.s2, 3));\
    val.s3 = 0.797885 * (val.s3 + 0.044715 * pown(val.s3, 3));\
    val.s0 = (val.s0 + (T)1.0) * (T)0.5;\
    val.s1 = (val.s1 + (T)1.0) * (T)0.5;\
    val.s2 = (val.s2 + (T)1.0) * (T)0.5;\
    val.s3 = (val.s3 + (T)1.0) * (T)0.5;\
    val.s0 = val.s0 * tmp.s0;\
    val.s1 = val.s1 * tmp.s1;\
    val.s2 = val.s2 * tmp.s2;\
    val.s3 = val.s3 * tmp.s3;\
}
#elif defined(USE_TANH)
#define calCore(val) {\
    val.s0 = 1.0 - 2.0 / (exp(2.0 * val.s0) + 1.0);\
    val.s1 = 1.0 - 2.0 / (exp(2.0 * val.s1) + 1.0);\
    val.s2 = 1.0 - 2.0 / (exp(2.0 * val.s2) + 1.0);\
    val.s3 = 1.0 - 2.0 / (exp(2.0 * val.s3) + 1.0);\
}
#elif defined(USE_SIGMOID)
#define calCore(val) {\
    val.s0 = 1.0 / (1.0 + exp(-1.0 * val.s0));\
    val.s1 = 1.0 / (1.0 + exp(-1.0 * val.s1));\
    val.s2 = 1.0 / (1.0 + exp(-1.0 * val.s2));\
    val.s3 = 1.0 / (1.0 + exp(-1.0 * val.s3));\
}
#endif
#endif

#if(H == 2)
#if defined(USE_RELU)
#define calCore(val) {\
    if(val.s0 < 0) val.s0 = 0;\
    if(val.s1 < 0) val.s1 = 0;\
    if(val.s2 < 0) val.s2 = 0;\
    if(val.s3 < 0) val.s3 = 0;\
    if(val.s4 < 0) val.s4 = 0;\
    if(val.s5 < 0) val.s5 = 0;\
    if(val.s6 < 0) val.s6 = 0;\
    if(val.s7 < 0) val.s7 = 0;\
}
#elif defined(USE_RELU6)
#define calCore(val) {\
    val.s0 = clamp(val.s0, (T)0, (T)6.0);\
    val.s1 = clamp(val.s1, (T)0, (T)6.0);\
    val.s2 = clamp(val.s2, (T)0, (T)6.0);\
    val.s3 = clamp(val.s3, (T)0, (T)6.0);\
    val.s4 = clamp(val.s4, (T)0, (T)6.0);\
    val.s5 = clamp(val.s5, (T)0, (T)6.0);\
    val.s6 = clamp(val.s6, (T)0, (T)6.0);\
    val.s7 = clamp(val.s7, (T)0, (T)6.0);\
}
#elif defined(USE_HSIGMOID)
#define calCore(val) {\
    val.s0 = val.s0 + (T)3.0;\
    val.s1 = val.s1 + (T)3.0;\
    val.s2 = val.s2 + (T)3.0;\
    val.s3 = val.s3 + (T)3.0;\
    val.s4 = val.s4 + (T)3.0;\
    val.s5 = val.s5 + (T)3.0;\
    val.s6 = val.s6 + (T)3.0;\
    val.s7 = val.s7 + (T)3.0;\
    val.s0 = clamp(val.s0, (T)0, (T)6.0);\
    val.s1 = clamp(val.s1, (T)0, (T)6.0);\
    val.s2 = clamp(val.s2, (T)0, (T)6.0);\
    val.s3 = clamp(val.s3, (T)0, (T)6.0);\
    val.s4 = clamp(val.s4, (T)0, (T)6.0);\
    val.s5 = clamp(val.s5, (T)0, (T)6.0);\
    val.s6 = clamp(val.s6, (T)0, (T)6.0);\
    val.s7 = clamp(val.s7, (T)0, (T)6.0);\
    val.s0 = val.s0 * 0.166667;\
    val.s1 = val.s1 * 0.166667;\
    val.s2 = val.s2 * 0.166667;\
    val.s3 = val.s3 * 0.166667;\
    val.s4 = val.s4 * 0.166667;\
    val.s5 = val.s5 * 0.166667;\
    val.s6 = val.s6 * 0.166667;\
    val.s7 = val.s7 * 0.166667;\
}
#elif defined(USE_HSWISH)
#define calCore(val) {\
    T8 tmp = val;\
    val.s0 = val.s0 + (T)3.0;\
    val.s1 = val.s1 + (T)3.0;\
    val.s2 = val.s2 + (T)3.0;\
    val.s3 = val.s3 + (T)3.0;\
    val.s4 = val.s4 + (T)3.0;\
    val.s5 = val.s5 + (T)3.0;\
    val.s6 = val.s6 + (T)3.0;\
    val.s7 = val.s7 + (T)3.0;\
    val.s0 = clamp(val.s0, (T)0, (T)6.0);\
    val.s1 = clamp(val.s1, (T)0, (T)6.0);\
    val.s2 = clamp(val.s2, (T)0, (T)6.0);\
    val.s3 = clamp(val.s3, (T)0, (T)6.0);\
    val.s4 = clamp(val.s4, (T)0, (T)6.0);\
    val.s5 = clamp(val.s5, (T)0, (T)6.0);\
    val.s6 = clamp(val.s6, (T)0, (T)6.0);\
    val.s7 = clamp(val.s7, (T)0, (T)6.0);\
    val.s0 = tmp.s0 * (val.s0 * 0.166667);\
    val.s1 = tmp.s1 * (val.s1 * 0.166667);\
    val.s2 = tmp.s2 * (val.s2 * 0.166667);\
    val.s3 = tmp.s3 * (val.s3 * 0.166667);\
    val.s4 = tmp.s4 * (val.s4 * 0.166667);\
    val.s5 = tmp.s5 * (val.s5 * 0.166667);\
    val.s6 = tmp.s6 * (val.s6 * 0.166667);\
    val.s7 = tmp.s7 * (val.s7 * 0.166667);\
}
#elif defined(USE_GELU)
#define calCore(val) {\
    T8 tmp = val;\
    val.s0 = 0.797885 * (val.s0 + 0.044715 * pown(val.s0, 3));\
    val.s1 = 0.797885 * (val.s1 + 0.044715 * pown(val.s1, 3));\
    val.s2 = 0.797885 * (val.s2 + 0.044715 * pown(val.s2, 3));\
    val.s3 = 0.797885 * (val.s3 + 0.044715 * pown(val.s3, 3));\
    val.s4 = 0.797885 * (val.s4 + 0.044715 * pown(val.s4, 3));\
    val.s5 = 0.797885 * (val.s5 + 0.044715 * pown(val.s5, 3));\
    val.s6 = 0.797885 * (val.s6 + 0.044715 * pown(val.s6, 3));\
    val.s7 = 0.797885 * (val.s7 + 0.044715 * pown(val.s7, 3));\
    val.s0 = (val.s0 + (T)1.0) * (T)0.5;\
    val.s1 = (val.s1 + (T)1.0) * (T)0.5;\
    val.s2 = (val.s2 + (T)1.0) * (T)0.5;\
    val.s3 = (val.s3 + (T)1.0) * (T)0.5;\
    val.s4 = (val.s4 + (T)1.0) * (T)0.5;\
    val.s5 = (val.s5 + (T)1.0) * (T)0.5;\
    val.s6 = (val.s6 + (T)1.0) * (T)0.5;\
    val.s7 = (val.s7 + (T)1.0) * (T)0.5;\
    val.s0 = val.s0 * tmp.s0;\
    val.s1 = val.s1 * tmp.s1;\
    val.s2 = val.s2 * tmp.s2;\
    val.s3 = val.s3 * tmp.s3;\
    val.s4 = val.s4 * tmp.s4;\
    val.s5 = val.s5 * tmp.s5;\
    val.s6 = val.s6 * tmp.s6;\
    val.s7 = val.s7 * tmp.s7;\
}
#elif defined(USE_TANH)
#define calCore(val) {\
    val.s0 = 1.0 - 2.0 / (exp(2.0 * val.s0) + 1.0);\
    val.s1 = 1.0 - 2.0 / (exp(2.0 * val.s1) + 1.0);\
    val.s2 = 1.0 - 2.0 / (exp(2.0 * val.s2) + 1.0);\
    val.s3 = 1.0 - 2.0 / (exp(2.0 * val.s3) + 1.0);\
    val.s4 = 1.0 - 2.0 / (exp(2.0 * val.s4) + 1.0);\
    val.s5 = 1.0 - 2.0 / (exp(2.0 * val.s5) + 1.0);\
    val.s6 = 1.0 - 2.0 / (exp(2.0 * val.s6) + 1.0);\
    val.s7 = 1.0 - 2.0 / (exp(2.0 * val.s7) + 1.0);\
}
#elif defined(USE_SIGMOID)
#define calCore(val) {\
    val.s0 = 1.0 / (1.0 + exp(-1.0 * val.s0));\
    val.s1 = 1.0 / (1.0 + exp(-1.0 * val.s1));\
    val.s2 = 1.0 / (1.0 + exp(-1.0 * val.s2));\
    val.s3 = 1.0 / (1.0 + exp(-1.0 * val.s3));\
    val.s4 = 1.0 / (1.0 + exp(-1.0 * val.s4));\
    val.s5 = 1.0 / (1.0 + exp(-1.0 * val.s5));\
    val.s6 = 1.0 / (1.0 + exp(-1.0 * val.s6));\
    val.s7 = 1.0 / (1.0 + exp(-1.0 * val.s7));\
}
#endif
#endif

__kernel void MANGLE_NAME(activation_, AC, H)(const int h, const int ih_str, const int iw_str, const int ih_off, const int iw_off, __global T* data) {
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);

    T4  val;
    int index = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;
    val = vload4(index, data);
    calCore(val);
    vstore4(val, index, data);
}
