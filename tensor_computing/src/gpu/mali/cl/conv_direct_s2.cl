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

#if (W == 1)
#define dirCalCore1(a0, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
}
#endif

#if (W == 2)
#define dirCalCore2(a0, a1, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[1].x += (a1.x * b.s0 + a1.y * b.s1 + a1.z * b.s2 + a1.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[1].y += (a1.x * b.s4 + a1.y * b.s5 + a1.z * b.s6 + a1.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[1].z += (a1.x * b.s8 + a1.y * b.s9 + a1.z * b.sa + a1.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
    c[1].w += (a1.x * b.sc + a1.y * b.sd + a1.z * b.se + a1.w * b.sf);\
}
#endif

#if(W == 3)
#define dirCalCore3(a0, a1, a2, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[1].x += (a1.x * b.s0 + a1.y * b.s1 + a1.z * b.s2 + a1.w * b.s3);\
    c[2].x += (a2.x * b.s0 + a2.y * b.s1 + a2.z * b.s2 + a2.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[1].y += (a1.x * b.s4 + a1.y * b.s5 + a1.z * b.s6 + a1.w * b.s7);\
    c[2].y += (a2.x * b.s4 + a2.y * b.s5 + a2.z * b.s6 + a2.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[1].z += (a1.x * b.s8 + a1.y * b.s9 + a1.z * b.sa + a1.w * b.sb);\
    c[2].z += (a2.x * b.s8 + a2.y * b.s9 + a2.z * b.sa + a2.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
    c[1].w += (a1.x * b.sc + a1.y * b.sd + a1.z * b.se + a1.w * b.sf);\
    c[2].w += (a2.x * b.sc + a2.y * b.sd + a2.z * b.se + a2.w * b.sf);\
}
#endif

#if(W == 4)
#define dirCalCore4(a0, a1, a2, a3, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[1].x += (a1.x * b.s0 + a1.y * b.s1 + a1.z * b.s2 + a1.w * b.s3);\
    c[2].x += (a2.x * b.s0 + a2.y * b.s1 + a2.z * b.s2 + a2.w * b.s3);\
    c[3].x += (a3.x * b.s0 + a3.y * b.s1 + a3.z * b.s2 + a3.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[1].y += (a1.x * b.s4 + a1.y * b.s5 + a1.z * b.s6 + a1.w * b.s7);\
    c[2].y += (a2.x * b.s4 + a2.y * b.s5 + a2.z * b.s6 + a2.w * b.s7);\
    c[3].y += (a3.x * b.s4 + a3.y * b.s5 + a3.z * b.s6 + a3.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[1].z += (a1.x * b.s8 + a1.y * b.s9 + a1.z * b.sa + a1.w * b.sb);\
    c[2].z += (a2.x * b.s8 + a2.y * b.s9 + a2.z * b.sa + a2.w * b.sb);\
    c[3].z += (a3.x * b.s8 + a3.y * b.s9 + a3.z * b.sa + a3.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
    c[1].w += (a1.x * b.sc + a1.y * b.sd + a1.z * b.se + a1.w * b.sf);\
    c[2].w += (a2.x * b.sc + a2.y * b.sd + a2.z * b.se + a2.w * b.sf);\
    c[3].w += (a3.x * b.sc + a3.y * b.sd + a3.z * b.se + a3.w * b.sf);\
}
#endif

#if(W == 5)
#define dirCalCore5(a0, a1, a2, a3, a4, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[1].x += (a1.x * b.s0 + a1.y * b.s1 + a1.z * b.s2 + a1.w * b.s3);\
    c[2].x += (a2.x * b.s0 + a2.y * b.s1 + a2.z * b.s2 + a2.w * b.s3);\
    c[3].x += (a3.x * b.s0 + a3.y * b.s1 + a3.z * b.s2 + a3.w * b.s3);\
    c[4].x += (a4.x * b.s0 + a4.y * b.s1 + a4.z * b.s2 + a4.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[1].y += (a1.x * b.s4 + a1.y * b.s5 + a1.z * b.s6 + a1.w * b.s7);\
    c[2].y += (a2.x * b.s4 + a2.y * b.s5 + a2.z * b.s6 + a2.w * b.s7);\
    c[3].y += (a3.x * b.s4 + a3.y * b.s5 + a3.z * b.s6 + a3.w * b.s7);\
    c[4].y += (a4.x * b.s4 + a4.y * b.s5 + a4.z * b.s6 + a4.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[1].z += (a1.x * b.s8 + a1.y * b.s9 + a1.z * b.sa + a1.w * b.sb);\
    c[2].z += (a2.x * b.s8 + a2.y * b.s9 + a2.z * b.sa + a2.w * b.sb);\
    c[3].z += (a3.x * b.s8 + a3.y * b.s9 + a3.z * b.sa + a3.w * b.sb);\
    c[4].z += (a4.x * b.s8 + a4.y * b.s9 + a4.z * b.sa + a4.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
    c[1].w += (a1.x * b.sc + a1.y * b.sd + a1.z * b.se + a1.w * b.sf);\
    c[2].w += (a2.x * b.sc + a2.y * b.sd + a2.z * b.se + a2.w * b.sf);\
    c[3].w += (a3.x * b.sc + a3.y * b.sd + a3.z * b.se + a3.w * b.sf);\
    c[4].w += (a4.x * b.sc + a4.y * b.sd + a4.z * b.se + a4.w * b.sf);\
}
#endif

#if(W == 6)
#define dirCalCore6(a0, a1, a2, a3, a4, a5, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[1].x += (a1.x * b.s0 + a1.y * b.s1 + a1.z * b.s2 + a1.w * b.s3);\
    c[2].x += (a2.x * b.s0 + a2.y * b.s1 + a2.z * b.s2 + a2.w * b.s3);\
    c[3].x += (a3.x * b.s0 + a3.y * b.s1 + a3.z * b.s2 + a3.w * b.s3);\
    c[4].x += (a4.x * b.s0 + a4.y * b.s1 + a4.z * b.s2 + a4.w * b.s3);\
    c[5].x += (a5.x * b.s0 + a5.y * b.s1 + a5.z * b.s2 + a5.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[1].y += (a1.x * b.s4 + a1.y * b.s5 + a1.z * b.s6 + a1.w * b.s7);\
    c[2].y += (a2.x * b.s4 + a2.y * b.s5 + a2.z * b.s6 + a2.w * b.s7);\
    c[3].y += (a3.x * b.s4 + a3.y * b.s5 + a3.z * b.s6 + a3.w * b.s7);\
    c[4].y += (a4.x * b.s4 + a4.y * b.s5 + a4.z * b.s6 + a4.w * b.s7);\
    c[5].y += (a5.x * b.s4 + a5.y * b.s5 + a5.z * b.s6 + a5.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[1].z += (a1.x * b.s8 + a1.y * b.s9 + a1.z * b.sa + a1.w * b.sb);\
    c[2].z += (a2.x * b.s8 + a2.y * b.s9 + a2.z * b.sa + a2.w * b.sb);\
    c[3].z += (a3.x * b.s8 + a3.y * b.s9 + a3.z * b.sa + a3.w * b.sb);\
    c[4].z += (a4.x * b.s8 + a4.y * b.s9 + a4.z * b.sa + a4.w * b.sb);\
    c[5].z += (a5.x * b.s8 + a5.y * b.s9 + a5.z * b.sa + a5.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
    c[1].w += (a1.x * b.sc + a1.y * b.sd + a1.z * b.se + a1.w * b.sf);\
    c[2].w += (a2.x * b.sc + a2.y * b.sd + a2.z * b.se + a2.w * b.sf);\
    c[3].w += (a3.x * b.sc + a3.y * b.sd + a3.z * b.se + a3.w * b.sf);\
    c[4].w += (a4.x * b.sc + a4.y * b.sd + a4.z * b.se + a4.w * b.sf);\
    c[5].w += (a5.x * b.sc + a5.y * b.sd + a5.z * b.se + a5.w * b.sf);\
}
#endif

#if(W == 7)
#define dirCalCore7(a0, a1, a2, a3, a4, a5, a6, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[1].x += (a1.x * b.s0 + a1.y * b.s1 + a1.z * b.s2 + a1.w * b.s3);\
    c[2].x += (a2.x * b.s0 + a2.y * b.s1 + a2.z * b.s2 + a2.w * b.s3);\
    c[3].x += (a3.x * b.s0 + a3.y * b.s1 + a3.z * b.s2 + a3.w * b.s3);\
    c[4].x += (a4.x * b.s0 + a4.y * b.s1 + a4.z * b.s2 + a4.w * b.s3);\
    c[5].x += (a5.x * b.s0 + a5.y * b.s1 + a5.z * b.s2 + a5.w * b.s3);\
    c[6].x += (a6.x * b.s0 + a6.y * b.s1 + a6.z * b.s2 + a6.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[1].y += (a1.x * b.s4 + a1.y * b.s5 + a1.z * b.s6 + a1.w * b.s7);\
    c[2].y += (a2.x * b.s4 + a2.y * b.s5 + a2.z * b.s6 + a2.w * b.s7);\
    c[3].y += (a3.x * b.s4 + a3.y * b.s5 + a3.z * b.s6 + a3.w * b.s7);\
    c[4].y += (a4.x * b.s4 + a4.y * b.s5 + a4.z * b.s6 + a4.w * b.s7);\
    c[5].y += (a5.x * b.s4 + a5.y * b.s5 + a5.z * b.s6 + a5.w * b.s7);\
    c[6].y += (a6.x * b.s4 + a6.y * b.s5 + a6.z * b.s6 + a6.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[1].z += (a1.x * b.s8 + a1.y * b.s9 + a1.z * b.sa + a1.w * b.sb);\
    c[2].z += (a2.x * b.s8 + a2.y * b.s9 + a2.z * b.sa + a2.w * b.sb);\
    c[3].z += (a3.x * b.s8 + a3.y * b.s9 + a3.z * b.sa + a3.w * b.sb);\
    c[4].z += (a4.x * b.s8 + a4.y * b.s9 + a4.z * b.sa + a4.w * b.sb);\
    c[5].z += (a5.x * b.s8 + a5.y * b.s9 + a5.z * b.sa + a5.w * b.sb);\
    c[6].z += (a6.x * b.s8 + a6.y * b.s9 + a6.z * b.sa + a6.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
    c[1].w += (a1.x * b.sc + a1.y * b.sd + a1.z * b.se + a1.w * b.sf);\
    c[2].w += (a2.x * b.sc + a2.y * b.sd + a2.z * b.se + a2.w * b.sf);\
    c[3].w += (a3.x * b.sc + a3.y * b.sd + a3.z * b.se + a3.w * b.sf);\
    c[4].w += (a4.x * b.sc + a4.y * b.sd + a4.z * b.se + a4.w * b.sf);\
    c[5].w += (a5.x * b.sc + a5.y * b.sd + a5.z * b.se + a5.w * b.sf);\
    c[6].w += (a6.x * b.sc + a6.y * b.sd + a6.z * b.se + a6.w * b.sf);\
}
#endif

#if(W == 8)
#define dirCalCore8(a0, a1, a2, a3, a4, a5, a6, a7, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[1].x += (a1.x * b.s0 + a1.y * b.s1 + a1.z * b.s2 + a1.w * b.s3);\
    c[2].x += (a2.x * b.s0 + a2.y * b.s1 + a2.z * b.s2 + a2.w * b.s3);\
    c[3].x += (a3.x * b.s0 + a3.y * b.s1 + a3.z * b.s2 + a3.w * b.s3);\
    c[4].x += (a4.x * b.s0 + a4.y * b.s1 + a4.z * b.s2 + a4.w * b.s3);\
    c[5].x += (a5.x * b.s0 + a5.y * b.s1 + a5.z * b.s2 + a5.w * b.s3);\
    c[6].x += (a6.x * b.s0 + a6.y * b.s1 + a6.z * b.s2 + a6.w * b.s3);\
    c[7].x += (a7.x * b.s0 + a7.y * b.s1 + a7.z * b.s2 + a7.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[1].y += (a1.x * b.s4 + a1.y * b.s5 + a1.z * b.s6 + a1.w * b.s7);\
    c[2].y += (a2.x * b.s4 + a2.y * b.s5 + a2.z * b.s6 + a2.w * b.s7);\
    c[3].y += (a3.x * b.s4 + a3.y * b.s5 + a3.z * b.s6 + a3.w * b.s7);\
    c[4].y += (a4.x * b.s4 + a4.y * b.s5 + a4.z * b.s6 + a4.w * b.s7);\
    c[5].y += (a5.x * b.s4 + a5.y * b.s5 + a5.z * b.s6 + a5.w * b.s7);\
    c[6].y += (a6.x * b.s4 + a6.y * b.s5 + a6.z * b.s6 + a6.w * b.s7);\
    c[7].y += (a7.x * b.s4 + a7.y * b.s5 + a7.z * b.s6 + a7.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[1].z += (a1.x * b.s8 + a1.y * b.s9 + a1.z * b.sa + a1.w * b.sb);\
    c[2].z += (a2.x * b.s8 + a2.y * b.s9 + a2.z * b.sa + a2.w * b.sb);\
    c[3].z += (a3.x * b.s8 + a3.y * b.s9 + a3.z * b.sa + a3.w * b.sb);\
    c[4].z += (a4.x * b.s8 + a4.y * b.s9 + a4.z * b.sa + a4.w * b.sb);\
    c[5].z += (a5.x * b.s8 + a5.y * b.s9 + a5.z * b.sa + a5.w * b.sb);\
    c[6].z += (a6.x * b.s8 + a6.y * b.s9 + a6.z * b.sa + a6.w * b.sb);\
    c[7].z += (a7.x * b.s8 + a7.y * b.s9 + a7.z * b.sa + a7.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
    c[1].w += (a1.x * b.sc + a1.y * b.sd + a1.z * b.se + a1.w * b.sf);\
    c[2].w += (a2.x * b.sc + a2.y * b.sd + a2.z * b.se + a2.w * b.sf);\
    c[3].w += (a3.x * b.sc + a3.y * b.sd + a3.z * b.se + a3.w * b.sf);\
    c[4].w += (a4.x * b.sc + a4.y * b.sd + a4.z * b.se + a4.w * b.sf);\
    c[5].w += (a5.x * b.sc + a5.y * b.sd + a5.z * b.se + a5.w * b.sf);\
    c[6].w += (a6.x * b.sc + a6.y * b.sd + a6.z * b.se + a6.w * b.sf);\
    c[7].w += (a7.x * b.sc + a7.y * b.sd + a7.z * b.se + a7.w * b.sf);\
}
#endif

#if(W == 10)
#define dirCalCore10(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, b, c){\
    c[0].x += (a0.x * b.s0 + a0.y * b.s1 + a0.z * b.s2 + a0.w * b.s3);\
    c[1].x += (a1.x * b.s0 + a1.y * b.s1 + a1.z * b.s2 + a1.w * b.s3);\
    c[2].x += (a2.x * b.s0 + a2.y * b.s1 + a2.z * b.s2 + a2.w * b.s3);\
    c[3].x += (a3.x * b.s0 + a3.y * b.s1 + a3.z * b.s2 + a3.w * b.s3);\
    c[4].x += (a4.x * b.s0 + a4.y * b.s1 + a4.z * b.s2 + a4.w * b.s3);\
    c[5].x += (a5.x * b.s0 + a5.y * b.s1 + a5.z * b.s2 + a5.w * b.s3);\
    c[6].x += (a6.x * b.s0 + a6.y * b.s1 + a6.z * b.s2 + a6.w * b.s3);\
    c[7].x += (a7.x * b.s0 + a7.y * b.s1 + a7.z * b.s2 + a7.w * b.s3);\
    c[8].x += (a8.x * b.s0 + a8.y * b.s1 + a8.z * b.s2 + a8.w * b.s3);\
    c[9].x += (a9.x * b.s0 + a9.y * b.s1 + a9.z * b.s2 + a9.w * b.s3);\
    c[0].y += (a0.x * b.s4 + a0.y * b.s5 + a0.z * b.s6 + a0.w * b.s7);\
    c[1].y += (a1.x * b.s4 + a1.y * b.s5 + a1.z * b.s6 + a1.w * b.s7);\
    c[2].y += (a2.x * b.s4 + a2.y * b.s5 + a2.z * b.s6 + a2.w * b.s7);\
    c[3].y += (a3.x * b.s4 + a3.y * b.s5 + a3.z * b.s6 + a3.w * b.s7);\
    c[4].y += (a4.x * b.s4 + a4.y * b.s5 + a4.z * b.s6 + a4.w * b.s7);\
    c[5].y += (a5.x * b.s4 + a5.y * b.s5 + a5.z * b.s6 + a5.w * b.s7);\
    c[6].y += (a6.x * b.s4 + a6.y * b.s5 + a6.z * b.s6 + a6.w * b.s7);\
    c[7].y += (a7.x * b.s4 + a7.y * b.s5 + a7.z * b.s6 + a7.w * b.s7);\
    c[8].y += (a8.x * b.s4 + a8.y * b.s5 + a8.z * b.s6 + a8.w * b.s7);\
    c[9].y += (a9.x * b.s4 + a9.y * b.s5 + a9.z * b.s6 + a9.w * b.s7);\
    c[0].z += (a0.x * b.s8 + a0.y * b.s9 + a0.z * b.sa + a0.w * b.sb);\
    c[1].z += (a1.x * b.s8 + a1.y * b.s9 + a1.z * b.sa + a1.w * b.sb);\
    c[2].z += (a2.x * b.s8 + a2.y * b.s9 + a2.z * b.sa + a2.w * b.sb);\
    c[3].z += (a3.x * b.s8 + a3.y * b.s9 + a3.z * b.sa + a3.w * b.sb);\
    c[4].z += (a4.x * b.s8 + a4.y * b.s9 + a4.z * b.sa + a4.w * b.sb);\
    c[5].z += (a5.x * b.s8 + a5.y * b.s9 + a5.z * b.sa + a5.w * b.sb);\
    c[6].z += (a6.x * b.s8 + a6.y * b.s9 + a6.z * b.sa + a6.w * b.sb);\
    c[7].z += (a7.x * b.s8 + a7.y * b.s9 + a7.z * b.sa + a7.w * b.sb);\
    c[8].z += (a8.x * b.s8 + a8.y * b.s9 + a8.z * b.sa + a8.w * b.sb);\
    c[9].z += (a9.x * b.s8 + a9.y * b.s9 + a9.z * b.sa + a9.w * b.sb);\
    c[0].w += (a0.x * b.sc + a0.y * b.sd + a0.z * b.se + a0.w * b.sf);\
    c[1].w += (a1.x * b.sc + a1.y * b.sd + a1.z * b.se + a1.w * b.sf);\
    c[2].w += (a2.x * b.sc + a2.y * b.sd + a2.z * b.se + a2.w * b.sf);\
    c[3].w += (a3.x * b.sc + a3.y * b.sd + a3.z * b.se + a3.w * b.sf);\
    c[4].w += (a4.x * b.sc + a4.y * b.sd + a4.z * b.se + a4.w * b.sf);\
    c[5].w += (a5.x * b.sc + a5.y * b.sd + a5.z * b.se + a5.w * b.sf);\
    c[6].w += (a6.x * b.sc + a6.y * b.sd + a6.z * b.se + a6.w * b.sf);\
    c[7].w += (a7.x * b.sc + a7.y * b.sd + a7.z * b.se + a7.w * b.sf);\
    c[8].w += (a8.x * b.sc + a8.y * b.sd + a8.z * b.se + a8.w * b.sf);\
    c[9].w += (a9.x * b.sc + a9.y * b.sd + a9.z * b.se + a9.w * b.sf);\
}
#endif

#if(F == 1)
#if(W == 1)
#define calCore0 dirCalCore1(in_val[0], flt_val, out_val);
#elif(W == 2)
#define calCore0 dirCalCore2(in_val[0], in_val[1], flt_val, out_val);
#elif(W == 3)
#define calCore0 dirCalCore3(in_val[0], in_val[1], in_val[2], flt_val, out_val);
#elif(W == 4)
#define calCore0 dirCalCore4(in_val[0], in_val[1], in_val[2], in_val[3], flt_val, out_val);
#elif(W == 5)
#define calCore0 dirCalCore5(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], flt_val, out_val);
#elif(W == 6)
#define calCore0 dirCalCore6(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], flt_val, out_val);
#elif(W == 7)
#define calCore0 dirCalCore7(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], flt_val, out_val);
#elif(W == 8)
#define calCore0 dirCalCore8(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], flt_val, out_val);
#elif(W == 10)
#define calCore0 dirCalCore10(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8],  in_val[9],  flt_val, out_val);
#endif
#endif

#if(F == 3)
#if(W == 1)
#define calCore0 dirCalCore1(in_val[0], flt_val, out_val);
#define calCore1 dirCalCore1(in_val[1], flt_val, out_val);
#define calCore2 dirCalCore1(in_val[2], flt_val, out_val);
#elif(W == 2)
#define calCore0 dirCalCore2(in_val[0], in_val[2], flt_val, out_val);
#define calCore1 dirCalCore2(in_val[1], in_val[3], flt_val, out_val);
#define calCore2 dirCalCore2(in_val[2], in_val[4], flt_val, out_val);
#elif(W == 3)
#define calCore0 dirCalCore3(in_val[0], in_val[2], in_val[4], flt_val, out_val);
#define calCore1 dirCalCore3(in_val[1], in_val[3], in_val[5], flt_val, out_val);
#define calCore2 dirCalCore3(in_val[2], in_val[4], in_val[6], flt_val, out_val);
#elif(W == 4)
#define calCore0 dirCalCore4(in_val[0], in_val[2], in_val[4], in_val[6], flt_val, out_val);
#define calCore1 dirCalCore4(in_val[1], in_val[3], in_val[5], in_val[7], flt_val, out_val);
#define calCore2 dirCalCore4(in_val[2], in_val[4], in_val[6], in_val[8], flt_val, out_val);
#elif(W == 5)
#define calCore0 dirCalCore4(in_val[0], in_val[2], in_val[4], in_val[6], in_val[8],  flt_val, out_val);
#define calCore1 dirCalCore4(in_val[1], in_val[3], in_val[5], in_val[7], in_val[9],  flt_val, out_val);
#define calCore2 dirCalCore4(in_val[2], in_val[4], in_val[6], in_val[8], in_val[10], flt_val, out_val);
#elif(W == 6)
#define calCore0 dirCalCore6(in_val[0], in_val[2], in_val[4], in_val[6], in_val[8],  in_val[10], flt_val, out_val);
#define calCore1 dirCalCore6(in_val[1], in_val[3], in_val[5], in_val[7], in_val[9],  in_val[11], flt_val, out_val);
#define calCore2 dirCalCore6(in_val[2], in_val[4], in_val[6], in_val[8], in_val[10], in_val[12], flt_val, out_val);
#endif
#endif

#if(F == 5)
#if(W == 1)
#define calCore0 dirCalCore1(in_val[0], flt_val, out_val);
#define calCore1 dirCalCore1(in_val[1], flt_val, out_val);
#define calCore2 dirCalCore1(in_val[2], flt_val, out_val);
#elif(W == 2)
#define calCore0 dirCalCore2(in_val[0], in_val[1], flt_val, out_val);
#define calCore1 dirCalCore2(in_val[1], in_val[2], flt_val, out_val);
#define calCore2 dirCalCore2(in_val[2], in_val[0], flt_val, out_val);
#elif(W == 3)
#define calCore0 dirCalCore3(in_val[0], in_val[1], in_val[2], flt_val, out_val);
#define calCore1 dirCalCore3(in_val[1], in_val[2], in_val[0], flt_val, out_val);
#define calCore2 dirCalCore3(in_val[2], in_val[0], in_val[1], flt_val, out_val);
#elif(W == 4)
#define calCore0 dirCalCore4(in_val[0], in_val[1], in_val[2], in_val[3], flt_val, out_val);
#define calCore1 dirCalCore4(in_val[1], in_val[2], in_val[3], in_val[4], flt_val, out_val);
#define calCore2 dirCalCore4(in_val[2], in_val[3], in_val[4], in_val[5], flt_val, out_val);
#elif(W == 5)
#define calCore0 dirCalCore4(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], flt_val, out_val);
#define calCore1 dirCalCore4(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], flt_val, out_val);
#define calCore2 dirCalCore4(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], flt_val, out_val);
#elif(W == 6)
#define calCore0 dirCalCore6(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], flt_val, out_val);
#define calCore1 dirCalCore6(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[0], flt_val, out_val);
#define calCore2 dirCalCore6(in_val[2], in_val[3], in_val[4], in_val[5], in_val[0], in_val[1], flt_val, out_val);
#elif(W == 7)
#define calCore0 dirCalCore7(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], flt_val, out_val);
#define calCore1 dirCalCore7(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[0], flt_val, out_val);
#define calCore2 dirCalCore7(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[0], in_val[1], flt_val, out_val);
#elif(W == 8)
#define calCore0 dirCalCore8(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], flt_val, out_val);
#define calCore1 dirCalCore8(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[0], flt_val, out_val);
#define calCore2 dirCalCore8(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[0], in_val[1], flt_val, out_val);
#elif(W == 10)
#define calCore0 dirCalCore10(in_val[0], in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6],  in_val[7],  in_val[8],  in_val[9], flt_val, out_val);
#define calCore1 dirCalCore10(in_val[1], in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7],  in_val[8],  in_val[9],  in_val[0], flt_val, out_val);
#define calCore2 dirCalCore10(in_val[2], in_val[3], in_val[4], in_val[5], in_val[6], in_val[7], in_val[8],  in_val[9],  in_val[0], in_val[1], flt_val, out_val);
#endif
#endif

#if(N == 1)
#define loadInval(val, off, str, in){\
    val[0] = vload4(off, in);\
}
#elif(N == 2)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off + str, in);\
}
#elif(N == 3)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
}
#elif(N == 4)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
}
#elif(N == 5)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
}
#elif(N == 6)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
    val[5] = vload4(off +  str * 5,   in);\
}
#elif(N == 7)
#define loadInval(val, off, str, in) {\
    val[0] = vload4(off, in);\
    val[1] = vload4(off +  str,       in);\
    val[2] = vload4(off + (str << 1), in);\
    val[3] = vload4(off +  str * 3,   in);\
    val[4] = vload4(off + (str << 2), in);\
    val[5] = vload4(off +  str * 5,   in);\
    val[6] = vload4(off +  str * 6,   in);\
}
#elif(N == 8)
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
#elif(N == 9)
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
#elif(N == 10)
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

#if (F == 1)
#define calCore(flt, off, val){\
    val = vload16(off, flt);\
    calCore0;\
    off += 1;\
}
#elif (F == 3)
#define calCore(flt, off, val){\
    val = vload16(off, flt);\
    calCore0;\
    val = vload16(off + 1, flt);\
    calCore1;\
    val = vload16(off + 2, flt);\
    calCore2;\
    off += 3;\
}
#elif (F == 5)
#define calCoreP1(flt, off, val, iv, ioff, istr, in){\
    val = vload16(off, flt);\
    calCore0;\
    val = vload16(off + 1, flt);\
    iv[0] = vload4(ioff + (W << 1) * istr, in);\
    calCore1;\
    iv[1] = vload4(ioff + ((W + 1) << 1) * istr, in);\
    val = vload16(off + 2, flt);\
    calCore2;\
}
#define calCoreP2(flt, off, val, iv, ioff, istr, in){\
    val = vload16(off, flt);\
    calCore0;\
    val = vload16(off + 1, flt);\
    iv[0] = vload4(ioff + ((W << 1) + 1) * istr, in);\
    calCore1;\
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
#if(F == 1 || F == 3)
#if defined(USE_RELU)
__kernel void MANGLE_NAME(conv_direct_s2_relu_, F, W)
#else
__kernel void MANGLE_NAME(conv_direct_s2_, F, W)
#endif
(const int ih_str, const int ihw_str, const int ic_str, const int ih_off, const int iw_off, const int oh_str, const int ow_str, const int oh_off, const int ow_off, const int ow, 
    __global const T* in, __global const T* flt, __read_only image1d_t bias, __global T* out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    T4  in_val[N];
    T16 flt_val;
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

    int in_off  = ((idy << 1) * W + iw_off) * ih_str + (idx << 1) + ih_off;
    int flt_off = idz * ic_str * Fsq;

    for(int i = 0; i < ic_str; ++i){
#if(F == 1)
        loadInval(in_val, in_off, (ih_str << 1), in);
        calCore(flt, flt_off, flt_val);
#elif(F == 3)
        loadInval(in_val, in_off, ih_str, in);
        calCore(flt, flt_off, flt_val);
        loadInval(in_val, in_off + 1, ih_str, in);
        calCore(flt, flt_off, flt_val);
        loadInval(in_val, in_off + 2, ih_str, in);
        calCore(flt, flt_off, flt_val);
#endif
        in_off += ihw_str;
    }

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
#endif

#if(F == 5)
#if defined(USE_RELU)
__kernel void MANGLE_NAME(conv_direct_s2_relu_, F, W)
#else
__kernel void MANGLE_NAME(conv_direct_s2_, F, W)
#endif
(const int ih_str, const int ihw_str, const int ic_str, const int ih_off, const int iw_off, const int oh_str, const int ow_str, const int oh_off, const int ow_off, const int ow, 
    __global const T* in, __global const T* flt, __read_only image1d_t bias, __global T* out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    T4  in_val[W];
    T16 flt_val;
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

    int in_off  = ((idy << 1) * W + iw_off) * ih_str + (idx << 1) + ih_off;
    int flt_off = idz * ic_str * Fsq;

    for(int i = 0; i < ic_str; ++i){    
        loadInval(in_val, in_off, (ih_str << 1), in);
        calCoreP1(flt, flt_off, flt_val, in_val, in_off, ih_str, in);
        loadInval(in_val, in_off + ih_str, (ih_str << 1), in);
        calCoreP2(flt, flt_off, flt_val, in_val, in_off, ih_str, in);

        loadInval(in_val, in_off + 1, (ih_str << 1), in);
        calCoreP1(flt, flt_off, flt_val, in_val, in_off + 1, ih_str, in);
        loadInval(in_val, in_off + 1 + ih_str, (ih_str << 1), in);
        calCoreP2(flt, flt_off, flt_val, in_val, in_off + 1, ih_str, in);

        loadInval(in_val, in_off + 2, (ih_str << 1), in);
        calCoreP1(flt, flt_off, flt_val, in_val, in_off + 2, ih_str, in);
        loadInval(in_val, in_off + 2 + ih_str, (ih_str << 1), in);
        calCoreP2(flt, flt_off, flt_val, in_val, in_off + 2, ih_str, in);

        loadInval(in_val, in_off + 3, (ih_str << 1), in);
        calCoreP1(flt, flt_off, flt_val, in_val, in_off + 3, ih_str, in);
        loadInval(in_val, in_off + 3 + ih_str, (ih_str << 1), in);
        calCoreP2(flt, flt_off, flt_val, in_val, in_off + 3, ih_str, in);

        loadInval(in_val, in_off + 4, (ih_str << 1), in);
        calCoreP1(flt, flt_off, flt_val, in_val, in_off + 4, ih_str, in);
        loadInval(in_val, in_off + 4 + ih_str, (ih_str << 1), in);
        calCoreP2(flt, flt_off, flt_val, in_val, in_off + 4, ih_str, in);
        in_off += ihw_str;
    }

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
#endif
