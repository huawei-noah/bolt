// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





#if defined(USE_HALF)
#define READ_IMAGE(image, sampler, coord)    read_imageh(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data)      write_imageh(image, coord, data)
#else
#define READ_IMAGE(image, sampler, coord)    read_imagef(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data)      write_imagef(image, coord, data)
#endif
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define calCore(iv, fv, res) {\
    res.x += iv.x * fv.s0 + iv.y * fv.s1 + iv.z * fv.s2 + iv.w * fv.s3;\
    res.y += iv.x * fv.s4 + iv.y * fv.s5 + iv.z * fv.s6 + iv.w * fv.s7;\
    res.z += iv.x * fv.s8 + iv.y * fv.s9 + iv.z * fv.sa + iv.w * fv.sb;\
    res.w += iv.x * fv.sc + iv.y * fv.sd + iv.z * fv.se + iv.w * fv.sf;\
}
__kernel void fc_p2(const int loop, const int len, const int oh_str, const int ow_str, const int oh_off, const int ow_off, 
    __global const T* in, __read_only image1d_t bias, __global T* out) { 
    const int idx = get_global_id(0);

    T4 sum = READ_IMAGE(bias, sampler, idx);
    T4 val;
    for(int i = 0; i < loop; i++) {
        val = vload4(idx + i * len, in);
        sum.x += val.x;
        sum.y += val.y;
        sum.z += val.z;
        sum.w += val.w;
    }

    const int out_off = (idx * ow_str + ow_off) * oh_str + oh_off;
    vstore4(sum, out_off, out);
}
