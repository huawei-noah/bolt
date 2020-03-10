// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.





#define MANGLE_NAME_IMPL(base, W, H, C) base ## W ## H ## C
#define MANGLE_NAME(base, W, H, C) MANGLE_NAME_IMPL(base, W, H, C)

__kernel void softmax(const int h, const int w, const int cd4, const int ih_str, const int ihw_str, const int ih_off, const int iw_off, 
    const int oh_str, const int ohw_str, const int oh_off, const int ow_off, __global const T* in, __global T* out) {
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    float4 maxval = (float4)(-FLT_MAX);
    float4 tmp;
    T4  val;
    int index = (idy + iw_off) * ih_str + idx + ih_off;
    for (int i = 0; i < cd4; i++) {
        val   = vload4(index + i * ihw_str, in);
        tmp.x = (float)val.x;
        tmp.y = (float)val.y;
        tmp.z = (float)val.z;
        tmp.w = (float)val.w;
        maxval = fmax(maxval, tmp);
    }
    float sumexp = 0;
    for (int i = 0; i < cd4; i++) {
        val   = vload4(index + i * ihw_str, in);
        sumexp += exp((float)val.x - maxval.x);
        sumexp += exp((float)val.y - maxval.y);
        sumexp += exp((float)val.z - maxval.z);
        sumexp += exp((float)val.w - maxval.w);
    }

    sumexp = 1.0 / sumexp;
    T4 res;
    int out_off = (idy + ow_off) * oh_str + idx + oh_off;
    for (int i = 0; i < cd4; i++) {
        val   = vload4(index + i * ihw_str, in);
        res.x = (T)exp(val.x - maxval.x) * sumexp;
        res.y = (T)exp(val.y - maxval.y) * sumexp;
        res.z = (T)exp(val.z - maxval.z) * sumexp;
        res.w = (T)exp(val.w - maxval.w) * sumexp;
        vstore4(res, out_off + i * ohw_str, out);
    }
}
