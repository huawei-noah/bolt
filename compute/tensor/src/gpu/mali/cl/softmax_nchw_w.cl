// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void softmax_nchw_w(const int wd4,
    const int we4,
    const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
    const int bx,
    const int by,
    __global T *in,
    __global T *out)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }

    float4 maxval = (float4)(-FLT_MAX);
    float4 tmp;
    float4 lval;
    T4 val;

    int index = (idy * ih_str + idx + ih_off) * iw_str + iw_off;
    for (int i = 0; i < wd4 - 1; i++) {
        val = vload4(i, in + index);
        tmp.x = (float)val.x;
        tmp.y = (float)val.y;
        tmp.z = (float)val.z;
        tmp.w = (float)val.w;
        maxval = fmax(maxval, tmp);
    }
    val = vload4(wd4 - 1, in + index);
    lval.x = (float)val.x;
    lval.y = (float)val.y;
    lval.z = (float)val.z;
    lval.w = (float)val.w;
    if (maxval.x < maxval.y) {
        maxval.x = maxval.y;
    }
    if (maxval.x < maxval.z) {
        maxval.x = maxval.z;
    }
    if (maxval.x < maxval.w) {
        maxval.x = maxval.w;
    }
    if (maxval.x < lval.x) {
        maxval.x = lval.x;
    }
    if (we4 > 1 && maxval.x < lval.y) {
        maxval.x = lval.y;
    }
    if (we4 > 2 && maxval.x < lval.z) {
        maxval.x = lval.z;
    }
    if (we4 > 3 && maxval.x < lval.w) {
        maxval.x = lval.w;
    }

    float sumexp = 0;
    for (int i = 0; i < wd4 - 1; i++) {
        val = vload4(i, in + index);
        sumexp += exp((float)val.x - maxval.x);
        sumexp += exp((float)val.y - maxval.x);
        sumexp += exp((float)val.z - maxval.x);
        sumexp += exp((float)val.w - maxval.x);
    }
    sumexp += exp(lval.x - maxval.x);
    if (we4 > 1) {
        sumexp += exp(lval.y - maxval.x);
    }
    if (we4 > 2) {
        sumexp += exp(lval.z - maxval.x);
    }
    if (we4 > 3) {
        sumexp += exp(lval.w - maxval.x);
    }

    sumexp = 1.0 / sumexp;
    int out_off = (idy * oh_str + idx + oh_off) * ow_str + ow_off;
    for (int i = 0; i < wd4 - 1; i++) {
        val = vload4(i, in + index);
        val.x = exp((float)val.x - maxval.x) * sumexp;
        val.y = exp((float)val.y - maxval.x) * sumexp;
        val.z = exp((float)val.z - maxval.x) * sumexp;
        val.w = exp((float)val.w - maxval.x) * sumexp;
        vstore4(val, i, out + out_off);
    }
    val.x = exp(lval.x - maxval.x) * sumexp;
    if (we4 > 1) {
        val.y = exp(lval.y - maxval.x) * sumexp;
    }
    if (we4 > 2) {
        val.z = exp(lval.z - maxval.x) * sumexp;
    }
    if (we4 > 3) {
        val.w = exp(lval.w - maxval.x) * sumexp;
    }
    vstore4(val, wd4 - 1, out + out_off);
}
