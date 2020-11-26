// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void softmax_h1w1_max_part(
    const int cd4, const int ce4, const int kn, __global const T *in, __global T *out)
{
    int idx = get_global_id(0);
    if (idx >= kn) {
        return;
    }

    float4 maxval = (float4)(-FLT_MAX);
    float4 tmp;
    T4 val;

    for (int i = idx; i < cd4 - 1; i = i + kn) {
        val = vload4(i, in);
        tmp.x = (float)val.x;
        tmp.y = (float)val.y;
        tmp.z = (float)val.z;
        tmp.w = (float)val.w;
        maxval = fmax(maxval, tmp);
    }

    if (maxval.x < maxval.y) {
        maxval.x = maxval.y;
    }
    if (maxval.x < maxval.z) {
        maxval.x = maxval.z;
    }
    if (maxval.x < maxval.w) {
        maxval.x = maxval.w;
    }

    if (idx == kn - 1) {
        val = vload4(cd4 - 1, in);
        maxval.x = fmax((float)val.x, maxval.x);
        if (ce4 >= 2) {
            maxval.x = fmax((float)val.y, maxval.x);
        }
        if (ce4 >= 3) {
            maxval.x = fmax((float)val.z, maxval.x);
        }
        if (ce4 >= 4) {
            maxval.x = fmax((float)val.w, maxval.x);
        }
    }

    out[idx] = (T)maxval.x;
}
