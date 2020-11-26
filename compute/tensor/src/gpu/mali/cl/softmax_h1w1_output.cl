// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void softmax_h1w1_output(const int cd4,
    const int ce4,
    const int kn,
    __global const T *in,
    __global const T *tmp,
    __global T *out)
{
    int idx = get_global_id(0);
    if (idx >= cd4) {
        return;
    }
    T4 val;

    val = vload4(idx, in);
    float maxv = (float)(tmp[kn + 1]);
    float sumexp = (float)(tmp[kn]);

    val.x = (T)(exp((float)val.x - maxv) * sumexp);
    val.y = (T)(exp((float)val.y - maxv) * sumexp);
    val.z = (T)(exp((float)val.z - maxv) * sumexp);
    val.w = (T)(exp((float)val.w - maxv) * sumexp);

    vstore4(val, idx, out);
}