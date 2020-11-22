// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void softmax_h1w1_sum_part(
    const int cd4, const int ce4, const int kn, __global const T *in, __global T *out)
{
    int idx = get_global_id(0);
    if (idx >= kn) {
        return;
    }

    T4 val;
    float maxval = (float)(out[kn + 1]);
    float sumexp = 0.0f;
    for (int i = idx; i < cd4 - 1; i = i + kn) {
        val = vload4(i, in);

        sumexp += exp((float)val.x - maxval);
        sumexp += exp((float)val.y - maxval);
        sumexp += exp((float)val.z - maxval);
        sumexp += exp((float)val.w - maxval);
    }

    if (idx == kn - 1) {
        val = vload4(cd4 - 1, in);
        sumexp += exp((float)val.x - maxval);
        if (ce4 >= 2) {
            sumexp += exp((float)val.y - maxval);
        }
        if (ce4 >= 3) {
            sumexp += exp((float)val.z - maxval);
        }
        if (ce4 >= 4) {
            sumexp += exp((float)val.w - maxval);
        }
    }

    out[idx] = (T)sumexp;
}