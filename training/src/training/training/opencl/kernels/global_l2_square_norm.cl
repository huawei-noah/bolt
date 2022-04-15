R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void global_l2_square_norm(
    const int fullSize,
    const int loopSize,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= 1 || idy >= 1) {
        return;
    }

    T8 res;
    int stride = 8;
    int offset = 0;
    char ew;
    for (int i = 0; i < loopSize; ++i)
    {
        res = vload8(0, in + offset);
        ew = (offset + 8 <= fullSize) ? 8 : ((fullSize - offset) & 7);
        if (ew > 0)
        {
            res.s0 = res.s0 * res.s0;
        } 
        if (ew > 1)
        {
            res.s0 += res.s1 * res.s1;
        }
        if (ew > 2)
        {
            res.s0 += res.s2 * res.s2;
        } 
        if (ew > 3)
        {
            res.s0 += res.s3 * res.s3;
        }
        if (ew > 4)
        {
            res.s0 += res.s4 * res.s4;
        }
        if (ew > 5)
        {
            res.s0 += res.s5 * res.s5;
        }
        if (ew > 6)
        {
            res.s0 += res.s6 * res.s6;
        }
        if (ew > 7)
        {
            res.s0 += res.s7 * res.s7;
        }
        
        out[0] += res.s0;
        offset += stride;
    }
}
)"
