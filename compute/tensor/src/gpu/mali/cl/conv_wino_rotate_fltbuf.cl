// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, F) base##F
#define MANGLE_NAME(base, F) MANGLE_NAME_IMPL(base, F)

__kernel void MANGLE_NAME(conv_wino_rotate_fltbuf_, F)(
    const int fwhc, const int fnc, const int fn, __global const T *fltdata, __global T *fltbuf)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);

    T val = 0;
    if (idy < fn) {
        const int in_off = idy * fwhc + idx;
        val = fltdata[in_off];
    }

    const int ox = idy;
    const int oy = idx / Fsq;
    const int oz = idx % Fsq;
    const int out_off = oz * fnc + oy * fn + ox;
    fltbuf[out_off] = val;
}
