R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//#include "kernel_def.h"

__kernel void init_alignment(const T ival,
    const int h,
    const int bs,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    
    if (idx >= h || idy >= bs) {
        return;
    }
    
    char ew = 0;
    ew = ((idx << 2) + 4 <= h) ? 4 : (h & 3);

    T4 val = 0.0f;
    int off = idy * h + (idx << 2);
    if (off % h == 0)
    {
        val.x = 1.0f;
    }
    if (ew == 4) {
        vstore4(val, 0, output + off);
    } else {
        if (ew == 1)
            output[off] = val.x;
        if (ew == 2) {
            vstore2((T2)(val.x, val.y), 0, output + off);
        }
        if (ew == 3) {
            vstore3((T3)(val.x, val.y, val.z), 0, output + off);
        }
    }
}
)"