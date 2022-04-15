R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"

__kernel void iota(const T start,
    const int len,
    __global T *output)
{
    int idx = get_global_id(0);
    
    if (idx >= len) {
        return;
    }

    const int x_off = idx << 2;
    output[x_off] = start + (T)x_off;

    if (x_off + 1 >= len) {
        return;
    }
    output[x_off + 1] = start + (T)(x_off + 1);
    if (x_off + 2 >= len) {
        return;
    }
    output[x_off + 2] = start + (T)(x_off + 2);
    if (x_off + 3 >= len) {
        return;
    }
    output[x_off + 3] = start + (T)(x_off + 3);
}
)"