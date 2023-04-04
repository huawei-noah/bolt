// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"

__kernel void KERNEL_NAME(
    const int len, const int offset, const int bx, const int by, KERNEL_MEM data)
{
#if defined(USE_OUTPUT_IMG)
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    int idz = get_global_id(2);
    T4 val = (T4)0;
    STORE_MEM_V4(val, (int4)(idx, idy, idz, 0), data);
#else
    int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    char el = ((idx << 2) + 4 <= len) ? 4 : (len & 3);
    const int off = offset + (idx << 2);
    T4 val = (T4)0;
    STORE_MEM_V4_C1(val, off, el, data);
#endif
}
