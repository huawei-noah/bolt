// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, DT) base##DT
#define MANGLE_NAME(base, DT) MANGLE_NAME_IMPL(base, DT)
__kernel void MANGLE_NAME(fill_memory_zero_vec4_, DT)(
    const int len, const int offset, const int bx, __global T *data)
{
    int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    char el = ((idx << 2) + 4 <= len) ? 4 : (len & 3);
    const int off = offset + (idx << 2);
    if (el == 4) {
        vstore4((T4)0, 0, data + off);
    } else {
        if (el == 1) {
            data[off] = 0;
        }
        if (el == 2) {
            vstore2((T2)0, 0, data + off);
        }
        if (el == 3) {
            vstore3((T3)0, 0, data + off);
        }
    }
}
