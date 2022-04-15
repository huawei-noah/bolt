// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, BIND, DT) base##BIND##DT
#define MANGLE_NAME(base, BIND, DT) MANGLE_NAME_IMPL(base, BIND, DT)

#define BIND
#if defined(USE_BLOCK_INDEX)
#define BIND with_block_index_
#endif

__kernel void MANGLE_NAME(copy_, BIND, DT)(const int s_len,
    const int d_len,
    int s_off,
    int d_off,
    const int bx,
#if defined(USE_BLOCK_INDEX)
    const int s_str,
    const int d_str,
    __global const int *srcBlockIndex,
    __global const int *dstBlockIndex,
#endif
    __global const T *src,
    __global T *dst)
{
    int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    char s_ex = (((idx << 2) + 4) <= s_len) ? 4 : (s_len & 3);
    char d_ex = (((idx << 2) + 4) <= d_len) ? 4 : (d_len & 3);
    if ((idx << 2) >= s_len) {
        s_ex = 0;
    }
    if ((idx << 2) >= d_len) {
        d_ex = 0;
    }
#if defined(USE_BLOCK_INDEX)
    s_off = s_off + s_str * srcBlockIndex[0];
    d_off = d_off + d_str * dstBlockIndex[0];
#endif
    int src_off = s_off + (idx << 2);
    int dst_off = d_off + (idx << 2);

    T4 val = 0;
    if (s_ex == 4) {
        val = vload4(0, src + src_off);
    } else {
        if (s_ex == 1) {
            val.x = src[src_off];
        }
        if (s_ex == 2) {
            val.xy = vload2(0, src + src_off);
        }
        if (s_ex == 3) {
            val.xyz = vload3(0, src + src_off);
        }
    }

    if (d_ex == 4) {
        vstore4(val, 0, dst + dst_off);
    } else {
        if (d_ex == 1) {
            dst[dst_off] = val.x;
        }
        if (d_ex == 2) {
            vstore2(val.xy, 0, dst + dst_off);
        }
        if (d_ex == 3) {
            vstore3(val.xyz, 0, dst + dst_off);
        }
    }
}
