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
#define MANGLE_NAME_IMPL(base, IOM) base##IOM
#define MANGLE_NAME(base, IOM) MANGLE_NAME_IMPL(base, IOM)

#define SWAP_VAL(av, bv) \
    {                    \
        T tv = av;       \
        av = bv;         \
        bv = tv;         \
    }

/*trans wh for set T to N or set N to T*/
__kernel void MANGLE_NAME(matmul_trans_input_, IOM)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int iw,
    const int ih,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 v0 = 0;
    T4 v1 = 0;
    T4 v2 = 0;
    T4 v3 = 0;

    int y_off = idy << 2;
    LOAD_MEM_V4_C1_COMMON(v0, idx, y_off, idz, iw_str, ih_str, i_off, iw, in);
    if (y_off + 1 < ih) {
        LOAD_MEM_V4_C1_COMMON(v1, idx, y_off + 1, idz, iw_str, ih_str, i_off, iw, in);
    }
    if (y_off + 2 < ih) {
        LOAD_MEM_V4_C1_COMMON(v2, idx, y_off + 2, idz, iw_str, ih_str, i_off, iw, in);
    }
    if (y_off + 3 < ih) {
        LOAD_MEM_V4_C1_COMMON(v3, idx, y_off + 3, idz, iw_str, ih_str, i_off, iw, in);
    }
    SWAP_VAL(v0.s1, v1.s0);
    SWAP_VAL(v0.s2, v2.s0);
    SWAP_VAL(v0.s3, v3.s0);
    SWAP_VAL(v1.s2, v2.s1);
    SWAP_VAL(v1.s3, v3.s1);
    SWAP_VAL(v2.s3, v3.s2);

    int x_off = idx << 2;
    STORE_MEM_V4_C1_COMMON(v0, idy, x_off, idz, ow_str, oh_str, o_off, ih, out);
    if (x_off + 1 < iw) {
        STORE_MEM_V4_C1_COMMON(v1, idy, x_off + 1, idz, ow_str, oh_str, o_off, ih, out);
    }
    if (x_off + 2 < iw) {
        STORE_MEM_V4_C1_COMMON(v2, idy, x_off + 2, idz, ow_str, oh_str, o_off, ih, out);
    }
    if (x_off + 3 < iw) {
        STORE_MEM_V4_C1_COMMON(v3, idy, x_off + 3, idz, ow_str, oh_str, o_off, ih, out);
    }
}
