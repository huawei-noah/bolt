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
#define MANGLE_NAME_IMPL(base, IOM, FM) base##IOM##FM
#define MANGLE_NAME(base, IOM, FM) MANGLE_NAME_IMPL(base, IOM, FM)
#define FM
#if defined(USE_NCHW)
#define FM nchw_
#endif

__kernel void MANGLE_NAME(channel_resize_, IOM, FM)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int ic,
    const int oc,
    const int w,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM input,
    KERNEL_MEM output)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 val = 0;
#if defined(USE_NCHW)
    const int idc = idz % oc;
    const int idn = idz / oc;
    if (idc < ic) {
        LOAD_MEM_V4_C1_COMMON(val, idx, idy, (idc + idn * ic), iw_str, ih_str, i_off, w, input);
    }
    STORE_MEM_V4_C1_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, w, output);
#else
    const int ic_str = (ic + 3) >> 2;
    const int oc_str = (oc + 3) >> 2;
    const int idc = idz % oc_str;
    const int idn = idz / oc_str;
    if (idc < ic_str) {
        LOAD_MEM_V4_COMMON(val, idx, idy, (idc + idn * ic_str), iw_str, ih_str, i_off, input);
    }
    STORE_MEM_V4_COMMON(val, idx, idy, idz, ow_str, oh_str, o_off, output);
#endif
}
