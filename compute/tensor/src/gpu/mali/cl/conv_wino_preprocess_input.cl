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
#define FM nchw
#endif
__kernel void MANGLE_NAME(conv_wino_preprocess_input_, IOM, FM)(const int iw_str,
    const int ih_str,
    const int i_off,
    const int ow_str,
    const int oh_str,
    const int iw,
    const int ih,
    const int ic,
    const int pw,
    const int ph,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
#if defined(USE_NCHW) 
{   
    //if input is image and nchw, trans to buffer for add padding
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    int ix = idx - pw;
    int iy = idy - ph;
    if (ix < 0 || ix >= iw || iy < 0 || iy >= ih) {
        int out_off = (idz * oh_str + idy) * ow_str + idx;
        out[out_off] = 0;//add padding
    } else {
        T4 val = READ_IMAGE(in, sampler, (int4)(ix, iy, idz, 0));
        int out_off = (idz * oh_str + idy + ph) * ow_str + (idx << 2) + pw;
        vstore4(val, 0, out_off + out);
    }
}
#else
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    T4 val[4] = {0};
    int ix = (idx << 2) - pw;
    int iy = idy - ph;
    if (iy >= 0 && iy < ih) {
        if (ix >= 0 && ix < iw) {
            LOAD_MEM_V4_COMMON(val[0], ix, iy, idz, iw_str, ih_str, i_off, in);  
        }
        if (ix + 1 >= 0 && ix + 1 < iw) {
            LOAD_MEM_V4_COMMON(val[1], (ix + 1), iy, idz, iw_str, ih_str, i_off, in);  
        }
        if (ix + 2 >= 0 && ix + 2 < iw) {
            LOAD_MEM_V4_COMMON(val[2], (ix + 2), iy, idz, iw_str, ih_str, i_off, in);  
        }
        if (ix + 3 >= 0 && ix + 3 < iw) {
            LOAD_MEM_V4_COMMON(val[3], (ix + 3), iy, idz, iw_str, ih_str, i_off, in);  
        }
    }
    int oz = idz << 2;
    T4 v = (val[0].x, val[1].x, val[2].x, val[3].x);
    STORE_MEM_V4_C1_COMMON(v,
        idx, idy, oz, ow_str, oh_str, 0, ow_str, out);
    if (oz + 1 < ic) {
        T4 v = (val[0].y, val[1].y, val[2].y, val[3].y);
        STORE_MEM_V4_C1_COMMON(v,
            idx, idy, (oz + 1), ow_str, oh_str, 0, ow_str, out);
    }
    if (oz + 2 < ic) {
        T4 v = (val[0].z, val[1].z, val[2].z, val[3].z);
        STORE_MEM_V4_C1_COMMON(v,
            idx, idy, (oz + 2), ow_str, oh_str, 0, ow_str, out);
    }
    if (oz + 3 < ic) {
        T4 v = (val[0].w, val[1].w, val[2].w, val[3].w);
        STORE_MEM_V4_C1_COMMON(v,
            idx, idy, (oz + 3), ow_str, oh_str, 0, ow_str, out);
    }
}
#endif
