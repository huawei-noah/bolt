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
#define MANGLE_NAME_IMPL(base, IOM, OFM) base##IOM##OFM
#define MANGLE_NAME(base, IOM, OFM) MANGLE_NAME_IMPL(base, IOM, OFM)

#define OFM
#if defined(OUT_NCHW)
#define OFM nchw
#endif

__kernel void MANGLE_NAME(depth2space_nchwc4_2x2_, IOM, OFM)(const int blockSize,
    const int iw_str,
    const int ihw_str,
    const int ic_str,
    const int i_off,
    const int ow_str,
    const int oh_str,
    const int ohw_str,
    const int o_off,
    const int iw,
    const int ih,
    const int oc,
    READ_ONLY_KERNEL_MEM input,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= iw || idy >= ih) {
        return;
    }
    const int idz = get_global_id(2);
    T4 val[4] = {0};
    T4 val_0, val_1, val_2, val_3;

    LOAD_MEM_V4_COMMON(val[0], idx, idy, idz * 4, iw_str, ih_str, i_off, input);
    if (idz * 4 + 1 < ic_str) {
        LOAD_MEM_V4_COMMON(val[1], idx, idy, (idz * 4 + 1), iw_str, ih_str, i_off, input);
    }
    if (idz * 4 + 2 < ic_str) {
        LOAD_MEM_V4_COMMON(val[2], idx, idy, (idz * 4 + 2), iw_str, ih_str, i_off, input);
    }
    if (idz * 4 + 3 < ic_str) {
        LOAD_MEM_V4_COMMON(val[3], idx, idy, (idz * 4 + 3), iw_str, ih_str, i_off, input);
    }

    val_0.x = val[0].x;
    val_1.x = val[0].y;
    val_2.x = val[0].z;
    val_3.x = val[0].w;

    val_0.y = val[1].x;
    val_1.y = val[1].y;
    val_2.y = val[1].z;
    val_3.y = val[1].w;

    val_0.z = val[2].x;
    val_1.z = val[2].y;
    val_2.z = val[2].z;
    val_3.z = val[2].w;

    val_0.w = val[3].x;
    val_1.w = val[3].y;
    val_2.w = val[3].z;
    val_3.w = val[3].w;

#if defined(OUT_NCHW)
    char ez = (((idz << 2) + 4) <= oc) ? 4 : (oc & 3);
    const int out_off = (idz << 2) * ohw_str + (idy << 1) * ow_str + (idx << 1) + o_off;
    vstore2((T2)(val_0.x, val_1.x), 0, out + out_off);
    vstore2((T2)(val_2.x, val_3.x), 0, out + out_off + ow_str);
    if (ez > 1) {
        vstore2((T2)(val_0.y, val_1.y), 0, out + out_off + ohw_str);
        vstore2((T2)(val_2.y, val_3.y), 0, out + out_off + ohw_str + ow_str);
    }
    if (ez > 2) {
        vstore2((T2)(val_0.z, val_1.z), 0, out + out_off + ohw_str * 2);
        vstore2((T2)(val_2.z, val_3.z), 0, out + out_off + ohw_str * 2 + ow_str);
    }
    if (ez > 3) {
        vstore2((T2)(val_0.w, val_1.w), 0, out + out_off + ohw_str * 3);
        vstore2((T2)(val_2.w, val_3.w), 0, out + out_off + ohw_str * 3 + ow_str);
    }
#else
    const int out_off = idz * ohw_str + (idy << 1) * ow_str + (idx << 1) + o_off;
    vstore4(val_0, out_off, out);
    vstore4(val_2, out_off + 1, out);
    vstore4(val_1, out_off + ow_str, out);
    vstore4(val_3, out_off + ow_str + 1, out);
#endif
}
