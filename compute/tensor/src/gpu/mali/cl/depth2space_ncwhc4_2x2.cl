// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void
#if defined(OUT_NCHW)
depth2space_ncwhc4_2x2_nchw
#else
depth2space_ncwhc4_2x2
#endif
    (const int blockSize,
        const int ih_str,
        const int ihw_str,
        const int ic_str,
        const int ih_off,
        const int iw_off,
        const int oh_str,
        const int ow_str,
        const int ohw_str,
        const int oh_off,
        const int ow_off,
        const int ih,
        const int iw,
        const int oc,
        __global const T *in,
        __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= ih || idy >= iw) {
        return;
    }
    const int idz = get_global_id(2);
    const int in_off = idz * 4 * ihw_str + (idy + iw_off) * ih_str + idx + ih_off;
    T4 val[4] = {0};
    T4 val_0, val_1, val_2, val_3;

    val[0] = vload4(in_off, in);
    if (idz * 4 + 1 < ic_str) {
        val[1] = vload4(in_off + ihw_str, in);
    }
    if (idz * 4 + 2 < ic_str) {
        val[2] = vload4(in_off + ihw_str * 2, in);
    }
    if (idz * 4 + 3 < ic_str) {
        val[3] = vload4(in_off + ihw_str * 3, in);
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
    const int out_off = (idz << 2) * ohw_str + ((idx << 1) + oh_off) * ow_str + (idy << 1) + ow_off;
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
    const int out_off = idz * ohw_str + ((idy << 1) + ow_off) * oh_str + (idx << 1) + oh_off;
    vstore4(val_0, out_off, out);
    vstore4(val_2, out_off + 1, out);
    vstore4(val_1, out_off + oh_str, out);
    vstore4(val_3, out_off + oh_str + 1, out);
#endif
}
