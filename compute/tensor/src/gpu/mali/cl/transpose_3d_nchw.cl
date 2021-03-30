// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void transpose_3d_nchw(const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
    const int dim0,
    const int dim1,
    const int dim2,
    const int dim3,
    const int iw,
    const int it,
    const int ot,
    const int bx,
    const int by,
    __global const T *in,
    __global T *out)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    int idt = idz % it;
    int idc = idz / it;
    if (idx >= bx || idy >= by) {
        return;
    }
    char ew = (((idx << 2) + 4) <= iw) ? 4 : (iw & 3);
    T4 val = 0;
    const int in_off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off;
    if (ew == 4) {
        val = vload4(0, in + in_off);
    } else {
        if (ew == 1) {
            val.x = in[in_off];
        }
        if (ew == 2) {
            val.xy = vload2(0, in + in_off);
        }
        if (ew == 3) {
            val.xyz = vload3(0, in + in_off);
        }
    }
    int idox = idx << 2;
    int idoy = idy;
    int idot = idt;
    int idoc = idc;

    int out_str = 1;

    if (dim0 == 1) {
        idox = idy;
    }
    if (dim0 == 2) {
        idox = idt;
    }
    if (dim0 == 3) {
        idox = idc;
    }

    if (dim1 == 0) {
        idoy = idx << 2;
        out_str = ow_str;
    }
    if (dim1 == 2) {
        idoy = idt;
    }
    if (dim1 == 3) {
        idoy = idc;
    }

    if (dim2 == 0) {
        idot = idx << 2;
        out_str = ow_str * oh_str;
    }
    if (dim2 == 1) {
        idot = idy;
    }
    if (dim2 == 3) {
        idot = idc;
    }

    if (dim3 == 0) {
        idoc = idx << 2;
        out_str = ow_str * oh_str * ot;
    }
    if (dim3 == 1) {
        idoc = idy;
    }
    if (dim3 == 2) {
        idoc = idt;
    }

    int out_off = ((idoc * ot + idot) * oh_str + idoy + oh_off) * ow_str + idox + ow_off;
    out[out_off] = val.x;
    if (ew > 1) {
        out[out_off + out_str] = val.y;
    }
    if (ew > 2) {
        out[out_off + out_str * 2] = val.z;
    }
    if (ew > 3) {
        out[out_off + out_str * 3] = val.w;
    }
}
