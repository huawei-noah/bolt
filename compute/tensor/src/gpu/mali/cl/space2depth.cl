// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, FM) base##FM
#define MANGLE_NAME(base, FM) MANGLE_NAME_IMPL(base, FM)

#define FM
#if defined(USE_NCHW)
#define FM _nchw
#endif

__kernel void MANGLE_NAME(space2depth, FM)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int blockSize,
    const int oc,
    const int bx,
    const int by,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }
    const int in_off = (idz * ih_str + idy) * iw_str + idx + i_off;
    const int ox = idx / blockSize;
    const int oy = idy / blockSize;
#if defined(USE_NCHW)
    T val = in[in_off];
    const int oz = (idz * blockSize + (idy % blockSize)) * blockSize + (idx % blockSize);
    const int out_off = (oz * oh_str + oy) * ow_str + ox + o_off;
    out[out_off] = val;
#else
    T4 val = vload4[in_off];
    const int oz = ((idz << 2) * blockSize + (idy % blockSize)) * blockSize + (idx % blockSize);
    const int out_off = (oz * oh_str + oy) * ow_str + ox + o_off;
    const int bs2 = blockSize * blockSize;
    const int out_str = bs2 * oh_str * ow_str;
    out[out_off] = val.x;
    if (oz + bs2 < oc) {
        out[out_off + out_str] = val.y;
    }
    if (oz + 2 * bs2 < oc) {
        out[out_off + out_str * 2] = val.z;
    }
    if (oz + 3 * bs2 < oc) {
        out[out_off + out_str * 3] = val.w;
    }
#endif
}
