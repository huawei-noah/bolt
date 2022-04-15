R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void seqMask(
    const int iw_str,
    const int ih_str,
    const int bx,
    const int by,
    __global const T* lengths,
    __global T *out)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);

    if (idx >= bx || idy >= by) {
        return;
    }
    
    char ew = (((idx << 2) + 4) <= iw_str) ? 4 : (iw_str & 3);
    const int in_off = (idz * ih_str + idy) * iw_str + (idx << 2);
    const int j = idy * iw_str + (idx << 2);

    T4 val = 0;
    val.x = (j < (int)(lengths[idz]) * iw_str);
    val.y = (j + 1 < (int)(lengths[idz]) * iw_str);
    val.z = (j + 2 < (int)(lengths[idz]) * iw_str);
    val.w = (j + 3 < (int)(lengths[idz]) * iw_str);
    int out_str = 1;
    out[in_off] = val.x;
    if (ew > 1) {
        out[in_off + out_str] = val.y;
    }
    if (ew > 2) {
        out[in_off + out_str * 2] = val.z;
    }
    if (ew > 3) {
        out[in_off + out_str * 3] = val.w;
    }
}

)"
