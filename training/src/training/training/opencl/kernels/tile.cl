R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, PATH) base##PATH
#define MANGLE_NAME(base, PATH) MANGLE_NAME_IMPL(base, PATH)
#if defined(BACKWARD)
#define PATH Backward
#else
#define PATH Forward
#endif

__kernel void MANGLE_NAME(tile, PATH)(const int bx,
    const int by,
    const int w,
    const int id_str,
    const int ih_str,
    const int iw_str,
    const int od_str,
    const int oh_str,
    const int ow_str,
    const int out_off,
    const int in_off,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
    char ew = 0;
    ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
#if defined(BACKWARD)
    int ioff = 0;
    int off = (idz * oh_str + idy) * ow_str + (idx << 2) + out_off;
    T4 val = vload4(0, out + off);
    for (int i = 0; i < id_str / od_str; ++i)
    {
        for (int j = 0; j < ih_str / oh_str; ++j)
        {
            for (int k = 0; k < iw_str / ow_str; ++k)
            {
                T4 tmp = 0;
                ioff = ((idz + i * od_str) * ih_str + (idy + j * oh_str)) * iw_str + (idx << 2) + k * ow_str + in_off;
                tmp = vload4(0, in + ioff);
                val += tmp;
            }
        }
    }
    if (ew == 4) {
        vstore4(val, 0, out + off);
    }
    else {
        if (ew == 1) {
            out[off] = val.x;
        } if (ew == 2) {
            vstore2((T2)(val.x, val.y), 0, out + off);
        } if (ew == 3) {
            vstore3((T3)(val.x, val.y, val.z), 0, out + off);
        }
    }
#else
    int ioff = ((idz % id_str) * ih_str + (idy % ih_str)) * iw_str + (idx << 2) + in_off;
    for (int k = 0; k < ow_str / iw_str; ++k)
    {
        int off = (idz * oh_str + idy) * ow_str + (idx << 2) + k * iw_str + out_off;
        T4 val = 0;
        if (ew == 4) {
            val = vload4(0, in + ioff);
        } else {
            if (ew == 1)
                val.x = in[ioff];
            if (ew == 2) {
                T2 tmp = vload2(0, in + ioff);
                val.x = tmp.x;
                val.y = tmp.y;
            }
            if (ew == 3) {
                T3 tmp = vload3(0, in + ioff);
                val.x = tmp.x;
                val.y = tmp.y;
                val.z = tmp.z;
            }
        }
        if (ew == 4) {
            vstore4(val, 0, out + off);
        }
        else {
            if (ew == 1) {
                out[off] = val.x;
            } if (ew == 2) {
                vstore2((T2)(val.x, val.y), 0, out + off);
            } if (ew == 3) {
                vstore3((T3)(val.x, val.y, val.z), 0, out + off);
            }
        }
    }
#endif
}

)"