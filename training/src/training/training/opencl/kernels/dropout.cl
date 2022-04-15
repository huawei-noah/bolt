R"(// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"

#define LOAD_INT(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val)   \
    {                                                                           \
        int off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off; \
        val = 0;                                                                \
        if (ew == 4) {                                                          \
            val = vload4(0, buf + off);                                         \
        } else {                                                                \
            if (ew == 1)                                                        \
                val.x = buf[off];                                               \
            if (ew == 2) {                                                      \
                int2 tmp = vload2(0, buf + off);                                  \
                val.x = tmp.x;                                                  \
                val.y = tmp.y;                                                  \
            }                                                                   \
            if (ew == 3) {                                                      \
                int3 tmp = vload3(0, buf + off);                                  \
                val.x = tmp.x;                                                  \
                val.y = tmp.y;                                                  \
                val.z = tmp.z;                                                  \
            }                                                                   \
        }                                                                       \
    }

__kernel void dropoutForward(const int w, 
    const int h, 
    const int ih_str, 
    const int iw_str, 
    const int ih_off, 
    const int iw_off, 
    const int oh_str, 
    const int ow_str, 
    const int oh_off, 
    const int ow_off, 
    const T scale, 
    __global T *input, 
    __global int* condition, 
    __global T *output) 
{ 
    int idx = get_global_id(0); 
    int idy = get_global_id(1); 
    int idz = get_global_id(2); 
    if (idx >= w || idy >= h) { 
        return; 
    } 
    char ew = ((idx << 2) + 4 <= iw_str) ? 4 : (iw_str & 3); 
    int4 cond; 
    LOAD_VAL_T4(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, input, val);
	LOAD_INT(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, condition, cond);
    val.x = scale * (cond.x == 0) * val.x; 
    val.y = scale * (cond.y == 0) * val.y; 
    val.z = scale * (cond.z == 0) * val.z; 
    val.w = scale * (cond.w == 0) * val.w; 
    STORE_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, output, val);
}

__kernel void dropoutBackward(const int w, 
    const int h, 
    const int ih_str, 
    const int iw_str, 
    const int ih_off, 
    const int iw_off, 
    const int oh_str, 
    const int ow_str, 
    const int oh_off, 
    const int ow_off, 
    const T scale, 
    __global T *deltas, 
    __global int* condition, 
    __global T *prevLayerDelta) 
{ 
    int idx = get_global_id(0); 
    int idy = get_global_id(1); 
    int idz = get_global_id(2); 
    if (idx >= w || idy >= h) { 
        return; 
    } 
	char ew = ((idx << 2) + 4 <= iw_str) ? 4 : (iw_str & 3); 
    int4 cond; 
    LOAD_VAL_T4(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, deltas, del);
	LOAD_VAL_T4(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, prevLayerDelta, prev);
	LOAD_INT(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, condition, cond);
    del.x = scale * (cond.x == 0) * del.x; 
    del.y = scale * (cond.y == 0) * del.y; 
    del.z = scale * (cond.z == 0) * del.z; 
    del.w = scale * (cond.w == 0) * del.w; 
    
    del.x += prev.x; 
    del.y += prev.y; 
    del.z += prev.z; 
    del.w += prev.w; 
    STORE_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, prevLayerDelta, del);
}

)"