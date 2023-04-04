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

#ifdef USE_INPUT_IMG
#define LOAD_INPUT LOAD_MEM_V4(v, (int4)(i, idx, id, 0), input)
#else
#define LOAD_INPUT LOAD_MEM_V4(v, in_off + i, input)
#endif
#ifdef USE_OUTPUT_IMG
#define STORE_OUTPUT STORE_MEM_V4(v, (int4)(i, idx, id, 0), output)
#else
#define STORE_OUTPUT STORE_MEM_V4(v, out_off + i, output)
#endif

__kernel void KERNEL_NAME(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int iw,
    const int ih,
    const int ic,
    const int in,
    const int inch,
    const float para,
    __global float *tmp,
    __global const T *alpha,
    __global const T *beta,
    READ_ONLY_KERNEL_MEM input,
    KERNEL_MEM output)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= ih || idy >= ic || idz >= in) {
        return;
    }
    int id = idz * ic + idy;
    int in_off = (id * ih_str + idx) * iw_str + i_off;
    int out_off = (id * oh_str + idx) * ow_str + o_off;
    int tmp_off = id * ih;
    const float eps = 1e-6;
#if defined(USE_NCHW)
    float mean = 0;
    for (int i = in_off; i < in_off + iw; i++) {
        mean += input[i];
    }
    tmp[tmp_off + idx] = mean;
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    mean = 0;
    for (int i = tmp_off; i < tmp_off + ih; i++) {
        mean += tmp[i];
    }
    mean = mean * para;
    float var = 0;
    for (int i = in_off; i < in_off + iw; i++) {
        float v = input[i];
        v = v - mean;
        var += v * v;
    }
    tmp_off += inch;
    tmp[tmp_off + idx] = var;
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    var = 0;
    for (int i = tmp_off; i < tmp_off + ih; i++) {
        var += tmp[i];
    }
    float s = alpha[idy] / sqrt(FMA(var, para, eps));
    float b = beta[idy] - s * mean;
    for (int i = 0; i < iw; i++) {
        float v = input[i + in_off];
        output[i + out_off] = FMA(s, v, b);
        ;
    }
#else
    float4 mean = 0;
    T4 v;
    for (int i = 0; i < iw; i++) {
        LOAD_INPUT

        mean.x += v.x;
        mean.y += v.y;
        mean.z += v.z;
        mean.w += v.w;
    }
    vstore4(mean, tmp_off + idx, tmp);
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    mean = 0;
    for (int i = tmp_off; i < tmp_off + ih; i++) {
        mean += vload4(i, tmp);
    }
    mean = mean * para;
    float4 var = 0, r;
    for (int i = 0; i < iw; i++) {
        LOAD_INPUT
        r.x = v.x - mean.x;
        r.y = v.y - mean.y;
        r.z = v.z - mean.z;
        r.w = v.w - mean.w;
        var += r * r;
    }
    tmp_off += inch;
    vstore4(var, tmp_off + idx, tmp);
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    var = 0;
    for (int i = tmp_off; i < tmp_off + ih; i++) {
        var += vload4(i, tmp);
    }
    T4 s = vload4(idy, alpha);
    T4 b = vload4(idy, beta);
    s.x = s.x / sqrt(FMA(var.x, para, eps));
    s.y = s.y / sqrt(FMA(var.y, para, eps));
    s.z = s.z / sqrt(FMA(var.z, para, eps));
    s.w = s.w / sqrt(FMA(var.w, para, eps));
    b.x = b.x - s.x * mean.x;
    b.y = b.y - s.y * mean.y;
    b.z = b.z - s.z * mean.z;
    b.w = b.w - s.w * mean.w;
    for (int i = 0; i < iw; i++) {
        LOAD_INPUT
        v = FMA(s, v, b);
        STORE_OUTPUT
    }
#endif
}
