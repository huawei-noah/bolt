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

#if defined(USE_NCHW)
#define REDUCE_VEC16(vec, res)                    \
    {                                             \
        res += vec.s0 + vec.s1 + vec.s2 + vec.s3; \
        res += vec.s4 + vec.s5 + vec.s6 + vec.s7; \
        res += vec.s8 + vec.s9 + vec.sa + vec.sb; \
        res += vec.sc + vec.sd + vec.se + vec.sf; \
    }
#endif

__kernel void KERNEL_NAME(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int axis_len,
    const int bx,
    const int by,
    const float para,
    __global float *tmp,
    __global const T *alpha,
    __global const T *beta,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }
#if defined(USE_NCHW)
    float mean = 0;
    float var = 0;
    float std_val;
    float16 tv;
    int in_off = (idz * ih_str + idy) * iw_str + i_off;
    int out_off = (idz * oh_str + idy) * ow_str + o_off;
    for (int i = idx; i < axis_len; i += 16) {
        T val = in[i + in_off];
        mean += val;
    }
    int tmp_off = (idz * by + idy) * bx;
    tmp[tmp_off + idx] = mean;
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    tv = vload16(0, tmp + tmp_off);
    mean = 0;
    REDUCE_VEC16(tv, mean);
    mean = mean * para;

    for (int i = idx; i < axis_len; i += 16) {
        T val = in[i + in_off];
        float valf = val;
        valf = valf - mean;
        var += valf * valf;
    }
    tmp_off += bx * by;
    tmp[tmp_off + idx] = var;
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    tv = vload16(0, tmp + tmp_off);
    var = 0;
    REDUCE_VEC16(tv, var);
    var = var * para;

    std_val = sqrt(var + 1e-6);
    std_val = 1.0 / std_val;
    for (int i = idx; i < axis_len; i += 16) {
        T val = in[i + in_off];
        T alp = alpha[i];
        T bet = beta[i];
        val = alp * (val - mean) * std_val + bet;
        out[i + out_off] = val;
    }
#else
    float4 mean = 0;
    float4 var = 0;
    float4 std_val;
    int in_off = (idz * ih_str + idy) * iw_str + i_off;
    int out_off = (idz * oh_str + idy) * ow_str + o_off;
    for (int i = idx; i < axis_len; i += 16) {
        T4 tmp = vload4(in_off + i, in);
        mean.x += tmp.x;
        mean.y += tmp.y;
        mean.z += tmp.z;
        mean.w += tmp.w;
    }
    int tmp_off = (idz * by + idy) * bx;
    vstore4(mean, tmp_off + idx, tmp);
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    mean = 0;
    for (int i = 0; i < 16; i++) {
        mean += vload4(i, tmp + tmp_off);
    }
    mean.x = mean.x * para;
    mean.y = mean.y * para;
    mean.z = mean.z * para;
    mean.w = mean.w * para;

    for (int i = idx; i < axis_len; i += 16) {
        T4 tmp = vload4(in_off + i, in);
        float4 tmpf;
        tmpf.x = tmp.x - mean.x;
        tmpf.y = tmp.y - mean.y;
        tmpf.z = tmp.z - mean.z;
        tmpf.w = tmp.w - mean.w;
        var.x = tmpf.x * tmpf.x;
        var.y = tmpf.y * tmpf.y;
        var.z = tmpf.z * tmpf.z;
        var.w = tmpf.w * tmpf.w;
    }
    tmp_off += bx * by;
    vstore4(var, tmp_off + idx, tmp);
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    var = 0;
    for (int i = 0; i < 16; i++) {
        var += vload4(i, tmp + tmp_off);
    }
    var.x = var.x * para;
    var.y = var.y * para;
    var.z = var.z * para;
    var.w = var.w * para;
    std_val.x = 1.0 / sqrt(var.x + 1e-6);
    std_val.y = 1.0 / sqrt(var.y + 1e-6);
    std_val.z = 1.0 / sqrt(var.z + 1e-6);
    std_val.w = 1.0 / sqrt(var.w + 1e-6);
    for (int i = idx; i < axis_len; i += 16) {
        T4 val = vload4(in_off + i, in);
        T alp = alpha[i];
        T bet = beta[i];
        val.x = alp * (val.x - mean.x) * std_val.x + bet;
        val.y = alp * (val.y - mean.y) * std_val.y + bet;
        val.z = alp * (val.z - mean.z) * std_val.z + bet;
        val.w = alp * (val.w - mean.w) * std_val.w + bet;
        vstore4(val, out_off + i, out);
    }
#endif
}
