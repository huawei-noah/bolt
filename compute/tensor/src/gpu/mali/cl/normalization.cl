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

#define REDUCE_VEC16(vec, res)                    \
    {                                             \
        res += vec.s0 + vec.s1 + vec.s2 + vec.s3; \
        res += vec.s4 + vec.s5 + vec.s6 + vec.s7; \
        res += vec.s8 + vec.s9 + vec.sa + vec.sb; \
        res += vec.sc + vec.sd + vec.se + vec.sf; \
    }
__kernel void MANGLE_NAME(normalization, FM)(const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
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
    float mean = 0;
    float var = 0;
    float std_val;
#if defined(USE_NCHW)
    float16 tv;
    int in_off = (idz * ih_str + idy + ih_off) * iw_str + iw_off;
    int out_off = (idz * oh_str + idy + oh_off) * ow_str + ow_off;
    for (int i = idx; i < axis_len; i += 16) {
        T val = in[i + in_off];
        float valf = val;
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
    int in_off = (idz * iw_str + iw_off) * ih_str + idx + ih_off;
    for (int i = 0; i < axis_len; ++i) {
        T4 tmp = vload4(in_off + i * ih_str, in);
        float4 tmpf;
        tmpf.x = tmp.x;
        tmpf.y = tmp.y;
        tmpf.z = tmp.z;
        tmpf.w = tmp.w;
        mean += (float)(tmpf.x + tmpf.y + tmpf.z + tmpf.w);
    }
    mean = mean * para;

    for (int i = 0; i < axis_len; ++i) {
        T4 tmp = vload4(in_off + i * ih_str, in);
        float4 tmpf;
        tmpf.x = tmp.x;
        tmpf.y = tmp.y;
        tmpf.z = tmp.z;
        tmpf.w = tmp.w;
        tmpf.x = tmpf.x - mean;
        tmpf.y = tmpf.y - mean;
        tmpf.z = tmpf.z - mean;
        tmpf.w = tmpf.w - mean;
        var += tmpf.x * tmpf.x + tmpf.y * tmpf.y + tmpf.z * tmpf.z + tmpf.w * tmpf.w;
    }
    var = var * para;

    float std_val = sqrt(var + 1e-6);
    std_val = 1.0 / std_val;
    int out_off = (idz * ow_str + ow_off) * oh_str + idx + oh_off;
    for (int i = 0; i < axis_len; ++i) {
        T4 out_val = vload4(in_off + i * ih_str, in);
        T alp = alpha[i];
        T bet = beta[i];
        out_val.x = alp * (out_val.x - mean) * std_val + bet;
        out_val.y = alp * (out_val.y - mean) * std_val + bet;
        out_val.z = alp * (out_val.z - mean) * std_val + bet;
        out_val.w = alp * (out_val.w - mean) * std_val + bet;
        vstore4(out_val, out_off + i * oh_str, out);
    }
#endif
}
