// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, OF, TP, AXIS) base##OF##TP##AXIS
#define MANGLE_NAME(base, OF, TP, AXIS) MANGLE_NAME_IMPL(base, OF, TP, AXIS)
#define OF
#if defined(USE_OUT_C4)
#define OF oc4_
#endif

#define SUM(v, res)     \
    {                   \
        res.s0 += v.s0; \
        res.s1 += v.s1; \
        res.s2 += v.s2; \
        res.s3 += v.s3; \
    }

#define VAR(mean, v, res)          \
    {                              \
        float4 tmp;                \
        tmp.s0 = v.s0 - mean.x;    \
        tmp.s1 = v.s1 - mean.y;    \
        tmp.s2 = v.s2 - mean.z;    \
        tmp.s3 = v.s3 - mean.w;    \
        res.s0 += tmp.s0 * tmp.s0; \
        res.s1 += tmp.s1 * tmp.s1; \
        res.s2 += tmp.s2 * tmp.s2; \
        res.s3 += tmp.s3 * tmp.s3; \
    }

__kernel void MANGLE_NAME(reduction_, OF, TP, AXIS)(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int iw,
    const int ih,
    const int ic,
    const int oc,
    const int keep_dim,
    const int od,
    const int bx,
    const int by,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    int in_off = (idz * ih_str + idy) * iw_str + idx + i_off;
#if (AXIS == 0)
    int in_str = 1;
    int loop = iw;
#elif (AXIS == 1)
    int in_str = iw_str;
    int loop = ih;
#elif (AXIS == 2)
    int in_str = ih_str * iw_str;
    int loop = (ic + 3) >> 2;
#endif
    T4 val = 0;
    float4 mean = 0;
    float4 res = 0;
    for (int i = 0; i < loop; ++i) {
        val = vload4(in_off + in_str * i, in);
#if defined(USE_SUM) || defined(USE_MEAN) || defined(USE_STD_DEVIATION)
        SUM(val, res);
#elif defined(USE_SCALAR_PRODUCT)
        VAR(mean, val, res);
#endif
    }

    float para;
#if (AXIS == 2)
    res.x = res.x + res.y + res.z + res.w;
    res.y = 0;
    res.z = 0;
    res.w = 0;
    para = 1.0 / ic;
#else
    para = 1.0 / loop;
#endif

#if defined(USE_MEAN) || defined(USE_SCALAR_PRODUCT)
    res.x = res.x * para;
#if (AXIS != 2)
    res.y = res.y * para;
    res.z = res.z * para;
    res.w = res.w * para;
#endif
#endif

#if defined(USE_STD_DEVIATION)
    mean = res;
    mean.x = mean.x * para;
#if (AXIS == 2)
    mean.y = mean.x;
    mean.z = mean.x;
    mean.w = mean.x;
#else
    mean.y = mean.y * para;
    mean.z = mean.z * para;
    mean.w = mean.w * para;
#endif
    res = 0;
    for (int i = 0; i < loop; ++i) {
        val = vload4(in_off + in_str * i, in);
        VAR(mean, val, res);
    }
#if (AXIS == 2)
    res.x = res.x + res.y + res.z + res.w;
    res.y = 0;
    res.z = 0;
    res.w = 0;
    res.x = sqrt(res.x * para);
#else
    res.x = sqrt(res.x * para);
    res.y = sqrt(res.y * para);
    res.z = sqrt(res.z * para);
    res.w = sqrt(res.w * para);
#endif
#endif

    T4 res_t;
    res_t.x = (T)res.x;
    res_t.y = (T)res.y;
    res_t.z = (T)res.z;
    res_t.w = (T)res.w;
    int out_off, out_str;

    /*input is c4, id must be 4*/
#if defined(USE_OUT_C4)
    /*reduce on axis 0/1 and keep dim, out as c4*/
    out_off = (idz * oh_str + idy) * ow_str + idx + o_off;
    vstore4(res_t, out_off, out);
#else
#if (AXIS == 0)
    /*id = 4 od = 3*/
    char ec = (((idz << 2) + 4) <= oc) ? 4 : (oc & 3);
    out_off = (idz << 2) * ow_str + idy + o_off;
    out_str = ow_str;
    out[out_off] = res_t.x;
    if (ec > 1) {
        out[out_off + out_str] = res_t.y;
    }
    if (ec > 2) {
        out[out_off + out_str * 2] = res_t.z;
    }
    if (ec > 3) {
        out[out_off + out_str * 3] = res_t.w;
    }
#elif (AXIS == 1)
    /*id = 4 od = 3*/
    char ec = (((idz << 2) + 4) <= oc) ? 4 : (oc & 3);
    out_off = (idz << 2) * ow_str + idx + o_off;
    out_str = ow_str;
    out[out_off] = res_t.x;
    if (ec > 1) {
        out[out_off + out_str] = res_t.y;
    }
    if (ec > 2) {
        out[out_off + out_str * 2] = res_t.z;
    }
    if (ec > 3) {
        out[out_off + out_str * 3] = res_t.w;
    }
#elif (AXIS == 2)
    /*id = 4 od = 4 / 3*/
    out_off = idy * ow_str + idx + o_off;
    out[out_off] = res_t.x;
#endif
#endif
}
