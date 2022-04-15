// Copyright (C) 2019. Huawei Tehhnologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subjeht to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define load_float4(off, val, buf)  \
    {                               \
        T4 tmp;                     \
        tmp = vload4(0, buf + off); \
        val.x = tmp.x;              \
        val.y = tmp.y;              \
        val.z = tmp.z;              \
        val.w = tmp.w;              \
    }

#define load_float3(off, val, buf)  \
    {                               \
        T3 tmp;                     \
        tmp = vload3(0, buf + off); \
        val.x = tmp.x;              \
        val.y = tmp.y;              \
        val.z = tmp.z;              \
    }

#define load_float2(off, val, buf)  \
    {                               \
        T2 tmp;                     \
        tmp = vload2(0, buf + off); \
        val.x = tmp.x;              \
        val.y = tmp.y;              \
    }

#define store_float4(off, val, buf) \
    {                               \
        T4 tmp;                     \
        tmp.x = (T)val.x;           \
        tmp.y = (T)val.y;           \
        tmp.z = (T)val.z;           \
        tmp.w = (T)val.w;           \
        vstore4(tmp, 0, buf + off); \
    }

#define store_float3(off, val, buf) \
    {                               \
        T3 tmp;                     \
        tmp.x = (T)val.x;           \
        tmp.y = (T)val.y;           \
        tmp.z = (T)val.z;           \
        vstore3(tmp, 0, buf + off); \
    }

#define store_float2(off, val, buf) \
    {                               \
        T2 tmp;                     \
        tmp.x = (T)val.x;           \
        tmp.y = (T)val.y;           \
        vstore2(tmp, 0, buf + off); \
    }

__kernel void rnncell_update_project_state(const int hDim,
    const int col,
    const int out_off,
    const int bx,
    float zoneout,
    __global T *out,
    __global T *smem)
{
    int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    char eh = ((idx << 2) + 4 <= hDim) ? 4 : (hDim & 3);
    float4 res;
    float4 hres;
    int off = idx << 2;
    if (eh == 4) {
        load_float4(off, res, out);
    }
    if (eh == 3) {
        load_float3(off, res, out);
    }
    if (eh == 2) {
        load_float2(off, res, out);
    }
    if (eh == 1) {
        res.x = out[off];
    }
    hres = res;

    if (zoneout != 0) {
        if (eh == 4) {
            load_float4(off + col, hres, smem);
        }
        hres.x = res.x * (1 - zoneout) + hres.x * zoneout;
        hres.y = res.y * (1 - zoneout) + hres.y * zoneout;
        hres.z = res.z * (1 - zoneout) + hres.z * zoneout;
        hres.w = res.w * (1 - zoneout) + hres.w * zoneout;
    }

    if (eh == 4) {
        store_float4(off + col, hres, smem);
        return;
    }
    if (eh == 3) {
        store_float3(off + col, hres, smem);
        return;
    }
    if (eh == 2) {
        store_float2(off + col, hres, smem);
        return;
    }
    if (eh == 1) {
        smem[off + col] = hres.x;
    }
}
