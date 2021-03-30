// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

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

__kernel void rnncell_update_res(const int col,
    const uchar noproject,
    const int out_off,
    const int bx,
    float fbias,
    float zonecell,
    float zoneout,
    __global T *smem,
    __global T *imem,
    __global T *out)
{
    int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }
    char ec = ((idx << 2) + 4 <= col) ? 4 : (col & 3);
    float4 cval;
    float4 lcval;
    float4 ival;
    float4 gval;
    float4 fval;
    float4 oval;
    float4 res;
    float4 hres;
    int off = idx << 2;
    load_float4(off, cval, smem);
    load_float4(off, ival, imem);
    load_float4(off + col, gval, imem);
    load_float4(off + col * 2, fval, imem);
    load_float4(off + col * 3, oval, imem);
    ival.x = 1.0 / (1.0 + exp(-ival.x));
    ival.y = 1.0 / (1.0 + exp(-ival.y));
    ival.z = 1.0 / (1.0 + exp(-ival.z));
    ival.w = 1.0 / (1.0 + exp(-ival.w));
    gval.x = tanh(gval.x);
    gval.y = tanh(gval.y);
    gval.z = tanh(gval.z);
    gval.w = tanh(gval.w);
    fval.x = 1.0 / (1.0 + exp(-(fval.x + fbias)));
    fval.y = 1.0 / (1.0 + exp(-(fval.y + fbias)));
    fval.z = 1.0 / (1.0 + exp(-(fval.z + fbias)));
    fval.w = 1.0 / (1.0 + exp(-(fval.w + fbias)));
    oval.x = 1.0 / (1.0 + exp(-oval.x));
    oval.y = 1.0 / (1.0 + exp(-oval.y));
    oval.z = 1.0 / (1.0 + exp(-oval.z));
    oval.w = 1.0 / (1.0 + exp(-oval.w));
    lcval = cval;
    cval.x = cval.x * fval.x + ival.x * gval.x;
    cval.y = cval.y * fval.y + ival.y * gval.y;
    cval.z = cval.z * fval.z + ival.z * gval.z;
    cval.w = cval.w * fval.w + ival.w * gval.w;
    res.x = oval.x * tanh(cval.x);
    res.y = oval.y * tanh(cval.y);
    res.z = oval.z * tanh(cval.z);
    res.w = oval.w * tanh(cval.w);
    hres = res;

    if (zonecell != 0) {
        cval.x = cval.x * (1 - zonecell) + lcval.x * zonecell;
        cval.y = cval.y * (1 - zonecell) + lcval.y * zonecell;
        cval.z = cval.z * (1 - zonecell) + lcval.z * zonecell;
        cval.w = cval.w * (1 - zonecell) + lcval.w * zonecell;
    }

    if (zoneout != 0 && noproject) {
        load_float4(off + col, hres, smem);
        hres.x = res.x * (1 - zoneout) + hres.x * zoneout;
        hres.y = res.y * (1 - zoneout) + hres.y * zoneout;
        hres.z = res.z * (1 - zoneout) + hres.z * zoneout;
        hres.w = res.w * (1 - zoneout) + hres.w * zoneout;
    }

    if (ec == 4) {
        store_float4(off, cval, smem);
        store_float4(off + out_off, res, out);
        if (noproject) {
            store_float4(off + col, hres, smem);
        }
    } else {
        if (ec == 1) {
            smem[off] = (T)cval.x;
            out[off + out_off] = (T)res.x;
            if (noproject) {
                smem[off + col] = (T)hres.x;
            }
        }
        if (ec == 2) {
            store_float2(off, cval, smem);
            store_float2(off + out_off, res, out);
            if (noproject) {
                store_float2(off + col, hres, smem);
            }
        }
        if (ec == 3) {
            store_float3(off, cval, smem);
            store_float3(off + out_off, res, out);
            if (noproject) {
                store_float3(off + col, hres, smem);
            }
        }
    }
}
