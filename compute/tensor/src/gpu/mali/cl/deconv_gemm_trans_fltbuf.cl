// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, C, K) base##C##K
#define MANGLE_NAME(base, C, K) MANGLE_NAME_IMPL(base, C, K)
__kernel void MANGLE_NAME(deconv_gemm_trans_fltbuf_, C, K)(const int fw,
    const int fwh,
    const int fwhc,
    const int fc,
    const int fn,
    __global const T *fltdata,
    __global T *fltbuf)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);  // (fn + 3) / 4;
    const int idx_wh = idx % fwh;      // fwh
    const int idx_c = idx / fwh;       // (fc + 3) / 4;
    uchar ec = ((idx_c + 1) * 4 <= fc) ? 4 : (fc % 4);
    uchar ek = ((idy + 1) * K <= fn) ? K : (fn % K);

    T16 val = 0;
    int flt_off = idy * fwhc * 4 + idx_c * fwh * 4 + idx_wh;
    val.s0 = fltdata[flt_off];
    if (ec > 1) {
        val.s4 = fltdata[flt_off + fwh];
    }
    if (ec > 2) {
        val.s8 = fltdata[flt_off + fwh * 2];
    }
    if (ec > 3) {
        val.sc = fltdata[flt_off + fwh * 3];
    }

    if (ek > 1) {
        flt_off += fwhc;
        val.s1 = fltdata[flt_off];
        if (ec > 1) {
            val.s5 = fltdata[flt_off + fwh];
        }
        if (ec > 2) {
            val.s9 = fltdata[flt_off + fwh * 2];
        }
        if (ec > 3) {
            val.sd = fltdata[flt_off + fwh * 3];
        }
    }

    if (ek > 2) {
        flt_off += fwhc;
        val.s2 = fltdata[flt_off];
        if (ec > 1) {
            val.s6 = fltdata[flt_off + fwh];
        }
        if (ec > 2) {
            val.sa = fltdata[flt_off + fwh * 2];
        }
        if (ec > 3) {
            val.se = fltdata[flt_off + fwh * 3];
        }
    }

    if (ek > 3) {
        flt_off += fwhc;
        val.s3 = fltdata[flt_off];
        if (ec > 1) {
            val.s7 = fltdata[flt_off + fwh];
        }
        if (ec > 2) {
            val.sb = fltdata[flt_off + fwh * 2];
        }
        if (ec > 3) {
            val.sf = fltdata[flt_off + fwh * 3];
        }
    }

    /*C = 1 C = 2 C = 4*/
    const int idx_w = idx_wh % fw;
    const int idx_h = idx_wh / fw;
    const int idx_tran = idx_c * fwh + idx_w * fw + idx_h;
    int out_off = (idx_tran / C) * ((fn + 3) >> 2) * C + idy * C + (idx_tran % C);
    vstore16(val, out_off, fltbuf);
}
