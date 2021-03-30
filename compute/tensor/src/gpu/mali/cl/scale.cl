// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define MANGLE_NAME_IMPL(base, FM, ALPHA, BETA) base##FM##ALPHA##BETA
#define MANGLE_NAME(base, FM, ALPHA, BETA) MANGLE_NAME_IMPL(base, FM, ALPHA, BETA)

#define FM
#define ALPHA
#define BETA

#if defined(USE_NCHW)
#define FM _nchw
#endif

#if defined(USE_ALPHA)
#define ALPHA _alpha
#endif

#if defined(USE_BETA)
#define BETA _beta
#endif

__kernel void MANGLE_NAME(scale, FM, ALPHA, BETA)(const int w,
    const int ih_str,
    const int iw_str,
    const int ih_off,
    const int iw_off,
    const int oh_str,
    const int ow_str,
    const int oh_off,
    const int ow_off,
    const int bx,
    const int by,
    __global const T *alpha,
    __global const T *beta,
    __global T *input,
    __global T *output)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    T4 val;
#if defined(USE_NCHW)
    T alp = 1.0;
    T bet = 0.0;
#if defined(USE_ALPHA)
    alp = alpha[idz];
#endif
#if defined(USE_BETA)
    bet = beta[idz];
#endif
    int in_off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off;
    int out_off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off;
    val = vload4(0, input + in_off);
    val.s0 = val.s0 * alp + bet;
    val.s1 = val.s1 * alp + bet;
    val.s2 = val.s2 * alp + bet;
    val.s3 = val.s3 * alp + bet;
#else
    T4 alp = 1.0;
    T4 bet = 0.0;
#if defined(USE_ALPHA)
    alp = vload4(idz, alpha);
#endif
#if defined(USE_BETA)
    bet = vload4(idz, beta);
#endif
    int in_off = (idz * iw_str + idy + iw_off) * ih_str + idx + ih_off;
    int out_off = (idz * ow_str + idy + ow_off) * oh_str + idx + oh_off;
    val = vload4(in_off, input);
    val.s0 = val.s0 * alp.x + bet.x;
    val.s1 = val.s1 * alp.y + bet.y;
    val.s2 = val.s2 * alp.z + bet.z;
    val.s3 = val.s3 * alp.w + bet.w;
#endif

#if defined(USE_NCHW)
    char ew = ((idx << 2) + 4 <= w) ? 4 : (w & 3);
    if (ew == 4) {
        vstore4(val, 0, output + out_off);
    } else {
        if (ew == 3) {
            vstore3(val.xyz, 0, output + out_off);
        } else if (ew == 2) {
            vstore2(val.xy, 0, output + out_off);
        } else if (ew == 1) {
            output[out_off] = val.x;
        }
    }
#else
    vstore4(val, out_off, output);
#endif
}
