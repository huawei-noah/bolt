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
#define MANGLE_NAME_IMPL(base, IOM) base##IOM
#define MANGLE_NAME(base, IOM) MANGLE_NAME_IMPL(base, IOM)
#if defined(USE_INPUT_IMG)
#define loadH(val, y_off, pic)                                                   \
    {                                                                            \
        val.s0123 = READ_IMAGE(pic, sampler, (int4)(idx, iy + y_off, idz, 0));     \
        val.s4567 = READ_IMAGE(pic, sampler, (int4)(idx + 1, iy + y_off, idz, 0)); \
    }
#else
#define loadH(val, y_off, pic)                                  \
    {                                                           \
        val.s0123 = vload4(0, pic + in_off + y_off * iw_str);     \
        val.s45   = vload2(0, pic + in_off + y_off * iw_str + 4); \
    }
#endif

__kernel void MANGLE_NAME(conv_wino_trans_picbuf_nchw_, IOM)(const int iw_str,
    const int ih_str,
    const int i_off,
    const int pw_str,
    const int pwh_str,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    __global T *pictran)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    if (idx >= bx || idy >= by) {
        return;
    }

    int iy = idy << 2;
#if !defined(USE_INPUT_IMG)
    const int in_off = (idz * ih_str + iy) * iw_str + (idx << 2) + i_off;
#endif
    const int pictran_off = idz * pw_str + idy * bx + idx;
    T16 tmp;
    T8 h0, h1, h2, h3, h4, h5;

    loadH(h0, 0, in);
    loadH(h1, 1, in);
    loadH(h2, 2, in);
    loadH(h3, 3, in);
    loadH(h4, 4, in);
    loadH(h5, 5, in);

    h0.s1234 = FMA(4, h0.s1234, FMA(-5, h2.s1234, h4.s1234));
    h5.s1234 = FMA(4, h1.s1234, FMA(-5, h3.s1234, h5.s1234));
    tmp.s0123 = FMA(-4, h1.s1234 + h2.s1234, h4.s1234) + h3.s1234;
    tmp.s4567 = FMA( 4, h1.s1234 - h2.s1234, h4.s1234) - h3.s1234;
    tmp.s89ab = FMA( 2, h3.s1234 - h1.s1234, h4.s1234) - h2.s1234;
    tmp.scdef = FMA( 2, h1.s1234 - h3.s1234, h4.s1234) - h2.s1234;
    pictran += pictran_off;
    *pictran = FMA(16, h0.s0, FMA(-20, h2.s0, FMA(4, h4.s0, FMA(-5, h0.s2, h0.s4))));
    pictran += pwh_str;
    *pictran = FMA(-4, (h0.s1 + h0.s2), h0.s4) + h0.s3;
    pictran += pwh_str;
    *pictran = FMA( 4, (h0.s1 - h0.s2), h0.s4) - h0.s3;
    pictran += pwh_str;
    *pictran = FMA( 2, (h0.s3 - h0.s1), h0.s4) - h0.s2;
    pictran += pwh_str;
    *pictran = FMA( 2, (h0.s1 - h0.s3), h0.s4) - h0.s2;
    pictran += pwh_str;
    *pictran = FMA( 4, (h0.s1 + h0.s5), FMA(-5, (h0.s3 + h2.s5), h4.s5));
    pictran += pwh_str;
    *pictran = FMA(-16, (h1.s0 + h2.s0), FMA(4, (h3.s0 + h4.s0), FMA(-5, tmp.s1, tmp.s3)));
    pictran += pwh_str;
    *pictran = FMA(-4, (tmp.s0 + tmp.s1), tmp.s3) + tmp.s2;
    pictran += pwh_str;
    *pictran = FMA( 4, (tmp.s0 - tmp.s1), tmp.s3) - tmp.s2;
    pictran += pwh_str;
    *pictran = FMA( 2, (tmp.s2 - tmp.s0), tmp.s3) - tmp.s1;
    pictran += pwh_str;
    *pictran = FMA( 2, (tmp.s0 - tmp.s2), tmp.s3) - tmp.s1;
    pictran += pwh_str;
    *pictran = FMA( 4, (tmp.s0 - h1.s5 - h2.s5), FMA(-5, tmp.s2, h3.s5)) + h4.s5;
    pictran += pwh_str;
    *pictran = FMA(16, (h1.s0 - h2.s0), FMA(4, (h4.s0 - h3.s0), FMA(-5, tmp.s5, tmp.s7)));
    pictran += pwh_str;
    *pictran = FMA(-4, (tmp.s4 + tmp.s5), tmp.s7) + tmp.s6;
    pictran += pwh_str;
    *pictran = FMA( 4, (tmp.s4 - tmp.s5), tmp.s7) - tmp.s6;
    pictran += pwh_str;
    *pictran = FMA( 2, (tmp.s6 - tmp.s4), tmp.s7) - tmp.s5;
    pictran += pwh_str;
    *pictran = FMA( 2, (tmp.s4 - tmp.s6), tmp.s7) - tmp.s5;
    pictran += pwh_str;
    *pictran = FMA( 4, (tmp.s4 + h1.s5 - h2.s5), FMA(-5, tmp.s6, - h3.s5)) + h4.s5;
    pictran += pwh_str;
    *pictran = FMA( 8, (h3.s0 - h1.s0), FMA(4, (h4.s0 - h2.s0), FMA(-5, tmp.s9, tmp.sb)));
    pictran += pwh_str;
    *pictran = FMA(-4, (tmp.s8 + tmp.s9), tmp.sb) + tmp.sa;
    pictran += pwh_str;
    *pictran = FMA( 4, (tmp.s8 - tmp.s9), tmp.sb) - tmp.sa;
    pictran += pwh_str;
    *pictran = FMA( 2, (tmp.sa - tmp.s8), tmp.sb) - tmp.s9;
    pictran += pwh_str;
    *pictran = FMA( 2, (tmp.s8 - tmp.sa), tmp.sb) - tmp.s9;
    pictran += pwh_str;
    *pictran = FMA( 4, tmp.s8, FMA(2, (h3.s5 - h1.s5), - h2.s5)) + FMA(-5, tmp.sa, h4.s5);
    pictran += pwh_str;
    *pictran = FMA( 8, (h1.s0 - h3.s0),  FMA(4, (h4.s0 - h2.s0), FMA(-5, tmp.sd, tmp.sf)));
    pictran += pwh_str;
    *pictran = FMA(-4, (tmp.sc + tmp.sd), tmp.sf) + tmp.se;
    pictran += pwh_str;
    *pictran = FMA( 4, (tmp.sc - tmp.sd), tmp.sf) - tmp.se;
    pictran += pwh_str;
    *pictran = FMA( 2, (tmp.se - tmp.sc), tmp.sf) - tmp.sd;
    pictran += pwh_str;
    *pictran = FMA( 2, (tmp.sc - tmp.se), tmp.sf) - tmp.sd;
    pictran += pwh_str;
    *pictran = FMA( 4, tmp.sc, FMA(2, (h1.s5 - h3.s5), - h2.s5)) + FMA(-5, tmp.se, h4.s5);
    pictran += pwh_str;
    *pictran = FMA(16, h1.s0, FMA(-20, h3.s0, FMA(4, h5.s0, FMA(-5, h5.s2, h5.s4))));
    pictran += pwh_str;
    *pictran = FMA(-4, (h5.s1 + h5.s2), h5.s4) + h5.s3;
    pictran += pwh_str;
    *pictran = FMA( 4, (h5.s1 - h5.s2), h5.s4) - h5.s3;
    pictran += pwh_str;
    *pictran = FMA( 2, (h5.s3 - h5.s1), h5.s4) - h5.s2;
    pictran += pwh_str;
    *pictran = FMA( 2, (h5.s1 - h5.s3), h5.s4) - h5.s2;
    pictran += pwh_str;
    *pictran = FMA( 4, (h5.s1 + h1.s5), FMA(-5, (h5.s3 + h3.s5), h5.s5));
}
