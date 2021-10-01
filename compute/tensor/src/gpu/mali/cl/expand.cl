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
#define MANGLE_NAME_IMPL(base, IOM, DN) base##IOM##DN
#define MANGLE_NAME(base, IOM, DN) MANGLE_NAME_IMPL(base, IOM, DN)

__kernel void MANGLE_NAME(expand_, IOM, DN)(const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
    const int iDimLen0,
    const int iDimLen1,
    const int iDimLen2,
#if (DN > 3)
    const int iDimLen3,
    const int oDimLen2,
#endif
#if (DN > 4)
    const int iDimLen4,
    const int oDimLen3,
#endif
#if (DN > 5)
    const int iDimLen5,
    const int oDimLen4,
#endif
#if (DN > 6)
    const int iDimLen6,
    const int oDimLen5,
#endif
#if (DN > 7)
    const int iDimLen7,
    const int oDimLen6,
#endif
    const int oDimLen0,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
{
    int od0 = get_global_id(0);
    int od1 = get_global_id(1);
    int od2 = get_global_id(2);
    if (od0 >= bx || od1 >= by) {
        return;
    }
    int id0 = od0;
    int id1 = od1;
    int id2 = od2;
#if (DN > 3)
    int tmpId = id2;
    id2 = tmpId % oDimLen2;
    int id3 = tmpId / oDimLen2;
#endif
#if (DN > 4)
    tmpId = id3;
    id3 = tmpId % oDimLen3;
    int id4 = tmpId / oDimLen3;
#endif
#if (DN > 5)
    tmpId = id4;
    id4 = tmpId % oDimLen4;
    int id5 = tmpId / oDimLen4;
#endif
#if (DN > 6)
    tmpId = id5;
    id5 = tmpId % oDimLen5;
    int id6 = tmpId / oDimLen5;
#endif
#if (DN > 7)
    tmpId = id6;
    id6 = tmpId % oDimLen6;
    int id7 = tmpId / oDimLen6;
#endif

    if (iDimLen0 == 1) {
        id0 = 0;
    }
    if (iDimLen1 == 1) {
        id1 = 0;
    }
    if (iDimLen2 == 1) {
        id2 = 0;
    }
#if (DN > 3)
    if (iDimLen3 == 1) {
        id3 = 0;
    }
#endif
#if (DN > 4)
    if (iDimLen4 == 1) {
        id4 = 0;
    }
#endif
#if (DN > 5)
    if (iDimLen5 == 1) {
        id5 = 0;
    }
#endif
#if (DN > 6)
    if (iDimLen6 == 1) {
        id6 = 0;
    }
#endif
#if (DN > 7)
    if (iDimLen7 == 1) {
        id7 = 0;
    }
#endif

    int z_off = id2;
#if (DN > 3)
    z_off += id3 * iDimLen2;
#endif
#if (DN > 4)
    z_off += id4 * iDimLen3;
#endif
#if (DN > 5)
    z_off += id5 * iDimLen4;
#endif
#if (DN > 6)
    z_off += id6 * iDimLen5;
#endif
#if (DN > 7)
    z_off += id7 * iDimLen6;
#endif

    T4 val = 0;
    char iew = (((id0 << 2) + 4) <= iDimLen0) ? 4 : (iDimLen0 & 3);
    char oew = (((od0 << 2) + 4) <= oDimLen0) ? 4 : (oDimLen0 & 3);
#if defined(USE_INPUT_IMG)
    int4 in_off = (int4)(id0, id1, z_off, 0);
#else
    const int in_off = (z_off * ih_str + id1 + ih_off) * iw_str + (id0 << 2) + iw_off;
#endif
    LOAD_MEM_V4_C1(val, in_off, iew, in);
    if (iDimLen0 == 1) {
        val.y = val.x;
        val.z = val.x;
        val.w = val.x;
    }
#if defined(USE_OUTPUT_IMG)
    int4 out_off = (int4)(od0, od1, od2, 0);
#else
    const int out_off = (od2 * oh_str + od1 + oh_off) * ow_str + (od0 << 2) + ow_off;
#endif
    STORE_MEM_V4_C1(val, out_off, oew, out);
}
