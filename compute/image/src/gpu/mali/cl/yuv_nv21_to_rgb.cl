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

#if defined(USE_BGR_0_255) || defined(USE_BGRA_0_255) || defined(USE_BGR_0_1) || \
    defined(USE_BGRA_0_1)
#define B_ID 0
#define R_VAL z
#define G_VAL y
#define B_VAL x
#elif defined(USE_RGB_0_255) || defined(USE_RGBA_0_255) || defined(USE_RGB_0_1) || \
    defined(USE_RGBA_0_1)
#define B_ID 2
#define R_VAL x
#define G_VAL y
#define B_VAL z
#else
#error you must define RGB format related macro.
#endif
#if defined(USE_BGRA_0_255) || defined(USE_RGBA_0_255) || defined(USE_BGRA_0_1) || \
    defined(USE_RGBA_0_1)
#define C_NUM 4
#elif defined(USE_BGR_0_255) || defined(USE_RGB_0_255) || defined(USE_BGR_0_1) || \
    defined(USE_RGB_0_1)
#define C_NUM 3
#else
#error you must define RGB format related macro.
#endif

#if defined(USE_RGB_0_255) || defined(USE_BGR_0_255) || defined(USE_RGBA_0_255) || \
    defined(USE_BGRA_0_255)
#define FUNC(x) convert_uchar_sat(x)
#elif defined(USE_RGB_0_1) || defined(USE_BGR_0_1) || defined(USE_RGBA_0_1) || defined(USE_BGRA_0_1)
#define FUNC(x) ((x) / (OT)255)
#endif

#define HALF_MAX_NUM 128
#define U_ID 1

const float c_YUV2RGBCoeffs_420[5] = {
    1.163999557f, 2.017999649f, -0.390999794f, -0.812999725f, 1.5959997177f};

__kernel void KERNEL_NAME(const int src_step,
    const int src_offset,
    const int dst_step,
    const int dst_offset,
    const int rows,
    const int cols,
    __global IT *srcptr,
    __global OT *dstptr)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int b = get_global_id(2);
    int rgb_size = rows * dst_step;
    srcptr = srcptr + b * rgb_size / 2;
    dstptr = dstptr + b * rgb_size;
    if (x < cols / 2) {
        if (y < rows / 2) {
            __global const IT *ysrc = srcptr + mad24(y << 1, src_step, (x << 1) + src_offset);
            __global const IT *usrc = srcptr + mad24(rows + y, src_step, (x << 1) + src_offset);
            __global OT *dst1 = dstptr + mad24(y << 1, dst_step, mad24(x, C_NUM << 1, dst_offset));
            __global OT *dst2 = dst1 + dst_step;

            float Y1 = ysrc[0];
            float Y2 = ysrc[1];
            float Y3 = ysrc[src_step];
            float Y4 = ysrc[src_step + 1];

            float U = ((float)usrc[U_ID]) - HALF_MAX_NUM;
            float V = ((float)usrc[1 - U_ID]) - HALF_MAX_NUM;

            __global const float *coeffs = c_YUV2RGBCoeffs_420;
            float ruv = fma(coeffs[4], V, 0.5f);
            float guv = fma(coeffs[3], V, fma(coeffs[2], U, 0.5f));
            float buv = fma(coeffs[1], U, 0.5f);

            Y1 = max(0.f, Y1 - 16.f) * coeffs[0];
            dst1[2 - B_ID] = FUNC(Y1 + ruv);
            dst1[1] = FUNC(Y1 + guv);
            dst1[B_ID] = FUNC(Y1 + buv);

            Y2 = max(0.f, Y2 - 16.f) * coeffs[0];
            dst1[C_NUM + 2 - B_ID] = FUNC(Y2 + ruv);
            dst1[C_NUM + 1] = FUNC(Y2 + guv);
            dst1[C_NUM + B_ID] = FUNC(Y2 + buv);

            Y3 = max(0.f, Y3 - 16.f) * coeffs[0];
            dst2[2 - B_ID] = FUNC(Y3 + ruv);
            dst2[1] = FUNC(Y3 + guv);
            dst2[B_ID] = FUNC(Y3 + buv);

            Y4 = max(0.f, Y4 - 16.f) * coeffs[0];
            dst2[C_NUM + 2 - B_ID] = FUNC(Y4 + ruv);
            dst2[C_NUM + 1] = FUNC(Y4 + guv);
            dst2[C_NUM + B_ID] = FUNC(Y4 + buv);
        }
    }
}
