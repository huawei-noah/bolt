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
#define CHANNEL 4
#elif defined(USE_BGR_0_255) || defined(USE_RGB_0_255) || defined(USE_BGR_0_1) || \
    defined(USE_RGB_0_1)
#define C_NUM 3
#else
#error you must define RGB format related macro.
#endif

#if defined(USE_RGB_0_255) || defined(USE_BGR_0_255) || defined(USE_RGBA_0_255) || \
    defined(USE_BGRA_0_255)
#define FUNC(x) (x)
#elif defined(USE_RGB_0_1) || defined(USE_BGR_0_1) || defined(USE_RGBA_0_1) || defined(USE_BGRA_0_1)
#define FUNC(x) ((x)*255)
#endif

#define HALF_MAX_NUM 128
#define U_ID 1

const float c_RGB2YUVCoeffs_420[8] = {0.256999969f, 0.50399971f, 0.09799957f, -0.1479988098f,
    -0.2909994125f, 0.438999176f, -0.3679990768f, -0.0709991455f};

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
    int yuv_size = rows * dst_step;
    srcptr = srcptr + b * yuv_size * 2;
    dstptr = dstptr + b * yuv_size;
    if (x < cols / 2) {
        if (y < rows / 3) {
            int y_rows = rows / 3 * 2;
            __global const float *coeffs = c_RGB2YUVCoeffs_420;

            __global const IT *src1 =
                srcptr + mad24(y << 1, src_step, mad24(x << 1, C_NUM, src_offset));
            __global const IT *src2 = src1 + src_step;
            __global OT *ydst1 = dstptr + mad24(y << 1, dst_step, (x << 1) + dst_offset);
            __global OT *ydst2 = ydst1 + dst_step;
            __global OT *usrc = dstptr + mad24(y_rows + y, dst_step, (x << 1) + src_offset);

            float3 src_pix1 = FUNC(convert_float3(vload3(0, src1)));
            float3 src_pix2 = FUNC(convert_float3(vload3(0, src1 + C_NUM)));
            float3 src_pix3 = FUNC(convert_float3(vload3(0, src2)));
            float3 src_pix4 = FUNC(convert_float3(vload3(0, src2 + C_NUM)));

            ydst1[0] = convert_uchar_sat(fma(coeffs[0], src_pix1.R_VAL,
                fma(coeffs[1], src_pix1.G_VAL, fma(coeffs[2], src_pix1.B_VAL, 16.5f))));
            ydst1[1] = convert_uchar_sat(fma(coeffs[0], src_pix2.R_VAL,
                fma(coeffs[1], src_pix2.G_VAL, fma(coeffs[2], src_pix2.B_VAL, 16.5f))));
            ydst2[0] = convert_uchar_sat(fma(coeffs[0], src_pix3.R_VAL,
                fma(coeffs[1], src_pix3.G_VAL, fma(coeffs[2], src_pix3.B_VAL, 16.5f))));
            ydst2[1] = convert_uchar_sat(fma(coeffs[0], src_pix4.R_VAL,
                fma(coeffs[1], src_pix4.G_VAL, fma(coeffs[2], src_pix4.B_VAL, 16.5f))));

            float uv[2] = {
                fma(coeffs[5], src_pix1.R_VAL,
                    fma(coeffs[6], src_pix1.G_VAL, fma(coeffs[7], src_pix1.B_VAL, 128.5f))),
                fma(coeffs[3], src_pix1.R_VAL,
                    fma(coeffs[4], src_pix1.G_VAL, fma(coeffs[5], src_pix1.B_VAL, 128.5f)))};

            usrc[U_ID] = convert_uchar_sat(uv[U_ID]);
            usrc[1 - U_ID] = convert_uchar_sat(uv[1 - U_ID]);
        }
    }
}
