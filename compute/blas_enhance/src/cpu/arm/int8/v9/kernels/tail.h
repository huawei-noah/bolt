// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MMM_TAIL
#define _H_MMM_TAIL

#include "data_type.h"
#include "arm_neon_expand.h"

template <int m, int n>
static void mmm_template(I32 offset, I32 K4, INT8 *A, INT8 *B, I32 *C)
{
    int k = 8;
    offset /= 4;
    for (int a = 0; a < m; a++) {
        for (int b = 0; b < n; b++) {
            float value = 0;
            for (int c = 0; c < K4; c++) {
                for (int d = 0; d < k; d++) {
                    float v0 = A[(c * m + a) * k + d];
                    float v1 = B[(c * n + b) * k + d];
                    value += v0 * v1;
                }
            }
            C[a * offset + b] += value;
        }
    }
}

inline void mmm_Nx1(U32 NInner, U32 stride, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    CHECK_REQUIREMENT(NInner <= 12);
    int32x4_t res[6] = {0};
    for (U32 i = 0; i < K; i += 8) {
        int8x8_t v = vld1_s8(matrix2);
        matrix2 += 8;
        int8x16_t b = vcombine_s8(v, v);

        for (U32 j = 0; j < NInner / 2; j++) {
            int8x16_t a = vld1q_s8(matrix1);
            matrix1 += 16;

            res[j] = vdotq_s32(res[j], a, b);
        }
    }
    I32 tmp[4];
    for (U32 i = 0, k = 0; i < NInner / 2; i += 2) {
        int32x4_t c = vpaddq_s32(res[i], res[i + 1]);
        vst1q_s32(tmp, c);
        for (int j = 0; j < 4 && k < NInner; j++, k++) {
            result[k * stride] += tmp[j];
        }
    }
}

inline void mmm_1x1(U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    int32x2_t c = {0};
    for (U32 i = 0; i < K; i += 8) {
        int8x8_t b = vld1_s8(matrix2);
        matrix2 += 8;

        int8x8_t a = vld1_s8(matrix1);
        matrix1 += 8;

        c = vdot_s32(c, a, b);
    }
    I32 tmp[2];
    c = vpadd_s32(c, c);
    vst1_s32(tmp, c);
    *result += tmp[0];
}
#endif
