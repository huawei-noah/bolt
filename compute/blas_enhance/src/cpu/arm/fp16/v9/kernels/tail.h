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
static void mmm_template(I32 offset, I32 K4, F16 *_A, F16 *_B, F16 *C)
{
    unsigned short *A = (unsigned short *)_A;
    unsigned short *B = (unsigned short *)_B;
    int k = 4;
    offset /= sizeof(F16);
    for (int a = 0; a < m; a++) {
        for (int b = 0; b < n; b++) {
            float value = 0;
            for (int c = 0; c < K4; c++) {
                for (int d = 0; d < k; d++) {
                    float v0 = bfloat16ToFloat32(A[(c * m + a) * k + d]);
                    float v1 = bfloat16ToFloat32(B[(c * n + b) * k + d]);
                    value += v0 * v1;
                }
            }
            C[a * offset + b] += value;
        }
    }
}

inline void mmm_Nx1(U32 NInner, U32 stride, U32 K, F16 *_matrix1, F16 *_matrix2, F16 *result)
{
    BF16 *matrix1 = (BF16 *)_matrix1;
    BF16 *matrix2 = (BF16 *)_matrix2;
    CHECK_REQUIREMENT(NInner <= 12);
    float32x4_t res[6] = {0};
    for (U32 i = 0; i < K; i += 4) {
        bfloat16x4_t v = vld1_bf16(matrix2);
        matrix2 += 4;
        bfloat16x8_t b = vcombine_bf16(v, v);

        for (U32 j = 0; j < NInner / 2; j++) {
            bfloat16x8_t a = vld1q_bf16(matrix1);
            matrix1 += 8;

            res[j] = vbfdotq_f32(res[j], a, b);
        }
    }
    F32 tmp[4];
    for (U32 i = 0, k = 0; i < NInner / 2; i += 2) {
        float32x4_t c = vpaddq_f32(res[i], res[i + 1]);
        vst1q_f32(tmp, c);
        for (int j = 0; j < 4 && k < NInner; j++, k++) {
            result[k * stride] += tmp[j];
        }
    }
}

inline void mmm_1x1(I32 K, F16 *_matrix1, F16 *_matrix2, F16 *result)
{
    BF16 *matrix1 = (BF16 *)_matrix1;
    BF16 *matrix2 = (BF16 *)_matrix2;
    float32x4_t c = vdupq_n_f32(0);
    int i = 0;
    for (; i < K - 7; i += 8) {
        bfloat16x8_t b = vld1q_bf16(matrix2);
        matrix2 += 8;

        bfloat16x8_t a = vld1q_bf16(matrix1);
        matrix1 += 8;

        c = vbfdotq_f32(c, a, b);
    }
    *result += vaddvq_f32(c);
    if (i < K) {
        float32x2_t c = vdup_n_f32(0);
        bfloat16x4_t b = vld1_bf16(matrix2);
        bfloat16x4_t a = vld1_bf16(matrix1);
        c = vbfdot_f32(c, a, b);
        *result += vaddv_f32(c);
    }
}
#endif
