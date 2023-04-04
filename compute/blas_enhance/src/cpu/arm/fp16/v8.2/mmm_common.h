// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MMM_COMMON
#define _H_MMM_COMMON

#include "cpu/arm/fp16/v8.2/blas_matrix_transpose.h"
#include "arm_neon_expand.h"

inline void mmm_NTail_M24(U32 M, U32 N, U32 K, F16 *matrix1, F16 *matrix2, F16 *result)
{
    for (U32 i = 0; i < N; i++) {
        float16x8x3_t res = vld3q_f16(result + i * M);
        for (U32 q = 0; q < K; q++) {
            float16x8x3_t mat2 = vld3q_f16(matrix2 + q * 24);
            res.val[0] = vfmaq_n_f16(res.val[0], mat2.val[0], matrix1[q * N + i]);
            res.val[1] = vfmaq_n_f16(res.val[1], mat2.val[1], matrix1[q * N + i]);
            res.val[2] = vfmaq_n_f16(res.val[2], mat2.val[2], matrix1[q * N + i]);
        }
        vst3q_f16(result + i * M, res);
    }
}

inline void mmm_NTail_M8(U32 M, U32 N, U32 K, F16 *matrix1, F16 *matrix2, F16 *result)
{
    for (U32 i = 0; i < N; i++) {
        float16x8_t res = vld1q_f16(result + i * M);
        for (U32 q = 0; q < K; q++) {
            float16x8_t mat2 = vld1q_f16(matrix2 + q * 8);
            res = vfmaq_n_f16(res, mat2, matrix1[q * N + i]);
        }
        vst1q_f16(result + i * M, res);
    }
}

inline void mmm_NTail_M4(U32 M, U32 N, U32 K, F16 *matrix1, F16 *matrix2, F16 *result)
{
    for (U32 i = 0; i < N; i++) {
        float16x4_t res = vld1_f16(result + i * M);
        for (U32 q = 0; q < K; q++) {
            float16x4_t mat2 = vld1_f16(matrix2 + q * 4);
            res = vfma_n_f16(res, mat2, matrix1[q * N + i]);
        }
        vst1_f16(result + i * M, res);
    }
}

inline void mmm_NTail_M(U32 MInner, U32 M, U32 N, U32 K, F16 *matrix1, F16 *matrix2, F16 *result)
{
    for (U32 i = 0; i < N; i++) {
        for (U32 j = 0; j < MInner; j++) {
            F16 value = 0;
            for (U32 k = 0; k < K; k++) {
                value += *(matrix1 + k * N + i) * *(matrix2 + k * MInner + j);
            }
            result[i * M + j] += value;
        }
    }
}

inline void mmm_N8_MTail(U32 MInner, U32 M, U32 K, F16 *matrix1, F16 *matrix2, F16 *result)
{
    CHECK_REQUIREMENT(MInner < 4);
    float16x8_t res[4] = {0};
    for (U32 i = 0; i < K; i++) {
        float16x8_t mat1 = vld1q_f16(matrix1 + i * 8);
#pragma unroll(4)
        for (U32 j = 0; j < MInner; j++) {
            res[j] = vfmaq_n_f16(res[j], mat1, matrix2[j + i * MInner]);
        }
    }
    F16 tmp[8];
    for (U32 p = 0; p < MInner; p++) {
        vst1q_f16(tmp, res[p]);
#pragma unroll(8)
        for (U32 q = 0; q < 8; q++) {
            result[q * M + p] += tmp[q];
        }
    }
}

inline void mmm_N4_MTail(U32 MInner, U32 M, U32 K, F16 *matrix1, F16 *matrix2, F16 *result)
{
    CHECK_REQUIREMENT(MInner < 4);
    float16x4_t res[4] = {0};
    for (U32 i = 0; i < K; i++) {
        float16x4_t mat1 = vld1_f16(matrix1 + i * 4);
#pragma unroll(4)
        for (U32 j = 0; j < MInner; j++) {
            res[j] = vfma_n_f16(res[j], mat1, matrix2[j + i * MInner]);
        }
    }
    F16 tmp[4];
    for (U32 p = 0; p < MInner; p++) {
        vst1_f16(tmp, res[p]);
#pragma unroll(4)
        for (U32 q = 0; q < 4; q++) {
            result[q * M + p] += tmp[q];
        }
    }
}
#endif
