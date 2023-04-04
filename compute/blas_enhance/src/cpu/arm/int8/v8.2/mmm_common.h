// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MMM_COMMON_V8
#define _H_MMM_COMMON_V8

#include "data_type.h"
#include "uni.h"
#include "arm_neon_expand.h"

inline void mmm_N8_MTail(U32 MInner, U32 M, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    int8x16_t mat1[2], mat2;
    int32x4_t res[4][2] = {{0}};
    I32 tmp[8] = {0};

    CHECK_REQUIREMENT(MInner < 4);

    for (U32 i = 0; i < K; i += 4) {
        mat1[0] = vld1q_s8(matrix1 + i * 8);
        mat1[1] = vld1q_s8(matrix1 + i * 8 + 16);

        mat2 = vld1q_s8(matrix2 + i * MInner);

        for (U32 j = 0; j < MInner; j++) {
            res[j][0] = vdotq_laneq_s32_builtin(res[j][0], mat1[0], mat2, j);
            res[j][1] = vdotq_laneq_s32_builtin(res[j][1], mat1[1], mat2, j);
        }
    }
    for (U32 p = 0; p < MInner; p++) {
        vst1q_s32(tmp, res[p][0]);
        vst1q_s32(tmp + 4, res[p][1]);
        for (U32 q = 0; q < 8; q++) {
            result[q * M + p] += tmp[q];
        }
    }
}

inline void mmm_N4_MTail(U32 MInner, U32 M, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    int8x16_t mat1, mat2;
    int32x4_t res[4] = {0};
    I32 tmp[8] = {0};

    CHECK_REQUIREMENT(MInner < 4);

    for (U32 i = 0; i < K; i += 4) {
        mat1 = vld1q_s8(matrix1 + i * 4);

        mat2 = vld1q_s8(matrix2 + i * MInner);

        for (U32 j = 0; j < MInner; j++) {
            res[j] = vdotq_laneq_s32_builtin(res[j], mat1, mat2, j);
        }
    }
    for (U32 p = 0; p < MInner; p++) {
        vst1q_s32(tmp, res[p]);
        for (U32 q = 0; q < 4; q++) {
            result[q * M + p] += tmp[q];
        }
    }
}

inline void mmm_NTail_M12(U32 M, U32 N, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    int8x16_t mat1, mat2[3];
    int32x4_t res[4][3];

    for (U32 i = 0; i < N; i++) {
        res[i][0] = vld1q_s32(result + i * M);
        res[i][1] = vld1q_s32(result + i * M + 4);
        res[i][2] = vld1q_s32(result + i * M + 8);
    }

    for (U32 q = 0; q < K; q += 4) {
        mat1 = vld1q_s8(matrix1 + q * N);

        mat2[0] = vld1q_s8(matrix2 + q * 12);
        mat2[1] = vld1q_s8(matrix2 + q * 12 + 16);
        mat2[2] = vld1q_s8(matrix2 + q * 12 + 32);

        for (U32 n = 0; n < N; n++) {
            res[n][0] = vdotq_laneq_s32_builtin(res[n][0], mat2[0], mat1, n);
            res[n][1] = vdotq_laneq_s32_builtin(res[n][1], mat2[1], mat1, n);
            res[n][2] = vdotq_laneq_s32_builtin(res[n][2], mat2[2], mat1, n);
        }
    }

    for (U32 i = 0; i < N; i++) {
        vst1q_s32(result + i * M, res[i][0]);
        vst1q_s32(result + i * M + 4, res[i][1]);
        vst1q_s32(result + i * M + 8, res[i][2]);
    }
}

inline void mmm_NTail_M8(U32 M, U32 N, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    int8x16_t mat1, mat2[2];
    int32x4_t res[4][2];

    for (U32 i = 0; i < N; i++) {
        res[i][0] = vld1q_s32(result + i * M);
        res[i][1] = vld1q_s32(result + i * M + 4);
    }

    for (U32 q = 0; q < K; q += 4) {
        mat1 = vld1q_s8(matrix1 + q * N);

        mat2[0] = vld1q_s8(matrix2 + q * 8);
        mat2[1] = vld1q_s8(matrix2 + q * 8 + 16);

        for (U32 n = 0; n < N; n++) {
            res[n][0] = vdotq_laneq_s32_builtin(res[n][0], mat2[0], mat1, n);
            res[n][1] = vdotq_laneq_s32_builtin(res[n][1], mat2[1], mat1, n);
        }
    }

    for (U32 i = 0; i < N; i++) {
        vst1q_s32(result + i * M, res[i][0]);
        vst1q_s32(result + i * M + 4, res[i][1]);
    }
}
inline void mmm_NTail_M4(U32 M, U32 N, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    int8x16_t mat1, mat2;
    int32x4_t res[4];

    for (U32 i = 0; i < N; i++) {
        res[i] = vld1q_s32(result + i * M);
    }

    for (U32 q = 0; q < K; q += 4) {
        mat1 = vld1q_s8(matrix1 + q * N);
        mat2 = vld1q_s8(matrix2 + q * 4);
        for (U32 n = 0; n < N; n++) {
            res[n] = vdotq_laneq_s32_builtin(res[n], mat2, mat1, n);
        }
    }

    for (U32 i = 0; i < N; i++) {
        vst1q_s32(result + i * M, res[i]);
    }
}

// matrix2 has been transformed to MKm(MInner)K4
inline void mmm_NTail_M(U32 MInner, U32 M, U32 N, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    int8x16_t mat1, mat2;
    int32x4_t res[3] = {0};
    I32 buf[4];
    for (U32 q = 0; q < K; q += 4) {
        mat1 = vld1q_s8(matrix1 + q * N);
        mat2 = vld1q_s8(matrix2 + q * MInner);
        for (U32 n = 0; n < N; n++) {
            res[n] = vdotq_laneq_s32_builtin(res[n], mat2, mat1, n);
        }
    }

    for (U32 i = 0; i < N; i++) {
        vst1q_s32(buf, res[i]);
        for (U32 j = 0; j < MInner; j++) {
            result[i * M + j] += buf[j];
        }
    }
}
#endif
