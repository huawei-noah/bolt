// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MVM
#define _H_MVM

#ifdef _USE_INT8
#include <arm_neon.h>
#include <string.h>

inline void mvm_col_tail(U32 N, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    for (U32 n = 0; n < N; n++) {
        I32 tmp = 0;
        for (U32 k = 0; k < K; k++) {
            tmp += vector[k] * matrix[k * N + n];
        }
        result[n] += tmp;
    }
}

inline void mvm_row_tail(U32 N, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    INT8 *cur_row = matrix;
    for (U32 n = 0; n < N; n++) {
        I32 tmp = 0;
        for (U32 k = 0; k < K; k++) {
            tmp += vector[k] * cur_row[k];
        }
        result[n] += tmp;
        cur_row += K;
    }
}

inline void mvm_row_unpack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    U32 N = Nbatch * 8;
    int8x16_t mat[8], v;
    U32 K_tail = K % 16;
    U32 K_inner = K - K_tail;
    for (U32 n = 0; n < N; n += 8) {
        int32x4_t res[8] = {0};
        int32x4_t bias[2];

        INT8 *w[8];
        for (int i = 0; i < 8; i++) {
            w[i] = matrix + (n + i) * K;
        }

        for (U32 k = 0; k < K_inner; k += 16) {
            v = vld1q_s8(vector + k);
            for (int i = 0; i < 8; i++) {
                mat[i] = vld1q_s8(w[i + k]);
            }
            for (int i = 0; i < 8; i++) {
                res[i] = vdotq_s32(res[i], mat[i], v);
            }
        }
        bias[0] = vld1q_s32(result + n);
        bias[1] = vld1q_s32(result + n + 4);

        res[0] = vpaddq_s32(res[0], res[1]);
        res[4] = vpaddq_s32(res[4], res[5]);
        res[2] = vpaddq_s32(res[2], res[3]);
        res[6] = vpaddq_s32(res[6], res[7]);
        res[0] = vpaddq_s32(res[0], res[2]);
        res[4] = vpaddq_s32(res[4], res[6]);
        res[0] = vaddq_s32(res[0], bias[0]);
        res[4] = vaddq_s32(res[4], bias[1]);

        vst1q_s32(result + n, res[0]);
        vst1q_s32(result + n + 4, res[4]);

        if (K_tail != 0) {
            for (int i = 0; i < 8; i++) {
                I32 tmp = 0;
                for (U32 p = K_inner; p < K; p++) {
                    tmp += vector[p] * w[i][p];
                }
                result[n + i] += tmp;
            }
        }
    }
}

inline void mvm_col(U32 numRows, U32 numColumns, INT8 *matrix, INT8 *vector, I32 *tmp, I32 *result)
{
    // Actual layout is KN, and vector is K
    U32 N = numRows;
    U32 K = numColumns;
    U32 NTail = N % 64;
    U32 NInner = N - NTail;

    for (U32 n = 0; n < NInner; n += 64) {
        memset(tmp, 0, sizeof(I32) * 64);
        for (U32 k = 0; k < K; k++) {
            for (U32 i = 0; i < 64; i++) {
                tmp[i] += vector[k] * matrix[k * N + n + i];
            }
        }

        for (U32 i = 0; i < 64; i++) {
            result[n + i] += tmp[i];
        }
    }

    memset(tmp, 0, sizeof(I32) * 64);
    for (U32 k = 0; k < K; k++) {
        for (U32 i = 0; i < NTail; i++) {
            tmp[i] += vector[k] * matrix[k * N + NInner + i];
        }
        for (U32 i = 0; i < NTail; i++) {
            result[NInner + i] += tmp[i];
        }
    }
}
#endif
#endif
