// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MVM_INT8
#define _H_MVM_INT8

#include "arm_neon_expand.h"

static const int ALIGN = 32;

#if 1
inline void mvm_row_pack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    U32 N = Nbatch * ALIGN;
    const int unroll = ALIGN / 4;
    int16x4_t mat[unroll];
    int32x4_t res[unroll];
    int align = ALIGN;
    for (U32 n = 0; n < N; n += align) {
        INT8 *bufMov = matrix + n * K;
        for (int i = 0; i < unroll; i++) {
            res[i] = vld1q_s32(result + n + i * 4);
        }
        for (U32 k = 0; k < K; k++) {
            int16x4_t v = vdup_n_s16(vector[k]);
            for (int i = 0; i < unroll / 2; i++) {
                int16x8_t tmp = vmovl_s8(vld1_s8(bufMov + i * 8));
                mat[i * 2] = vget_low_s16(tmp);
                mat[i * 2 + 1] = vget_high_s16(tmp);
            }
            for (int i = 0; i < unroll; i++) {
                res[i] = vmlal_s16(res[i], mat[i], v);
            }
            bufMov += align;
        }
        for (int i = 0; i < unroll; i++) {
            vst1q_s32(result + n + i * 4, res[i]);
        }
    }
}
#else
inline void mvm_row_pack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    U32 N = Nbatch * ALIGN;
    const int unroll = ALIGN / 8;
    int16x8_t res[unroll];
    int align = ALIGN;
    for (U32 n = 0; n < N; n += align) {
        INT8 *bufMov = matrix + n * K;
        for (int i = 0; i < unroll; i++) {
            res[i] = vdupq_n_s16(0);
        }
        for (U32 k = 0; k < K; k++) {
            int8x8_t v = vdup_n_s8(vector[k]);
            for (int i = 0; i < unroll; i++) {
                int8x8_t tmp = vld1_s8(bufMov + i * 8);
                res[i] = vmlal_s8(res[i], tmp, v);
            }
            bufMov += align;
        }
        for (int i = 0; i < unroll; i++) {
            int32x4_t a = vld1q_s32(result + n + i * 8);
            int32x4_t b = vld1q_s32(result + n + i * 8 + 4);
            a = vaddw_s16(a, vget_low_s16(res[i]));
            b = vaddw_s16(b, vget_high_s16(res[i]));
            vst1q_s32(result + n + i * 8, a);
            vst1q_s32(result + n + i * 8 + 4, b);
        }
    }
}
#endif

inline void mvm_row_unpack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    U32 N = Nbatch * 8;
    int16x4_t mat[8][2];
    U32 K_tail = K % 8;
    U32 K_inner = K - K_tail;
    INT8 *w[8];
    for (U32 n = 0; n < N; n += 8) {
        for (int i = 0; i < 8; i++) {
            w[i] = matrix + (n + i) * K;
        }

        int32x4_t bias0 = vld1q_s32(result + n);
        int32x4_t bias1 = vld1q_s32(result + n + 4);
        int32x4_t res[8] = {0};
        for (U32 k = 0; k < K_inner; k += 8) {
            int16x8_t v = vmovl_s8(vld1_s8(vector + k));
            int16x4_t v_l = vget_low_s16(v);
            int16x4_t v_h = vget_high_s16(v);
            for (int i = 0; i < 8; i++) {
                int16x8_t v = vmovl_s8(vld1_s8(w[i] + k));
                mat[i][0] = vget_low_s16(v);
                mat[i][1] = vget_high_s16(v);
            }
            for (int i = 0; i < 8; i++) {
                res[i] = vmlal_s16(res[i], mat[i][0], v_l);
                res[i] = vmlal_s16(res[i], mat[i][1], v_h);
            }
        }
        res[0] = vpaddq_s32(res[0], res[1]);
        res[4] = vpaddq_s32(res[4], res[5]);
        res[2] = vpaddq_s32(res[2], res[3]);
        res[6] = vpaddq_s32(res[6], res[7]);
        res[0] = vpaddq_s32(res[0], res[2]);
        res[4] = vpaddq_s32(res[4], res[6]);
        res[0] = vaddq_s32(res[0], bias0);
        res[4] = vaddq_s32(res[4], bias1);
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
#endif
