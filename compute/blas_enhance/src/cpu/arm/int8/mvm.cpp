// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/int8/blas_int8.h"
#include "cpu/arm/blas_arm.h"
#include "cpu/arm/int8/blas_matrix_transpose.h"
#include "arm_neon_expand.h"
#include "uni.h"

#define ALIGN 32

EE matrix_vector_multiply_transform_weight_int8(TensorDesc desc, INT8 *src, INT8 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    EE ret = SUCCESS;
    int i = 0;
    switch (desc.df) {
        case DF_NORMAL: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
#ifdef _USE_FP16
            U32 K4 = pad_to_4_multiple(K);
#else
            U32 K4 = K;
#endif
            for (; i < (int)N - ALIGN + 1; i += ALIGN) {
                matrix1_trans_int8(ALIGN, K, K, src + i * K, dst + i * K4);
            }
            if (i < (int)N) {
                UNI_MEMCPY(dst + i * K4, src + i * K, (N - i) * K * bytesOf(DT_I8));
            }
            break;
        }
        case DF_TRANSPOSE: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
#ifdef _USE_FP16
            U32 K4 = pad_to_4_multiple(K);
#else
            U32 K4 = K;
#endif
            for (; i < (int)N - ALIGN + 1; i += ALIGN) {
                matrix2_trans_int8(ALIGN, K, N, src + i, dst + i * K4);
            }
            if (i < (int)N) {
                int base = i;
                INT8 *basePtr = dst + i * K4;
                for (int j = 0; j < (int)K; j++) {
                    for (int k = base; k < (int)N; k++) {
                        basePtr[(k - base) * K + j] = src[j * N + k];
                    }
                }
            }
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

#ifndef _USE_FP16
#if 1
void mvm_row_pack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
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
void mvm_row_pack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
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
#else
void mvm_row_pack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    U32 N = Nbatch * 32;
    int8x16_t mat[16];
    int8x8_t v;
    int32x4_t res[8];
    U32 K_tail = K % 4;
    U32 K_inner = K - K_tail;
    U32 K4 = pad_to_4_multiple(K);

    for (U32 n = 0; n < N; n += 32) {
        INT8 *bufMov = matrix + n * K4;
        if (K_inner > 0) {
            for (int i = 0; i < 8; i++) {
                res[i] = vld1q_s32(result + n + i * 4);
            }
            U32 k = 0;
            for (; k < K_inner; k += 8) {
                v = vld1_s8(vector + k);
                for (int i = 0; i < 8; i++) {
                    mat[i] = vld1q_s8(bufMov + i * 16);
                }
                for (int i = 0; i < 8; i++) {
                    res[i] = vdotq_lane_s32(res[i], mat[i], v, 0);
                }
                for (int i = 8; i < 16; i++) {
                    mat[i] = vld1q_s8(bufMov + i * 16);
                }
                for (int i = 0; i < 8; i++) {
                    res[i] = vdotq_lane_s32(res[i], mat[i + 8], v, 1);
                }
                bufMov += 256;
            }
            if (K_inner > K) {
                v = vld1_s8(vector + k - 4);
                for (int i = 0; i < 8; i++) {
                    mat[i] = vld1q_s8(bufMov + i * 16);
                }
                for (int i = 0; i < 8; i++) {
                    res[i] = vdotq_lane_s32(res[i], mat[i], v, 1);
                }
                bufMov += 128;
            }

            for (int i = 0; i < 8; i++) {
                vst1q_s32(result + n + i * 4, res[i]);
            }
        }
        if (K_tail > 0) {
            for (int i = 0; i < 32; i++) {
                I32 tmp = 0;
                for (int j = 0; j < (int)K_tail; j++) {
                    tmp += vector[K_inner + j] * bufMov[j];
                }
                result[n + i] += tmp;
                bufMov += 4;
            }
        }
    }
}
#endif

inline void mvm_row_unpack(U32 Nbatch, U32 K, INT8 *matrix, INT8 *vector, I32 *result)
{
    U32 N = Nbatch * 8;
#ifdef _USE_FP16
    int8x16_t mat[8];
#else
    int16x4_t mat[8][2];
#endif
    U32 K_tail = K % 16;
    U32 K_inner = K - K_tail;
    INT8 *w[8];
    for (U32 n = 0; n < N; n += 8) {
        for (int i = 0; i < 8; i++) {
            w[i] = matrix + (n + i) * K;
        }

        int32x4_t bias0 = vld1q_s32(result + n);
        int32x4_t bias1 = vld1q_s32(result + n + 4);
        int32x4_t res[8] = {0};
#ifdef _USE_FP16
        for (U32 k = 0; k < K_inner; k += 16) {
            int8x16_t v = vld1q_s8(vector + k);
            for (int i = 0; i < 8; i++) {
                mat[i] = vld1q_s8(w[i + k]);
            }
            for (int i = 0; i < 8; i++) {
                res[i] = vdotq_s32(res[i], mat[i], v);
            }
        }
#else
        for (U32 k = 0; k < K_inner; k += 8) {
            int16x8_t v = vmovl_s8(vld1_s8(vector + k));
            int16x4_t v_l = vget_low_s16(v);
            int16x4_t v_h = vget_high_s16(v);
            for (int i = 0; i < 8; i++) {
                int16x8_t v = vmovl_s8(vld1_s8(w[i + k]));
                mat[i][0] = vget_low_s16(v);
                mat[i][1] = vget_high_s16(v);
            }
            for (int i = 0; i < 8; i++) {
                res[i] = vmlal_s16(res[i], mat[i][0], v_l);
                res[i] = vmlal_s16(res[i], mat[i][1], v_h);
            }
        }
#endif
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

void mvm_row(U32 numRows, U32 numColumns, DataFormat df, INT8 *matrix, INT8 *vector, I32 *result)
{
    // Actual layout is NK, and vector is K
    U32 N = numRows;
    U32 K = numColumns;
    switch (df) {
        case DF_NORMAL: {
            U32 Nbatch = N / 8;
            U32 NTail = N % 8;

            mvm_row_unpack(Nbatch, K, matrix, vector, result);

            if (NTail != 0) {
                mvm_row_tail(NTail, K, matrix + (N - NTail) * K, vector, result + N - NTail);
            }
            break;
        }
        case DF_NKN32K4: {
            U32 Nbatch = N / ALIGN;
            U32 NTail = N % ALIGN;

            mvm_row_pack(Nbatch, K, matrix, vector, result);

            if (NTail != 0) {
                U32 K4 = pad_to_4_multiple(K);
                mvm_row_tail(NTail, K, matrix + (N - NTail) * K4, vector, result + N - NTail);
            }
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
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
        UNI_MEMSET(tmp, 0, sizeof(I32) * 64);
        for (U32 k = 0; k < K; k++) {
            for (U32 i = 0; i < 64; i++) {
                tmp[i] += vector[k] * matrix[k * N + n + i];
            }
        }

        for (U32 i = 0; i < 64; i++) {
            result[n + i] += tmp[i];
        }
    }

    UNI_MEMSET(tmp, 0, sizeof(I32) * 64);
    for (U32 k = 0; k < K; k++) {
        for (U32 i = 0; i < NTail; i++) {
            tmp[i] += vector[k] * matrix[k * N + NInner + i];
        }
        for (U32 i = 0; i < NTail; i++) {
            result[NInner + i] += tmp[i];
        }
    }
}

EE mvm_int8(U32 row, U32 col, DataFormat df, INT8 *matrix, INT8 *vector, I32 *tmp, I32 *result)
{
    if (DF_TRANSPOSE == df) {
        mvm_col(row, col, matrix, vector, tmp, result);
    } else {
        mvm_row(row, col, df, matrix, vector, result);
    }
    return SUCCESS;
}
