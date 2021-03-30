// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_INT8
#include "cpu/arm/blas_arm.h"
#include "cpu/arm/int8/blas_int8.h"
#include "cpu/arm/int8/mvm.h"
#include "cpu/arm/int8/mmm_common.h"

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
            U32 K4 = pad_to_4_multiple(K);
            for (; i < (int)N - 31; i += 32) {
                matrix1_trans_int8(32, K, K, src + i * K, dst + i * K4);
            }
            if (i < (int)N) {
                memcpy(dst + i * K4, src + i * K, (N - i) * K * bytesOf(DT_I8));
            }
            break;
        }
        case DF_TRANSPOSE: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
            U32 K4 = pad_to_4_multiple(K);
            for (; i < (int)N - 31; i += 32) {
                matrix2_trans_int8(32, K, N, src + i, dst + i * K4);
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
            U32 Nbatch = N / 32;
            U32 NTail = N % 32;

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

EE mvm_int8(U32 row, U32 col, DataFormat df, INT8 *matrix, INT8 *vector, I32 *tmp, I32 *result)
{
    if (DF_TRANSPOSE == df) {
        mvm_col(row, col, matrix, vector, tmp, result);
    } else {
        mvm_row(row, col, df, matrix, vector, result);
    }
    return SUCCESS;
}
#endif
