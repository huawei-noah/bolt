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
#ifdef _USE_FP16
#include "cpu/arm/int8/v8.2/mvm.h"
#include "cpu/arm/int8/v8.2/blas_matrix_transpose.h"
#else
#include "cpu/arm/int8/v7/mvm.h"
#include "cpu/arm/int8/v7/blas_matrix_transpose.h"
#endif

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
            U32 K4 = UNI_ALIGN(K, 4);
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
            U32 K4 = UNI_ALIGN(K, 4);
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
#ifdef _USE_FP16
                U32 K4 = UNI_ALIGN(K, 4);
#else
                U32 K4 = K;
#endif
                mvm_row_tail(NTail, K, matrix + (N - NTail) * K4, vector, result + N - NTail);
            }
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
}

template <int tileN>
inline void mvm_col_kernel(int N, int K, int stride, INT8 *matrix, INT8 *vector, I32 *result)
{
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        int32x4_t res[tileN / 4];
        int16x4_t mat[tileN / 4];
#ifdef _USE_OPENMP
#pragma omp for
#endif
        for (int i = 0; i < N - tileN + 1; i += tileN) {
            for (int ii = 0; ii < tileN / 4; ii++) {
                res[ii] = vld1q_s32(result + i + ii * 4);
            }
            for (int k = 0; k < K; k++) {
                int16x4_t v = vdup_n_s16(vector[k]);

                INT8 *p = matrix + k * stride;
                for (int ii = 0; ii < tileN / 4; ii += 2) {
                    int16x8_t v = vmovl_s8(vld1_s8(p + i + ii * 4));
                    mat[ii] = vget_low_s16(v);
                    mat[ii + 1] = vget_high_s16(v);
                }
                for (int ii = 0; ii < tileN / 4; ii++) {
                    res[ii] = vmlal_s16(res[ii], mat[ii], v);
                }
            }
            for (int ii = 0; ii < tileN / 4; ii++) {
                vst1q_s32(result + i + ii * 4, res[ii]);
            }
        }
    }
}

inline void mvm_col(int N, int K, INT8 *matrix, INT8 *vector, I32 *result)
{
    //int tail = N;
    mvm_col_kernel<32>(N, K, N, matrix, vector, result);
    int loop = N / 32 * 32;
    int tail = N - loop;
    result += loop;
    matrix += loop;
    if (tail >= 8) {
        mvm_col_kernel<8>(tail, K, N, matrix, vector, result);
        loop = tail / 8 * 8;
        tail = tail - loop;
        result += loop;
        matrix += loop;
    }
    if (tail > 0) {
        for (int i = 0; i < tail; i++) {
            int value = 0;
            for (int k = 0; k < K; k++) {
                value += matrix[k * N + i] * vector[k];
            }
            result[i] += value;
        }
    }
}

EE mvm_int8(U32 row, U32 col, DataFormat df, INT8 *matrix, INT8 *vector, I32 *tmp, I32 *result)
{
    if (DF_TRANSPOSE == df) {
        mvm_col(row, col, matrix, vector, result);
    } else {
        mvm_row(row, col, df, matrix, vector, result);
    }
    return SUCCESS;
}
