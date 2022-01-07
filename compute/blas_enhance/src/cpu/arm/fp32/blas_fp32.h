// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_FP32
#define _H_BLAS_FP32

#include "sys.h"

#include "uni.h"
#include "error.h"
#include "tensor_desc.h"
#include "arm_neon_expand.h"

EE matrix_vector_multiply_transform_weight_fp32(TensorDesc desc, F32 *src, F32 *dst);

void mvm_col_fp32(U32 row, U32 col, F32 *matrix, F32 *vector, F32 *result);

void mvm_row_fp32(U32 row, U32 col, F32 *matrix, F32 *vector, F32 *result);

EE mvm_fp32(U32 row, U32 col, DataFormat df, F32 *matrix, F32 *vector, F32 *result);

void matrix_matrix_multiply_tmp_bytes_fp32(
    U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes);

EE matrix_matrix_multiply_transform_rhsN_fp32(TensorDesc desc, F32 *src, F32 *dst);

EE matrix_matrix_multiply_transform_rhsT_fp32(TensorDesc desc, F32 *src, F32 *dst);

#ifdef __aarch64__
EE mmm_fp32_V8(
    int M, int N, int K, bool transposeA, F32 *matrix1, F32 *matrix2, F32 *tmp, F32 *result);
#else
EE mmm_fp32_V7(
    int M, int N, int K, bool transposeA, F32 *matrix1, F32 *matrix2, F32 *tmp, F32 *result);
#endif

EE axpby_fp32(U32 len, F32 a, const F32 *x, F32 b, F32 *y);

inline void matrix1_trans(U32 size, U32 blockK, U32 K, F32 *src, F32 *dst)
{
    F32 *src1 = src;
    for (U32 i = 0; i < blockK; i++) {
        for (U32 j = 0; j < size; j++) {
            src1 = src + j * K;
            if (i % 16 == 0) {
                __builtin_prefetch(src1 + 16);
            }
            *dst++ = *(src1 + i);
        }
    }
}

inline void matrix2_trans(U32 size, U32 blockK, U32 M, F32 *src, F32 *dst)
{
    for (U32 i = 0; i < blockK; i++) {
        if (i % 16 == 0) {
            __builtin_prefetch(src + 16);
        }
        memcpy(dst, src, size * sizeof(F32));
        dst += size;
        src += M;
    }
}

inline void mvm_row_tail(U32 N, U32 K, F32 *matrix, F32 *vector, F32 *result)
{
    float32x4_t vec, res, mat;
    U32 KTail = K % 4;
    U32 KInner = K - KTail;

    for (U32 i = 0; i < N; i++) {
        res = vdupq_n_f32(0);

        for (U32 j = 0; j < KInner; j += 4) {
            vec = vld1q_f32(&vector[j]);
            mat = vld1q_f32(&matrix[j + K * i]);
            res = vfmaq_f32(res, vec, mat);
        }
        result[i] += vaddvq_f32(res);

        if (KTail != 0) {
            for (U32 p = 0; p < KTail; p++) {
                result[i] += vector[p + KInner] * matrix[KInner + p + K * i];
            }
        }
    }
}
#endif
