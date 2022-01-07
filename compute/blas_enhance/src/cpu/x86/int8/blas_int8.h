// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_INT8
#define _H_BLAS_INT8

#include "sys.h"
#include "error.h"
#include "tensor_desc.h"
#include "thread_affinity.h"
#include "uni.h"

#define SIMDW 8
#define align_size(size, unit) ((size + unit - 1) / unit * unit)

void matrix_matrix_multiply_tmp_bytes_int8(
    U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes);

// transform no-transposed B to K4, offline
inline void matrix1_trans_l(int size, int blockK, int K, int alignSize, INT8 *src, INT8 *dst)
{
    int alignedBlockK = align_size(blockK, alignSize);
    int blockKF32 = blockK / 4;
    __m256i vindex = _mm256_set_epi32(K * 7, K * 6, K * 5, K * 4, K * 3, K * 2, K, 0);
    int i;
    for (i = 0; i < blockKF32; ++i) {
        int j = 0;
        for (; j < size / 8; ++j) {
            if (i % 16 == 0) {
                _mm_prefetch(dst + i * 4 + j * 8 * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 1) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 2) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 3) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 4) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 5) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 6) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 7) * K + 16, _MM_HINT_NTA);
            }
            _mm256_storeu_ps(
                (float *)dst, _mm256_i32gather_ps((float *)(src + i * 4 + j * 8 * K), vindex, 1));
            dst += 32;
        }
        j *= 8;
        for (; j < size; ++j) {
            memcpy(dst, src + i * 4 + j * K, 4);
            dst += 4;
        }
    }
    i *= 4;
    for (; i < alignedBlockK; i += 4) {
        for (int j = 0; j < size; ++j) {
            for (int ii = i; ii < i + 4; ++ii) {
                if (ii < blockK) {
                    *(dst++) = src[ii + j * K];
                } else {
                    *(dst++) = 0;
                }
            }
        }
    }
}

// transform transposed B to K4, offline
inline void matrix2_trans_l(int size, int blockK, int N, int alignSize, INT8 *src, INT8 *dst)
{
    int alignedBlockK = align_size(blockK, alignSize);
    for (int i = 0; i < alignedBlockK; i += 4) {
        for (int j = 0; j < size; ++j) {
            for (int ii = i; ii < (i + 4); ++ii) {
                if (ii < blockK) {
                    *(dst++) = src[ii * N + j];
                } else {
                    *(dst++) = 0;
                }
            }
        }
    }
}

// transpose A, online
inline void matrix2_trans_r(int size, int blockK, int M, int alignSize, UINT8 *src, UINT8 *dst)
{
    // TODO: optimize
    int alignedBlockK = align_size(blockK, alignSize);
    for (int j = 0; j < size; ++j) {
        int i = 0;
        for (i = 0; i < blockK; ++i) {
            if (j % 64 == 0) {
                _mm_prefetch(src + i * M + j + 64, _MM_HINT_NTA);
            }
            *(dst++) = *(src + i * M + j);
        }
        for (; i < alignedBlockK; ++i) {
            *(dst++) = 0;
        }
    }
}

// transpose A, online
inline void matrix1_trans_r(int size, int blockK, int K, int alignSize, UINT8 *src, UINT8 *dst)
{
    int alignedBlockK = align_size(blockK, alignSize);
    if (alignedBlockK != blockK) {
        memset(dst, 0, alignedBlockK * size);
    }
    for (int j = 0; j < size; ++j) {
        memcpy(dst + j * alignedBlockK, src + j * K, blockK);
    }
}

EE matrix_vector_multiply_transform_weight_int8(
    TensorDesc desc, INT8 *src, INT8 *packB, I32 *offsetCBias);

EE matrix_matrix_multiply_transform_rhsN_int8(
    TensorDesc desc, INT8 *src, INT8 *dst, I32 *offsetCBias);

EE matrix_matrix_multiply_transform_rhsT_int8(
    TensorDesc desc, INT8 *src, INT8 *dst, I32 *offsetCBias);

EE mmm_avx512_vnni_int8(U32 M,
    U32 N,
    U32 K,
    DataFormat matrixADataFormat,
    UINT8 *matrix1,
    INT8 *matrix2,
    UINT8 *tmp,
    UINT8 *result,
    const F32 *scale);

EE mvm_avx512_int8(U32 numRows,
    U32 numColumns,
    INT8 *packB,
    UINT8 *vector,
    UINT8 *result,
    I32 *offsetCBias,
    const F32 *scale);

EE mvm_avx512_int8_row_i8u8(U32 numRows,
    U32 numColumns,
    DataFormat df,
    UINT8 *packB,
    INT8 *vector,
    UINT8 *result,
    I32 *tmp,
    const F32 *scale);

#endif
