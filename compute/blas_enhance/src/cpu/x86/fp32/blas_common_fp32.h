// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_COMMON_FP32
#define _H_BLAS_COMMON_FP32

#include "tensor_desc.h"
#include "uni.h"

void mvm_col_fp32(U32 row, U32 col, F32 *matrix, F32 *vector, F32 *result);

void mvm_row_fp32(U32 row, U32 col, F32 *matrix, F32 *vector, F32 *result);

void mvm_pack_fp32(U32 row, U32 col, F32 *matrix, F32 *vector, F32 *result);

inline void matrix1_trans_w(U32 size, U32 realSize, U32 blockK, U32 K, F32 *src, F32 *dst)
{
    U32 remain = realSize % 4;
    U32 mainSize = realSize / 4 * 4;
    __m128i vindex = _mm_set_epi32(K * 3, K * 2, K, 0);
    F32 *rdst = dst;
    for (U32 i = 0; i < blockK; ++i) {
        U32 j;
        for (j = 0; j < mainSize; j += 4) {
            if (i % 16 == 0) {
                _mm_prefetch(src + i + j * K + 16, _MM_HINT_NTA);
                _mm_prefetch(src + i + (j + 1) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(src + i + (j + 2) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(src + i + (j + 3) * K + 16, _MM_HINT_NTA);
            }
            _mm_store_ps(dst, _mm_i32gather_ps(src + i + j * K, vindex, 4));
            dst += 4;
        }
        for (; j < realSize; ++j) {
            if (i % 16 == 0) {
                _mm_prefetch(src + i + (j + mainSize) * K + 16, _MM_HINT_NTA);
            }
            *(dst++) = *(src + i + j * K);
        }

        for (; j < size; ++j) {
            *(dst++) = 0;
        }
    }
}

inline void matrix2_trans_w(U32 size, U32 realSize, U32 blockK, U32 M, F32 *src, F32 *dst)
{
    for (U32 i = 0; i < blockK; i++) {
        for (U32 j = 0; j < size; j += 16) {
            _mm_prefetch(src + M + j, _MM_HINT_NTA);
        }
        UNI_MEMCPY(dst, src, realSize * sizeof(F32));
        dst += size;
        src += M;
    }
}

inline void matrix1_trans(U32 size, U32 blockK, U32 K, F32 *src, F32 *dst)
{
    U32 remain = size % 8;
    size = size / 8 * 8;
    __m256i vindex = _mm256_set_epi32(K * 7, K * 6, K * 5, K * 4, K * 3, K * 2, K, 0);
    for (U32 i = 0; i < blockK; ++i) {
        U32 j;
        for (j = 0; j < size; j += 8) {
            if (i % 16 == 0) {
                _mm_prefetch(src + i + j * K + 16, _MM_HINT_NTA);
                _mm_prefetch(src + i + (j + 1) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(src + i + (j + 2) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(src + i + (j + 3) * K + 16, _MM_HINT_NTA);
            }
            _mm256_storeu_ps(dst, _mm256_i32gather_ps(src + i + j * K, vindex, 4));
            dst += 8;
        }
        for (; j < (remain + size); ++j) {
            if (i % 16 == 0) {
                _mm_prefetch(src + i + (j + size) * K + 16, _MM_HINT_NTA);
            }
            *(dst++) = *(src + i + j * K);
        }
    }
}

inline void matrix2_trans(U32 size, U32 blockK, U32 M, F32 *src, F32 *dst)
{
    for (U32 i = 0; i < blockK; i++) {
        for (U32 j = 0; j < size; j += 16) {
            _mm_prefetch(src + M + j, _MM_HINT_NTA);
        }
        UNI_MEMCPY(dst, src, size * sizeof(F32));
        dst += size;
        src += M;
    }
}

inline void matrix2_trans_c8(U32 size, U32 blockK, U32 M, F32 *src, F32 *dst)
{
    // KNK8, blockK % 8 == 0
    CHECK_REQUIREMENT(blockK % 8 == 0);
    U32 remain = size % 4;
    size = size / 4 * 4;
    __m128i vindex = _mm_set_epi32(8 * 3, 8 * 2, 8, 0);
    for (U32 i = 0; i < blockK; i += 8) {
        for (U32 i8 = 0; i8 < 8; ++i8) {
            U32 j;
            for (j = 0; j < size; j += 4) {
                _mm_store_ps(dst, _mm_i32gather_ps(src + i * M + j * 8 + i8, vindex, 4));
                dst += 4;
            }
            for (; j < remain; ++j) {
                *(dst++) = *(src + i * M + j * 8 + i8);
            }
        }
    }
}

#endif
