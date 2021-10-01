// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/int8/blas_int8.h"
#include "thread_affinity.h"

#define UNROLL_N 48
#define UNROLL_M 8
#define BOLCK_M_DIM 384
#define BOLCK_K_DIM 4096

typedef void (*kernel_func)(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags);

void matrix_matrix_multiply_tmp_bytes_int8(
    U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes)
{
    row1 = align_size(row1, SIMDW);
    row2 = align_size(row2, SIMDW);
    col1 = align_size(col1, SIMDW);
    col2 = align_size(col2, SIMDW);
    *bytes = row1 * col1 + row2 * col2 + UNI_MAX(row2, col2) * 4;
    *bytes *= sizeof(dt);
    *bytes += 64;
}

EE matrix_matrix_multiply_transform_rhsN_int8(
    TensorDesc desc, INT8 *src, INT8 *packB, I32 *offsetCBias)
{
    DataType dt;
    DataFormat df;
    U32 N, K, blockSizeK, unrollSizeN;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    U32 unrollSize[4] = {8, 16, 32, 48};
    INT8 *tmpS = src;
    bool hasBias = (offsetCBias != nullptr);
    I32 *sumB = nullptr;
    if (!hasBias) {
        sumB = (I32 *)packB;
        memset(sumB, 0, N * sizeof(I32));
    } else {
        sumB = offsetCBias;
    }
    packB += N * bytesOf(DT_I32);

    for (U32 bk = 0; bk < K; bk += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        blockSizeK = UNI_MAX(blockSizeK % SIMDW, blockSizeK - blockSizeK % SIMDW);
        U32 alignedBlockSizeK = align_size(blockSizeK, SIMDW);
        for (U32 un = 0; un < N; un += unrollSizeN) {
            unrollSizeN = UNI_MIN(UNROLL_N, N - un);
            unrollSizeN = UNI_MIN(unrollSize[unrollSizeN >> 4], unrollSizeN);
            matrix2_trans_l(unrollSizeN, blockSizeK, N, SIMDW, tmpS + un, packB);
            packB += unrollSizeN * alignedBlockSizeK;
        }
        tmpS += blockSizeK * N;
    }

    for (U32 n = 0; n < N; ++n) {
        I32 tmp = 0;
        for (U32 k = 0; k < K; ++k) {
            tmp += (I32)(src[k * N + n]);
        }
        sumB[n] += tmp * (-128);
    }
    return SUCCESS;
}

EE matrix_matrix_multiply_transform_rhsT_int8(
    TensorDesc desc, INT8 *src, INT8 *packB, I32 *offsetCBias)
{
    DataType dt;
    DataFormat df;
    U32 N, K, blockSizeK, unrollSizeN;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    U32 unrollSize[4] = {8, 16, 32, 48};
    INT8 *tmpS = src;
    bool hasBias = (offsetCBias != nullptr);
    I32 *sumB = nullptr;
    if (!hasBias) {
        sumB = (I32 *)packB;
        memset(sumB, 0, N * sizeof(I32));
    } else {
        sumB = offsetCBias;
    }
    packB += N * bytesOf(DT_I32);

    for (U32 bk = 0; bk < K; bk += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        blockSizeK = UNI_MAX(blockSizeK % SIMDW, blockSizeK - blockSizeK % SIMDW);
        U32 alignedBlockSizeK = align_size(blockSizeK, SIMDW);
        for (U32 un = 0; un < N; un += unrollSizeN) {
            unrollSizeN = UNI_MIN(UNROLL_N, N - un);
            unrollSizeN = UNI_MIN(unrollSize[unrollSizeN >> 4], unrollSizeN);
            matrix1_trans_l(unrollSizeN, blockSizeK, K, SIMDW, tmpS + un * K, packB);
            packB += unrollSizeN * alignedBlockSizeK;
        }
        tmpS += blockSizeK;
    }

    for (U32 n = 0; n < N; ++n) {
        I32 tmp = 0;
        for (U32 k = 0; k < K; ++k) {
            tmp += (I32)(src[n * K + k]);
        }
        sumB[n] += tmp * (-128);
    }

    return SUCCESS;
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel8x48                                             \
    "movq %0, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "prefetcht0 0xC0(%1)                              \n\t"       \
    "prefetcht0 0x100(%1)                              \n\t"      \
    "prefetcht0 0x140(%1)                              \n\t"      \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm0              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm1              \n\t"         \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm2              \n\t"         \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm3              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm4              \n\t"         \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm5              \n\t"         \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "vmovups (%1), %%zmm27                             \n\t"      \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm6              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm7              \n\t"         \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm8              \n\t"         \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm9              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm10              \n\t"        \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm11              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "vmovups 0x40(%1), %%zmm28                             \n\t"  \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm12              \n\t"        \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm13              \n\t"        \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm14              \n\t"        \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm15              \n\t"        \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm16              \n\t"        \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm17              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "vmovups 0x80(%1), %%zmm29                             \n\t"  \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm18              \n\t"        \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm19              \n\t"        \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm20              \n\t"        \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm21              \n\t"        \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm22              \n\t"        \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm23              \n\t"        \
    "movq %0, %%rax  \n\t"                                        \
    "addq $0x4, %%rax  \n\t"                                      \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "prefetcht0 0x180(%1)                              \n\t"      \
    "prefetcht0 0x1C0(%1)                              \n\t"      \
    "prefetcht0 0x200(%1)                              \n\t"      \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm0              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm1              \n\t"         \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm2              \n\t"         \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm3              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm4              \n\t"         \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm5              \n\t"         \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "vmovups 0xC0(%1), %%zmm24                             \n\t"  \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm6              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm7              \n\t"         \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm8              \n\t"         \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm9              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm10              \n\t"        \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm11              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "vmovups 0x100(%1), %%zmm25                             \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm12              \n\t"        \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm13              \n\t"        \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm14              \n\t"        \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm15              \n\t"        \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm16              \n\t"        \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm17              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "vmovups 0x140(%1), %%zmm26                             \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm18              \n\t"        \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm19              \n\t"        \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm20              \n\t"        \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm21              \n\t"        \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm22              \n\t"        \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm23              \n\t"
#else
#define mmmKernel8x48                                             \
    "movq %0, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "prefetcht0 0xC0(%1)                              \n\t"       \
    "prefetcht0 0x100(%1)                              \n\t"      \
    "prefetcht0 0x140(%1)                              \n\t"      \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm0, %%zmm27, %%zmm0              \n\t"            \
    "vpaddd %%zmm1, %%zmm28, %%zmm1              \n\t"            \
    "vpaddd %%zmm2, %%zmm29, %%zmm2              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"            \
    "vpaddd %%zmm4, %%zmm28, %%zmm4              \n\t"            \
    "vpaddd %%zmm5, %%zmm29, %%zmm5              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm6, %%zmm27, %%zmm6              \n\t"            \
    "vpaddd %%zmm7, %%zmm28, %%zmm7              \n\t"            \
    "vpaddd %%zmm8, %%zmm29, %%zmm8              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpaddd %%zmm9, %%zmm27, %%zmm9              \n\t"            \
    "vpaddd %%zmm10, %%zmm28, %%zmm10              \n\t"          \
    "vpaddd %%zmm11, %%zmm29, %%zmm11              \n\t"          \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm12, %%zmm27, %%zmm12              \n\t"          \
    "vpaddd %%zmm13, %%zmm28, %%zmm13              \n\t"          \
    "vpaddd %%zmm14, %%zmm29, %%zmm14              \n\t"          \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpaddd %%zmm15, %%zmm27, %%zmm15              \n\t"          \
    "vpaddd %%zmm16, %%zmm28, %%zmm16              \n\t"          \
    "vpaddd %%zmm17, %%zmm29, %%zmm17              \n\t"          \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm18, %%zmm27, %%zmm18              \n\t"          \
    "vpaddd %%zmm19, %%zmm28, %%zmm19              \n\t"          \
    "vpaddd %%zmm20, %%zmm29, %%zmm20              \n\t"          \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vmovups (%1), %%zmm24                             \n\t"      \
    "vmovups 0x40(%1), %%zmm25                             \n\t"  \
    "vmovups 0x80(%1), %%zmm26                             \n\t"  \
    "vpaddd %%zmm21, %%zmm27, %%zmm21              \n\t"          \
    "vpaddd %%zmm22, %%zmm28, %%zmm22              \n\t"          \
    "vpaddd %%zmm23, %%zmm29, %%zmm23              \n\t"          \
    "movq %0, %%rax  \n\t"                                        \
    "addq $0x4, %%rax  \n\t"                                      \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "prefetcht0 0x180(%1)                              \n\t"      \
    "prefetcht0 0x1C0(%1)                              \n\t"      \
    "prefetcht0 0x200(%1)                              \n\t"      \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm0, %%zmm27, %%zmm0              \n\t"            \
    "vpaddd %%zmm1, %%zmm28, %%zmm1              \n\t"            \
    "vpaddd %%zmm2, %%zmm29, %%zmm2              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"            \
    "vpaddd %%zmm4, %%zmm28, %%zmm4              \n\t"            \
    "vpaddd %%zmm5, %%zmm29, %%zmm5              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm6, %%zmm27, %%zmm6              \n\t"            \
    "vpaddd %%zmm7, %%zmm28, %%zmm7              \n\t"            \
    "vpaddd %%zmm8, %%zmm29, %%zmm8              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpaddd %%zmm9, %%zmm27, %%zmm9              \n\t"            \
    "vpaddd %%zmm10, %%zmm28, %%zmm10              \n\t"          \
    "vpaddd %%zmm11, %%zmm29, %%zmm11              \n\t"          \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm12, %%zmm27, %%zmm12              \n\t"          \
    "vpaddd %%zmm13, %%zmm28, %%zmm13              \n\t"          \
    "vpaddd %%zmm14, %%zmm29, %%zmm14              \n\t"          \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpaddd %%zmm15, %%zmm27, %%zmm15              \n\t"          \
    "vpaddd %%zmm16, %%zmm28, %%zmm16              \n\t"          \
    "vpaddd %%zmm17, %%zmm29, %%zmm17              \n\t"          \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm18, %%zmm27, %%zmm18              \n\t"          \
    "vpaddd %%zmm19, %%zmm28, %%zmm19              \n\t"          \
    "vpaddd %%zmm20, %%zmm29, %%zmm20              \n\t"          \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vmovups 0xC0(%1), %%zmm24                             \n\t"  \
    "vmovups 0x100(%1), %%zmm25                             \n\t" \
    "vmovups 0x140(%1), %%zmm26                             \n\t" \
    "vpaddd %%zmm21, %%zmm27, %%zmm21              \n\t"          \
    "vpaddd %%zmm22, %%zmm28, %%zmm22              \n\t"          \
    "vpaddd %%zmm23, %%zmm29, %%zmm23              \n\t"
#endif

inline void mmm_avx512_8x48_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0xC0(%1)                              \n\t"
        "prefetcht0 0x100(%1)                              \n\t"
        "prefetcht0 0x140(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "vmovups 0x40(%1), %%zmm25                             \n\t"
        "vmovups 0x80(%1), %%zmm26                             \n\t"
        "add $0xC0, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%eax \n\t"
        "vmovd %%eax, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif

        "movq %%rbx, %%rax          \n\t"
        "andq $0x1, %%rax          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%7), %%zmm0                       \n\t"
        "vmovups 0x40(%7), %%zmm1                   \n\t"
        "vmovups 0x80(%7), %%zmm2                   \n\t"
        "vmovups %%zmm0, %%zmm3                   \n\t"
        "vmovups %%zmm1, %%zmm4                   \n\t"
        "vmovups %%zmm2, %%zmm5                   \n\t"
        "vmovups %%zmm0, %%zmm6                   \n\t"
        "vmovups %%zmm1, %%zmm7                   \n\t"
        "vmovups %%zmm2, %%zmm8                   \n\t"
        "vmovups %%zmm0, %%zmm9                   \n\t"
        "vmovups %%zmm1, %%zmm10                   \n\t"
        "vmovups %%zmm2, %%zmm11                   \n\t"
        "vmovups %%zmm0, %%zmm12                   \n\t"
        "vmovups %%zmm1, %%zmm13                   \n\t"
        "vmovups %%zmm2, %%zmm14                   \n\t"
        "vmovups %%zmm0, %%zmm15                   \n\t"
        "vmovups %%zmm1, %%zmm16                   \n\t"
        "vmovups %%zmm2, %%zmm17                   \n\t"
        "vmovups %%zmm0, %%zmm18                   \n\t"
        "vmovups %%zmm1, %%zmm19                   \n\t"
        "vmovups %%zmm2, %%zmm20                   \n\t"
        "vmovups %%zmm0, %%zmm21                   \n\t"
        "vmovups %%zmm1, %%zmm22                   \n\t"
        "vmovups %%zmm2, %%zmm23                   \n\t"
        "jmp 1f          \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
        "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
        "vxorps %%zmm4, %%zmm4, %%zmm4                     \n\t"
        "vxorps %%zmm5, %%zmm5, %%zmm5                     \n\t"
        "vxorps %%zmm6, %%zmm6, %%zmm6                     \n\t"
        "vxorps %%zmm7, %%zmm7, %%zmm7                     \n\t"
        "vxorps %%zmm8, %%zmm8, %%zmm8                     \n\t"
        "vxorps %%zmm9, %%zmm9, %%zmm9                     \n\t"
        "vxorps %%zmm10, %%zmm10, %%zmm10                  \n\t"
        "vxorps %%zmm11, %%zmm11, %%zmm11                  \n\t"
        "vxorps %%zmm12, %%zmm12, %%zmm12                  \n\t"
        "vxorps %%zmm13, %%zmm13, %%zmm13                  \n\t"
        "vxorps %%zmm14, %%zmm14, %%zmm14                  \n\t"
        "vxorps %%zmm15, %%zmm15, %%zmm15                  \n\t"
        "vxorps %%zmm16, %%zmm16, %%zmm16                  \n\t"
        "vxorps %%zmm17, %%zmm17, %%zmm17                  \n\t"
        "vxorps %%zmm18, %%zmm18, %%zmm18                  \n\t"
        "vxorps %%zmm19, %%zmm19, %%zmm19                  \n\t"
        "vxorps %%zmm20, %%zmm20, %%zmm20                  \n\t"
        "vxorps %%zmm21, %%zmm21, %%zmm21                  \n\t"
        "vxorps %%zmm22, %%zmm22, %%zmm22                  \n\t"
        "vxorps %%zmm23, %%zmm23, %%zmm23                  \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 0x80(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "prefetcht0 0x80(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 0x80(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "prefetcht0 0x80(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 0x80(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "prefetcht0 0x80(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 0x80(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "prefetcht0 0x80(%%rax, %4)                              \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel8x48

        "add $0x180, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "movq %2, %%rax  \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpaddd (%%rax), %%zmm0, %%zmm0                       \n\t"
        "vpaddd 0x40(%%rax), %%zmm1, %%zmm1                   \n\t"
        "vpaddd 0x80(%%rax), %%zmm2, %%zmm2                   \n\t"
        "vpaddd (%%rax, %4), %%zmm3, %%zmm3                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm4, %%zmm4                   \n\t"
        "vpaddd 0x80(%%rax, %4), %%zmm5, %%zmm5                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm6, %%zmm6                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm7, %%zmm7                   \n\t"
        "vpaddd 0x80(%%rax), %%zmm8, %%zmm8                   \n\t"
        "vpaddd (%%rax, %4), %%zmm9, %%zmm9                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm10, %%zmm10                   \n\t"
        "vpaddd 0x80(%%rax, %4), %%zmm11, %%zmm11                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm12, %%zmm12                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm13, %%zmm13                   \n\t"
        "vpaddd 0x80(%%rax), %%zmm14, %%zmm14                   \n\t"
        "vpaddd (%%rax, %4), %%zmm15, %%zmm15                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm16, %%zmm16                   \n\t"
        "vpaddd 0x80(%%rax, %4), %%zmm17, %%zmm17                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm18, %%zmm18                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm19, %%zmm19                   \n\t"
        "vpaddd 0x80(%%rax), %%zmm20, %%zmm20                   \n\t"
        "vpaddd (%%rax, %4), %%zmm21, %%zmm21                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm22, %%zmm22                   \n\t"
        "vpaddd 0x80(%%rax, %4), %%zmm23, %%zmm23                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
        "vcvtdq2ps %%zmm2, %%zmm2                       \n\t"
        "vcvtdq2ps %%zmm3, %%zmm3                       \n\t"
        "vcvtdq2ps %%zmm4, %%zmm4                       \n\t"
        "vcvtdq2ps %%zmm5, %%zmm5                       \n\t"
        "vcvtdq2ps %%zmm6, %%zmm6                       \n\t"
        "vcvtdq2ps %%zmm7, %%zmm7                       \n\t"
        "vcvtdq2ps %%zmm8, %%zmm8                       \n\t"
        "vcvtdq2ps %%zmm9, %%zmm9                       \n\t"
        "vcvtdq2ps %%zmm10, %%zmm10                       \n\t"
        "vcvtdq2ps %%zmm11, %%zmm11                       \n\t"
        "vcvtdq2ps %%zmm12, %%zmm12                       \n\t"
        "vcvtdq2ps %%zmm13, %%zmm13                       \n\t"
        "vcvtdq2ps %%zmm14, %%zmm14                       \n\t"
        "vcvtdq2ps %%zmm15, %%zmm15                       \n\t"
        "vcvtdq2ps %%zmm16, %%zmm16                       \n\t"
        "vcvtdq2ps %%zmm17, %%zmm17                       \n\t"
        "vcvtdq2ps %%zmm18, %%zmm18                       \n\t"
        "vcvtdq2ps %%zmm19, %%zmm19                       \n\t"
        "vcvtdq2ps %%zmm20, %%zmm20                       \n\t"
        "vcvtdq2ps %%zmm21, %%zmm21                       \n\t"
        "vcvtdq2ps %%zmm22, %%zmm22                       \n\t"
        "vcvtdq2ps %%zmm23, %%zmm23                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vmulps %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vmulps %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vmulps %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vmulps %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vmulps %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vmulps %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vmulps %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vmulps %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vmulps %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vmulps %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vmulps %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "vmulps %%zmm12, %%zmm24, %%zmm12                     \n\t"
        "vmulps %%zmm13, %%zmm24, %%zmm13                     \n\t"
        "vmulps %%zmm14, %%zmm24, %%zmm14                     \n\t"
        "vmulps %%zmm15, %%zmm24, %%zmm15                     \n\t"
        "vmulps %%zmm16, %%zmm24, %%zmm16                     \n\t"
        "vmulps %%zmm17, %%zmm24, %%zmm17                     \n\t"
        "vmulps %%zmm18, %%zmm24, %%zmm18                     \n\t"
        "vmulps %%zmm19, %%zmm24, %%zmm19                     \n\t"
        "vmulps %%zmm20, %%zmm24, %%zmm20                     \n\t"
        "vmulps %%zmm21, %%zmm24, %%zmm21                     \n\t"
        "vmulps %%zmm22, %%zmm24, %%zmm22                     \n\t"
        "vmulps %%zmm23, %%zmm24, %%zmm23                     \n\t"

        "movq %%rbx, %%rax          \n\t"
        "andq $0x2, %%rax          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
        "vcvtps2dq %%zmm4, %%zmm4                       \n\t"
        "vcvtps2dq %%zmm5, %%zmm5                       \n\t"
        "vcvtps2dq %%zmm6, %%zmm6                       \n\t"
        "vcvtps2dq %%zmm7, %%zmm7                       \n\t"
        "vcvtps2dq %%zmm8, %%zmm8                       \n\t"
        "vcvtps2dq %%zmm9, %%zmm9                       \n\t"
        "vcvtps2dq %%zmm10, %%zmm10                       \n\t"
        "vcvtps2dq %%zmm11, %%zmm11                       \n\t"
        "vcvtps2dq %%zmm12, %%zmm12                       \n\t"
        "vcvtps2dq %%zmm13, %%zmm13                       \n\t"
        "vcvtps2dq %%zmm14, %%zmm14                       \n\t"
        "vcvtps2dq %%zmm15, %%zmm15                       \n\t"
        "vcvtps2dq %%zmm16, %%zmm16                       \n\t"
        "vcvtps2dq %%zmm17, %%zmm17                       \n\t"
        "vcvtps2dq %%zmm18, %%zmm18                       \n\t"
        "vcvtps2dq %%zmm19, %%zmm19                       \n\t"
        "vcvtps2dq %%zmm20, %%zmm20                       \n\t"
        "vcvtps2dq %%zmm21, %%zmm21                       \n\t"
        "vcvtps2dq %%zmm22, %%zmm22                       \n\t"
        "vcvtps2dq %%zmm23, %%zmm23                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpaddd %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vpaddd %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vpaddd %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vpaddd %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vpaddd %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vpaddd %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vpaddd %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vpaddd %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "vpaddd %%zmm12, %%zmm24, %%zmm12                     \n\t"
        "vpaddd %%zmm13, %%zmm24, %%zmm13                     \n\t"
        "vpaddd %%zmm14, %%zmm24, %%zmm14                     \n\t"
        "vpaddd %%zmm15, %%zmm24, %%zmm15                     \n\t"
        "vpaddd %%zmm16, %%zmm24, %%zmm16                     \n\t"
        "vpaddd %%zmm17, %%zmm24, %%zmm17                     \n\t"
        "vpaddd %%zmm18, %%zmm24, %%zmm18                     \n\t"
        "vpaddd %%zmm19, %%zmm24, %%zmm19                     \n\t"
        "vpaddd %%zmm20, %%zmm24, %%zmm20                     \n\t"
        "vpaddd %%zmm21, %%zmm24, %%zmm21                     \n\t"
        "vpaddd %%zmm22, %%zmm24, %%zmm22                     \n\t"
        "vpaddd %%zmm23, %%zmm24, %%zmm23                     \n\t"
        "movq %9, %%rax  \n\t"
        "shr $2, %4                                     \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpmovusdb %%zmm0,  (%%rax)                             \n\t"
        "vpmovusdb %%zmm1,  0x10(%%rax)                         \n\t"
        "vpmovusdb %%zmm2,  0x20(%%rax)                         \n\t"
        "vpmovusdb %%zmm3,  (%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm4,  0x10(%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm5,  0x20(%%rax, %4)                         \n\t"
        "add %%rcx, %%rax                                     \n\t"
        "vpmovusdb %%zmm6,  (%%rax)                         \n\t"
        "vpmovusdb %%zmm7,  0x10(%%rax)                         \n\t"
        "vpmovusdb %%zmm8,  0x20(%%rax)                         \n\t"
        "vpmovusdb %%zmm9,  (%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm10,  0x10(%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm11,  0x20(%%rax, %4)                         \n\t"
        "add %%rcx, %%rax                                     \n\t"
        "vpmovusdb %%zmm12,  (%%rax)                         \n\t"
        "vpmovusdb %%zmm13,  0x10(%%rax)                         \n\t"
        "vpmovusdb %%zmm14,  0x20(%%rax)                         \n\t"
        "vpmovusdb %%zmm15,  (%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm16,  0x10(%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm17,  0x20(%%rax, %4)                         \n\t"
        "add %%rcx, %%rax                                     \n\t"
        "vpmovusdb %%zmm18,  (%%rax)                             \n\t"
        "vpmovusdb %%zmm19,  0x10(%%rax)                             \n\t"
        "vpmovusdb %%zmm20,  0x20(%%rax)                             \n\t"
        "vpmovusdb %%zmm21,  (%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm22,  0x10(%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm23,  0x20(%%rax, %4)                         \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "vmovups %%zmm0,  (%%rax)                             \n\t"
        "vmovups %%zmm1,  0x40(%%rax)                         \n\t"
        "vmovups %%zmm2,  0x80(%%rax)                         \n\t"
        "vmovups %%zmm3,  (%%rax, %4)                         \n\t"
        "vmovups %%zmm4,  0x40(%%rax, %4)                         \n\t"
        "vmovups %%zmm5,  0x80(%%rax, %4)                         \n\t"
        "add %%rcx, %%rax                                     \n\t"
        "vmovups %%zmm6,  (%%rax)                         \n\t"
        "vmovups %%zmm7,  0x40(%%rax)                         \n\t"
        "vmovups %%zmm8,  0x80(%%rax)                         \n\t"
        "vmovups %%zmm9,  (%%rax, %4)                         \n\t"
        "vmovups %%zmm10,  0x40(%%rax, %4)                         \n\t"
        "vmovups %%zmm11,  0x80(%%rax, %4)                         \n\t"
        "add %%rcx, %%rax                                     \n\t"
        "vmovups %%zmm12,  (%%rax)                         \n\t"
        "vmovups %%zmm13,  0x40(%%rax)                         \n\t"
        "vmovups %%zmm14,  0x80(%%rax)                         \n\t"
        "vmovups %%zmm15,  (%%rax, %4)                         \n\t"
        "vmovups %%zmm16,  0x40(%%rax, %4)                         \n\t"
        "vmovups %%zmm17,  0x80(%%rax, %4)                         \n\t"
        "add %%rcx, %%rax                                     \n\t"
        "vmovups %%zmm18,  (%%rax)                             \n\t"
        "vmovups %%zmm19,  0x40(%%rax)                             \n\t"
        "vmovups %%zmm20,  0x80(%%rax)                             \n\t"
        "vmovups %%zmm21,  (%%rax, %4)                         \n\t"
        "vmovups %%zmm22,  0x40(%%rax, %4)                         \n\t"
        "vmovups %%zmm23,  0x80(%%rax, %4)                         \n\t"
        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"((int64_t)stepK), "r"(offsetC), "b"((int64_t)flags), "r"(u8Result)
        : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8",
        "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",
        "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
        "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel12x32                                              \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "prefetcht0 0xC0(%1)                              \n\t"         \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm0              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm1              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm2              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm3              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm4              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm5              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm6              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm7              \n\t"           \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31                     \n\t" \
    "vmovups (%1), %%zmm26                             \n\t"        \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm8              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm9              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm10              \n\t"          \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm11              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm12              \n\t"          \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm13              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm14              \n\t"          \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm15              \n\t"          \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31                     \n\t" \
    "vmovups 0x40(%1), %%zmm27                             \n\t"    \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm16              \n\t"          \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm17              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm18              \n\t"          \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm19              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm20              \n\t"          \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm21              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm22              \n\t"          \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm23              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31                     \n\t" \
    "prefetcht0 0x100(%1)                              \n\t"        \
    "prefetcht0 0x140(%1)                              \n\t"        \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm0              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm1              \n\t"           \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm2              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm3              \n\t"           \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm4              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm5              \n\t"           \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm6              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm7              \n\t"           \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31                     \n\t" \
    "vmovups 0x80(%1), %%zmm24                             \n\t"    \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm8              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm9              \n\t"           \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm10              \n\t"          \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm11              \n\t"          \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm12              \n\t"          \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm13              \n\t"          \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm14              \n\t"          \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm15              \n\t"          \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31                     \n\t" \
    "vmovups 0xC0(%1), %%zmm25                             \n\t"    \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm16              \n\t"          \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm17              \n\t"          \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm18              \n\t"          \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm19              \n\t"          \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm20              \n\t"          \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm21              \n\t"          \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm22              \n\t"          \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm23              \n\t"
#else
#define mmmKernel12x32                                              \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "prefetcht0 0xC0(%1)                              \n\t"         \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"              \
    "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"              \
    "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"              \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"        \
    "vpbroadcastd (%%rax, %%rbx), %%zmm29                     \n\t" \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpaddd %%zmm3, %%zmm26, %%zmm3              \n\t"              \
    "vpaddd %%zmm4, %%zmm27, %%zmm4              \n\t"              \
    "vpaddd %%zmm5, %%zmm28, %%zmm5              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpaddd %%zmm6, %%zmm26, %%zmm6              \n\t"              \
    "vpaddd %%zmm7, %%zmm27, %%zmm7              \n\t"              \
    "vpaddd %%zmm8, %%zmm28, %%zmm8              \n\t"              \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm28              \n\t"        \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %%rbx), %%zmm29                     \n\t" \
    "vpaddd %%zmm9, %%zmm26, %%zmm9              \n\t"              \
    "vpaddd %%zmm10, %%zmm27, %%zmm10              \n\t"            \
    "vpaddd %%zmm11, %%zmm28, %%zmm11              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpaddd %%zmm12, %%zmm26, %%zmm12              \n\t"            \
    "vpaddd %%zmm13, %%zmm27, %%zmm13              \n\t"            \
    "vpaddd %%zmm14, %%zmm28, %%zmm14              \n\t"            \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpaddd %%zmm15, %%zmm26, %%zmm15              \n\t"            \
    "vpaddd %%zmm16, %%zmm27, %%zmm16              \n\t"            \
    "vpaddd %%zmm17, %%zmm28, %%zmm17              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %%rbx), %%zmm29                     \n\t" \
    "vpaddd %%zmm18, %%zmm26, %%zmm18              \n\t"            \
    "vpaddd %%zmm19, %%zmm27, %%zmm19              \n\t"            \
    "vpaddd %%zmm20, %%zmm28, %%zmm20              \n\t"            \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm28              \n\t"        \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vmovups (%1), %%zmm24                             \n\t"        \
    "vmovups 0x40(%1), %%zmm25                             \n\t"    \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpaddd %%zmm21, %%zmm26, %%zmm21              \n\t"            \
    "vpaddd %%zmm22, %%zmm27, %%zmm22              \n\t"            \
    "vpaddd %%zmm23, %%zmm28, %%zmm23              \n\t"            \
    "prefetcht0 0x100(%1)                              \n\t"        \
    "prefetcht0 0x140(%1)                              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"              \
    "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"              \
    "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"              \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"        \
    "vpbroadcastd (%%rax, %%rbx), %%zmm29                     \n\t" \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpaddd %%zmm3, %%zmm26, %%zmm3              \n\t"              \
    "vpaddd %%zmm4, %%zmm27, %%zmm4              \n\t"              \
    "vpaddd %%zmm5, %%zmm28, %%zmm5              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpaddd %%zmm6, %%zmm26, %%zmm6              \n\t"              \
    "vpaddd %%zmm7, %%zmm27, %%zmm7              \n\t"              \
    "vpaddd %%zmm8, %%zmm28, %%zmm8              \n\t"              \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm28              \n\t"        \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %%rbx), %%zmm29                     \n\t" \
    "vpaddd %%zmm9, %%zmm26, %%zmm9              \n\t"              \
    "vpaddd %%zmm10, %%zmm27, %%zmm10              \n\t"            \
    "vpaddd %%zmm11, %%zmm28, %%zmm11              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpaddd %%zmm12, %%zmm26, %%zmm12              \n\t"            \
    "vpaddd %%zmm13, %%zmm27, %%zmm13              \n\t"            \
    "vpaddd %%zmm14, %%zmm28, %%zmm14              \n\t"            \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpaddd %%zmm15, %%zmm26, %%zmm15              \n\t"            \
    "vpaddd %%zmm16, %%zmm27, %%zmm16              \n\t"            \
    "vpaddd %%zmm17, %%zmm28, %%zmm17              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %%rbx), %%zmm29                     \n\t" \
    "vpaddd %%zmm18, %%zmm26, %%zmm18              \n\t"            \
    "vpaddd %%zmm19, %%zmm27, %%zmm19              \n\t"            \
    "vpaddd %%zmm20, %%zmm28, %%zmm20              \n\t"            \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vmovups 0x80(%1), %%zmm24                             \n\t"    \
    "vmovups 0xC0(%1), %%zmm25                             \n\t"    \
    "vpaddd %%zmm21, %%zmm26, %%zmm21              \n\t"            \
    "vpaddd %%zmm22, %%zmm27, %%zmm22              \n\t"            \
    "vpaddd %%zmm23, %%zmm28, %%zmm23              \n\t"
#endif

inline void mmm_avx512_12x32_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x80(%1)                              \n\t"
        "prefetcht0 0xC0(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "vmovups 0x40(%1), %%zmm25                             \n\t"
        "add $0x80, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%ebx \n\t"
        "vmovd %%ebx, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif

        "movq %8, %%rbx          \n\t"
        "andq $0x1, %%rbx          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%7), %%zmm0                       \n\t"
        "vmovups 0x40(%7), %%zmm1                   \n\t"
        "vmovups %%zmm0, %%zmm2                   \n\t"
        "vmovups %%zmm1, %%zmm3                   \n\t"
        "vmovups %%zmm0, %%zmm4                   \n\t"
        "vmovups %%zmm1, %%zmm5                   \n\t"
        "vmovups %%zmm0, %%zmm6                   \n\t"
        "vmovups %%zmm1, %%zmm7                   \n\t"
        "vmovups %%zmm0, %%zmm8                   \n\t"
        "vmovups %%zmm1, %%zmm9                   \n\t"
        "vmovups %%zmm0, %%zmm10                   \n\t"
        "vmovups %%zmm1, %%zmm11                   \n\t"
        "vmovups %%zmm0, %%zmm12                   \n\t"
        "vmovups %%zmm1, %%zmm13                   \n\t"
        "vmovups %%zmm0, %%zmm14                   \n\t"
        "vmovups %%zmm1, %%zmm15                   \n\t"
        "vmovups %%zmm0, %%zmm16                   \n\t"
        "vmovups %%zmm1, %%zmm17                   \n\t"
        "vmovups %%zmm0, %%zmm18                   \n\t"
        "vmovups %%zmm1, %%zmm19                   \n\t"
        "vmovups %%zmm0, %%zmm20                   \n\t"
        "vmovups %%zmm1, %%zmm21                   \n\t"
        "vmovups %%zmm0, %%zmm22                   \n\t"
        "vmovups %%zmm1, %%zmm23                   \n\t"
        "jmp 1f          \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
        "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
        "vxorps %%zmm4, %%zmm4, %%zmm4                     \n\t"
        "vxorps %%zmm5, %%zmm5, %%zmm5                     \n\t"
        "vxorps %%zmm6, %%zmm6, %%zmm6                     \n\t"
        "vxorps %%zmm7, %%zmm7, %%zmm7                     \n\t"
        "vxorps %%zmm8, %%zmm8, %%zmm8                     \n\t"
        "vxorps %%zmm9, %%zmm9, %%zmm9                     \n\t"
        "vxorps %%zmm10, %%zmm10, %%zmm10                  \n\t"
        "vxorps %%zmm11, %%zmm11, %%zmm11                  \n\t"
        "vxorps %%zmm12, %%zmm12, %%zmm12                  \n\t"
        "vxorps %%zmm13, %%zmm13, %%zmm13                  \n\t"
        "vxorps %%zmm14, %%zmm14, %%zmm14                  \n\t"
        "vxorps %%zmm15, %%zmm15, %%zmm15                  \n\t"
        "vxorps %%zmm16, %%zmm16, %%zmm16                  \n\t"
        "vxorps %%zmm17, %%zmm17, %%zmm17                  \n\t"
        "vxorps %%zmm18, %%zmm18, %%zmm18                  \n\t"
        "vxorps %%zmm19, %%zmm19, %%zmm19                  \n\t"
        "vxorps %%zmm20, %%zmm20, %%zmm20                  \n\t"
        "vxorps %%zmm21, %%zmm21, %%zmm21                  \n\t"
        "vxorps %%zmm22, %%zmm22, %%zmm22                  \n\t"
        "vxorps %%zmm23, %%zmm23, %%zmm23                  \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "movq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel12x32

        "add $0x100, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "movq %2, %%rax  \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpaddd (%%rax), %%zmm0, %%zmm0                       \n\t"
        "vpaddd 0x40(%%rax), %%zmm1, %%zmm1                   \n\t"
        "vpaddd (%%rax, %4), %%zmm2, %%zmm2                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm3, %%zmm3                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm4, %%zmm4                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm5, %%zmm5                   \n\t"
        "vpaddd (%%rax, %4), %%zmm6, %%zmm6                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm7, %%zmm7                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm8, %%zmm8                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm9, %%zmm9                   \n\t"
        "vpaddd (%%rax, %4), %%zmm10, %%zmm10                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm11, %%zmm11                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm12, %%zmm12                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm13, %%zmm13                   \n\t"
        "vpaddd (%%rax, %4), %%zmm14, %%zmm14                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm15, %%zmm15                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm16, %%zmm16                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm17, %%zmm17                   \n\t"
        "vpaddd (%%rax, %4), %%zmm18, %%zmm18                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm19, %%zmm19                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm20, %%zmm20                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm21, %%zmm21                   \n\t"
        "vpaddd (%%rax, %4), %%zmm22, %%zmm22                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm23, %%zmm23                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
        "vcvtdq2ps %%zmm2, %%zmm2                       \n\t"
        "vcvtdq2ps %%zmm3, %%zmm3                       \n\t"
        "vcvtdq2ps %%zmm4, %%zmm4                       \n\t"
        "vcvtdq2ps %%zmm5, %%zmm5                       \n\t"
        "vcvtdq2ps %%zmm6, %%zmm6                       \n\t"
        "vcvtdq2ps %%zmm7, %%zmm7                       \n\t"
        "vcvtdq2ps %%zmm8, %%zmm8                       \n\t"
        "vcvtdq2ps %%zmm9, %%zmm9                       \n\t"
        "vcvtdq2ps %%zmm10, %%zmm10                       \n\t"
        "vcvtdq2ps %%zmm11, %%zmm11                       \n\t"
        "vcvtdq2ps %%zmm12, %%zmm12                       \n\t"
        "vcvtdq2ps %%zmm13, %%zmm13                       \n\t"
        "vcvtdq2ps %%zmm14, %%zmm14                       \n\t"
        "vcvtdq2ps %%zmm15, %%zmm15                       \n\t"
        "vcvtdq2ps %%zmm16, %%zmm16                       \n\t"
        "vcvtdq2ps %%zmm17, %%zmm17                       \n\t"
        "vcvtdq2ps %%zmm18, %%zmm18                       \n\t"
        "vcvtdq2ps %%zmm19, %%zmm19                       \n\t"
        "vcvtdq2ps %%zmm20, %%zmm20                       \n\t"
        "vcvtdq2ps %%zmm21, %%zmm21                       \n\t"
        "vcvtdq2ps %%zmm22, %%zmm22                       \n\t"
        "vcvtdq2ps %%zmm23, %%zmm23                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vmulps %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vmulps %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vmulps %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vmulps %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vmulps %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vmulps %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vmulps %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vmulps %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vmulps %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vmulps %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vmulps %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "vmulps %%zmm12, %%zmm24, %%zmm12                     \n\t"
        "vmulps %%zmm13, %%zmm24, %%zmm13                     \n\t"
        "vmulps %%zmm14, %%zmm24, %%zmm14                     \n\t"
        "vmulps %%zmm15, %%zmm24, %%zmm15                     \n\t"
        "vmulps %%zmm16, %%zmm24, %%zmm16                     \n\t"
        "vmulps %%zmm17, %%zmm24, %%zmm17                     \n\t"
        "vmulps %%zmm18, %%zmm24, %%zmm18                     \n\t"
        "vmulps %%zmm19, %%zmm24, %%zmm19                     \n\t"
        "vmulps %%zmm20, %%zmm24, %%zmm20                     \n\t"
        "vmulps %%zmm21, %%zmm24, %%zmm21                     \n\t"
        "vmulps %%zmm22, %%zmm24, %%zmm22                     \n\t"
        "vmulps %%zmm23, %%zmm24, %%zmm23                     \n\t"

        "movq %8, %%rbx          \n\t"
        "andq $0x2, %%rbx          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
        "vcvtps2dq %%zmm4, %%zmm4                       \n\t"
        "vcvtps2dq %%zmm5, %%zmm5                       \n\t"
        "vcvtps2dq %%zmm6, %%zmm6                       \n\t"
        "vcvtps2dq %%zmm7, %%zmm7                       \n\t"
        "vcvtps2dq %%zmm8, %%zmm8                       \n\t"
        "vcvtps2dq %%zmm9, %%zmm9                       \n\t"
        "vcvtps2dq %%zmm10, %%zmm10                       \n\t"
        "vcvtps2dq %%zmm11, %%zmm11                       \n\t"
        "vcvtps2dq %%zmm12, %%zmm12                       \n\t"
        "vcvtps2dq %%zmm13, %%zmm13                       \n\t"
        "vcvtps2dq %%zmm14, %%zmm14                       \n\t"
        "vcvtps2dq %%zmm15, %%zmm15                       \n\t"
        "vcvtps2dq %%zmm16, %%zmm16                       \n\t"
        "vcvtps2dq %%zmm17, %%zmm17                       \n\t"
        "vcvtps2dq %%zmm18, %%zmm18                       \n\t"
        "vcvtps2dq %%zmm19, %%zmm19                       \n\t"
        "vcvtps2dq %%zmm20, %%zmm20                       \n\t"
        "vcvtps2dq %%zmm21, %%zmm21                       \n\t"
        "vcvtps2dq %%zmm22, %%zmm22                       \n\t"
        "vcvtps2dq %%zmm23, %%zmm23                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpaddd %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vpaddd %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vpaddd %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vpaddd %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vpaddd %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vpaddd %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vpaddd %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vpaddd %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "vpaddd %%zmm12, %%zmm24, %%zmm12                     \n\t"
        "vpaddd %%zmm13, %%zmm24, %%zmm13                     \n\t"
        "vpaddd %%zmm14, %%zmm24, %%zmm14                     \n\t"
        "vpaddd %%zmm15, %%zmm24, %%zmm15                     \n\t"
        "vpaddd %%zmm16, %%zmm24, %%zmm16                     \n\t"
        "vpaddd %%zmm17, %%zmm24, %%zmm17                     \n\t"
        "vpaddd %%zmm18, %%zmm24, %%zmm18                     \n\t"
        "vpaddd %%zmm19, %%zmm24, %%zmm19                     \n\t"
        "vpaddd %%zmm20, %%zmm24, %%zmm20                     \n\t"
        "vpaddd %%zmm21, %%zmm24, %%zmm21                     \n\t"
        "vpaddd %%zmm22, %%zmm24, %%zmm22                     \n\t"
        "vpaddd %%zmm23, %%zmm24, %%zmm23                     \n\t"
        "movq %9, %%rax  \n\t"
        "shr $2, %4                                     \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpmovusdb %%zmm0, (%%rax)                       \n\t"
        "vpmovusdb %%zmm1, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm2, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm3, 0x10(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm4, (%%rax)                       \n\t"
        "vpmovusdb %%zmm5, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm6, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm7, 0x10(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm8, (%%rax)                       \n\t"
        "vpmovusdb %%zmm9, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm10, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm11, 0x10(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm12, (%%rax)                       \n\t"
        "vpmovusdb %%zmm13, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm14, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm15, 0x10(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm16, (%%rax)                       \n\t"
        "vpmovusdb %%zmm17, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm18, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm19, 0x10(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm20, (%%rax)                       \n\t"
        "vpmovusdb %%zmm21, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm22, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm23, 0x10(%%rax, %4)                       \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "vmovups %%zmm0, (%%rax)                       \n\t"
        "vmovups %%zmm1, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm2, (%%rax, %4)                       \n\t"
        "vmovups %%zmm3, 0x40(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm4, (%%rax)                       \n\t"
        "vmovups %%zmm5, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm6, (%%rax, %4)                       \n\t"
        "vmovups %%zmm7, 0x40(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm8, (%%rax)                       \n\t"
        "vmovups %%zmm9, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm10, (%%rax, %4)                       \n\t"
        "vmovups %%zmm11, 0x40(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm12, (%%rax)                       \n\t"
        "vmovups %%zmm13, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm14, (%%rax, %4)                       \n\t"
        "vmovups %%zmm15, 0x40(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm16, (%%rax)                       \n\t"
        "vmovups %%zmm17, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm18, (%%rax, %4)                       \n\t"
        "vmovups %%zmm19, 0x40(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm20, (%%rax)                       \n\t"
        "vmovups %%zmm21, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm22, (%%rax, %4)                       \n\t"
        "vmovups %%zmm23, 0x40(%%rax, %4)                       \n\t"

        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"((int64_t)stepK), "r"(offsetC), "r"((int64_t)flags), "r"(u8Result)
        : "%rax", "%rbx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
        "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
        "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
        "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel24x16                                              \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "vpdpbusd %%zmm24, %%zmm25, %%zmm0              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm26, %%zmm1              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm27, %%zmm2              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm3              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm4              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm5              \n\t"           \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpdpbusd %%zmm24, %%zmm25, %%zmm6              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm26, %%zmm7              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm27, %%zmm8              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm9              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm10              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm11              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vmovups (%1), %%zmm31                             \n\t"        \
    "vpdpbusd %%zmm24, %%zmm25, %%zmm12              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm26, %%zmm13              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm27, %%zmm14              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm15              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm16              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm17              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpdpbusd %%zmm24, %%zmm25, %%zmm18              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm26, %%zmm19              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm27, %%zmm20              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm21              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm22              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm23              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "prefetcht0 0xC0(%1)                              \n\t"         \
    "vpdpbusd %%zmm31, %%zmm25, %%zmm0              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm26, %%zmm1              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm27, %%zmm2              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm28, %%zmm3              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm29, %%zmm4              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm30, %%zmm5              \n\t"           \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpdpbusd %%zmm31, %%zmm25, %%zmm6              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm26, %%zmm7              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm27, %%zmm8              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm28, %%zmm9              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm29, %%zmm10              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm30, %%zmm11              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vmovups 0x40(%1), %%zmm24                             \n\t"    \
    "vpdpbusd %%zmm31, %%zmm25, %%zmm12              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm26, %%zmm13              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm27, %%zmm14              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm28, %%zmm15              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm29, %%zmm16              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm30, %%zmm17              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpdpbusd %%zmm31, %%zmm25, %%zmm18              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm26, %%zmm19              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm27, %%zmm20              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm28, %%zmm21              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm29, %%zmm22              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm30, %%zmm23              \n\t"
#else
#define mmmKernel24x16                                              \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm0, %%zmm28, %%zmm0              \n\t"              \
    "vpaddd %%zmm1, %%zmm29, %%zmm1              \n\t"              \
    "vpaddd %%zmm2, %%zmm30, %%zmm2              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm3, %%zmm28, %%zmm3              \n\t"              \
    "vpaddd %%zmm4, %%zmm29, %%zmm4              \n\t"              \
    "vpaddd %%zmm5, %%zmm30, %%zmm5              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm6, %%zmm28, %%zmm6              \n\t"              \
    "vpaddd %%zmm7, %%zmm29, %%zmm7              \n\t"              \
    "vpaddd %%zmm8, %%zmm30, %%zmm8              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm9, %%zmm28, %%zmm9              \n\t"              \
    "vpaddd %%zmm10, %%zmm29, %%zmm10              \n\t"            \
    "vpaddd %%zmm11, %%zmm30, %%zmm11              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm12, %%zmm28, %%zmm12              \n\t"            \
    "vpaddd %%zmm13, %%zmm29, %%zmm13              \n\t"            \
    "vpaddd %%zmm14, %%zmm30, %%zmm14              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm15, %%zmm28, %%zmm15              \n\t"            \
    "vpaddd %%zmm16, %%zmm29, %%zmm16              \n\t"            \
    "vpaddd %%zmm17, %%zmm30, %%zmm17              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm18, %%zmm28, %%zmm18              \n\t"            \
    "vpaddd %%zmm19, %%zmm29, %%zmm19              \n\t"            \
    "vpaddd %%zmm20, %%zmm30, %%zmm20              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vmovups (%1), %%zmm24                             \n\t"        \
    "vpaddd %%zmm21, %%zmm28, %%zmm21              \n\t"            \
    "vpaddd %%zmm22, %%zmm29, %%zmm22              \n\t"            \
    "vpaddd %%zmm23, %%zmm30, %%zmm23              \n\t"            \
    "prefetcht0 0xC0(%1)                              \n\t"         \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm0, %%zmm28, %%zmm0              \n\t"              \
    "vpaddd %%zmm1, %%zmm29, %%zmm1              \n\t"              \
    "vpaddd %%zmm2, %%zmm30, %%zmm2              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm3, %%zmm28, %%zmm3              \n\t"              \
    "vpaddd %%zmm4, %%zmm29, %%zmm4              \n\t"              \
    "vpaddd %%zmm5, %%zmm30, %%zmm5              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm6, %%zmm28, %%zmm6              \n\t"              \
    "vpaddd %%zmm7, %%zmm29, %%zmm7              \n\t"              \
    "vpaddd %%zmm8, %%zmm30, %%zmm8              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm9, %%zmm28, %%zmm9              \n\t"              \
    "vpaddd %%zmm10, %%zmm29, %%zmm10              \n\t"            \
    "vpaddd %%zmm11, %%zmm30, %%zmm11              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm12, %%zmm28, %%zmm12              \n\t"            \
    "vpaddd %%zmm13, %%zmm29, %%zmm13              \n\t"            \
    "vpaddd %%zmm14, %%zmm30, %%zmm14              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm15, %%zmm28, %%zmm15              \n\t"            \
    "vpaddd %%zmm16, %%zmm29, %%zmm16              \n\t"            \
    "vpaddd %%zmm17, %%zmm30, %%zmm17              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm18, %%zmm28, %%zmm18              \n\t"            \
    "vpaddd %%zmm19, %%zmm29, %%zmm19              \n\t"            \
    "vpaddd %%zmm20, %%zmm30, %%zmm20              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "vmovups 0x40(%1), %%zmm24                             \n\t"    \
    "vpaddd %%zmm21, %%zmm28, %%zmm21              \n\t"            \
    "vpaddd %%zmm22, %%zmm29, %%zmm22              \n\t"            \
    "vpaddd %%zmm23, %%zmm30, %%zmm23              \n\t"
#endif

inline void mmm_avx512_24x16_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x80(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "add $0x40, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%ebx \n\t"
        "vmovd %%ebx, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif
        "movq %8, %%rbx          \n\t"
        "andq $0x1, %%rbx          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%7), %%zmm0                       \n\t"
        "vmovups %%zmm0, %%zmm1                   \n\t"
        "vmovups %%zmm0, %%zmm2                   \n\t"
        "vmovups %%zmm0, %%zmm3                   \n\t"
        "vmovups %%zmm0, %%zmm4                   \n\t"
        "vmovups %%zmm0, %%zmm5                   \n\t"
        "vmovups %%zmm0, %%zmm6                   \n\t"
        "vmovups %%zmm0, %%zmm7                   \n\t"
        "vmovups %%zmm0, %%zmm8                   \n\t"
        "vmovups %%zmm0, %%zmm9                   \n\t"
        "vmovups %%zmm0, %%zmm10                   \n\t"
        "vmovups %%zmm0, %%zmm11                   \n\t"
        "vmovups %%zmm0, %%zmm12                   \n\t"
        "vmovups %%zmm0, %%zmm13                   \n\t"
        "vmovups %%zmm0, %%zmm14                   \n\t"
        "vmovups %%zmm0, %%zmm15                   \n\t"
        "vmovups %%zmm0, %%zmm16                   \n\t"
        "vmovups %%zmm0, %%zmm17                   \n\t"
        "vmovups %%zmm0, %%zmm18                   \n\t"
        "vmovups %%zmm0, %%zmm19                   \n\t"
        "vmovups %%zmm0, %%zmm20                   \n\t"
        "vmovups %%zmm0, %%zmm21                   \n\t"
        "vmovups %%zmm0, %%zmm22                   \n\t"
        "vmovups %%zmm0, %%zmm23                   \n\t"
        "jmp 1f          \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
        "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
        "vxorps %%zmm4, %%zmm4, %%zmm4                     \n\t"
        "vxorps %%zmm5, %%zmm5, %%zmm5                     \n\t"
        "vxorps %%zmm6, %%zmm6, %%zmm6                     \n\t"
        "vxorps %%zmm7, %%zmm7, %%zmm7                     \n\t"
        "vxorps %%zmm8, %%zmm8, %%zmm8                     \n\t"
        "vxorps %%zmm9, %%zmm9, %%zmm9                     \n\t"
        "vxorps %%zmm10, %%zmm10, %%zmm10                  \n\t"
        "vxorps %%zmm11, %%zmm11, %%zmm11                  \n\t"
        "vxorps %%zmm12, %%zmm12, %%zmm12                  \n\t"
        "vxorps %%zmm13, %%zmm13, %%zmm13                  \n\t"
        "vxorps %%zmm14, %%zmm14, %%zmm14                  \n\t"
        "vxorps %%zmm15, %%zmm15, %%zmm15                  \n\t"
        "vxorps %%zmm16, %%zmm16, %%zmm16                  \n\t"
        "vxorps %%zmm17, %%zmm17, %%zmm17                  \n\t"
        "vxorps %%zmm18, %%zmm18, %%zmm18                  \n\t"
        "vxorps %%zmm19, %%zmm19, %%zmm19                  \n\t"
        "vxorps %%zmm20, %%zmm20, %%zmm20                  \n\t"
        "vxorps %%zmm21, %%zmm21, %%zmm21                  \n\t"
        "vxorps %%zmm22, %%zmm22, %%zmm22                  \n\t"
        "vxorps %%zmm23, %%zmm23, %%zmm23                  \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "movq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel24x16

        "add $0x80, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "movq %2, %%rax  \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpaddd (%%rax), %%zmm0, %%zmm0                       \n\t"
        "vpaddd (%%rax, %4), %%zmm1, %%zmm1                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm2, %%zmm2                   \n\t"
        "vpaddd (%%rax, %4), %%zmm3, %%zmm3                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm4, %%zmm4                   \n\t"
        "vpaddd (%%rax, %4), %%zmm5, %%zmm5                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm6, %%zmm6                   \n\t"
        "vpaddd (%%rax, %4), %%zmm7, %%zmm7                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm8, %%zmm8                   \n\t"
        "vpaddd (%%rax, %4), %%zmm9, %%zmm9                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm10, %%zmm10                   \n\t"
        "vpaddd (%%rax, %4), %%zmm11, %%zmm11                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm12, %%zmm12                   \n\t"
        "vpaddd (%%rax, %4), %%zmm13, %%zmm13                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm14, %%zmm14                   \n\t"
        "vpaddd (%%rax, %4), %%zmm15, %%zmm15                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm16, %%zmm16                   \n\t"
        "vpaddd (%%rax, %4), %%zmm17, %%zmm17                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm18, %%zmm18                   \n\t"
        "vpaddd (%%rax, %4), %%zmm19, %%zmm19                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm20, %%zmm20                   \n\t"
        "vpaddd (%%rax, %4), %%zmm21, %%zmm21                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm22, %%zmm22                   \n\t"
        "vpaddd (%%rax, %4), %%zmm23, %%zmm23                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
        "vcvtdq2ps %%zmm2, %%zmm2                       \n\t"
        "vcvtdq2ps %%zmm3, %%zmm3                       \n\t"
        "vcvtdq2ps %%zmm4, %%zmm4                       \n\t"
        "vcvtdq2ps %%zmm5, %%zmm5                       \n\t"
        "vcvtdq2ps %%zmm6, %%zmm6                       \n\t"
        "vcvtdq2ps %%zmm7, %%zmm7                       \n\t"
        "vcvtdq2ps %%zmm8, %%zmm8                       \n\t"
        "vcvtdq2ps %%zmm9, %%zmm9                       \n\t"
        "vcvtdq2ps %%zmm10, %%zmm10                       \n\t"
        "vcvtdq2ps %%zmm11, %%zmm11                       \n\t"
        "vcvtdq2ps %%zmm12, %%zmm12                       \n\t"
        "vcvtdq2ps %%zmm13, %%zmm13                       \n\t"
        "vcvtdq2ps %%zmm14, %%zmm14                       \n\t"
        "vcvtdq2ps %%zmm15, %%zmm15                       \n\t"
        "vcvtdq2ps %%zmm16, %%zmm16                       \n\t"
        "vcvtdq2ps %%zmm17, %%zmm17                       \n\t"
        "vcvtdq2ps %%zmm18, %%zmm18                       \n\t"
        "vcvtdq2ps %%zmm19, %%zmm19                       \n\t"
        "vcvtdq2ps %%zmm20, %%zmm20                       \n\t"
        "vcvtdq2ps %%zmm21, %%zmm21                       \n\t"
        "vcvtdq2ps %%zmm22, %%zmm22                       \n\t"
        "vcvtdq2ps %%zmm23, %%zmm23                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vmulps %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vmulps %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vmulps %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vmulps %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vmulps %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vmulps %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vmulps %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vmulps %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vmulps %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vmulps %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vmulps %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "vmulps %%zmm12, %%zmm24, %%zmm12                     \n\t"
        "vmulps %%zmm13, %%zmm24, %%zmm13                     \n\t"
        "vmulps %%zmm14, %%zmm24, %%zmm14                     \n\t"
        "vmulps %%zmm15, %%zmm24, %%zmm15                     \n\t"
        "vmulps %%zmm16, %%zmm24, %%zmm16                     \n\t"
        "vmulps %%zmm17, %%zmm24, %%zmm17                     \n\t"
        "vmulps %%zmm18, %%zmm24, %%zmm18                     \n\t"
        "vmulps %%zmm19, %%zmm24, %%zmm19                     \n\t"
        "vmulps %%zmm20, %%zmm24, %%zmm20                     \n\t"
        "vmulps %%zmm21, %%zmm24, %%zmm21                     \n\t"
        "vmulps %%zmm22, %%zmm24, %%zmm22                     \n\t"
        "vmulps %%zmm23, %%zmm24, %%zmm23                     \n\t"

        "movq %8, %%rbx          \n\t"
        "andq $0x2, %%rbx          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
        "vcvtps2dq %%zmm4, %%zmm4                       \n\t"
        "vcvtps2dq %%zmm5, %%zmm5                       \n\t"
        "vcvtps2dq %%zmm6, %%zmm6                       \n\t"
        "vcvtps2dq %%zmm7, %%zmm7                       \n\t"
        "vcvtps2dq %%zmm8, %%zmm8                       \n\t"
        "vcvtps2dq %%zmm9, %%zmm9                       \n\t"
        "vcvtps2dq %%zmm10, %%zmm10                       \n\t"
        "vcvtps2dq %%zmm11, %%zmm11                       \n\t"
        "vcvtps2dq %%zmm12, %%zmm12                       \n\t"
        "vcvtps2dq %%zmm13, %%zmm13                       \n\t"
        "vcvtps2dq %%zmm14, %%zmm14                       \n\t"
        "vcvtps2dq %%zmm15, %%zmm15                       \n\t"
        "vcvtps2dq %%zmm16, %%zmm16                       \n\t"
        "vcvtps2dq %%zmm17, %%zmm17                       \n\t"
        "vcvtps2dq %%zmm18, %%zmm18                       \n\t"
        "vcvtps2dq %%zmm19, %%zmm19                       \n\t"
        "vcvtps2dq %%zmm20, %%zmm20                       \n\t"
        "vcvtps2dq %%zmm21, %%zmm21                       \n\t"
        "vcvtps2dq %%zmm22, %%zmm22                       \n\t"
        "vcvtps2dq %%zmm23, %%zmm23                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpaddd %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vpaddd %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vpaddd %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vpaddd %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vpaddd %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vpaddd %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vpaddd %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vpaddd %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "vpaddd %%zmm12, %%zmm24, %%zmm12                     \n\t"
        "vpaddd %%zmm13, %%zmm24, %%zmm13                     \n\t"
        "vpaddd %%zmm14, %%zmm24, %%zmm14                     \n\t"
        "vpaddd %%zmm15, %%zmm24, %%zmm15                     \n\t"
        "vpaddd %%zmm16, %%zmm24, %%zmm16                     \n\t"
        "vpaddd %%zmm17, %%zmm24, %%zmm17                     \n\t"
        "vpaddd %%zmm18, %%zmm24, %%zmm18                     \n\t"
        "vpaddd %%zmm19, %%zmm24, %%zmm19                     \n\t"
        "vpaddd %%zmm20, %%zmm24, %%zmm20                     \n\t"
        "vpaddd %%zmm21, %%zmm24, %%zmm21                     \n\t"
        "vpaddd %%zmm22, %%zmm24, %%zmm22                     \n\t"
        "vpaddd %%zmm23, %%zmm24, %%zmm23                     \n\t"
        "movq %9, %%rax  \n\t"
        "shr $2, %4                                     \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpmovusdb %%zmm0, (%%rax)                       \n\t"
        "vpmovusdb %%zmm1, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm2, (%%rax)                       \n\t"
        "vpmovusdb %%zmm3, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm4, (%%rax)                       \n\t"
        "vpmovusdb %%zmm5, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm6, (%%rax)                       \n\t"
        "vpmovusdb %%zmm7, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm8, (%%rax)                       \n\t"
        "vpmovusdb %%zmm9, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm10, (%%rax)                       \n\t"
        "vpmovusdb %%zmm11, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm12, (%%rax)                       \n\t"
        "vpmovusdb %%zmm13, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm14, (%%rax)                       \n\t"
        "vpmovusdb %%zmm15, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm16, (%%rax)                       \n\t"
        "vpmovusdb %%zmm17, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm18, (%%rax)                       \n\t"
        "vpmovusdb %%zmm19, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm20, (%%rax)                       \n\t"
        "vpmovusdb %%zmm21, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm22, (%%rax)                       \n\t"
        "vpmovusdb %%zmm23, (%%rax, %4)                       \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "vmovups %%zmm0, (%%rax)                       \n\t"
        "vmovups %%zmm1, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm2, (%%rax)                       \n\t"
        "vmovups %%zmm3, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm4, (%%rax)                       \n\t"
        "vmovups %%zmm5, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm6, (%%rax)                       \n\t"
        "vmovups %%zmm7, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm8, (%%rax)                       \n\t"
        "vmovups %%zmm9, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm10, (%%rax)                       \n\t"
        "vmovups %%zmm11, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm12, (%%rax)                       \n\t"
        "vmovups %%zmm13, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm14, (%%rax)                       \n\t"
        "vmovups %%zmm15, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm16, (%%rax)                       \n\t"
        "vmovups %%zmm17, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm18, (%%rax)                       \n\t"
        "vmovups %%zmm19, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm20, (%%rax)                       \n\t"
        "vmovups %%zmm21, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm22, (%%rax)                       \n\t"
        "vmovups %%zmm23, (%%rax, %4)                       \n\t"

        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((int64_t)(N * 4)),
        "r"(scale), "r"((int64_t)stepK), "r"(offsetC), "r"((int64_t)flags), "r"(u8Result)
        : "%rax", "%rbx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
        "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
        "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
        "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel24x8                                               \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "vpdpbusd %%ymm24, %%ymm25, %%ymm0              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm26, %%ymm1              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm27, %%ymm2              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm28, %%ymm3              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm29, %%ymm4              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm30, %%ymm5              \n\t"           \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vpdpbusd %%ymm24, %%ymm25, %%ymm6              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm26, %%ymm7              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm27, %%ymm8              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm28, %%ymm9              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm29, %%ymm10              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm30, %%ymm11              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vmovups (%1), %%ymm31                             \n\t"        \
    "vpdpbusd %%ymm24, %%ymm25, %%ymm12              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm26, %%ymm13              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm27, %%ymm14              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm28, %%ymm15              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm29, %%ymm16              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm30, %%ymm17              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vpdpbusd %%ymm24, %%ymm25, %%ymm18              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm26, %%ymm19              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm27, %%ymm20              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm28, %%ymm21              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm29, %%ymm22              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm30, %%ymm23              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vpdpbusd %%ymm31, %%ymm25, %%ymm0              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm26, %%ymm1              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm27, %%ymm2              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm28, %%ymm3              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm29, %%ymm4              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm30, %%ymm5              \n\t"           \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vpdpbusd %%ymm31, %%ymm25, %%ymm6              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm26, %%ymm7              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm27, %%ymm8              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm28, %%ymm9              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm29, %%ymm10              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm30, %%ymm11              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vmovups 0x20(%1), %%ymm24                             \n\t"    \
    "vpdpbusd %%ymm31, %%ymm25, %%ymm12              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm26, %%ymm13              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm27, %%ymm14              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm28, %%ymm15              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm29, %%ymm16              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm30, %%ymm17              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vpdpbusd %%ymm31, %%ymm25, %%ymm18              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm26, %%ymm19              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm27, %%ymm20              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm28, %%ymm21              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm29, %%ymm22              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm30, %%ymm23              \n\t"
#else
#define mmmKernel24x8                                               \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm0, %%ymm28, %%ymm0              \n\t"              \
    "vpaddd %%ymm1, %%ymm29, %%ymm1              \n\t"              \
    "vpaddd %%ymm2, %%ymm30, %%ymm2              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm3, %%ymm28, %%ymm3              \n\t"              \
    "vpaddd %%ymm4, %%ymm29, %%ymm4              \n\t"              \
    "vpaddd %%ymm5, %%ymm30, %%ymm5              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm6, %%ymm28, %%ymm6              \n\t"              \
    "vpaddd %%ymm7, %%ymm29, %%ymm7              \n\t"              \
    "vpaddd %%ymm8, %%ymm30, %%ymm8              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm9, %%ymm28, %%ymm9              \n\t"              \
    "vpaddd %%ymm10, %%ymm29, %%ymm10              \n\t"            \
    "vpaddd %%ymm11, %%ymm30, %%ymm11              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm12, %%ymm28, %%ymm12              \n\t"            \
    "vpaddd %%ymm13, %%ymm29, %%ymm13              \n\t"            \
    "vpaddd %%ymm14, %%ymm30, %%ymm14              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm15, %%ymm28, %%ymm15              \n\t"            \
    "vpaddd %%ymm16, %%ymm29, %%ymm16              \n\t"            \
    "vpaddd %%ymm17, %%ymm30, %%ymm17              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm18, %%ymm28, %%ymm18              \n\t"            \
    "vpaddd %%ymm19, %%ymm29, %%ymm19              \n\t"            \
    "vpaddd %%ymm20, %%ymm30, %%ymm20              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vmovups (%1), %%ymm24                             \n\t"        \
    "vpaddd %%ymm21, %%ymm28, %%ymm21              \n\t"            \
    "vpaddd %%ymm22, %%ymm29, %%ymm22              \n\t"            \
    "vpaddd %%ymm23, %%ymm30, %%ymm23              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm0, %%ymm28, %%ymm0              \n\t"              \
    "vpaddd %%ymm1, %%ymm29, %%ymm1              \n\t"              \
    "vpaddd %%ymm2, %%ymm30, %%ymm2              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm3, %%ymm28, %%ymm3              \n\t"              \
    "vpaddd %%ymm4, %%ymm29, %%ymm4              \n\t"              \
    "vpaddd %%ymm5, %%ymm30, %%ymm5              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm6, %%ymm28, %%ymm6              \n\t"              \
    "vpaddd %%ymm7, %%ymm29, %%ymm7              \n\t"              \
    "vpaddd %%ymm8, %%ymm30, %%ymm8              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm9, %%ymm28, %%ymm9              \n\t"              \
    "vpaddd %%ymm10, %%ymm29, %%ymm10              \n\t"            \
    "vpaddd %%ymm11, %%ymm30, %%ymm11              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm12, %%ymm28, %%ymm12              \n\t"            \
    "vpaddd %%ymm13, %%ymm29, %%ymm13              \n\t"            \
    "vpaddd %%ymm14, %%ymm30, %%ymm14              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm15, %%ymm28, %%ymm15              \n\t"            \
    "vpaddd %%ymm16, %%ymm29, %%ymm16              \n\t"            \
    "vpaddd %%ymm17, %%ymm30, %%ymm17              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm18, %%ymm28, %%ymm18              \n\t"            \
    "vpaddd %%ymm19, %%ymm29, %%ymm19              \n\t"            \
    "vpaddd %%ymm20, %%ymm30, %%ymm20              \n\t"            \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "vmovups 0x20(%1), %%ymm24                             \n\t"    \
    "vpaddd %%ymm21, %%ymm28, %%ymm21              \n\t"            \
    "vpaddd %%ymm22, %%ymm29, %%ymm22              \n\t"            \
    "vpaddd %%ymm23, %%ymm30, %%ymm23              \n\t"
#endif

inline void mmm_avx512_24x8_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x40(%1)                              \n\t"
        "vmovups (%1), %%ymm24                             \n\t"
        "add $0x20, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%ebx \n\t"
        "vmovd %%ebx, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%ymm31            \n\t"
#endif
        "movq %8, %%rbx          \n\t"
        "andq $0x1, %%rbx          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%7), %%ymm0                       \n\t"
        "vmovups %%ymm0, %%ymm1                   \n\t"
        "vmovups %%ymm0, %%ymm2                   \n\t"
        "vmovups %%ymm0, %%ymm3                   \n\t"
        "vmovups %%ymm0, %%ymm4                   \n\t"
        "vmovups %%ymm0, %%ymm5                   \n\t"
        "vmovups %%ymm0, %%ymm6                   \n\t"
        "vmovups %%ymm0, %%ymm7                   \n\t"
        "vmovups %%ymm0, %%ymm8                   \n\t"
        "vmovups %%ymm0, %%ymm9                   \n\t"
        "vmovups %%ymm0, %%ymm10                   \n\t"
        "vmovups %%ymm0, %%ymm11                   \n\t"
        "vmovups %%ymm0, %%ymm12                   \n\t"
        "vmovups %%ymm0, %%ymm13                   \n\t"
        "vmovups %%ymm0, %%ymm14                   \n\t"
        "vmovups %%ymm0, %%ymm15                   \n\t"
        "vmovups %%ymm0, %%ymm16                   \n\t"
        "vmovups %%ymm0, %%ymm17                   \n\t"
        "vmovups %%ymm0, %%ymm18                   \n\t"
        "vmovups %%ymm0, %%ymm19                   \n\t"
        "vmovups %%ymm0, %%ymm20                   \n\t"
        "vmovups %%ymm0, %%ymm21                   \n\t"
        "vmovups %%ymm0, %%ymm22                   \n\t"
        "vmovups %%ymm0, %%ymm23                   \n\t"
        "jmp 1f          \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
        "vxorps %%ymm5, %%ymm5, %%ymm5                     \n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6                     \n\t"
        "vxorps %%ymm7, %%ymm7, %%ymm7                     \n\t"
        "vxorps %%ymm8, %%ymm8, %%ymm8                     \n\t"
        "vxorps %%ymm9, %%ymm9, %%ymm9                     \n\t"
        "vxorps %%ymm10, %%ymm10, %%ymm10                  \n\t"
        "vxorps %%ymm11, %%ymm11, %%ymm11                  \n\t"
        "vxorps %%ymm12, %%ymm12, %%ymm12                  \n\t"
        "vxorps %%ymm13, %%ymm13, %%ymm13                  \n\t"
        "vxorps %%ymm14, %%ymm14, %%ymm14                  \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vxorps %%ymm16, %%ymm16, %%ymm16                  \n\t"
        "vxorps %%ymm17, %%ymm17, %%ymm17                  \n\t"
        "vxorps %%ymm18, %%ymm18, %%ymm18                  \n\t"
        "vxorps %%ymm19, %%ymm19, %%ymm19                  \n\t"
        "vxorps %%ymm20, %%ymm20, %%ymm20                  \n\t"
        "vxorps %%ymm21, %%ymm21, %%ymm21                  \n\t"
        "vxorps %%ymm22, %%ymm22, %%ymm22                  \n\t"
        "vxorps %%ymm23, %%ymm23, %%ymm23                  \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "movq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel24x8

        "add $0x40, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "movq %2, %%rax  \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpaddd (%%rax), %%ymm0, %%ymm0                       \n\t"
        "vpaddd (%%rax, %4), %%ymm1, %%ymm1                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm2, %%ymm2                   \n\t"
        "vpaddd (%%rax, %4), %%ymm3, %%ymm3                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm4, %%ymm4                   \n\t"
        "vpaddd (%%rax, %4), %%ymm5, %%ymm5                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm6, %%ymm6                   \n\t"
        "vpaddd (%%rax, %4), %%ymm7, %%ymm7                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm8, %%ymm8                   \n\t"
        "vpaddd (%%rax, %4), %%ymm9, %%ymm9                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm10, %%ymm10                   \n\t"
        "vpaddd (%%rax, %4), %%ymm11, %%ymm11                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm12, %%ymm12                   \n\t"
        "vpaddd (%%rax, %4), %%ymm13, %%ymm13                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm14, %%ymm14                   \n\t"
        "vpaddd (%%rax, %4), %%ymm15, %%ymm15                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm16, %%ymm16                   \n\t"
        "vpaddd (%%rax, %4), %%ymm17, %%ymm17                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm18, %%ymm18                   \n\t"
        "vpaddd (%%rax, %4), %%ymm19, %%ymm19                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm20, %%ymm20                   \n\t"
        "vpaddd (%%rax, %4), %%ymm21, %%ymm21                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm22, %%ymm22                   \n\t"
        "vpaddd (%%rax, %4), %%ymm23, %%ymm23                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%ymm24                        \n\t"
        "vcvtdq2ps %%ymm0, %%ymm0                       \n\t"
        "vcvtdq2ps %%ymm1, %%ymm1                       \n\t"
        "vcvtdq2ps %%ymm2, %%ymm2                       \n\t"
        "vcvtdq2ps %%ymm3, %%ymm3                       \n\t"
        "vcvtdq2ps %%ymm4, %%ymm4                       \n\t"
        "vcvtdq2ps %%ymm5, %%ymm5                       \n\t"
        "vcvtdq2ps %%ymm6, %%ymm6                       \n\t"
        "vcvtdq2ps %%ymm7, %%ymm7                       \n\t"
        "vcvtdq2ps %%ymm8, %%ymm8                       \n\t"
        "vcvtdq2ps %%ymm9, %%ymm9                       \n\t"
        "vcvtdq2ps %%ymm10, %%ymm10                       \n\t"
        "vcvtdq2ps %%ymm11, %%ymm11                       \n\t"
        "vcvtdq2ps %%ymm12, %%ymm12                       \n\t"
        "vcvtdq2ps %%ymm13, %%ymm13                       \n\t"
        "vcvtdq2ps %%ymm14, %%ymm14                       \n\t"
        "vcvtdq2ps %%ymm15, %%ymm15                       \n\t"
        "vcvtdq2ps %%ymm16, %%ymm16                       \n\t"
        "vcvtdq2ps %%ymm17, %%ymm17                       \n\t"
        "vcvtdq2ps %%ymm18, %%ymm18                       \n\t"
        "vcvtdq2ps %%ymm19, %%ymm19                       \n\t"
        "vcvtdq2ps %%ymm20, %%ymm20                       \n\t"
        "vcvtdq2ps %%ymm21, %%ymm21                       \n\t"
        "vcvtdq2ps %%ymm22, %%ymm22                       \n\t"
        "vcvtdq2ps %%ymm23, %%ymm23                       \n\t"
        "vmulps %%ymm0, %%ymm24, %%ymm0                       \n\t"
        "vmulps %%ymm1, %%ymm24, %%ymm1                       \n\t"
        "vmulps %%ymm2, %%ymm24, %%ymm2                       \n\t"
        "vmulps %%ymm3, %%ymm24, %%ymm3                       \n\t"
        "vmulps %%ymm4, %%ymm24, %%ymm4                       \n\t"
        "vmulps %%ymm5, %%ymm24, %%ymm5                       \n\t"
        "vmulps %%ymm6, %%ymm24, %%ymm6                       \n\t"
        "vmulps %%ymm7, %%ymm24, %%ymm7                       \n\t"
        "vmulps %%ymm8, %%ymm24, %%ymm8                       \n\t"
        "vmulps %%ymm9, %%ymm24, %%ymm9                       \n\t"
        "vmulps %%ymm10, %%ymm24, %%ymm10                     \n\t"
        "vmulps %%ymm11, %%ymm24, %%ymm11                     \n\t"
        "vmulps %%ymm12, %%ymm24, %%ymm12                     \n\t"
        "vmulps %%ymm13, %%ymm24, %%ymm13                     \n\t"
        "vmulps %%ymm14, %%ymm24, %%ymm14                     \n\t"
        "vmulps %%ymm15, %%ymm24, %%ymm15                     \n\t"
        "vmulps %%ymm16, %%ymm24, %%ymm16                     \n\t"
        "vmulps %%ymm17, %%ymm24, %%ymm17                     \n\t"
        "vmulps %%ymm18, %%ymm24, %%ymm18                     \n\t"
        "vmulps %%ymm19, %%ymm24, %%ymm19                     \n\t"
        "vmulps %%ymm20, %%ymm24, %%ymm20                     \n\t"
        "vmulps %%ymm21, %%ymm24, %%ymm21                     \n\t"
        "vmulps %%ymm22, %%ymm24, %%ymm22                     \n\t"
        "vmulps %%ymm23, %%ymm24, %%ymm23                     \n\t"

        "movq %8, %%rbx          \n\t"
        "andq $0x2, %%rbx          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
        "vcvtps2dq %%zmm4, %%zmm4                       \n\t"
        "vcvtps2dq %%zmm5, %%zmm5                       \n\t"
        "vcvtps2dq %%zmm6, %%zmm6                       \n\t"
        "vcvtps2dq %%zmm7, %%zmm7                       \n\t"
        "vcvtps2dq %%zmm8, %%zmm8                       \n\t"
        "vcvtps2dq %%zmm9, %%zmm9                       \n\t"
        "vcvtps2dq %%zmm10, %%zmm10                       \n\t"
        "vcvtps2dq %%zmm11, %%zmm11                       \n\t"
        "vcvtps2dq %%zmm12, %%zmm12                       \n\t"
        "vcvtps2dq %%zmm13, %%zmm13                       \n\t"
        "vcvtps2dq %%zmm14, %%zmm14                       \n\t"
        "vcvtps2dq %%zmm15, %%zmm15                       \n\t"
        "vcvtps2dq %%zmm16, %%zmm16                       \n\t"
        "vcvtps2dq %%zmm17, %%zmm17                       \n\t"
        "vcvtps2dq %%zmm18, %%zmm18                       \n\t"
        "vcvtps2dq %%zmm19, %%zmm19                       \n\t"
        "vcvtps2dq %%zmm20, %%zmm20                       \n\t"
        "vcvtps2dq %%zmm21, %%zmm21                       \n\t"
        "vcvtps2dq %%zmm22, %%zmm22                       \n\t"
        "vcvtps2dq %%zmm23, %%zmm23                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpaddd %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vpaddd %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vpaddd %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vpaddd %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vpaddd %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vpaddd %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vpaddd %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vpaddd %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "vpaddd %%zmm12, %%zmm24, %%zmm12                     \n\t"
        "vpaddd %%zmm13, %%zmm24, %%zmm13                     \n\t"
        "vpaddd %%zmm14, %%zmm24, %%zmm14                     \n\t"
        "vpaddd %%zmm15, %%zmm24, %%zmm15                     \n\t"
        "vpaddd %%zmm16, %%zmm24, %%zmm16                     \n\t"
        "vpaddd %%zmm17, %%zmm24, %%zmm17                     \n\t"
        "vpaddd %%zmm18, %%zmm24, %%zmm18                     \n\t"
        "vpaddd %%zmm19, %%zmm24, %%zmm19                     \n\t"
        "vpaddd %%zmm20, %%zmm24, %%zmm20                     \n\t"
        "vpaddd %%zmm21, %%zmm24, %%zmm21                     \n\t"
        "vpaddd %%zmm22, %%zmm24, %%zmm22                     \n\t"
        "vpaddd %%zmm23, %%zmm24, %%zmm23                     \n\t"
        "movq %9, %%rax  \n\t"
        "shr $2, %4                                     \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpmovusdb %%zmm0, (%%rax)                       \n\t"
        "vpmovusdb %%zmm1, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm2, (%%rax)                       \n\t"
        "vpmovusdb %%zmm3, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm4, (%%rax)                       \n\t"
        "vpmovusdb %%zmm5, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm6, (%%rax)                       \n\t"
        "vpmovusdb %%zmm7, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm8, (%%rax)                       \n\t"
        "vpmovusdb %%zmm9, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm10, (%%rax)                       \n\t"
        "vpmovusdb %%zmm11, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm12, (%%rax)                       \n\t"
        "vpmovusdb %%zmm13, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm14, (%%rax)                       \n\t"
        "vpmovusdb %%zmm15, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm16, (%%rax)                       \n\t"
        "vpmovusdb %%zmm17, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm18, (%%rax)                       \n\t"
        "vpmovusdb %%zmm19, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm20, (%%rax)                       \n\t"
        "vpmovusdb %%zmm21, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm22, (%%rax)                       \n\t"
        "vpmovusdb %%zmm23, (%%rax, %4)                       \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "vmovups %%ymm0, (%%rax)                       \n\t"
        "vmovups %%ymm1, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm2, (%%rax)                       \n\t"
        "vmovups %%ymm3, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm4, (%%rax)                       \n\t"
        "vmovups %%ymm5, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm6, (%%rax)                       \n\t"
        "vmovups %%ymm7, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm8, (%%rax)                       \n\t"
        "vmovups %%ymm9, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm10, (%%rax)                       \n\t"
        "vmovups %%ymm11, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm12, (%%rax)                       \n\t"
        "vmovups %%ymm13, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm14, (%%rax)                       \n\t"
        "vmovups %%ymm15, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm16, (%%rax)                       \n\t"
        "vmovups %%ymm17, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm18, (%%rax)                       \n\t"
        "vmovups %%ymm19, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm20, (%%rax)                       \n\t"
        "vmovups %%ymm21, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm22, (%%rax)                       \n\t"
        "vmovups %%ymm23, (%%rax, %4)                       \n\t"

        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"((int64_t)stepK), "r"(offsetC), "r"((int64_t)flags), "r"(u8Result)
        : "%rax", "%rbx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
        "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%ymm16",
        "%ymm17", "%ymm18", "%ymm19", "%ymm20", "%ymm21", "%ymm22", "%ymm23", "%ymm24", "%ymm25",
        "%ymm26", "%ymm27", "%ymm28", "%ymm29", "%ymm30", "%ymm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel4x48                                             \
    "movq %0, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "prefetcht0 0xC0(%1)                              \n\t"       \
    "prefetcht0 0x100(%1)                              \n\t"      \
    "prefetcht0 0x140(%1)                              \n\t"      \
    "vmovups (%1), %%zmm27                             \n\t"      \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm0              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm1              \n\t"         \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm2              \n\t"         \
    "vmovups 0x40(%1), %%zmm28                             \n\t"  \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm3              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm4              \n\t"         \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm5              \n\t"         \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "vmovups 0x80(%1), %%zmm29                             \n\t"  \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm6              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm7              \n\t"         \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm8              \n\t"         \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm9              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm10              \n\t"        \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm11              \n\t"        \
    "movq %0, %%rax  \n\t"                                        \
    "addq $0x4, %%rax  \n\t"                                      \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "prefetcht0 0x180(%1)                              \n\t"      \
    "prefetcht0 0x1C0(%1)                              \n\t"      \
    "prefetcht0 0x200(%1)                              \n\t"      \
    "vmovups 0xC0(%1), %%zmm24                             \n\t"  \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm0              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm1              \n\t"         \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm2              \n\t"         \
    "vmovups 0x100(%1), %%zmm25                             \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm3              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm4              \n\t"         \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm5              \n\t"         \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd (%%rax, %6), %%zmm31                     \n\t"  \
    "vmovups 0x140(%1), %%zmm26                             \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm6              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm7              \n\t"         \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm8              \n\t"         \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm9              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm10              \n\t"        \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm11              \n\t"
#else
#define mmmKernel4x48                                             \
    "movq %0, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "prefetcht0 0xC0(%1)                              \n\t"       \
    "prefetcht0 0x100(%1)                              \n\t"      \
    "prefetcht0 0x140(%1)                              \n\t"      \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm0, %%zmm27, %%zmm0              \n\t"            \
    "vpaddd %%zmm1, %%zmm28, %%zmm1              \n\t"            \
    "vpaddd %%zmm2, %%zmm29, %%zmm2              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"            \
    "vpaddd %%zmm4, %%zmm28, %%zmm4              \n\t"            \
    "vpaddd %%zmm5, %%zmm29, %%zmm5              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm6, %%zmm27, %%zmm6              \n\t"            \
    "vpaddd %%zmm7, %%zmm28, %%zmm7              \n\t"            \
    "vpaddd %%zmm8, %%zmm29, %%zmm8              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vmovups (%1), %%zmm24                             \n\t"      \
    "vmovups 0x40(%1), %%zmm25                             \n\t"  \
    "vmovups 0x80(%1), %%zmm26                             \n\t"  \
    "vpaddd %%zmm9, %%zmm27, %%zmm9              \n\t"            \
    "vpaddd %%zmm10, %%zmm28, %%zmm10              \n\t"          \
    "vpaddd %%zmm11, %%zmm29, %%zmm11              \n\t"          \
    "movq %0, %%rax  \n\t"                                        \
    "addq $0x4, %%rax  \n\t"                                      \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "prefetcht0 0x180(%1)                              \n\t"      \
    "prefetcht0 0x1C0(%1)                              \n\t"      \
    "prefetcht0 0x200(%1)                              \n\t"      \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm0, %%zmm27, %%zmm0              \n\t"            \
    "vpaddd %%zmm1, %%zmm28, %%zmm1              \n\t"            \
    "vpaddd %%zmm2, %%zmm29, %%zmm2              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "addq %6, %%rax  \n\t"                                        \
    "addq %6, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"            \
    "vpaddd %%zmm4, %%zmm28, %%zmm4              \n\t"            \
    "vpaddd %%zmm5, %%zmm29, %%zmm5              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm30                     \n\t"  \
    "vpaddd %%zmm6, %%zmm27, %%zmm6              \n\t"            \
    "vpaddd %%zmm7, %%zmm28, %%zmm7              \n\t"            \
    "vpaddd %%zmm8, %%zmm29, %%zmm8              \n\t"            \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vmovups 0xC0(%1), %%zmm24                             \n\t"  \
    "vmovups 0x100(%1), %%zmm25                             \n\t" \
    "vmovups 0x140(%1), %%zmm26                             \n\t" \
    "vpaddd %%zmm9, %%zmm27, %%zmm9              \n\t"            \
    "vpaddd %%zmm10, %%zmm28, %%zmm10              \n\t"          \
    "vpaddd %%zmm11, %%zmm29, %%zmm11              \n\t"
#endif

inline void mmm_avx512_4x48_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0xC0(%1)                              \n\t"
        "prefetcht0 0x100(%1)                              \n\t"
        "prefetcht0 0x140(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "vmovups 0x40(%1), %%zmm25                             \n\t"
        "vmovups 0x80(%1), %%zmm26                             \n\t"
        "add $0xC0, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%eax \n\t"
        "vmovd %%eax, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif
        "movq %%rbx, %%rax          \n\t"
        "andq $0x1, %%rax          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%7), %%zmm0                       \n\t"
        "vmovups 0x40(%7), %%zmm1                   \n\t"
        "vmovups 0x80(%7), %%zmm2                   \n\t"
        "vmovups %%zmm0, %%zmm3                   \n\t"
        "vmovups %%zmm1, %%zmm4                   \n\t"
        "vmovups %%zmm2, %%zmm5                   \n\t"
        "vmovups %%zmm0, %%zmm6                   \n\t"
        "vmovups %%zmm1, %%zmm7                   \n\t"
        "vmovups %%zmm2, %%zmm8                   \n\t"
        "vmovups %%zmm0, %%zmm9                   \n\t"
        "vmovups %%zmm1, %%zmm10                   \n\t"
        "vmovups %%zmm2, %%zmm11                   \n\t"
        "jmp 1f          \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
        "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
        "vxorps %%zmm4, %%zmm4, %%zmm4                     \n\t"
        "vxorps %%zmm5, %%zmm5, %%zmm5                     \n\t"
        "vxorps %%zmm6, %%zmm6, %%zmm6                     \n\t"
        "vxorps %%zmm7, %%zmm7, %%zmm7                     \n\t"
        "vxorps %%zmm8, %%zmm8, %%zmm8                     \n\t"
        "vxorps %%zmm9, %%zmm9, %%zmm9                     \n\t"
        "vxorps %%zmm10, %%zmm10, %%zmm10                  \n\t"
        "vxorps %%zmm11, %%zmm11, %%zmm11                  \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 0x80(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "prefetcht0 0x80(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 0x80(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "prefetcht0 0x80(%%rax, %4)                              \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel4x48

        "add $0x180, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "movq %2, %%rax  \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpaddd (%%rax), %%zmm0, %%zmm0                       \n\t"
        "vpaddd 0x40(%%rax), %%zmm1, %%zmm1                   \n\t"
        "vpaddd 0x80(%%rax), %%zmm2, %%zmm2                   \n\t"
        "vpaddd (%%rax, %4), %%zmm3, %%zmm3                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm4, %%zmm4                   \n\t"
        "vpaddd 0x80(%%rax, %4), %%zmm5, %%zmm5                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm6, %%zmm6                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm7, %%zmm7                   \n\t"
        "vpaddd 0x80(%%rax), %%zmm8, %%zmm8                   \n\t"
        "vpaddd (%%rax, %4), %%zmm9, %%zmm9                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm10, %%zmm10                   \n\t"
        "vpaddd 0x80(%%rax, %4), %%zmm11, %%zmm11                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
        "vcvtdq2ps %%zmm2, %%zmm2                       \n\t"
        "vcvtdq2ps %%zmm3, %%zmm3                       \n\t"
        "vcvtdq2ps %%zmm4, %%zmm4                       \n\t"
        "vcvtdq2ps %%zmm5, %%zmm5                       \n\t"
        "vcvtdq2ps %%zmm6, %%zmm6                       \n\t"
        "vcvtdq2ps %%zmm7, %%zmm7                       \n\t"
        "vcvtdq2ps %%zmm8, %%zmm8                       \n\t"
        "vcvtdq2ps %%zmm9, %%zmm9                       \n\t"
        "vcvtdq2ps %%zmm10, %%zmm10                       \n\t"
        "vcvtdq2ps %%zmm11, %%zmm11                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vmulps %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vmulps %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vmulps %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vmulps %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vmulps %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vmulps %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vmulps %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vmulps %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vmulps %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vmulps %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vmulps %%zmm11, %%zmm24, %%zmm11                     \n\t"

        "movq %%rbx, %%rax          \n\t"
        "andq $0x2, %%rax          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
        "vcvtps2dq %%zmm4, %%zmm4                       \n\t"
        "vcvtps2dq %%zmm5, %%zmm5                       \n\t"
        "vcvtps2dq %%zmm6, %%zmm6                       \n\t"
        "vcvtps2dq %%zmm7, %%zmm7                       \n\t"
        "vcvtps2dq %%zmm8, %%zmm8                       \n\t"
        "vcvtps2dq %%zmm9, %%zmm9                       \n\t"
        "vcvtps2dq %%zmm10, %%zmm10                       \n\t"
        "vcvtps2dq %%zmm11, %%zmm11                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpaddd %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vpaddd %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vpaddd %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vpaddd %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vpaddd %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vpaddd %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vpaddd %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vpaddd %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "movq %9, %%rax  \n\t"
        "shr $2, %4                                     \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpmovusdb %%zmm0,  (%%rax)                             \n\t"
        "vpmovusdb %%zmm1,  0x10(%%rax)                         \n\t"
        "vpmovusdb %%zmm2,  0x20(%%rax)                         \n\t"
        "vpmovusdb %%zmm3,  (%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm4,  0x10(%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm5,  0x20(%%rax, %4)                         \n\t"
        "add %%rcx, %%rax                                     \n\t"
        "vpmovusdb %%zmm6,  (%%rax)                         \n\t"
        "vpmovusdb %%zmm7,  0x10(%%rax)                         \n\t"
        "vpmovusdb %%zmm8,  0x20(%%rax)                         \n\t"
        "vpmovusdb %%zmm9,  (%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm10,  0x10(%%rax, %4)                         \n\t"
        "vpmovusdb %%zmm11,  0x20(%%rax, %4)                         \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "vmovups %%zmm0,  (%%rax)                             \n\t"
        "vmovups %%zmm1,  0x40(%%rax)                         \n\t"
        "vmovups %%zmm2,  0x80(%%rax)                         \n\t"
        "vmovups %%zmm3,  (%%rax, %4)                         \n\t"
        "vmovups %%zmm4,  0x40(%%rax, %4)                         \n\t"
        "vmovups %%zmm5,  0x80(%%rax, %4)                         \n\t"
        "add %%rcx, %%rax                                     \n\t"
        "vmovups %%zmm6,  (%%rax)                         \n\t"
        "vmovups %%zmm7,  0x40(%%rax)                         \n\t"
        "vmovups %%zmm8,  0x80(%%rax)                         \n\t"
        "vmovups %%zmm9,  (%%rax, %4)                         \n\t"
        "vmovups %%zmm10,  0x40(%%rax, %4)                         \n\t"
        "vmovups %%zmm11,  0x80(%%rax, %4)                         \n\t"
        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"((int64_t)stepK), "r"(offsetC), "b"((int64_t)flags), "r"(u8Result)
        : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8",
        "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",
        "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
        "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel6x32                                               \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "prefetcht0 0xC0(%1)                              \n\t"         \
    "vmovups (%1), %%zmm26                             \n\t"        \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm0              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm1              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm2              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm3              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm4              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm5              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm6              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm7              \n\t"           \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vmovups 0x40(%1), %%zmm27                             \n\t"    \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm8              \n\t"           \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm9              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm10              \n\t"          \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm11              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31                     \n\t" \
    "prefetcht0 0x100(%1)                              \n\t"        \
    "prefetcht0 0x140(%1)                              \n\t"        \
    "vmovups 0x80(%1), %%zmm24                             \n\t"    \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm0              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm1              \n\t"           \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm2              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm3              \n\t"           \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm4              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm5              \n\t"           \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm6              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm7              \n\t"           \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vmovups 0xC0(%1), %%zmm25                             \n\t"    \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm8              \n\t"           \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm9              \n\t"           \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm10              \n\t"          \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm11              \n\t"
#else
#define mmmKernel6x32                                               \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "prefetcht0 0xC0(%1)                              \n\t"         \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"              \
    "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"              \
    "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"              \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"        \
    "vpbroadcastd (%%rax, %%rbx), %%zmm29                     \n\t" \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpaddd %%zmm3, %%zmm26, %%zmm3              \n\t"              \
    "vpaddd %%zmm4, %%zmm27, %%zmm4              \n\t"              \
    "vpaddd %%zmm5, %%zmm28, %%zmm5              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpaddd %%zmm6, %%zmm26, %%zmm6              \n\t"              \
    "vpaddd %%zmm7, %%zmm27, %%zmm7              \n\t"              \
    "vpaddd %%zmm8, %%zmm28, %%zmm8              \n\t"              \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm28              \n\t"        \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vmovups (%1), %%zmm24                             \n\t"        \
    "vmovups 0x40(%1), %%zmm25                             \n\t"    \
    "vpaddd %%zmm9, %%zmm26, %%zmm9              \n\t"              \
    "vpaddd %%zmm10, %%zmm27, %%zmm10              \n\t"            \
    "vpaddd %%zmm11, %%zmm28, %%zmm11              \n\t"            \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "prefetcht0 0x100(%1)                              \n\t"        \
    "prefetcht0 0x140(%1)                              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"              \
    "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"              \
    "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"              \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"        \
    "vpbroadcastd (%%rax, %%rbx), %%zmm29                     \n\t" \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "addq %6, %%rax  \n\t"                                          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"        \
    "vpaddd %%zmm3, %%zmm26, %%zmm3              \n\t"              \
    "vpaddd %%zmm4, %%zmm27, %%zmm4              \n\t"              \
    "vpaddd %%zmm5, %%zmm28, %%zmm5              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpaddd %%zmm6, %%zmm26, %%zmm6              \n\t"              \
    "vpaddd %%zmm7, %%zmm27, %%zmm7              \n\t"              \
    "vpaddd %%zmm8, %%zmm28, %%zmm8              \n\t"              \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm26              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm29, %%zmm27              \n\t"        \
    "vpmaddubsw %%zmm25, %%zmm29, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"          \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"          \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vmovups 0x80(%1), %%zmm24                             \n\t"    \
    "vmovups 0xC0(%1), %%zmm25                             \n\t"    \
    "vpaddd %%zmm9, %%zmm26, %%zmm9              \n\t"              \
    "vpaddd %%zmm10, %%zmm27, %%zmm10              \n\t"            \
    "vpaddd %%zmm11, %%zmm28, %%zmm11              \n\t"
#endif

inline void mmm_avx512_6x32_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x80(%1)                              \n\t"
        "prefetcht0 0xC0(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "vmovups 0x40(%1), %%zmm25                             \n\t"
        "add $0x80, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%ebx \n\t"
        "vmovd %%ebx, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif
        "movq %8, %%rbx          \n\t"
        "andq $0x1, %%rbx          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%7), %%zmm0                       \n\t"
        "vmovups 0x40(%7), %%zmm1                   \n\t"
        "vmovups %%zmm0, %%zmm2                   \n\t"
        "vmovups %%zmm1, %%zmm3                   \n\t"
        "vmovups %%zmm0, %%zmm4                   \n\t"
        "vmovups %%zmm1, %%zmm5                   \n\t"
        "vmovups %%zmm0, %%zmm6                   \n\t"
        "vmovups %%zmm1, %%zmm7                   \n\t"
        "vmovups %%zmm0, %%zmm8                   \n\t"
        "vmovups %%zmm1, %%zmm9                   \n\t"
        "vmovups %%zmm0, %%zmm10                   \n\t"
        "vmovups %%zmm1, %%zmm11                   \n\t"
        "jmp 1f          \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
        "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
        "vxorps %%zmm4, %%zmm4, %%zmm4                     \n\t"
        "vxorps %%zmm5, %%zmm5, %%zmm5                     \n\t"
        "vxorps %%zmm6, %%zmm6, %%zmm6                     \n\t"
        "vxorps %%zmm7, %%zmm7, %%zmm7                     \n\t"
        "vxorps %%zmm8, %%zmm8, %%zmm8                     \n\t"
        "vxorps %%zmm9, %%zmm9, %%zmm9                     \n\t"
        "vxorps %%zmm10, %%zmm10, %%zmm10                  \n\t"
        "vxorps %%zmm11, %%zmm11, %%zmm11                  \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "movq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel6x32

        "add $0x100, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "movq %2, %%rax  \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpaddd (%%rax), %%zmm0, %%zmm0                       \n\t"
        "vpaddd 0x40(%%rax), %%zmm1, %%zmm1                   \n\t"
        "vpaddd (%%rax, %4), %%zmm2, %%zmm2                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm3, %%zmm3                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm4, %%zmm4                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm5, %%zmm5                   \n\t"
        "vpaddd (%%rax, %4), %%zmm6, %%zmm6                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm7, %%zmm7                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm8, %%zmm8                   \n\t"
        "vpaddd 0x40(%%rax), %%zmm9, %%zmm9                   \n\t"
        "vpaddd (%%rax, %4), %%zmm10, %%zmm10                   \n\t"
        "vpaddd 0x40(%%rax, %4), %%zmm11, %%zmm11                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
        "vcvtdq2ps %%zmm2, %%zmm2                       \n\t"
        "vcvtdq2ps %%zmm3, %%zmm3                       \n\t"
        "vcvtdq2ps %%zmm4, %%zmm4                       \n\t"
        "vcvtdq2ps %%zmm5, %%zmm5                       \n\t"
        "vcvtdq2ps %%zmm6, %%zmm6                       \n\t"
        "vcvtdq2ps %%zmm7, %%zmm7                       \n\t"
        "vcvtdq2ps %%zmm8, %%zmm8                       \n\t"
        "vcvtdq2ps %%zmm9, %%zmm9                       \n\t"
        "vcvtdq2ps %%zmm10, %%zmm10                       \n\t"
        "vcvtdq2ps %%zmm11, %%zmm11                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vmulps %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vmulps %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vmulps %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vmulps %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vmulps %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vmulps %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vmulps %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vmulps %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vmulps %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vmulps %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vmulps %%zmm11, %%zmm24, %%zmm11                     \n\t"

        "movq %8, %%rbx          \n\t"
        "andq $0x2, %%rbx          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
        "vcvtps2dq %%zmm4, %%zmm4                       \n\t"
        "vcvtps2dq %%zmm5, %%zmm5                       \n\t"
        "vcvtps2dq %%zmm6, %%zmm6                       \n\t"
        "vcvtps2dq %%zmm7, %%zmm7                       \n\t"
        "vcvtps2dq %%zmm8, %%zmm8                       \n\t"
        "vcvtps2dq %%zmm9, %%zmm9                       \n\t"
        "vcvtps2dq %%zmm10, %%zmm10                       \n\t"
        "vcvtps2dq %%zmm11, %%zmm11                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpaddd %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vpaddd %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vpaddd %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vpaddd %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vpaddd %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vpaddd %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vpaddd %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vpaddd %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "movq %9, %%rax  \n\t"
        "shr $2, %4                                     \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpmovusdb %%zmm0, (%%rax)                       \n\t"
        "vpmovusdb %%zmm1, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm2, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm3, 0x10(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm4, (%%rax)                       \n\t"
        "vpmovusdb %%zmm5, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm6, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm7, 0x10(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm8, (%%rax)                       \n\t"
        "vpmovusdb %%zmm9, 0x10(%%rax)                       \n\t"
        "vpmovusdb %%zmm10, (%%rax, %4)                       \n\t"
        "vpmovusdb %%zmm11, 0x10(%%rax, %4)                       \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "vmovups %%zmm0, (%%rax)                       \n\t"
        "vmovups %%zmm1, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm2, (%%rax, %4)                       \n\t"
        "vmovups %%zmm3, 0x40(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm4, (%%rax)                       \n\t"
        "vmovups %%zmm5, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm6, (%%rax, %4)                       \n\t"
        "vmovups %%zmm7, 0x40(%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm8, (%%rax)                       \n\t"
        "vmovups %%zmm9, 0x40(%%rax)                       \n\t"
        "vmovups %%zmm10, (%%rax, %4)                       \n\t"
        "vmovups %%zmm11, 0x40(%%rax, %4)                       \n\t"

        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"((int64_t)stepK), "r"(offsetC), "r"((int64_t)flags), "r"(u8Result)
        : "%rax", "%rbx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
        "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
        "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
        "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel12x16                                              \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "vpdpbusd %%zmm24, %%zmm25, %%zmm0              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm26, %%zmm1              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm27, %%zmm2              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm3              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm4              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm5              \n\t"           \
    "vmovups (%1), %%zmm31                             \n\t"        \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpdpbusd %%zmm24, %%zmm25, %%zmm6              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm26, %%zmm7              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm27, %%zmm8              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm9              \n\t"           \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm10              \n\t"          \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm11              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "prefetcht0 0xC0(%1)                              \n\t"         \
    "vpdpbusd %%zmm31, %%zmm25, %%zmm0              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm26, %%zmm1              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm27, %%zmm2              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm28, %%zmm3              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm29, %%zmm4              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm30, %%zmm5              \n\t"           \
    "vmovups 0x40(%1), %%zmm24                             \n\t"    \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm30                     \n\t" \
    "vpdpbusd %%zmm31, %%zmm25, %%zmm6              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm26, %%zmm7              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm27, %%zmm8              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm28, %%zmm9              \n\t"           \
    "vpdpbusd %%zmm31, %%zmm29, %%zmm10              \n\t"          \
    "vpdpbusd %%zmm31, %%zmm30, %%zmm11              \n\t"
#else
#define mmmKernel12x16                                              \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm0, %%zmm28, %%zmm0              \n\t"              \
    "vpaddd %%zmm1, %%zmm29, %%zmm1              \n\t"              \
    "vpaddd %%zmm2, %%zmm30, %%zmm2              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm3, %%zmm28, %%zmm3              \n\t"              \
    "vpaddd %%zmm4, %%zmm29, %%zmm4              \n\t"              \
    "vpaddd %%zmm5, %%zmm30, %%zmm5              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm6, %%zmm28, %%zmm6              \n\t"              \
    "vpaddd %%zmm7, %%zmm29, %%zmm7              \n\t"              \
    "vpaddd %%zmm8, %%zmm30, %%zmm8              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm9, %%zmm28, %%zmm9              \n\t"              \
    "vpaddd %%zmm10, %%zmm29, %%zmm10              \n\t"            \
    "vpaddd %%zmm11, %%zmm30, %%zmm11              \n\t"            \
    "vmovups (%1), %%zmm24                             \n\t"        \
    "prefetcht0 0xC0(%1)                              \n\t"         \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm0, %%zmm28, %%zmm0              \n\t"              \
    "vpaddd %%zmm1, %%zmm29, %%zmm1              \n\t"              \
    "vpaddd %%zmm2, %%zmm30, %%zmm2              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm3, %%zmm28, %%zmm3              \n\t"              \
    "vpaddd %%zmm4, %%zmm29, %%zmm4              \n\t"              \
    "vpaddd %%zmm5, %%zmm30, %%zmm5              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%zmm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%zmm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%zmm27                     \n\t" \
    "vpaddd %%zmm6, %%zmm28, %%zmm6              \n\t"              \
    "vpaddd %%zmm7, %%zmm29, %%zmm7              \n\t"              \
    "vpaddd %%zmm8, %%zmm30, %%zmm8              \n\t"              \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm26, %%zmm29              \n\t"        \
    "vpmaddubsw %%zmm24, %%zmm27, %%zmm30              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"          \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"          \
    "vpmaddwd %%zmm30, %%zmm31, %%zmm30              \n\t"          \
    "vmovups 0x40(%1), %%zmm24                             \n\t"    \
    "vpaddd %%zmm9, %%zmm28, %%zmm9              \n\t"              \
    "vpaddd %%zmm10, %%zmm29, %%zmm10              \n\t"            \
    "vpaddd %%zmm11, %%zmm30, %%zmm11              \n\t"
#endif

inline void mmm_avx512_12x16_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x80(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "add $0x40, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%ebx \n\t"
        "vmovd %%ebx, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif
        "movq %8, %%rbx          \n\t"
        "andq $0x1, %%rbx          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%7), %%zmm0                       \n\t"
        "vmovups %%zmm0, %%zmm1                   \n\t"
        "vmovups %%zmm0, %%zmm2                   \n\t"
        "vmovups %%zmm0, %%zmm3                   \n\t"
        "vmovups %%zmm0, %%zmm4                   \n\t"
        "vmovups %%zmm0, %%zmm5                   \n\t"
        "vmovups %%zmm0, %%zmm6                   \n\t"
        "vmovups %%zmm0, %%zmm7                   \n\t"
        "vmovups %%zmm0, %%zmm8                   \n\t"
        "vmovups %%zmm0, %%zmm9                   \n\t"
        "vmovups %%zmm0, %%zmm10                   \n\t"
        "vmovups %%zmm0, %%zmm11                   \n\t"
        "jmp 1f          \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
        "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
        "vxorps %%zmm4, %%zmm4, %%zmm4                     \n\t"
        "vxorps %%zmm5, %%zmm5, %%zmm5                     \n\t"
        "vxorps %%zmm6, %%zmm6, %%zmm6                     \n\t"
        "vxorps %%zmm7, %%zmm7, %%zmm7                     \n\t"
        "vxorps %%zmm8, %%zmm8, %%zmm8                     \n\t"
        "vxorps %%zmm9, %%zmm9, %%zmm9                     \n\t"
        "vxorps %%zmm10, %%zmm10, %%zmm10                  \n\t"
        "vxorps %%zmm11, %%zmm11, %%zmm11                  \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "movq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel12x16

        "add $0x80, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "movq %2, %%rax  \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpaddd (%%rax), %%zmm0, %%zmm0                       \n\t"
        "vpaddd (%%rax, %4), %%zmm1, %%zmm1                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm2, %%zmm2                   \n\t"
        "vpaddd (%%rax, %4), %%zmm3, %%zmm3                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm4, %%zmm4                   \n\t"
        "vpaddd (%%rax, %4), %%zmm5, %%zmm5                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm6, %%zmm6                   \n\t"
        "vpaddd (%%rax, %4), %%zmm7, %%zmm7                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm8, %%zmm8                   \n\t"
        "vpaddd (%%rax, %4), %%zmm9, %%zmm9                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%zmm10, %%zmm10                   \n\t"
        "vpaddd (%%rax, %4), %%zmm11, %%zmm11                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
        "vcvtdq2ps %%zmm2, %%zmm2                       \n\t"
        "vcvtdq2ps %%zmm3, %%zmm3                       \n\t"
        "vcvtdq2ps %%zmm4, %%zmm4                       \n\t"
        "vcvtdq2ps %%zmm5, %%zmm5                       \n\t"
        "vcvtdq2ps %%zmm6, %%zmm6                       \n\t"
        "vcvtdq2ps %%zmm7, %%zmm7                       \n\t"
        "vcvtdq2ps %%zmm8, %%zmm8                       \n\t"
        "vcvtdq2ps %%zmm9, %%zmm9                       \n\t"
        "vcvtdq2ps %%zmm10, %%zmm10                       \n\t"
        "vcvtdq2ps %%zmm11, %%zmm11                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vmulps %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vmulps %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vmulps %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vmulps %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vmulps %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vmulps %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vmulps %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vmulps %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vmulps %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vmulps %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vmulps %%zmm11, %%zmm24, %%zmm11                     \n\t"

        "movq %8, %%rbx          \n\t"
        "andq $0x2, %%rbx          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
        "vcvtps2dq %%zmm4, %%zmm4                       \n\t"
        "vcvtps2dq %%zmm5, %%zmm5                       \n\t"
        "vcvtps2dq %%zmm6, %%zmm6                       \n\t"
        "vcvtps2dq %%zmm7, %%zmm7                       \n\t"
        "vcvtps2dq %%zmm8, %%zmm8                       \n\t"
        "vcvtps2dq %%zmm9, %%zmm9                       \n\t"
        "vcvtps2dq %%zmm10, %%zmm10                       \n\t"
        "vcvtps2dq %%zmm11, %%zmm11                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpaddd %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vpaddd %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vpaddd %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vpaddd %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vpaddd %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vpaddd %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vpaddd %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vpaddd %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "movq %9, %%rax  \n\t"
        "shr $2, %4                                     \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpmovusdb %%zmm0, (%%rax)                       \n\t"
        "vpmovusdb %%zmm1, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm2, (%%rax)                       \n\t"
        "vpmovusdb %%zmm3, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm4, (%%rax)                       \n\t"
        "vpmovusdb %%zmm5, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm6, (%%rax)                       \n\t"
        "vpmovusdb %%zmm7, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm8, (%%rax)                       \n\t"
        "vpmovusdb %%zmm9, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm10, (%%rax)                       \n\t"
        "vpmovusdb %%zmm11, (%%rax, %4)                       \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "vmovups %%zmm0, (%%rax)                       \n\t"
        "vmovups %%zmm1, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm2, (%%rax)                       \n\t"
        "vmovups %%zmm3, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm4, (%%rax)                       \n\t"
        "vmovups %%zmm5, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm6, (%%rax)                       \n\t"
        "vmovups %%zmm7, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm8, (%%rax)                       \n\t"
        "vmovups %%zmm9, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%zmm10, (%%rax)                       \n\t"
        "vmovups %%zmm11, (%%rax, %4)                       \n\t"

        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((int64_t)(N * 4)),
        "r"(scale), "r"((int64_t)stepK), "r"(offsetC), "r"((int64_t)flags), "r"(u8Result)
        : "%rax", "%rbx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
        "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16",
        "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25",
        "%zmm26", "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel12x8                                               \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "vpdpbusd %%ymm24, %%ymm25, %%ymm0              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm26, %%ymm1              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm27, %%ymm2              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm28, %%ymm3              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm29, %%ymm4              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm30, %%ymm5              \n\t"           \
    "vmovups (%1), %%ymm31                             \n\t"        \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vpdpbusd %%ymm24, %%ymm25, %%ymm6              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm26, %%ymm7              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm27, %%ymm8              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm28, %%ymm9              \n\t"           \
    "vpdpbusd %%ymm24, %%ymm29, %%ymm10              \n\t"          \
    "vpdpbusd %%ymm24, %%ymm30, %%ymm11              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vpdpbusd %%ymm31, %%ymm25, %%ymm0              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm26, %%ymm1              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm27, %%ymm2              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm28, %%ymm3              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm29, %%ymm4              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm30, %%ymm5              \n\t"           \
    "vmovups 0x20(%1), %%ymm24                             \n\t"    \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm28                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm29                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm30                     \n\t" \
    "vpdpbusd %%ymm31, %%ymm25, %%ymm6              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm26, %%ymm7              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm27, %%ymm8              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm28, %%ymm9              \n\t"           \
    "vpdpbusd %%ymm31, %%ymm29, %%ymm10              \n\t"          \
    "vpdpbusd %%ymm31, %%ymm30, %%ymm11              \n\t"
#else
#define mmmKernel12x8                                               \
    "movq %0, %%rax  \n\t"                                          \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "prefetcht0 0x80(%1)                              \n\t"         \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm0, %%ymm28, %%ymm0              \n\t"              \
    "vpaddd %%ymm1, %%ymm29, %%ymm1              \n\t"              \
    "vpaddd %%ymm2, %%ymm30, %%ymm2              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm3, %%ymm28, %%ymm3              \n\t"              \
    "vpaddd %%ymm4, %%ymm29, %%ymm4              \n\t"              \
    "vpaddd %%ymm5, %%ymm30, %%ymm5              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm6, %%ymm28, %%ymm6              \n\t"              \
    "vpaddd %%ymm7, %%ymm29, %%ymm7              \n\t"              \
    "vpaddd %%ymm8, %%ymm30, %%ymm8              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "movq %0, %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm9, %%ymm28, %%ymm9              \n\t"              \
    "vpaddd %%ymm10, %%ymm29, %%ymm10              \n\t"            \
    "vpaddd %%ymm11, %%ymm30, %%ymm11              \n\t"            \
    "vmovups (%1), %%ymm24                             \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm0, %%ymm28, %%ymm0              \n\t"              \
    "vpaddd %%ymm1, %%ymm29, %%ymm1              \n\t"              \
    "vpaddd %%ymm2, %%ymm30, %%ymm2              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm3, %%ymm28, %%ymm3              \n\t"              \
    "vpaddd %%ymm4, %%ymm29, %%ymm4              \n\t"              \
    "vpaddd %%ymm5, %%ymm30, %%ymm5              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), %%ymm25                     \n\t"        \
    "vpbroadcastd (%%rax, %6), %%ymm26                     \n\t"    \
    "vpbroadcastd (%%rax, %6, 2), %%ymm27                     \n\t" \
    "vpaddd %%ymm6, %%ymm28, %%ymm6              \n\t"              \
    "vpaddd %%ymm7, %%ymm29, %%ymm7              \n\t"              \
    "vpaddd %%ymm8, %%ymm30, %%ymm8              \n\t"              \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm26, %%ymm29              \n\t"        \
    "vpmaddubsw %%ymm24, %%ymm27, %%ymm30              \n\t"        \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"          \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"          \
    "vpmaddwd %%ymm30, %%ymm31, %%ymm30              \n\t"          \
    "vmovups 0x20(%1), %%ymm24                             \n\t"    \
    "vpaddd %%ymm9, %%ymm28, %%ymm9              \n\t"              \
    "vpaddd %%ymm10, %%ymm29, %%ymm10              \n\t"            \
    "vpaddd %%ymm11, %%ymm30, %%ymm11              \n\t"
#endif

inline void mmm_avx512_12x8_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x40(%1)                              \n\t"
        "vmovups (%1), %%ymm24                             \n\t"
        "add $0x20, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%ebx \n\t"
        "vmovd %%ebx, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%ymm31            \n\t"
#endif
        "movq %8, %%rbx          \n\t"
        "andq $0x1, %%rbx          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%7), %%ymm0                       \n\t"
        "vmovups %%ymm0, %%ymm1                   \n\t"
        "vmovups %%ymm0, %%ymm2                   \n\t"
        "vmovups %%ymm0, %%ymm3                   \n\t"
        "vmovups %%ymm0, %%ymm4                   \n\t"
        "vmovups %%ymm0, %%ymm5                   \n\t"
        "vmovups %%ymm0, %%ymm6                   \n\t"
        "vmovups %%ymm0, %%ymm7                   \n\t"
        "vmovups %%ymm0, %%ymm8                   \n\t"
        "vmovups %%ymm0, %%ymm9                   \n\t"
        "vmovups %%ymm0, %%ymm10                   \n\t"
        "vmovups %%ymm0, %%ymm11                   \n\t"
        "jmp 1f          \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
        "vxorps %%ymm5, %%ymm5, %%ymm5                     \n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6                     \n\t"
        "vxorps %%ymm7, %%ymm7, %%ymm7                     \n\t"
        "vxorps %%ymm8, %%ymm8, %%ymm8                     \n\t"
        "vxorps %%ymm9, %%ymm9, %%ymm9                     \n\t"
        "vxorps %%ymm10, %%ymm10, %%ymm10                  \n\t"
        "vxorps %%ymm11, %%ymm11, %%ymm11                  \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "movq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"
        "addq %6, %%rbx  \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel12x8

        "add $0x40, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "movq %2, %%rax  \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpaddd (%%rax), %%ymm0, %%ymm0                       \n\t"
        "vpaddd (%%rax, %4), %%ymm1, %%ymm1                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm2, %%ymm2                   \n\t"
        "vpaddd (%%rax, %4), %%ymm3, %%ymm3                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm4, %%ymm4                   \n\t"
        "vpaddd (%%rax, %4), %%ymm5, %%ymm5                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm6, %%ymm6                   \n\t"
        "vpaddd (%%rax, %4), %%ymm7, %%ymm7                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm8, %%ymm8                   \n\t"
        "vpaddd (%%rax, %4), %%ymm9, %%ymm9                   \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpaddd (%%rax), %%ymm10, %%ymm10                   \n\t"
        "vpaddd (%%rax, %4), %%ymm11, %%ymm11                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%ymm24                        \n\t"
        "vcvtdq2ps %%ymm0, %%ymm0                       \n\t"
        "vcvtdq2ps %%ymm1, %%ymm1                       \n\t"
        "vcvtdq2ps %%ymm2, %%ymm2                       \n\t"
        "vcvtdq2ps %%ymm3, %%ymm3                       \n\t"
        "vcvtdq2ps %%ymm4, %%ymm4                       \n\t"
        "vcvtdq2ps %%ymm5, %%ymm5                       \n\t"
        "vcvtdq2ps %%ymm6, %%ymm6                       \n\t"
        "vcvtdq2ps %%ymm7, %%ymm7                       \n\t"
        "vcvtdq2ps %%ymm8, %%ymm8                       \n\t"
        "vcvtdq2ps %%ymm9, %%ymm9                       \n\t"
        "vcvtdq2ps %%ymm10, %%ymm10                       \n\t"
        "vcvtdq2ps %%ymm11, %%ymm11                       \n\t"
        "vmulps %%ymm0, %%ymm24, %%ymm0                       \n\t"
        "vmulps %%ymm1, %%ymm24, %%ymm1                       \n\t"
        "vmulps %%ymm2, %%ymm24, %%ymm2                       \n\t"
        "vmulps %%ymm3, %%ymm24, %%ymm3                       \n\t"
        "vmulps %%ymm4, %%ymm24, %%ymm4                       \n\t"
        "vmulps %%ymm5, %%ymm24, %%ymm5                       \n\t"
        "vmulps %%ymm6, %%ymm24, %%ymm6                       \n\t"
        "vmulps %%ymm7, %%ymm24, %%ymm7                       \n\t"
        "vmulps %%ymm8, %%ymm24, %%ymm8                       \n\t"
        "vmulps %%ymm9, %%ymm24, %%ymm9                       \n\t"
        "vmulps %%ymm10, %%ymm24, %%ymm10                     \n\t"
        "vmulps %%ymm11, %%ymm24, %%ymm11                     \n\t"

        "movq %8, %%rbx          \n\t"
        "andq $0x2, %%rbx          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
        "vcvtps2dq %%zmm4, %%zmm4                       \n\t"
        "vcvtps2dq %%zmm5, %%zmm5                       \n\t"
        "vcvtps2dq %%zmm6, %%zmm6                       \n\t"
        "vcvtps2dq %%zmm7, %%zmm7                       \n\t"
        "vcvtps2dq %%zmm8, %%zmm8                       \n\t"
        "vcvtps2dq %%zmm9, %%zmm9                       \n\t"
        "vcvtps2dq %%zmm10, %%zmm10                       \n\t"
        "vcvtps2dq %%zmm11, %%zmm11                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpaddd %%zmm3, %%zmm24, %%zmm3                       \n\t"
        "vpaddd %%zmm4, %%zmm24, %%zmm4                       \n\t"
        "vpaddd %%zmm5, %%zmm24, %%zmm5                       \n\t"
        "vpaddd %%zmm6, %%zmm24, %%zmm6                       \n\t"
        "vpaddd %%zmm7, %%zmm24, %%zmm7                       \n\t"
        "vpaddd %%zmm8, %%zmm24, %%zmm8                       \n\t"
        "vpaddd %%zmm9, %%zmm24, %%zmm9                       \n\t"
        "vpaddd %%zmm10, %%zmm24, %%zmm10                     \n\t"
        "vpaddd %%zmm11, %%zmm24, %%zmm11                     \n\t"
        "movq %9, %%rax  \n\t"
        "shr $2, %4                                     \n\t"
        "movq %4, %%rcx  \n\t"
        "addq %4, %%rcx                                     \n\t"
        "vpmovusdb %%zmm0, (%%rax)                       \n\t"
        "vpmovusdb %%zmm1, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm2, (%%rax)                       \n\t"
        "vpmovusdb %%zmm3, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm4, (%%rax)                       \n\t"
        "vpmovusdb %%zmm5, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm6, (%%rax)                       \n\t"
        "vpmovusdb %%zmm7, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm8, (%%rax)                       \n\t"
        "vpmovusdb %%zmm9, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vpmovusdb %%zmm10, (%%rax)                       \n\t"
        "vpmovusdb %%zmm11, (%%rax, %4)                       \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "vmovups %%ymm0, (%%rax)                       \n\t"
        "vmovups %%ymm1, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm2, (%%rax)                       \n\t"
        "vmovups %%ymm3, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm4, (%%rax)                       \n\t"
        "vmovups %%ymm5, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm6, (%%rax)                       \n\t"
        "vmovups %%ymm7, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm8, (%%rax)                       \n\t"
        "vmovups %%ymm9, (%%rax, %4)                       \n\t"
        "addq %%rcx, %%rax  \n\t"
        "vmovups %%ymm10, (%%rax)                       \n\t"
        "vmovups %%ymm11, (%%rax, %4)                       \n\t"

        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"((int64_t)stepK), "r"(offsetC), "r"((int64_t)flags), "r"(u8Result)
        : "%rax", "%rbx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
        "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%ymm16",
        "%ymm17", "%ymm18", "%ymm19", "%ymm20", "%ymm21", "%ymm22", "%ymm23", "%ymm24", "%ymm25",
        "%ymm26", "%ymm27", "%ymm28", "%ymm29", "%ymm30", "%ymm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel1x48                                             \
    "movq %0, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "vpbroadcastd 0x4(%%rax), %%zmm31                     \n\t"   \
    "prefetcht0 0xC0(%1)                              \n\t"       \
    "prefetcht0 0x100(%1)                              \n\t"      \
    "prefetcht0 0x140(%1)                              \n\t"      \
    "vmovups (%1), %%zmm27                             \n\t"      \
    "vmovups 0x40(%1), %%zmm28                             \n\t"  \
    "vmovups 0x80(%1), %%zmm29                             \n\t"  \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm0              \n\t"         \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm1              \n\t"         \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm2              \n\t"         \
    "prefetcht0 0x180(%1)                              \n\t"      \
    "prefetcht0 0x1C0(%1)                              \n\t"      \
    "prefetcht0 0x200(%1)                              \n\t"      \
    "vmovups 0xC0(%1), %%zmm24                             \n\t"  \
    "vmovups 0x100(%1), %%zmm25                             \n\t" \
    "vmovups 0x140(%1), %%zmm26                             \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm0              \n\t"         \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm1              \n\t"         \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm2              \n\t"
#else
#define mmmKernel1x48                                             \
    "movq %0, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), %%zmm30                     \n\t"      \
    "prefetcht0 0xC0(%1)                              \n\t"       \
    "prefetcht0 0x100(%1)                              \n\t"      \
    "prefetcht0 0x140(%1)                              \n\t"      \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vmovups (%1), %%zmm24                             \n\t"      \
    "vmovups 0x40(%1), %%zmm25                             \n\t"  \
    "vmovups 0x80(%1), %%zmm26                             \n\t"  \
    "vpbroadcastd 0x4(%%rax), %%zmm30                     \n\t"   \
    "vpaddd %%zmm0, %%zmm27, %%zmm0              \n\t"            \
    "vpaddd %%zmm1, %%zmm28, %%zmm1              \n\t"            \
    "vpaddd %%zmm2, %%zmm29, %%zmm2              \n\t"            \
    "prefetcht0 0x180(%1)                              \n\t"      \
    "prefetcht0 0x1C0(%1)                              \n\t"      \
    "prefetcht0 0x200(%1)                              \n\t"      \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm27              \n\t"      \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm28              \n\t"      \
    "vpmaddubsw %%zmm26, %%zmm30, %%zmm29              \n\t"      \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"        \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"        \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"        \
    "vmovups 0xC0(%1), %%zmm24                             \n\t"  \
    "vmovups 0x100(%1), %%zmm25                             \n\t" \
    "vmovups 0x140(%1), %%zmm26                             \n\t" \
    "vpaddd %%zmm0, %%zmm27, %%zmm0              \n\t"            \
    "vpaddd %%zmm1, %%zmm28, %%zmm1              \n\t"            \
    "vpaddd %%zmm2, %%zmm29, %%zmm2              \n\t"
#endif

inline void mmm_avx512_1x48_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0xC0(%1)                              \n\t"
        "prefetcht0 0x100(%1)                              \n\t"
        "prefetcht0 0x140(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "vmovups 0x40(%1), %%zmm25                             \n\t"
        "vmovups 0x80(%1), %%zmm26                             \n\t"
        "add $0xC0, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%eax \n\t"
        "vmovd %%eax, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif
        "movq %%rbx, %%rax          \n\t"
        "andq $0x1, %%rax          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%6), %%zmm0                       \n\t"
        "vmovups 0x40(%6), %%zmm1                   \n\t"
        "vmovups 0x80(%6), %%zmm2                   \n\t"
        "jmp 1f                                         \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "movq %2, %%rax  \n\t"
        "add %4, %%rax                                     \n\t"
        "prefetcht0 (%%rax)                              \n\t"
        "prefetcht0 0x40(%%rax)                              \n\t"
        "prefetcht0 0x80(%%rax)                              \n\t"
        "prefetcht0 (%%rax, %4)                              \n\t"
        "prefetcht0 0x40(%%rax, %4)                              \n\t"
        "prefetcht0 0x80(%%rax, %4)                              \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t" mmmKernel1x48

        "add $0x180, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 2b                                             \n\t"

        "vpaddd (%2), %%zmm0, %%zmm0                       \n\t"
        "vpaddd 0x40(%2), %%zmm1, %%zmm1                   \n\t"
        "vpaddd 0x80(%2), %%zmm2, %%zmm2                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 3f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
        "vcvtdq2ps %%zmm2, %%zmm2                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vmulps %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vmulps %%zmm2, %%zmm24, %%zmm2                       \n\t"

        "movq %%rbx, %%rax          \n\t"
        "andq $0x2, %%rax          \n\t"
        "je 3f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpaddd %%zmm2, %%zmm24, %%zmm2                       \n\t"
        "vpmovusdb %%zmm0,  (%8)                             \n\t"
        "vpmovusdb %%zmm1,  0x10(%8)                         \n\t"
        "vpmovusdb %%zmm2,  0x20(%8)                         \n\t"
        "jmp 4f                                         \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        "vmovups %%zmm0,  (%2)                             \n\t"
        "vmovups %%zmm1,  0x40(%2)                         \n\t"
        "vmovups %%zmm2,  0x80(%2)                         \n\t"
        ".align 16                                         \n\t"
        "4:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"(offsetC), "b"((int64_t)flags), "r"(u8Result)
        : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8",
        "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",
        "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
        "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel1x32                                            \
    "vpbroadcastd (%0), %%zmm28                     \n\t"        \
    "vpbroadcastd 0x4(%0), %%zmm29                     \n\t"     \
    "prefetcht0 0x80(%1)                              \n\t"      \
    "prefetcht0 0xC0(%1)                              \n\t"      \
    "vmovups (%1), %%zmm26                             \n\t"     \
    "vmovups 0x40(%1), %%zmm27                             \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm0              \n\t"        \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm1              \n\t"        \
    "prefetcht0 0x100(%1)                              \n\t"     \
    "prefetcht0 0x140(%1)                              \n\t"     \
    "vmovups 0x80(%1), %%zmm24                             \n\t" \
    "vmovups 0xC0(%1), %%zmm25                             \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm0              \n\t"        \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm1              \n\t"
#else
#define mmmKernel1x32                                            \
    "vpbroadcastd (%0), %%zmm30                     \n\t"        \
    "prefetcht0 0x80(%1)                              \n\t"      \
    "prefetcht0 0xC0(%1)                              \n\t"      \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm26              \n\t"     \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm27              \n\t"     \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"       \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"       \
    "vpbroadcastd 0x4(%0), %%zmm30                     \n\t"     \
    "vmovups (%1), %%zmm24                             \n\t"     \
    "vmovups 0x40(%1), %%zmm25                             \n\t" \
    "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"           \
    "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"           \
    "prefetcht0 0x100(%1)                              \n\t"     \
    "prefetcht0 0x140(%1)                              \n\t"     \
    "vpmaddubsw %%zmm24, %%zmm30, %%zmm26              \n\t"     \
    "vpmaddubsw %%zmm25, %%zmm30, %%zmm27              \n\t"     \
    "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"       \
    "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"       \
    "vmovups 0x80(%1), %%zmm24                             \n\t" \
    "vmovups 0xC0(%1), %%zmm25                             \n\t" \
    "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"           \
    "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"
#endif

inline void mmm_avx512_1x32_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x80(%1)                              \n\t"
        "prefetcht0 0xC0(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "vmovups 0x40(%1), %%zmm25                             \n\t"
        "add $0x80, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%eax \n\t"
        "vmovd %%eax, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif
        "movq %%rbx, %%rax          \n\t"
        "andq $0x1, %%rax          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%6), %%zmm0                       \n\t"
        "vmovups 0x40(%6), %%zmm1                   \n\t"
        "jmp 1f                                         \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t" mmmKernel1x32

        "add $0x100, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vpaddd (%2), %%zmm0, %%zmm0                       \n\t"
        "vpaddd 0x40(%2), %%zmm1, %%zmm1                   \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 2f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vmulps %%zmm1, %%zmm24, %%zmm1                       \n\t"

        "movq %%rbx, %%rax          \n\t"
        "andq $0x2, %%rax          \n\t"
        "je 2f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpaddd %%zmm1, %%zmm24, %%zmm1                       \n\t"
        "vpmovusdb %%zmm0,  (%8)                             \n\t"
        "vpmovusdb %%zmm1,  0x10(%8)                         \n\t"
        "jmp 3f                                         \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t"
        "vmovups %%zmm0, (%2)                       \n\t"
        "vmovups %%zmm1, 0x40(%2)                       \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"(offsetC), "b"((int64_t)flags), "r"(u8Result)
        : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8",
        "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",
        "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
        "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel1x16                                            \
    "vpbroadcastd (%0), %%zmm25                     \n\t"        \
    "vpbroadcastd 0x4(%0), %%zmm26                     \n\t"     \
    "prefetcht0 0x80(%1)                              \n\t"      \
    "vmovups (%1), %%zmm31                             \n\t"     \
    "vpdpbusd %%zmm24, %%zmm25, %%zmm0              \n\t"        \
    "prefetcht0 0xC0(%1)                              \n\t"      \
    "vmovups 0x40(%1), %%zmm24                             \n\t" \
    "vpdpbusd %%zmm31, %%zmm26, %%zmm0              \n\t"
#else
#define mmmKernel1x16                                            \
    "vpbroadcastd (%0), %%zmm25                     \n\t"        \
    "vpbroadcastd 0x4(%0), %%zmm26                     \n\t"     \
    "prefetcht0 0x80(%1)                              \n\t"      \
    "vmovups (%1), %%zmm30                             \n\t"     \
    "vpmaddubsw %%zmm24, %%zmm25, %%zmm28              \n\t"     \
    "vpmaddubsw %%zmm30, %%zmm26, %%zmm29              \n\t"     \
    "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"       \
    "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"       \
    "prefetcht0 0xC0(%1)                              \n\t"      \
    "vmovups 0x40(%1), %%zmm24                             \n\t" \
    "vpaddd %%zmm0, %%zmm28, %%zmm0              \n\t"           \
    "vpaddd %%zmm0, %%zmm29, %%zmm0              \n\t"
#endif

inline void mmm_avx512_1x16_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x80(%1)                              \n\t"
        "vmovups (%1), %%zmm24                             \n\t"
        "add $0x40, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%eax \n\t"
        "vmovd %%eax, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%zmm31            \n\t"
#endif
        "movq %%rbx, %%rax          \n\t"
        "andq $0x1, %%rax          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%6), %%zmm0                       \n\t"
        "jmp 1f                                         \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t" mmmKernel1x16

        "add $0x80, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vpaddd (%2), %%zmm0, %%zmm0                       \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 2f      \n\t"

        "vbroadcastss (%5), %%zmm24                        \n\t"
        "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
        "vmulps %%zmm0, %%zmm24, %%zmm0                       \n\t"

        "movq %%rbx, %%rax          \n\t"
        "andq $0x2, %%rax          \n\t"
        "je 2f                                         \n\t"
        "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%zmm24            \n\t"
        "vpaddd %%zmm0, %%zmm24, %%zmm0                       \n\t"
        "vpmovusdb %%zmm0,  (%8)                             \n\t"
        "jmp 3f                                         \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t"
        "vmovups %%zmm0, (%2)                       \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"(offsetC), "b"((int64_t)flags), "r"(u8Result)
        : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "%zmm8",
        "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",
        "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
        "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define mmmKernel1x8                                             \
    "vpbroadcastd (%0), %%ymm25                     \n\t"        \
    "vpbroadcastd 0x4(%0), %%ymm26                     \n\t"     \
    "prefetcht0 0x40(%1)                              \n\t"      \
    "vmovups (%1), %%ymm31                             \n\t"     \
    "vpdpbusd %%ymm24, %%ymm25, %%ymm0              \n\t"        \
    "vmovups 0x20(%1), %%ymm24                             \n\t" \
    "vpdpbusd %%ymm31, %%ymm26, %%ymm0              \n\t"
#else
#define mmmKernel1x8                                             \
    "vpbroadcastd (%0), %%ymm25                     \n\t"        \
    "vpbroadcastd 0x4(%0), %%ymm26                     \n\t"     \
    "prefetcht0 0x80(%1)                              \n\t"      \
    "vmovups (%1), %%ymm30                             \n\t"     \
    "vpmaddubsw %%ymm24, %%ymm25, %%ymm28              \n\t"     \
    "vpmaddubsw %%ymm30, %%ymm26, %%ymm29              \n\t"     \
    "vpmaddwd %%ymm28, %%ymm31, %%ymm28              \n\t"       \
    "vpmaddwd %%ymm29, %%ymm31, %%ymm29              \n\t"       \
    "vmovups 0x20(%1), %%ymm24                             \n\t" \
    "vpaddd %%ymm0, %%ymm28, %%ymm0              \n\t"           \
    "vpaddd %%ymm0, %%ymm29, %%ymm0              \n\t"
#endif

inline void mmm_avx512_1x8_asm(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__(
        "prefetcht0 0x40(%1)                              \n\t"
        "vmovups (%1), %%ymm24                             \n\t"
        "add $0x20, %1                                    \n\t"
#ifndef _USE_AVX512_VNNI
        "mov $1, %%eax \n\t"
        "vmovd %%eax, %%xmm0                    \n\t"
        "vpbroadcastw %%xmm0, %%ymm31            \n\t"
#endif
        "movq %%rbx, %%rax          \n\t"
        "andq $0x1, %%rax          \n\t"
        "jne 0f                                         \n\t"
        "vmovups (%6), %%ymm0                       \n\t"
        "jmp 1f                                         \n\t"

        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t" mmmKernel1x8

        "add $0x40, %1                                    \n\t"
        "add $0x8, %0                                     \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vpaddd (%2), %%ymm0, %%ymm0                       \n\t"

        "cmpq $0x0, %5 \n\t"
        "je 2f      \n\t"

        "vbroadcastss (%5), %%ymm24                        \n\t"
        "vcvtdq2ps %%ymm0, %%ymm0                       \n\t"
        "vmulps %%ymm0, %%ymm24, %%ymm0                       \n\t"
        "movq %%rbx, %%rax          \n\t"
        "andq $0x2, %%rax          \n\t"
        "je 2f                                         \n\t"
        "vcvtps2dq %%ymm0, %%ymm0                       \n\t"
        "mov $128, %%eax \n\t"
        "vmovd %%eax, %%xmm25                    \n\t"
        "vbroadcastss %%xmm25, %%ymm24            \n\t"
        "vpaddd %%ymm0, %%ymm24, %%ymm0                       \n\t"
        "vpmovusdb %%ymm0,  (%8)                             \n\t"
        "jmp 3f                                         \n\t"

        ".align 16                                         \n\t"
        "2:                                                \n\t"
        "vmovups %%ymm0, (%2)                       \n\t"

        ".align 16                                         \n\t"
        "3:                                                \n\t"
        :
        : "r"(matrixA), "r"(matrixB), "r"(matrixC), "c"((int64_t)bk), "r"((long long)(N * 4)),
        "r"(scale), "r"(offsetC), "b"((int64_t)flags), "r"(u8Result)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%ymm16", "%ymm17",
        "%ymm18", "%ymm19", "%ymm20", "%ymm21", "%ymm22", "%ymm23", "%ymm24", "%ymm25", "%ymm26",
        "%ymm27", "%ymm28", "%ymm29", "%ymm30", "%ymm31", "memory", "cc");
}

void mmm_avx512_n_mtail(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 flags)
{
    I32 *result = (I32 *)matrixC;
    F32 *resultF32 = (F32 *)matrixC;
    for (U32 i = 0; i < um; ++i) {
        for (U32 j = 0; j < un; ++j) {
            I32 tmp = result[i * N + j];
            for (U32 k = 0; k < bk * 8; k += 4) {
                if (((flags & 0x1) == 0) && (k == 0)) {
                    tmp += offsetC[j];
                }
                for (U32 k4 = 0; k4 < 4; ++k4) {
                    tmp += (int)matrixA[i * stepK + k4 + k] * (int)matrixB[k * un + j * 4 + k4];
                }
            }
            if (scale != nullptr) {
                resultF32[i * N + j] = tmp * scale[0];
                if ((flags & 0x2) != 0) {
                    tmp = (I32)(resultF32[i * N + j] + 128);
                    u8Result[i * N + j] = (tmp > 255) ? 255 : tmp;
                }
            } else {
                result[i * N + j] = tmp;
            }
        }
    }
}

//TODO: matrixC alloc
EE mmm_avx512_vnni_int8(U32 N,
    U32 M,
    U32 K,
    DataFormat matrix1Df,
    UINT8 *matrix1,
    INT8 *packB,
    UINT8 *tmp,
    UINT8 *result,
    const F32 *scale)
{
    UINT8 *packA = matrix1;
    kernel_func kernel[3][5] = {{mmm_avx512_n_mtail, mmm_avx512_1x8_asm, mmm_avx512_1x16_asm,
                                    mmm_avx512_1x32_asm, mmm_avx512_1x48_asm},
        {mmm_avx512_n_mtail, mmm_avx512_12x8_asm, mmm_avx512_12x16_asm, mmm_avx512_6x32_asm,
            mmm_avx512_4x48_asm},
        {mmm_avx512_n_mtail, mmm_avx512_24x8_asm, mmm_avx512_24x16_asm, mmm_avx512_12x32_asm,
            mmm_avx512_8x48_asm}};
    U32 unrollNSizes[5] = {8, 8, 16, 32, 48};
    U32 unrollMSize[5] = {M, 24, 24, 12, 8};
    U32 alignedK = (K + 7) / 8 * 8;

    I32 *offsetC = (I32 *)(tmp);
    tmp += N * bytesOf(DT_I32);

    UINT8 *tmpA = tmp;
    tmp += M * alignedK * bytesOf(DT_U8_Q);
    packB += N * bytesOf(DT_I32);
    if (uintptr_t(tmp + N * bytesOf(DT_I32)) == uintptr_t(packB)) {  // matmul
        tmp += N * alignedK * bytesOf(DT_I8) + N * bytesOf(DT_I32);
    }

    U32 flags = 0;
    F32 *factorPtr = nullptr;
    F32 factor = 0;
    I32 *i32Result = (I32 *)result;
    UINT8 *u8Result = result;
    if (scale != nullptr) {
        if (scale[0] <
            0) {  // when use offline scale, the output datatype is U8_Q, you need more tmp buffer
            flags |= 1 << 1;
            factor = scale[1];
            i32Result = (I32 *)tmp;
            memset(i32Result, 0, M * N * bytesOf(DT_I32));
            tmp += M * N * bytesOf(DT_I32);
        } else {
            factor = 1 / (scale[0]);
        }
        factorPtr = &factor;
    }

    auto computeMNums = [=](U32 block, U32 unit) {
        return block / unit + (block % unit >= (unit / 2)) + (block % (unit / 2));
    };

    U32 mNum = M / BOLCK_M_DIM;
    U32 unNum = N / UNROLL_N;
    U32 unArrays[4] = {0};
    U32 umArrays[4] = {0};
    U32 umNums[4] = {0};
    U32 umResNums[4] = {0};
    U32 res = N % UNROLL_N;
    unArrays[0] = UNROLL_N;
    umArrays[0] = unrollMSize[(UNROLL_N >> 4) + 1];
    umNums[0] = computeMNums(BOLCK_M_DIM, umArrays[0]);
    U32 idx = 1;
    while (res > 0) {
        unArrays[idx] = UNI_MIN(unrollNSizes[(res >> 4) + 1], res);
        umArrays[idx] = unrollMSize[(res >> 4) + 1];
        umNums[idx] = computeMNums(BOLCK_M_DIM, umArrays[idx]);
        if (unArrays[idx] < 8) {
            umArrays[idx] = UNI_MIN(unrollMSize[0], BOLCK_M_DIM);
            umNums[idx] = 1;
        }
        res -= unArrays[idx++];
    }
    U32 nLoopNum = unNum * umNums[0] + umNums[1] + umNums[2] + umNums[3];
    U32 mLoopNum = nLoopNum * mNum;
    U32 nLoopResNum = 0;
    if (M % BOLCK_M_DIM > 0) {
        res = M % BOLCK_M_DIM;
        for (U32 i = 0; i < 4 && umArrays[i] > 0; ++i) {
            if (unArrays[i] < 8) {
                umResNums[i] = 1;
            } else {
                umResNums[i] = computeMNums(res, umArrays[i]);
            }
        }
        nLoopResNum = (unNum * umResNums[0] + umResNums[1] + umResNums[2] + umResNums[3]);
    }
    idx = (unNum > 0) ? 0 : 1;
    U32 umUnit = umArrays[idx];
    U32 firstLoopNum = (unArrays[idx] >= 8) ? computeMNums(M, umUnit) : 1;
    U32 loopNum = mLoopNum + nLoopResNum - firstLoopNum;
    if (unNum >= 1) {
        unNum -= 1;
        nLoopNum -= umNums[0];
        nLoopResNum -= umResNums[0];
    } else {
        nLoopNum -= umNums[1];
        nLoopResNum -= umResNums[1];
    }
    mLoopNum = nLoopNum * mNum;

#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (mLoopNum + nLoopResNum > OMP_NUM_THREADS)
    {
#endif
        U32 blockSizeK = 0;
        for (U32 k = 0; k < K; k += blockSizeK) {
            blockSizeK = UNI_MIN(BOLCK_K_DIM, K - k);
            blockSizeK = UNI_MAX(blockSizeK % SIMDW, blockSizeK - blockSizeK % SIMDW);
            U32 alignedBlockSizeK = align_size(blockSizeK, SIMDW);
            F32 *useFactor = nullptr;
            flags |= (k > 0);
            if (k == K - blockSizeK) {
                useFactor = factorPtr;
            }

#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
            for (U32 l = 0; l < firstLoopNum; ++l) {
                U32 umNum = M / umUnit;
                U32 idxM = 2;
                U32 m = 0;
                U32 unrollSizeM = 0;
                if (l < umNum) {
                    m = l * umUnit;
                    unrollSizeM = umUnit;
                } else if (l == umNum) {
                    m = umNum * umUnit;
                    if ((M - umNum * umUnit) >= (umUnit / 2)) {
                        unrollSizeM = umUnit / 2;
                        idxM = 1;
                    } else {
                        unrollSizeM = 1;
                        idxM = 0;
                    }
                } else {
                    if (M >= (umNum * umUnit + umUnit / 2)) {
                        m = umNum * umUnit + umUnit / 2 + (l - umNum - 1);
                    } else {
                        m = umNum * umUnit + (l - umNum);
                    }
                    unrollSizeM = 1;
                    idxM = 0;
                }

                U32 stepK = K;
                INT8 *curB = packB + k * N;
                UINT8 *curA = packA + m * stepK + k;
                if (matrix1Df == DF_TRANSPOSE) {
                    curA = tmpA + m * alignedBlockSizeK;
                    matrix2_trans_r(unrollSizeM, blockSizeK, M, SIMDW, matrix1 + m + k * M, curA);
                    stepK = alignedBlockSizeK;
                } else if (matrix1Df == DF_NORMAL && blockSizeK < SIMDW) {
                    curA = tmpA + m * alignedBlockSizeK;
                    matrix1_trans_r(unrollSizeM, blockSizeK, K, SIMDW, matrix1 + k + m * K, curA);
                    stepK = alignedBlockSizeK;
                }
                kernel[idxM][(unArrays[idx] >> 4) + (unArrays[idx] >= 8)](unrollSizeM,
                    unArrays[idx], alignedBlockSizeK / 8, curA, curB, i32Result + m * N,
                    u8Result + m * N, offsetC, N, stepK, useFactor, flags);
            }

#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
            for (U32 l = 0; l < loopNum; ++l) {
                U32 bm = l / nLoopNum * BOLCK_M_DIM;
                U32 nLoop = l % nLoopNum;
                U32 unrollSizeN = 0;
                U32 blockSizeM = 0;
                U32 unrollM = 0;
                U32 m = 0, n = 0;
                U32 *umNumsPtr;
                if (l < mLoopNum) {
                    blockSizeM = BOLCK_M_DIM;
                    umNumsPtr = umNums;
                } else {
                    blockSizeM = M % BOLCK_M_DIM;
                    umNumsPtr = umResNums;
                }

                if (nLoop < unNum * umNumsPtr[0]) {
                    n = nLoop / umNumsPtr[0] * unArrays[0];
                    m = nLoop % umNumsPtr[0];
                    unrollSizeN = unArrays[0];
                    unrollM = umArrays[0];
                } else {
                    n = unNum * unArrays[0];
                    U32 x = unNum * umNumsPtr[0];
                    for (int j = idx + 1; j < 4; x += umNumsPtr[j], n += unArrays[j], ++j) {
                        if (nLoop < x + umNumsPtr[j]) {
                            m = nLoop - x;
                            unrollSizeN = unArrays[j];
                            unrollM = umArrays[j];
                            break;
                        }
                    }
                }

                U32 unrollSizeM = 0;
                U32 umNum = blockSizeM / unrollM;
                U32 idxM = 2;
                if (m < umNum) {
                    m = m * unrollM;
                    unrollSizeM = unrollM;
                } else if (m == umNum) {
                    m = umNum * unrollM;
                    if ((blockSizeM - umNum * unrollM) >= (unrollM / 2)) {
                        unrollSizeM = unrollM / 2;
                        idxM = 1;
                    } else {
                        unrollSizeM = 1;
                        idxM = 0;
                    }
                } else {
                    if (blockSizeM >= (umNum * unrollM + unrollM / 2)) {
                        m = umNum * unrollM + unrollM / 2 + (m - umNum - 1);
                    } else {
                        m = umNum * unrollM + (m - umNum);
                    }
                    unrollSizeM = 1;
                    idxM = 0;
                }

                n += unArrays[idx];
                INT8 *curB = packB + k * N + n * alignedBlockSizeK;
                UINT8 *curA = packA + (m + bm) * K + k;
                kernel[idxM][(unrollSizeN >> 4) + (unrollSizeN >= 8)](unrollSizeM, unrollSizeN,
                    alignedBlockSizeK / 8, curA, curB, i32Result + (m + bm) * N + n,
                    u8Result + (m + bm) * N + n, offsetC + n, N, K, useFactor, flags);
            }
        }
#ifdef _USE_OPENMP
    }
#endif

    return SUCCESS;
}
