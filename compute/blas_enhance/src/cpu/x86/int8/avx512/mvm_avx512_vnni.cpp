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
#define UNROLL_N 64
#define BOLCK_K_DIM 2048

typedef void (*kernel_func)(U32 bn,
    U32 bk,
    INT8 *matrix,
    UINT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags);

EE matrix_vector_multiply_transform_weight_int8(
    TensorDesc desc, INT8 *src, INT8 *packB, I32 *offsetCBias)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    U32 unrollSize[5] = {8, 16, 32, 32, 64};
    U32 unrollSizeN = 0;
    EE ret = SUCCESS;
    bool hasBias = (offsetCBias != nullptr);
    switch (desc.df) {
        case DF_NORMAL: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
            I32 *sumB = nullptr;
            if (!hasBias) {
                sumB = (I32 *)(packB + UNI_ALIGN(N, 16) * UNI_ALIGN(K, 8));
                UNI_MEMSET(sumB, 0, N * sizeof(I32));
            } else {
                sumB = offsetCBias;
            }
            U32 blockKSize = 0;
            for (U32 bk = 0; bk < K; bk += blockKSize) {
                blockKSize = UNI_MIN(K - bk, BOLCK_K_DIM);
                U32 alignedBlockSizeK = UNI_ALIGN(blockKSize, 4);
                for (U32 un = 0; un < N; un += unrollSizeN) {
                    unrollSizeN = UNI_MIN(UNROLL_N, N - un);
                    unrollSizeN = unrollSize[unrollSizeN >> 4];
                    if (N - un < unrollSizeN) {
                        unrollSizeN = N - un;
                        UNI_MEMSET(packB, 0, unrollSizeN * alignedBlockSizeK);
                        for (U32 k = 0; k < alignedBlockSizeK; k += 4) {
                            for (U32 i = 0; i < unrollSizeN; ++i) {
                                for (U32 ii = 0; ii < 4 && k + ii < blockKSize; ++ii) {
                                    packB[(k / 4) * (unrollSizeN * 4) + i * 4 + ii] =
                                        src[(un + i) * K + k + bk + ii];
                                }
                            }
                        }
                    } else {
                        matrix1_trans_l(
                            unrollSizeN, unrollSizeN, blockKSize, K, 4, src + un * K + bk, packB);
                    }
                    packB += unrollSizeN * alignedBlockSizeK;
                }
            }
            for (U32 i = 0; i < N; ++i) {
                I32 sumT = 0;
                for (U32 j = 0; j < K; ++j) {
                    sumT += (I32)(src[i * K + j]);
                }
                sumB[i] = sumT * (-128);
            }
            break;
        }
        case DF_TRANSPOSE: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
            I32 *sumB = nullptr;
            if (!hasBias) {
		sumB = (I32 *)(packB + UNI_ALIGN(N, 16) * UNI_ALIGN(K, 8));
                UNI_MEMSET(sumB, 0, N * sizeof(I32));
            } else {
                sumB = offsetCBias;
            }
            U32 blockKSize = 0;
            for (U32 bk = 0; bk < K; bk += blockKSize) {
                blockKSize = UNI_MIN(K - bk, BOLCK_K_DIM);
                U32 alignedBlockSizeK = UNI_ALIGN(blockKSize, 4);
                for (U32 un = 0; un < N; un += unrollSizeN) {
                    unrollSizeN = UNI_MIN(UNROLL_N, N - un);
                    unrollSizeN = unrollSize[unrollSizeN >> 4];
                    if (N - un < unrollSizeN) {
                        unrollSizeN = N - un;
                        UNI_MEMSET(packB, 0, unrollSizeN * alignedBlockSizeK);
                        for (U32 k = 0; k < blockKSize; k += 4) {
                            for (U32 i = 0; i < unrollSizeN; ++i) {
                                for (U32 ii = 0; ii < 4 && k + ii < blockKSize; ++ii) {
                                    packB[(k / 4) * (unrollSizeN * 4) + i * 4 + ii] =
                                        src[(k + bk + ii) * N + un + i];
                                }
                            }
                        }
                    } else {
                        matrix2_trans_l(
                            unrollSizeN, unrollSizeN, blockKSize, N, 4, src + un + bk * N, packB);
                    }
                    packB += unrollSizeN * alignedBlockSizeK;
                }
            }

            for (U32 i = 0; i < N; ++i) {
                I32 sumT = 0;
                for (U32 j = 0; j < K; ++j) {
                    sumT += (I32)(src[j * N + i]);
                }
                sumB[i] = sumT * (-128);
            }
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

void mvm_row_avx512_64(U32 bn,
    U32 bk,
    INT8 *matrix,
    UINT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("mov %%ebx, %%eax          \n\t"
                         "and $0x1, %%eax          \n\t"
                         "jne 0f                                         \n\t"
                         "vmovups (%4), %%zmm0                       \n\t"
                         "vmovups 0x40(%4), %%zmm1                   \n\t"
                         "vmovups 0x80(%4), %%zmm2                   \n\t"
                         "vmovups 0xC0(%4), %%zmm3                   \n\t"
                         "jmp 1f          \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
                         "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
                         "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
#ifndef _USE_AVX512_VNNI
                         "mov $1, %%eax \n\t"
                         "vmovd %%eax, %%xmm24                    \n\t"
                         "vpbroadcastw %%xmm24, %%zmm31            \n\t"
#endif
                         "vpbroadcastd (%0), %%zmm4                     \n\t"
                         "vmovups (%1), %%zmm5                             \n\t"
                         "vmovups 0x40(%1), %%zmm6                             \n\t"
                         "vmovups 0x80(%1), %%zmm7                             \n\t"
                         "vmovups 0xC0(%1), %%zmm8                             \n\t"
                         "vpaddd (%2), %%zmm0, %%zmm0                    \n\t"
                         "vpaddd 0x40(%2), %%zmm1, %%zmm1                \n\t"
                         "vpaddd 0x80(%2), %%zmm2, %%zmm2                \n\t"
                         "vpaddd 0xC0(%2), %%zmm3, %%zmm3                \n\t"
                         "add $0x100, %1                                    \n\t"
                         "mov %3, %%ecx                                    \n\t"
                         "shr $4, %%ecx                                    \n\t"
                         "jz 3f                                      \n\t"

                         ".align 16                                         \n\t"
                         "2:                                      \n\t"
                         "vpbroadcastd 0x4(%0), %%zmm9                     \n\t"
                         "vmovups (%1), %%zmm10                             \n\t"
                         "vmovups 0x40(%1), %%zmm11                             \n\t"
                         "vmovups 0x80(%1), %%zmm12                             \n\t"
                         "vmovups 0xC0(%1), %%zmm13                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm5, %%zmm4, %%zmm0              \n\t"
                         "vpdpbusd %%zmm6, %%zmm4, %%zmm1              \n\t"
                         "vpdpbusd %%zmm7, %%zmm4, %%zmm2              \n\t"
                         "vpdpbusd %%zmm8, %%zmm4, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm5, %%zmm4, %%zmm24              \n\t"
                         "vpmaddubsw %%zmm6, %%zmm4, %%zmm25              \n\t"
                         "vpmaddubsw %%zmm7, %%zmm4, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm8, %%zmm4, %%zmm27              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpmaddwd %%zmm25, %%zmm31, %%zmm25              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm25, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm26, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"
#endif

                         "vpbroadcastd 0x8(%0), %%zmm14                     \n\t"
                         "vmovups 0x100(%1), %%zmm15                        \n\t"
                         "vmovups 0x140(%1), %%zmm16                        \n\t"
                         "vmovups 0x180(%1), %%zmm17                        \n\t"
                         "vmovups 0x1C0(%1), %%zmm18                        \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm10, %%zmm9, %%zmm0              \n\t"
                         "vpdpbusd %%zmm11, %%zmm9, %%zmm1              \n\t"
                         "vpdpbusd %%zmm12, %%zmm9, %%zmm2              \n\t"
                         "vpdpbusd %%zmm13, %%zmm9, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm10, %%zmm9, %%zmm24              \n\t"
                         "vpmaddubsw %%zmm11, %%zmm9, %%zmm25              \n\t"
                         "vpmaddubsw %%zmm12, %%zmm9, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm13, %%zmm9, %%zmm27              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpmaddwd %%zmm25, %%zmm31, %%zmm25              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm25, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm26, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"
#endif

                         "vpbroadcastd 0xC(%0), %%zmm19                     \n\t"
                         "vmovups 0x200(%1), %%zmm20                             \n\t"
                         "vmovups 0x240(%1), %%zmm21                             \n\t"
                         "vmovups 0x280(%1), %%zmm22                             \n\t"
                         "vmovups 0x2C0(%1), %%zmm23                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm15, %%zmm14, %%zmm0              \n\t"
                         "vpdpbusd %%zmm16, %%zmm14, %%zmm1              \n\t"
                         "vpdpbusd %%zmm17, %%zmm14, %%zmm2              \n\t"
                         "vpdpbusd %%zmm18, %%zmm14, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm15, %%zmm14, %%zmm24              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm14, %%zmm25              \n\t"
                         "vpmaddubsw %%zmm17, %%zmm14, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm18, %%zmm14, %%zmm27              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpmaddwd %%zmm25, %%zmm31, %%zmm25              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm25, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm26, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"
#endif

                         "vpbroadcastd 0x10(%0), %%zmm4                  \n\t"
                         "vmovups 0x300(%1), %%zmm5                      \n\t"
                         "vmovups 0x340(%1), %%zmm6                      \n\t"
                         "vmovups 0x380(%1), %%zmm7                      \n\t"
                         "vmovups 0x3C0(%1), %%zmm8                      \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm20, %%zmm19, %%zmm0              \n\t"
                         "vpdpbusd %%zmm21, %%zmm19, %%zmm1              \n\t"
                         "vpdpbusd %%zmm22, %%zmm19, %%zmm2              \n\t"
                         "vpdpbusd %%zmm23, %%zmm19, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm20, %%zmm19, %%zmm24              \n\t"
                         "vpmaddubsw %%zmm21, %%zmm19, %%zmm25              \n\t"
                         "vpmaddubsw %%zmm22, %%zmm19, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm23, %%zmm19, %%zmm27              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpmaddwd %%zmm25, %%zmm31, %%zmm25              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm25, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm26, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"
#endif

                         "add $0x10, %0                                  \n\t"
                         "add $0x400, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 2b                                          \n\t"

                         ".align 16                                         \n\t"
                         "3:                                           \n\t"
                         "mov %3, %%ecx                                    \n\t"
                         "and $0xF, %%ecx                                    \n\t"
                         "shr $2, %%ecx                                    \n\t"
                         "jz 5f                                          \n\t"
                         ".align 16                                         \n\t"
                         "4:                                           \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm5, %%zmm4, %%zmm0              \n\t"
                         "vpdpbusd %%zmm6, %%zmm4, %%zmm1              \n\t"
                         "vpdpbusd %%zmm7, %%zmm4, %%zmm2              \n\t"
                         "vpdpbusd %%zmm8, %%zmm4, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm5, %%zmm4, %%zmm24              \n\t"
                         "vpmaddubsw %%zmm6, %%zmm4, %%zmm25              \n\t"
                         "vpmaddubsw %%zmm7, %%zmm4, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm8, %%zmm4, %%zmm27              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpmaddwd %%zmm25, %%zmm31, %%zmm25              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm25, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm26, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm27, %%zmm3              \n\t"
#endif

                         "vpbroadcastd 0x4(%0), %%zmm4                  \n\t"
                         "vmovups (%1), %%zmm5                          \n\t"
                         "vmovups 0x40(%1), %%zmm6                      \n\t"
                         "vmovups 0x80(%1), %%zmm7                      \n\t"
                         "vmovups 0xC0(%1), %%zmm8                      \n\t"

                         "add $0x4, %0                                  \n\t"
                         "add $0x100, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 4b                                          \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "cmpq $0x0, %6 \n\t"
                         "je 6f      \n\t"
                         "vbroadcastss (%6), %%zmm5                        \n\t"
                         "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
                         "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
                         "vcvtdq2ps %%zmm2, %%zmm2                       \n\t"
                         "vcvtdq2ps %%zmm3, %%zmm3                       \n\t"
                         "vmulps %%zmm0, %%zmm5, %%zmm0                       \n\t"
                         "vmulps %%zmm1, %%zmm5, %%zmm1                       \n\t"
                         "vmulps %%zmm2, %%zmm5, %%zmm2                       \n\t"
                         "vmulps %%zmm3, %%zmm5, %%zmm3                       \n\t"

                         "mov %%ebx, %%eax          \n\t"
                         "and $0x2, %%eax          \n\t"
                         "je 6f                                         \n\t"
                         "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
                         "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
                         "vcvtps2dq %%zmm2, %%zmm2                       \n\t"
                         "vcvtps2dq %%zmm3, %%zmm3                       \n\t"
                         "mov $128, %%eax \n\t"
                         "vmovd %%eax, %%xmm5                    \n\t"
                         "vbroadcastss %%xmm5, %%zmm4            \n\t"
                         "vpaddd %%zmm0, %%zmm4, %%zmm0                       \n\t"
                         "vpaddd %%zmm1, %%zmm4, %%zmm1                       \n\t"
                         "vpaddd %%zmm2, %%zmm4, %%zmm2                       \n\t"
                         "vpaddd %%zmm3, %%zmm4, %%zmm3                       \n\t"
                         "vpmovusdb %%zmm0,  (%7)                             \n\t"
                         "vpmovusdb %%zmm1,  0x10(%7)                         \n\t"
                         "vpmovusdb %%zmm2,  0x20(%7)                         \n\t"
                         "vpmovusdb %%zmm3,  0x30(%7)                         \n\t"
                         "jmp 7f                                         \n\t"

                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         "vmovups %%zmm0, (%2)                           \n\t"
                         "vmovups %%zmm1, 0x40(%2)                       \n\t"
                         "vmovups %%zmm2, 0x80(%2)                       \n\t"
                         "vmovups %%zmm3, 0xC0(%2)                       \n\t"
                         ".align 16                                         \n\t"
                         "7:                                      \n\t"
                         :
                         : "r"(vector), "r"(matrix), "r"(result), "r"(bk), "r"(offsetC), "b"(flags),
                         "r"(scale), "r"(u8Result)
                         : "%eax", "%ecx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                         "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13",
                         "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20",
                         "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27",
                         "%zmm31", "memory");
}

void mvm_row_avx512_32(U32 bn,
    U32 bk,
    INT8 *matrix,
    UINT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("mov %%ebx, %%eax          \n\t"
                         "and $0x1, %%eax          \n\t"
                         "jne 0f                                         \n\t"
                         "vmovups (%4), %%zmm0                       \n\t"
                         "vmovups 0x40(%4), %%zmm1                   \n\t"
                         "jmp 1f          \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
#ifndef _USE_AVX512_VNNI
                         "mov $1, %%eax \n\t"
                         "vmovd %%eax, %%xmm24                    \n\t"
                         "vpbroadcastw %%xmm24, %%zmm31            \n\t"
#endif
                         "vpbroadcastd (%0), %%zmm2                     \n\t"
                         "vmovups (%1), %%zmm3                             \n\t"
                         "vmovups 0x40(%1), %%zmm4                             \n\t"
                         "vpaddd (%2), %%zmm0, %%zmm0                    \n\t"
                         "vpaddd 0x40(%2), %%zmm1, %%zmm1                \n\t"
                         "add $0x80, %1                                    \n\t"
                         "mov %3, %%ecx                                    \n\t"
                         "shr $3, %%ecx                                    \n\t"
                         "jz 3f                                      \n\t"

                         ".align 16                                         \n\t"
                         "2:                                      \n\t"
                         "vpbroadcastd 0x4(%0), %%zmm5                     \n\t"
                         "vmovups (%1), %%zmm6                             \n\t"
                         "vmovups 0x40(%1), %%zmm7                         \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm3, %%zmm2, %%zmm0              \n\t"
                         "vpdpbusd %%zmm4, %%zmm2, %%zmm1              \n\t"
#else
                         "vpmaddubsw %%zmm3, %%zmm2, %%zmm24              \n\t"
                         "vpmaddubsw %%zmm4, %%zmm2, %%zmm25              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpmaddwd %%zmm25, %%zmm31, %%zmm25              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm25, %%zmm1              \n\t"
#endif

                         "vpbroadcastd 0x8(%0), %%zmm2                     \n\t"
                         "vmovups 0x80(%1), %%zmm3                         \n\t"
                         "vmovups 0xC0(%1), %%zmm4                        \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm6, %%zmm5, %%zmm0                 \n\t"
                         "vpdpbusd %%zmm7, %%zmm5, %%zmm1                 \n\t"
#else
                         "vpmaddubsw %%zmm6, %%zmm5, %%zmm24              \n\t"
                         "vpmaddubsw %%zmm7, %%zmm5, %%zmm25              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpmaddwd %%zmm25, %%zmm31, %%zmm25              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm25, %%zmm1              \n\t"
#endif

                         "add $0x8, %0                                  \n\t"
                         "add $0x100, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 2b                                          \n\t"

                         ".align 16                                         \n\t"
                         "3:                                           \n\t"
                         "mov %3, %%ecx                                    \n\t"
                         "and $0x7, %%ecx                                    \n\t"
                         "shr $2, %%ecx                                    \n\t"
                         "jz 5f                                          \n\t"
                         ".align 16                                         \n\t"
                         "4:                                           \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm3, %%zmm2, %%zmm0              \n\t"
                         "vpdpbusd %%zmm4, %%zmm2, %%zmm1              \n\t"
#else
                         "vpmaddubsw %%zmm3, %%zmm2, %%zmm24              \n\t"
                         "vpmaddubsw %%zmm4, %%zmm2, %%zmm25              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpmaddwd %%zmm25, %%zmm31, %%zmm25              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm25, %%zmm1              \n\t"
#endif

                         "vpbroadcastd 0x4(%0), %%zmm2                  \n\t"
                         "vmovups (%1), %%zmm3                          \n\t"
                         "vmovups 0x40(%1), %%zmm4                      \n\t"

                         "add $0x4, %0                                  \n\t"
                         "add $0x80, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 4b                                          \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "cmpq $0x0, %6 \n\t"
                         "je 6f      \n\t"
                         "vbroadcastss (%6), %%zmm3                        \n\t"
                         "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
                         "vcvtdq2ps %%zmm1, %%zmm1                       \n\t"
                         "vmulps %%zmm0, %%zmm3, %%zmm0                       \n\t"
                         "vmulps %%zmm1, %%zmm3, %%zmm1                       \n\t"

                         "mov %%ebx, %%eax          \n\t"
                         "and $0x2, %%eax          \n\t"
                         "je 6f                                         \n\t"
                         "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
                         "vcvtps2dq %%zmm1, %%zmm1                       \n\t"
                         "mov $128, %%eax \n\t"
                         "vmovd %%eax, %%xmm5                    \n\t"
                         "vbroadcastss %%xmm5, %%zmm4            \n\t"
                         "vpaddd %%zmm0, %%zmm4, %%zmm0                       \n\t"
                         "vpaddd %%zmm1, %%zmm4, %%zmm1                       \n\t"
                         "vpmovusdb %%zmm0,  (%7)                             \n\t"
                         "vpmovusdb %%zmm1,  0x10(%7)                         \n\t"
                         "jmp 7f                                         \n\t"

                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         "vmovups %%zmm0, (%2)                           \n\t"
                         "vmovups %%zmm1, 0x40(%2)                       \n\t"
                         ".align 16                                         \n\t"
                         "7:                                      \n\t"
                         :
                         : "r"(vector), "r"(matrix), "r"(result), "r"(bk), "r"(offsetC), "b"(flags),
                         "r"(scale), "r"(u8Result)
                         : "%eax", "%ecx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                         "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13",
                         "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20",
                         "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm31", "memory");
}

void mvm_row_avx512_16(U32 bn,
    U32 bk,
    INT8 *matrix,
    UINT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("mov %%ebx, %%eax          \n\t"
                         "and $0x1, %%eax          \n\t"
                         "jne 0f                                         \n\t"
                         "vmovups (%4), %%zmm0                       \n\t"
                         "jmp 1f          \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
#ifndef _USE_AVX512_VNNI
                         "mov $1, %%eax \n\t"
                         "vmovd %%eax, %%xmm24                    \n\t"
                         "vpbroadcastw %%xmm24, %%zmm31            \n\t"
#endif
                         "vpaddd (%2), %%zmm0, %%zmm0                    \n\t"
                         "shr $2, %%ecx                                    \n\t"
                         "jz 3f                                      \n\t"

                         ".align 16                                         \n\t"
                         "2:                                      \n\t"
                         "vpbroadcastd (%0), %%zmm1                     \n\t"
                         "vmovups (%1), %%zmm2                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm2, %%zmm1, %%zmm0              \n\t"
#else
                         "vpmaddubsw %%zmm2, %%zmm1, %%zmm24              \n\t"
                         "vpmaddwd %%zmm24, %%zmm31, %%zmm24              \n\t"
                         "vpaddd %%zmm0, %%zmm24, %%zmm0              \n\t"
#endif
                         "add $0x4, %0                                  \n\t"
                         "add $0x40, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 2b                                          \n\t"

                         ".align 16                                         \n\t"
                         "3:                                      \n\t"
                         "cmpq $0x0, %6 \n\t"
                         "je 4f      \n\t"
                         "vbroadcastss (%6), %%zmm3                        \n\t"
                         "vcvtdq2ps %%zmm0, %%zmm0                       \n\t"
                         "vmulps %%zmm0, %%zmm3, %%zmm0                       \n\t"

                         "mov %%ebx, %%eax          \n\t"
                         "and $0x2, %%eax          \n\t"
                         "je 4f                                         \n\t"
                         "vcvtps2dq %%zmm0, %%zmm0                       \n\t"
                         "mov $128, %%eax \n\t"
                         "vmovd %%eax, %%xmm5                    \n\t"
                         "vbroadcastss %%xmm5, %%zmm4            \n\t"
                         "vpaddd %%zmm0, %%zmm4, %%zmm0                       \n\t"
                         "vpmovusdb %%zmm0,  (%7)                             \n\t"
                         "jmp 5f                                         \n\t"

                         ".align 16                                         \n\t"
                         "4:                                      \n\t"
                         "vmovups %%zmm0, (%2)                           \n\t"
                         ".align 16                                         \n\t"
                         "5:                                      \n\t"
                         :
                         : "r"(vector), "r"(matrix), "r"(result), "c"(bk), "r"(offsetC), "b"(flags),
                         "r"(scale), "r"(u8Result)
                         : "%eax", "%zmm0", "%zmm1", "%zmm2", "%zmm24", "%zmm31", "memory");
}

void mvm_row_avx512_8(U32 bn,
    U32 bk,
    INT8 *matrix,
    UINT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("mov %%ebx, %%eax          \n\t"
                         "and $0x1, %%eax          \n\t"
                         "jne 0f                                         \n\t"
                         "vmovups (%4), %%ymm0                       \n\t"
                         "jmp 1f          \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vxorps %%ymm0, %%ymm0, %%ymm0               \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
#ifndef _USE_AVX512_VNNI
                         "mov $1, %%eax \n\t"
                         "vmovd %%eax, %%xmm24                    \n\t"
                         "vpbroadcastw %%xmm24, %%ymm31            \n\t"
#endif
                         "vpaddd (%2), %%ymm0, %%ymm0                 \n\t"
                         "shr $2, %%ecx                                    \n\t"
                         "jz 3f                                      \n\t"

                         ".align 16                                   \n\t"
                         "2:                                          \n\t"
                         "vpbroadcastd (%0), %%ymm1                   \n\t"
                         "vmovups (%1), %%ymm2                        \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%ymm2, %%ymm1, %%ymm0             \n\t"
#else
                         "vpmaddubsw %%ymm2, %%ymm1, %%ymm24              \n\t"
                         "vpmaddwd %%ymm24, %%ymm31, %%ymm24              \n\t"
                         "vpaddd %%ymm0, %%ymm24, %%ymm0              \n\t"
#endif
                         "add $0x4, %0                                \n\t"
                         "add $0x20, %1                               \n\t"
                         "dec %%ecx                                   \n\t"
                         "jg 2b                                       \n\t"

                         ".align 16                                         \n\t"
                         "3:                                      \n\t"
                         "cmpq $0x0, %6 \n\t"
                         "je 4f      \n\t"
                         "vbroadcastss (%6), %%ymm3                        \n\t"
                         "vcvtdq2ps %%ymm0, %%ymm0                       \n\t"
                         "vmulps %%ymm0, %%ymm3, %%ymm0                       \n\t"

                         "mov %%ebx, %%eax          \n\t"
                         "and $0x2, %%eax          \n\t"
                         "je 4f                                         \n\t"
                         "vcvtps2dq %%ymm0, %%ymm0                       \n\t"
                         "mov $128, %%eax \n\t"
                         "vmovd %%eax, %%xmm5                    \n\t"
                         "vbroadcastss %%xmm5, %%ymm4            \n\t"
                         "vpaddd %%ymm0, %%ymm4, %%ymm0                       \n\t"
                         "vpmovusdb %%ymm0,  (%7)                             \n\t"
                         "jmp 5f                                         \n\t"

                         ".align 16                                         \n\t"
                         "4:                                      \n\t"
                         "vmovups %%ymm0, (%2)                        \n\t"
                         ".align 16                                         \n\t"
                         "5:                                      \n\t"
                         :
                         : "r"(vector), "r"(matrix), "r"(result), "c"(bk), "r"(offsetC), "b"(flags),
                         "r"(scale), "r"(u8Result)
                         : "%eax", "%ymm0", "%ymm1", "%ymm2", "%ymm24", "%ymm31", "memory");
}

void mvm_row_avx512_tail(U32 bn,
    U32 bk,
    INT8 *matrix,
    UINT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    F32 *resultF32 = (F32 *)result;
    for (U32 n = 0; n < bn; ++n) {
        I32 tmp = 0;
        if ((flags & 0x1) == 0) {
            tmp += offsetC[n];
        } else {
            tmp = ((I32 *)result)[n];
        }
        for (U32 k = 0; k < bk; k += 4) {
            for (U32 k4 = 0; k4 < 4; ++k4) {
                tmp += (int)matrix[k * bn + n * 4 + k4] * (int)vector[k + k4];
            }
        }
        if (scale != nullptr) {
            resultF32[n] = tmp * scale[0];
            if (flags & 0x2) {
                tmp = (I32)(resultF32[n] + 128);
                u8Result[n] = (tmp > 255) ? 255 : tmp;
            }
        } else {
            result[n] = tmp;
        }
    }
}

EE mvm_avx512_int8(U32 numRows,
    U32 numColumns,
    INT8 *packB,
    UINT8 *vector,
    UINT8 *result,
    I32 *offsetCBias,
    const F32 *scale)
{
    // Actual layout is NKN64, and vector is K
    kernel_func kernel[6] = {mvm_row_avx512_tail, mvm_row_avx512_8, mvm_row_avx512_16,
        mvm_row_avx512_32, mvm_row_avx512_32, mvm_row_avx512_64};
    U32 unrollSize[5] = {8, 16, 32, 32, 64};
    U32 blockSizeK = 0, blockSizeN = 0;
    U32 flags = 0;
    F32 *factorPtr = nullptr;
    F32 factor;
    I32 *i32Result = (I32 *)result;
    UINT8 *u8Result = result;
    if (uintptr_t(packB) == uintptr_t(offsetCBias)) {
        packB += numRows * bytesOf(DT_I32);
    }
    if (scale != nullptr) {
        // when use offline scale, the output datatype is U8_Q, you need more tmp buffer
        if (scale[0] < 0) {
            flags |= 1 << 1;
            factor = scale[1];
            i32Result = offsetCBias + numRows;
            UNI_MEMSET(i32Result, 0, numRows * bytesOf(DT_I32));
        } else {
            factor = 1 / (*scale);
        }
        factorPtr = &factor;
    }
    for (U32 k = 0; k < numColumns; k += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, numColumns - k);
        U32 alignedBlockSizeK = UNI_ALIGN(blockSizeK, 4);
        flags |= (k > 0);
        F32 *useFactor = nullptr;
        if (k == numColumns - blockSizeK) {
            useFactor = factorPtr;
        }
        for (U32 j = 0; j < numRows; j += blockSizeN) {
            blockSizeN = UNI_MIN(UNROLL_N, numRows - j);
            int idx = blockSizeN >> 4;
            if (blockSizeN < unrollSize[idx]) {
                idx = -1;
            } else {
                blockSizeN = unrollSize[idx];
            }
            INT8 *curM = packB + k * numRows + alignedBlockSizeK * j;
            kernel[idx + 1](blockSizeN, alignedBlockSizeK, curM, vector + k, i32Result + j,
                u8Result + j, offsetCBias + j, useFactor, flags);
        }
    }

    return SUCCESS;
}
