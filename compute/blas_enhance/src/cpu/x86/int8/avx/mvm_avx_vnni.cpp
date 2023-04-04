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
#define UNROLL_N 32
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
    U32 unrollSize[5] = {8, 8, 16, 24, 32};
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
                    unrollSizeN = unrollSize[unrollSizeN >> 3];
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
                    unrollSizeN = unrollSize[unrollSizeN >> 3];
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
                         "vmovups (%[offsetC]), %%ymm0                       \n\t"
                         "vmovups 0x20(%[offsetC]), %%ymm1                   \n\t"
                         "vmovups 0x40(%[offsetC]), %%ymm2                   \n\t"
                         "vmovups 0x60(%[offsetC]), %%ymm3                   \n\t"
                         "vpaddd (%[result]), %%ymm0, %%ymm0                    \n\t"
                         "vpaddd 0x20(%[result]), %%ymm1, %%ymm1                \n\t"
                         "vpaddd 0x40(%[result]), %%ymm2, %%ymm2                \n\t"
                         "vpaddd 0x60(%[result]), %%ymm3, %%ymm3                \n\t"
                         "jmp 1f          \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%[result]), %%ymm0                     \n\t"
                         "vmovups 0x20(%[result]), %%ymm1                     \n\t"
                         "vmovups 0x40(%[result]), %%ymm2                     \n\t"
                         "vmovups 0x60(%[result]), %%ymm3                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %[bk], %%ecx                                    \n\t"
                         "shr $3, %%ecx                                    \n\t"
                         "jz 3f                                      \n\t"

                         ".align 16                                         \n\t"
                         "2:                                      \n\t"
                         "vpbroadcastd (%[vector]), %%ymm4                     \n\t"
                         "vmovups (%[matrix]), %%ymm5                             \n\t"
                         "vmovups 0x20(%[matrix]), %%ymm6                             \n\t"
                         "vmovups 0x40(%[matrix]), %%ymm7                             \n\t"
                         "vmovups 0x60(%[matrix]), %%ymm8                             \n\t"

                         "%{vex%} vpdpbusd %%ymm5, %%ymm4, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm6, %%ymm4, %%ymm1              \n\t"
                         "%{vex%} vpdpbusd %%ymm7, %%ymm4, %%ymm2              \n\t"
                         "%{vex%} vpdpbusd %%ymm8, %%ymm4, %%ymm3              \n\t"

                         "vpbroadcastd 0x4(%[vector]), %%ymm9                     \n\t"
                         "vmovups 0x80(%[matrix]), %%ymm10                             \n\t"
                         "vmovups 0xA0(%[matrix]), %%ymm11                             \n\t"
                         "vmovups 0xC0(%[matrix]), %%ymm12                             \n\t"
                         "vmovups 0xE0(%[matrix]), %%ymm13                             \n\t"

                         "%{vex%} vpdpbusd %%ymm10, %%ymm9, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm11, %%ymm9, %%ymm1              \n\t"
                         "%{vex%} vpdpbusd %%ymm12, %%ymm9, %%ymm2              \n\t"
                         "%{vex%} vpdpbusd %%ymm13, %%ymm9, %%ymm3              \n\t"

                         "add $0x8, %[vector]                                  \n\t"
                         "add $0x100, %[matrix]                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 2b                                          \n\t"

                         ".align 16                                         \n\t"
                         "3:                                           \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 4f      \n\t"
                         "vbroadcastss (%[scale]), %%ymm5                        \n\t"
                         "vcvtdq2ps %%ymm0, %%ymm0                       \n\t"
                         "vcvtdq2ps %%ymm1, %%ymm1                       \n\t"
                         "vcvtdq2ps %%ymm2, %%ymm2                       \n\t"
                         "vcvtdq2ps %%ymm3, %%ymm3                       \n\t"
                         "vmulps %%ymm0, %%ymm5, %%ymm0                       \n\t"
                         "vmulps %%ymm1, %%ymm5, %%ymm1                       \n\t"
                         "vmulps %%ymm2, %%ymm5, %%ymm2                       \n\t"
                         "vmulps %%ymm3, %%ymm5, %%ymm3                       \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "vmovups %%ymm0, (%[result])                           \n\t"
                         "vmovups %%ymm1, 0x20(%[result])                       \n\t"
                         "vmovups %%ymm2, 0x40(%[result])                       \n\t"
                         "vmovups %%ymm3, 0x60(%[result])                       \n\t"
                         ".align 16                                         \n\t"
                         "5:                                      \n\t"
                         :
                         : [vector] "r" (vector),
                           [matrix] "r" (matrix),
                           [result] "r" (result),
                           [bk] "r" (bk),
                           [offsetC] "r" (offsetC),
                           "b" (flags),
                           [scale] "r" (scale),
                           [u8Result] "r" (u8Result)
                         : "%eax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory");
}

void mvm_row_avx512_24(U32 bn,
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
                         "vmovups (%[offsetC]), %%ymm0                       \n\t"
                         "vmovups 0x20(%[offsetC]), %%ymm1                   \n\t"
                         "vmovups 0x40(%[offsetC]), %%ymm2                   \n\t"
                         "vpaddd (%[result]), %%ymm0, %%ymm0                    \n\t"
                         "vpaddd 0x20(%[result]), %%ymm1, %%ymm1                \n\t"
                         "vpaddd 0x40(%[result]), %%ymm2, %%ymm2                \n\t"
                         "jmp 1f          \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%[result]), %%ymm0                     \n\t"
                         "vmovups 0x20(%[result]), %%ymm1                     \n\t"
                         "vmovups 0x40(%[result]), %%ymm2                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %[bk], %%ecx                                    \n\t"
                         "shr $3, %%ecx                                    \n\t"
                         "jz 3f                                      \n\t"

                         ".align 16                                         \n\t"
                         "2:                                      \n\t"
                         "vpbroadcastd (%[vector]), %%ymm4                     \n\t"
                         "vmovups (%[matrix]), %%ymm5                             \n\t"
                         "vmovups 0x20(%[matrix]), %%ymm6                             \n\t"
                         "vmovups 0x40(%[matrix]), %%ymm7                             \n\t"

                         "%{vex%} vpdpbusd %%ymm5, %%ymm4, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm6, %%ymm4, %%ymm1              \n\t"
                         "%{vex%} vpdpbusd %%ymm7, %%ymm4, %%ymm2              \n\t"

                         "vpbroadcastd 0x4(%[vector]), %%ymm9                     \n\t"
                         "vmovups 0x60(%[matrix]), %%ymm10                             \n\t"
                         "vmovups 0x80(%[matrix]), %%ymm11                             \n\t"
                         "vmovups 0xA0(%[matrix]), %%ymm12                             \n\t"

                         "%{vex%} vpdpbusd %%ymm10, %%ymm9, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm11, %%ymm9, %%ymm1              \n\t"
                         "%{vex%} vpdpbusd %%ymm12, %%ymm9, %%ymm2              \n\t"

                         "add $0x8, %[vector]                                  \n\t"
                         "add $0xC0, %[matrix]                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 2b                                          \n\t"

                         ".align 16                                         \n\t"
                         "3:                                           \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 4f      \n\t"
                         "vbroadcastss (%[scale]), %%ymm5                        \n\t"
                         "vcvtdq2ps %%ymm0, %%ymm0                       \n\t"
                         "vcvtdq2ps %%ymm1, %%ymm1                       \n\t"
                         "vcvtdq2ps %%ymm2, %%ymm2                       \n\t"
                         "vmulps %%ymm0, %%ymm5, %%ymm0                       \n\t"
                         "vmulps %%ymm1, %%ymm5, %%ymm1                       \n\t"
                         "vmulps %%ymm2, %%ymm5, %%ymm2                       \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "vmovups %%ymm0, (%[result])                           \n\t"
                         "vmovups %%ymm1, 0x20(%[result])                       \n\t"
                         "vmovups %%ymm2, 0x40(%[result])                       \n\t"
                         ".align 16                                         \n\t"
                         "5:                                      \n\t"
                         :
                         : [vector] "r" (vector),
                           [matrix] "r" (matrix),
                           [result] "r" (result),
                           [bk] "r" (bk),
                           [offsetC] "r" (offsetC),
                           "b" (flags),
                           [scale] "r" (scale),
                           [u8Result] "r" (u8Result)
                         : "%eax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory");
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
                         "vmovups (%[offsetC]), %%ymm0                       \n\t"
                         "vmovups 0x20(%[offsetC]), %%ymm1                   \n\t"
                         "vpaddd (%[result]), %%ymm0, %%ymm0                    \n\t"
                         "vpaddd 0x20(%[result]), %%ymm1, %%ymm1                \n\t"
                         "jmp 1f          \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%[result]), %%ymm0                     \n\t"
                         "vmovups 0x20(%[result]), %%ymm1                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %[bk], %%ecx                                    \n\t"
                         "shr $3, %%ecx                                    \n\t"
                         "jz 3f                                      \n\t"

                         ".align 16                                         \n\t"
                         "2:                                      \n\t"
                         "vpbroadcastd (%[vector]), %%ymm4                     \n\t"
                         "vmovups (%[matrix]), %%ymm5                             \n\t"
                         "vmovups 0x20(%[matrix]), %%ymm6                             \n\t"

                         "%{vex%} vpdpbusd %%ymm5, %%ymm4, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm6, %%ymm4, %%ymm1              \n\t"

                         "vpbroadcastd 0x4(%[vector]), %%ymm9                     \n\t"
                         "vmovups 0x40(%[matrix]), %%ymm10                             \n\t"
                         "vmovups 0x60(%[matrix]), %%ymm11                             \n\t"

                         "%{vex%} vpdpbusd %%ymm10, %%ymm9, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm11, %%ymm9, %%ymm1              \n\t"

                         "add $0x8, %[vector]                                  \n\t"
                         "add $0x80, %[matrix]                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 2b                                          \n\t"

                         ".align 16                                         \n\t"
                         "3:                                           \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 4f      \n\t"
                         "vbroadcastss (%[scale]), %%ymm5                        \n\t"
                         "vcvtdq2ps %%ymm0, %%ymm0                       \n\t"
                         "vcvtdq2ps %%ymm1, %%ymm1                       \n\t"
                         "vmulps %%ymm0, %%ymm5, %%ymm0                       \n\t"
                         "vmulps %%ymm1, %%ymm5, %%ymm1                       \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "vmovups %%ymm0, (%[result])                           \n\t"
                         "vmovups %%ymm1, 0x20(%[result])                       \n\t"
                         ".align 16                                         \n\t"
                         "5:                                      \n\t"
                         :
                         : [vector] "r" (vector),
                           [matrix] "r" (matrix),
                           [result] "r" (result),
                           [bk] "r" (bk),
                           [offsetC] "r" (offsetC),
                           "b" (flags),
                           [scale] "r" (scale),
                           [u8Result] "r" (u8Result)
                         : "%eax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory");
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
                         "vmovups (%[offsetC]), %%ymm0                       \n\t"
                         "vpaddd (%[result]), %%ymm0, %%ymm0                    \n\t"
                         "jmp 1f          \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%[result]), %%ymm0                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %[bk], %%ecx                                    \n\t"
                         "shr $3, %%ecx                                    \n\t"
                         "jz 3f                                      \n\t"

                         ".align 16                                         \n\t"
                         "2:                                      \n\t"
                         "vpbroadcastd (%[vector]), %%ymm4                     \n\t"
                         "vmovups (%[matrix]), %%ymm5                             \n\t"
                         "%{vex%} vpdpbusd %%ymm5, %%ymm4, %%ymm0              \n\t"
                         "vpbroadcastd 0x4(%[vector]), %%ymm9                     \n\t"
                         "vmovups 0x20(%[matrix]), %%ymm10                             \n\t"
                         "%{vex%} vpdpbusd %%ymm10, %%ymm9, %%ymm0              \n\t"

                         "add $0x8, %[vector]                                  \n\t"
                         "add $0x40, %[matrix]                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 2b                                          \n\t"

                         ".align 16                                         \n\t"
                         "3:                                           \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 4f      \n\t"
                         "vbroadcastss (%[scale]), %%ymm5                        \n\t"
                         "vcvtdq2ps %%ymm0, %%ymm0                       \n\t"
                         "vmulps %%ymm0, %%ymm5, %%ymm0                       \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "vmovups %%ymm0, (%[result])                           \n\t"
                         ".align 16                                         \n\t"
                         "5:                                      \n\t"
                         :
                         : [vector] "r" (vector),
                           [matrix] "r" (matrix),
                           [result] "r" (result),
                           [bk] "r" (bk),
                           [offsetC] "r" (offsetC),
                           "b" (flags),
                           [scale] "r" (scale),
                           [u8Result] "r" (u8Result)
                         : "%eax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory");
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
    kernel_func kernel[5] = {mvm_row_avx512_tail, mvm_row_avx512_8, mvm_row_avx512_16,
        mvm_row_avx512_24, mvm_row_avx512_32};
    U32 unrollSize[5] = {8, 8, 16, 24, 32};
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
            int idx = blockSizeN >> 3;
            INT8 *curM = packB + k * numRows + alignedBlockSizeK * j;
            kernel[idx](blockSizeN, alignedBlockSizeK, curM, vector + k, i32Result + j,
                u8Result + j, offsetCBias + j, useFactor, flags);
        }
    }

    return SUCCESS;
}
