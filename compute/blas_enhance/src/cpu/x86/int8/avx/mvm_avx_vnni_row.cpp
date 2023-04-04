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

#define UNROLL_N 8
#define BOLCK_K_DIM 2048

typedef void (*kernel_func)(U32 K,
    U32 bk,
    UINT8 *matrix,
    INT8 *vector,
    UINT8 *matrixEdge,
    INT8 *vectorEdge,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags);

// I8 U8
void mvm_avx512_8_row(U32 K,
    U32 bk,
    UINT8 *matrix,
    INT8 *vector,
    UINT8 *matrixEdge,
    INT8 *vectorEdge,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                         "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
                         "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
                         "vxorps %%ymm5, %%ymm5, %%ymm5                     \n\t"
                         "vxorps %%ymm6, %%ymm6, %%ymm6                     \n\t"
                         "vxorps %%ymm7, %%ymm7, %%ymm7                     \n\t"
                         "vxorps %%ymm8, %%ymm8, %%ymm8                     \n\t"

                         "cmp $0x0, %%ecx                         \n\t"
                         "je 2f                                    \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %[matrix], %%rax \n\t"
                         "add %[k], %%rax \n\t"
                         "add %[k], %%rax \n\t"
                         "vmovups (%[vector]), %%ymm9                     \n\t"
                         "vmovups (%[matrix]), %%ymm10                             \n\t"
                         "vmovups (%[matrix], %[k]), %%ymm11                             \n\t"
                         "vmovups (%%rax), %%ymm12                             \n\t"
                         "vmovups (%%rax, %[k]), %%ymm13                             \n\t"

                         "%{vex%} vpdpbusd %%ymm9, %%ymm10, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm11, %%ymm1              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm12, %%ymm2              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm13, %%ymm3              \n\t"
                         "add %[k], %%rax \n\t"
                         "add %[k], %%rax \n\t"
                         "mov %%rax, %%rbx \n\t"
                         "add %[k], %%rbx \n\t"
                         "add %[k], %%rbx \n\t"

                         "vmovups (%%rax), %%ymm10                             \n\t"
                         "vmovups (%%rax, %[k]), %%ymm11                             \n\t"
                         "vmovups (%%rbx), %%ymm12                             \n\t"
                         "vmovups (%%rbx, %[k]), %%ymm13                             \n\t"

                         "%{vex%} vpdpbusd %%ymm9, %%ymm10, %%ymm4              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm11, %%ymm5              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm12, %%ymm6              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm13, %%ymm7              \n\t"

                         "add $0x20, %[vector]                                  \n\t"
                         "add $0x20, %[matrix]                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 1b                                          \n\t"

                         ".align 16                                      \n\t"
                         "2:                                             \n\t"

                         "cmp $0x0, %[vectorEdge]                       \n\t"
                         "je 3f                                  \n\t"

                         "vmovups (%[vectorEdge]), %%ymm9                     \n\t"
                         "vmovups (%[matrixEdge]), %%ymm10                             \n\t"
                         "vmovups 0x20(%[matrixEdge]), %%ymm11                             \n\t"
                         "vmovups 0x40(%[matrixEdge]), %%ymm12                             \n\t"
                         "vmovups 0x60(%[matrixEdge]), %%ymm13                             \n\t"

                         "%{vex%} vpdpbusd %%ymm9, %%ymm10, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm11, %%ymm1              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm12, %%ymm2              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm13, %%ymm3              \n\t"

                         "vmovups 0x80(%[matrixEdge]), %%ymm10                             \n\t"
                         "vmovups 0xA0(%[matrixEdge]), %%ymm11                             \n\t"
                         "vmovups 0xC0(%[matrixEdge]), %%ymm12                             \n\t"
                         "vmovups 0xE0(%[matrixEdge]), %%ymm13                             \n\t"

                         "%{vex%} vpdpbusd %%ymm9, %%ymm10, %%ymm4              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm11, %%ymm5              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm12, %%ymm6              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm13, %%ymm7              \n\t"

                         ".align 16                                      \n\t"
                         "3:                                             \n\t"
                         "vpunpckhdq %%ymm2, %%ymm0, %%ymm9              \n\t"
                         "vpunpckhdq %%ymm3, %%ymm1, %%ymm10              \n\t"
                         "vpunpckldq %%ymm2, %%ymm0, %%ymm11              \n\t"
                         "vpunpckldq %%ymm3, %%ymm1, %%ymm12              \n\t"
                         "vpaddd %%ymm9, %%ymm11, %%ymm13              \n\t"
                         "vpaddd %%ymm10, %%ymm12, %%ymm14              \n\t"
                         "vpunpckhdq %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vpunpckldq %%ymm14, %%ymm13, %%ymm10              \n\t"
                         "vpaddd %%ymm9, %%ymm10, %%ymm11              \n\t"
                         "vextractf128 $0x1, %%ymm11, %%xmm12              \n\t"
                         "vpaddd %%xmm12, %%xmm11, %%xmm0              \n\t"

                         "vpunpckhdq %%ymm6, %%ymm4, %%ymm9              \n\t"
                         "vpunpckhdq %%ymm7, %%ymm5, %%ymm10              \n\t"
                         "vpunpckldq %%ymm6, %%ymm4, %%ymm11              \n\t"
                         "vpunpckldq %%ymm7, %%ymm5, %%ymm12              \n\t"
                         "vpaddd %%ymm9, %%ymm11, %%ymm13              \n\t"
                         "vpaddd %%ymm10, %%ymm12, %%ymm14              \n\t"
                         "vpunpckhdq %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vpunpckldq %%ymm14, %%ymm13, %%ymm10              \n\t"
                         "vpaddd %%ymm9, %%ymm10, %%ymm11              \n\t"
                         "vextractf128 $0x1, %%ymm11, %%xmm12              \n\t"
                         "vpaddd %%xmm12, %%xmm11, %%xmm1              \n\t"

                         "mov %[flags], %%ebx \n\t"
                         "and $0x1, %%ebx \n\t"
                         "cmp $0x0, %%ebx \n\t"
                         "jne 4f      \n\t"
                         "vbroadcastss (%[offsetC]), %%xmm4              \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0              \n\t"
                         "vpaddd %%xmm1, %%xmm4, %%xmm1              \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 5f      \n\t"
                         "vbroadcastss (%[scale]), %%xmm5                        \n\t"
                         "vcvtdq2ps %%xmm0, %%xmm0                       \n\t"
                         "vcvtdq2ps %%xmm1, %%xmm1                       \n\t"
                         "vmulps %%xmm0, %%xmm5, %%xmm0                       \n\t"
                         "vmulps %%xmm1, %%xmm5, %%xmm1                       \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "vmovups %%xmm0, (%[result])                           \n\t"
                         "vmovups %%xmm1, 0x10(%[result])                       \n\t"
                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         :
                         : [vector] "r" (vector),
                           [matrix] "r" (matrix),
                           [result] "r" (result),
                           "c" (bk),
                           [offsetC] "r" (offsetC),
                           [flags] "r" (flags),
                           [scale] "r" (scale),
                           [k] "r" ((I64)K),
                           [matrixEdge] "r" (matrixEdge),
                           [vectorEdge] "r" (vectorEdge)
                         : "%rax", "%rbx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                           "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
                           "%ymm13", "%ymm14", "%ymm15", "memory");
}

void mvm_avx512_4_row(U32 K,
    U32 bk,
    UINT8 *matrix,
    INT8 *vector,
    UINT8 *matrixEdge,
    INT8 *vectorEdge,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                         "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"

                         "cmp $0x0, %%ecx                         \n\t"
                         "je 2f                                    \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %[matrix], %%rax \n\t"
                         "add %[k], %%rax \n\t"
                         "add %[k], %%rax \n\t"
                         "vmovups (%[vector]), %%ymm9                     \n\t"
                         "vmovups (%[matrix]), %%ymm10                             \n\t"
                         "vmovups (%[matrix], %[k]), %%ymm11                             \n\t"
                         "vmovups (%%rax), %%ymm12                             \n\t"
                         "vmovups (%%rax, %[k]), %%ymm13                             \n\t"

                         "%{vex%} vpdpbusd %%ymm9, %%ymm10, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm11, %%ymm1              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm12, %%ymm2              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm13, %%ymm3              \n\t"

                         "add $0x20, %[vector]                                  \n\t"
                         "add $0x20, %[matrix]                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 1b                                          \n\t"

                         ".align 16                                      \n\t"
                         "2:                                             \n\t"

                         "cmp $0x0, %[vectorEdge]                       \n\t"
                         "je 3f                                  \n\t"

                         "vmovups (%[vectorEdge]), %%ymm9                     \n\t"
                         "vmovups (%[matrixEdge]), %%ymm10                             \n\t"
                         "vmovups 0x20(%[matrixEdge]), %%ymm11                             \n\t"
                         "vmovups 0x40(%[matrixEdge]), %%ymm12                             \n\t"
                         "vmovups 0x60(%[matrixEdge]), %%ymm13                             \n\t"

                         "%{vex%} vpdpbusd %%ymm9, %%ymm10, %%ymm0              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm11, %%ymm1              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm12, %%ymm2              \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm13, %%ymm3              \n\t"

                         ".align 16                                      \n\t"
                         "3:                                             \n\t"
                         "vpunpckhdq %%ymm2, %%ymm0, %%ymm9              \n\t"
                         "vpunpckhdq %%ymm3, %%ymm1, %%ymm10              \n\t"
                         "vpunpckldq %%ymm2, %%ymm0, %%ymm11              \n\t"
                         "vpunpckldq %%ymm3, %%ymm1, %%ymm12              \n\t"
                         "vpaddd %%ymm9, %%ymm11, %%ymm13              \n\t"
                         "vpaddd %%ymm10, %%ymm12, %%ymm14              \n\t"
                         "vpunpckhdq %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vpunpckldq %%ymm14, %%ymm13, %%ymm10              \n\t"
                         "vpaddd %%ymm9, %%ymm10, %%ymm11              \n\t"
                         "vextractf128 $0x1, %%ymm11, %%xmm12              \n\t"
                         "vpaddd %%xmm12, %%xmm11, %%xmm0              \n\t"

                         "mov %[flags], %%ebx \n\t"
                         "and $0x1, %%ebx \n\t"
                         "cmp $0x0, %%ebx \n\t"
                         "jne 4f      \n\t"
                         "vbroadcastss (%[offsetC]), %%xmm4              \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0              \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 5f      \n\t"
                         "vbroadcastss (%[scale]), %%xmm5                        \n\t"
                         "vcvtdq2ps %%xmm0, %%xmm0                       \n\t"
                         "vcvtdq2ps %%xmm1, %%xmm1                       \n\t"
                         "vmulps %%xmm0, %%xmm5, %%xmm0                       \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "vmovups %%xmm0, (%[result])                           \n\t"
                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         :
                         : [vector] "r" (vector),
                           [matrix] "r" (matrix),
                           [result] "r" (result),
                           "c" (bk),
                           [offsetC] "r" (offsetC),
                           [flags] "r" (flags),
                           [scale] "r" (scale),
                           [k] "r" ((I64)K),
                           [matrixEdge] "r" (matrixEdge),
                           [vectorEdge] "r" (vectorEdge)
                         : "%rax", "%rbx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                           "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
                           "%ymm13", "%ymm14", "%ymm15", "memory");
}

void mvm_avx512_1_row(U32 K,
    U32 bk,
    UINT8 *matrix,
    INT8 *vector,
    UINT8 *matrixEdge,
    INT8 *vectorEdge,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"

                         "cmp $0x0, %%ecx                         \n\t"
                         "je 2f                                    \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "vmovups (%[vector]), %%ymm9                     \n\t"
                         "vmovups (%[matrix]), %%ymm10                             \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm10, %%ymm0              \n\t"

                         "add $0x20, %[vector]                                  \n\t"
                         "add $0x20, %[matrix]                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 1b                                          \n\t"

                         ".align 16                                      \n\t"
                         "2:                                             \n\t"

                         "cmp $0x0, %[vectorEdge]                       \n\t"
                         "je 3f                                  \n\t"

                         "vmovups (%[vectorEdge]), %%ymm9                     \n\t"
                         "vmovups (%[matrixEdge]), %%ymm10                             \n\t"
                         "%{vex%} vpdpbusd %%ymm9, %%ymm10, %%ymm0              \n\t"

                         ".align 16                                      \n\t"
                         "3:                                             \n\t"
                         "vextractf128 $0x1, %%ymm0, %%xmm1       \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00001011, %%xmm0, %%xmm1    \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00000001, %%xmm0, %%xmm1    \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"

                         "mov %[flags], %%ebx \n\t"
                         "and $0x1, %%ebx \n\t"
                         "cmp $0x0, %%ebx \n\t"
                         "jne 4f      \n\t"
                         "vbroadcastss (%[offsetC]), %%xmm4              \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0              \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "je 5f      \n\t"
                         "vbroadcastss (%[scale]), %%xmm5                        \n\t"
                         "vcvtdq2ps %%xmm0, %%xmm0                       \n\t"
                         "vmulps %%xmm0, %%xmm5, %%xmm0                       \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "vmovss %%xmm0, (%[result])                           \n\t"
                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         :
                         : [vector] "r" (vector),
                           [matrix] "r" (matrix),
                           [result] "r" (result),
                           "c" (bk),
                           [offsetC] "r" (offsetC),
                           [flags] "r" (flags),
                           [scale] "r" (scale),
                           [k] "r" ((I64)K),
                           [matrixEdge] "r" (matrixEdge),
                           [vectorEdge] "r" (vectorEdge)
                         : "%rax", "%rbx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                           "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
                           "%ymm13", "%ymm14", "%ymm15", "memory");
}

inline void transpose(UINT8 *matrix, UINT8 *transMatrix, U32 N, U32 K)
{
    U32 blockSizeN = 0;
    for (U32 n = 0; n < N; n += blockSizeN) {
        blockSizeN = UNI_MIN(BOLCK_K_DIM, N - n);
        for (U32 k = 0; k < K; ++k) {
            for (U32 in = n; in < n + blockSizeN; ++in) {
                transMatrix[in * K + k] = matrix[k * N + in];
            }
        }
    }
}

EE mvm_avx512_int8_row_i8u8(U32 numRows,
    U32 numColumns,
    DataFormat df,
    UINT8 *packB,
    INT8 *vector,
    UINT8 *result,
    I32 *tmp,
    const F32 *scale)
{
    I32 offsetC = 0;
    U32 num8 = numColumns / 8;
    U64 resNum = numColumns % 8;
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0            \n\t"
                         "mov %[vector], %%rdx \n\t"
                         "mov %[num8], %%ebx \n\t"
                         "cmp $0x0, %%ebx                         \n\t"
                         "je 1f                                    \n\t"
                         ".align 16                                \n\t"
                         "0:                                       \n\t"
                         "vmovsd (%%rdx), %%xmm2              \n\t"
                         "vpmovsxbd %%xmm2, %%ymm1              \n\t"
                         "vpaddd %%ymm1, %%ymm0, %%ymm0            \n\t"
                         "add $0x8, %%rdx                         \n\t"
                         "dec %%ebx                                \n\t"
                         "jg 0b                                    \n\t"

                         "vextractf128 $0x1, %%ymm0, %%xmm1       \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00001011, %%xmm0, %%xmm1    \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00000001, %%xmm0, %%xmm1    \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vmovss %%xmm0, (%[offsetC])                      \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : [vector] "r" (vector),
                           [offsetC] "r" (&offsetC),
                           [num8] "r" (num8)
                         : "%ebx", "%rdx", "%ymm0", "%ymm1", "%ymm2", "memory", "cc");
    num8 *= 8;
    for (U32 i = 0; i < resNum; ++i) {
        offsetC += vector[num8 + i];
    }
    offsetC *= -128;
    if (df == DF_TRANSPOSE) {
        transpose(packB, (UINT8 *)tmp, numRows, numColumns);
        packB = (UINT8 *)tmp;
        tmp = (I32 *)((U8 *)tmp + numColumns * numRows);
    }

    INT8 *vectorEdge = nullptr;
    UINT8 *matrixEdge = nullptr;
    if ((numColumns % 32) != 0) {
        U32 idx = numColumns / 32 * 32;
        U32 res = numColumns % 32;
        vectorEdge = (INT8 *)tmp;
        UNI_MEMCPY(vectorEdge, vector + idx, res);
        UNI_MEMSET(vectorEdge + res, 0, 32 - res);
        tmp = (I32 *)((U8 *)tmp + 32);
        matrixEdge = (UINT8 *)tmp;
        for (U32 i = 0; i < numRows; ++i) {
            UNI_MEMCPY(matrixEdge + i * 32, packB + i * numColumns + idx, res);
            UNI_MEMSET(matrixEdge + i * 32 + res, 128, 32 - res);
        }
    }

    U32 blockSizeK = 0, blockSizeN = 0, flags = 0;
    kernel_func kernel[3] = {
        mvm_avx512_1_row, mvm_avx512_4_row, mvm_avx512_8_row};
    U32 unrollSize[3] = {1, 4, 8};
    F32 *factorPtr = nullptr;
    F32 factor;
    I32 *i32Result = (I32 *)result;
    UINT8 *u8Result = result;
    if (scale != nullptr) {
        if (scale[0] <
            0) {  // when use offline scale, the output datatype is U8_Q, you need more tmp buffer
            flags |= 1 << 1;
            factor = scale[1];
            i32Result = (I32 *)((UINT8 *)tmp + numRows * numColumns);
            UNI_MEMSET(i32Result, 0, numRows * bytesOf(DT_I32));
        } else {
            factor = 1 / (*scale);
        }
        factorPtr = &factor;
    }
    for (U32 k = 0; k < numColumns; k += blockSizeK) {
        blockSizeK = numColumns;
        flags |= (k > 0);
        U32 num32 = blockSizeK / 32;
        F32 *useFactor = nullptr;
        if (k == numColumns - blockSizeK) {
            useFactor = factorPtr;
        }
        for (U32 j = 0; j < numRows; j += blockSizeN) {
            blockSizeN = UNI_MIN(UNROLL_N, numRows - j);
            blockSizeN = unrollSize[blockSizeN >> 2];
            UINT8 *curM = packB + j * numColumns;
            kernel[blockSizeN >> 2](numColumns, num32, curM, vector + k, matrixEdge + j * 32, vectorEdge, i32Result + j,
                u8Result + j, &offsetC, useFactor, flags);
        }
    }
    return SUCCESS;
}
