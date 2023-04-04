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

#define UNROLL_N 16
#define BOLCK_K_DIM 2048

typedef void (*kernel_func)(U32 K,
    U32 bk,
    U64 kmask,
    UINT8 *matrix,
    INT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags);

// I8 U8
void mvm_avx512_16_row(U32 K,
    U32 bk,
    U64 kmask,
    UINT8 *matrix,
    INT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
                         "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
                         "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
                         "vxorps %%zmm4, %%zmm4, %%zmm4                     \n\t"
                         "vxorps %%zmm5, %%zmm5, %%zmm5                     \n\t"
                         "vxorps %%zmm6, %%zmm6, %%zmm6                     \n\t"
                         "vxorps %%zmm7, %%zmm7, %%zmm7                     \n\t"
                         "vxorps %%zmm8, %%zmm8, %%zmm8                     \n\t"
                         "vxorps %%zmm9, %%zmm9, %%zmm9                     \n\t"
                         "vxorps %%zmm10, %%zmm10, %%zmm10                     \n\t"
                         "vxorps %%zmm11, %%zmm11, %%zmm11                     \n\t"
                         "vxorps %%zmm12, %%zmm12, %%zmm12                     \n\t"
                         "vxorps %%zmm13, %%zmm13, %%zmm13                     \n\t"
                         "vxorps %%zmm14, %%zmm14, %%zmm14                     \n\t"
                         "vxorps %%zmm15, %%zmm15, %%zmm15                     \n\t"
#ifndef _USE_AVX512_VNNI
                         "mov $1, %%eax \n\t"
                         "vmovd %%eax, %%xmm26                    \n\t"
                         "vpbroadcastw %%xmm26, %%zmm31            \n\t"
#endif
                         "cmp $0x0, %%ecx                         \n\t"
                         "je 2f                                    \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %1, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "vmovups (%0), %%zmm16                     \n\t"
                         "vmovups (%1), %%zmm17                             \n\t"
                         "vmovups (%1, %7), %%zmm18                             \n\t"
                         "vmovups (%%rax), %%zmm19                             \n\t"
                         "vmovups (%%rax, %7), %%zmm20                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0              \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm1              \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm2              \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm18, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm19, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm20, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm29, %%zmm3              \n\t"
#endif

                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "mov %%rax, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "vmovups (%%rax), %%zmm22                             \n\t"
                         "vmovups (%%rax, %7), %%zmm23                             \n\t"
                         "vmovups (%%rbx), %%zmm24                             \n\t"
                         "vmovups (%%rbx, %7), %%zmm25                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm22, %%zmm4              \n\t"
                         "vpdpbusd %%zmm16, %%zmm23, %%zmm5              \n\t"
                         "vpdpbusd %%zmm16, %%zmm24, %%zmm6              \n\t"
                         "vpdpbusd %%zmm16, %%zmm25, %%zmm7              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm22, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm23, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm25, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm4, %%zmm26, %%zmm4              \n\t"
                         "vpaddd %%zmm5, %%zmm27, %%zmm5              \n\t"
                         "vpaddd %%zmm6, %%zmm28, %%zmm6              \n\t"
                         "vpaddd %%zmm7, %%zmm29, %%zmm7              \n\t"
#endif

                         "add %7, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "mov %%rbx, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "vmovups (%%rbx), %%zmm17                             \n\t"
                         "vmovups (%%rbx, %7), %%zmm18                             \n\t"
                         "vmovups (%%rax), %%zmm19                             \n\t"
                         "vmovups (%%rax, %7), %%zmm20                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm8              \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm9              \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm10              \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm11              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm18, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm19, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm20, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm8, %%zmm26, %%zmm8              \n\t"
                         "vpaddd %%zmm9, %%zmm27, %%zmm9              \n\t"
                         "vpaddd %%zmm10, %%zmm28, %%zmm10              \n\t"
                         "vpaddd %%zmm11, %%zmm29, %%zmm11              \n\t"
#endif

                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "mov %%rax, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "vmovups (%%rax), %%zmm22                             \n\t"
                         "vmovups (%%rax, %7), %%zmm23                             \n\t"
                         "vmovups (%%rbx), %%zmm24                             \n\t"
                         "vmovups (%%rbx, %7), %%zmm25                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm22, %%zmm12              \n\t"
                         "vpdpbusd %%zmm16, %%zmm23, %%zmm13              \n\t"
                         "vpdpbusd %%zmm16, %%zmm24, %%zmm14              \n\t"
                         "vpdpbusd %%zmm16, %%zmm25, %%zmm15              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm22, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm23, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm25, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm12, %%zmm26, %%zmm12              \n\t"
                         "vpaddd %%zmm13, %%zmm27, %%zmm13              \n\t"
                         "vpaddd %%zmm14, %%zmm28, %%zmm14              \n\t"
                         "vpaddd %%zmm15, %%zmm29, %%zmm15              \n\t"
#endif

                         "add $0x40, %0                                  \n\t"
                         "add $0x40, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 1b                                          \n\t"

                         ".align 16                                      \n\t"
                         "2:                                             \n\t"

                         "kmovq %8, %%k1                      \n\t"
                         "cmp $0x0, %8                       \n\t"
                         "je 3f                                  \n\t"

                         "mov %1, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "vxorps %%zmm16, %%zmm16, %%zmm16                     \n\t"
                         "vmovdqu8 (%0), %%zmm16 %{%%k1%}                     \n\t"
                         "vmovdqu8 (%1), %%zmm17 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%1, %7), %%zmm18 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rax), %%zmm19 %{%%k1%}                            \n\t"
                         "vmovdqu8 (%%rax, %7), %%zmm20 %{%%k1%}                            \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0              \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm1              \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm2              \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm18, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm19, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm20, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm29, %%zmm3              \n\t"
#endif

                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "mov %%rax, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "vmovdqu8 (%%rax), %%zmm22 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rax, %7), %%zmm23 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rbx), %%zmm24 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rbx, %7), %%zmm25 %{%%k1%}                            \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm22, %%zmm4              \n\t"
                         "vpdpbusd %%zmm16, %%zmm23, %%zmm5              \n\t"
                         "vpdpbusd %%zmm16, %%zmm24, %%zmm6              \n\t"
                         "vpdpbusd %%zmm16, %%zmm25, %%zmm7              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm22, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm23, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm25, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm4, %%zmm26, %%zmm4              \n\t"
                         "vpaddd %%zmm5, %%zmm27, %%zmm5              \n\t"
                         "vpaddd %%zmm6, %%zmm28, %%zmm6              \n\t"
                         "vpaddd %%zmm7, %%zmm29, %%zmm7              \n\t"
#endif

                         "add %7, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "mov %%rbx, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "vmovdqu8 (%%rbx), %%zmm17 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rbx, %7), %%zmm18 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rax), %%zmm19 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rax, %7), %%zmm20 %{%%k1%}                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm8              \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm9              \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm10              \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm11              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm18, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm19, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm20, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm8, %%zmm26, %%zmm8              \n\t"
                         "vpaddd %%zmm9, %%zmm27, %%zmm9              \n\t"
                         "vpaddd %%zmm10, %%zmm28, %%zmm10              \n\t"
                         "vpaddd %%zmm11, %%zmm29, %%zmm11            \n\t"
#endif

                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "mov %%rax, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "vmovdqu8 (%%rax), %%zmm22 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rax, %7), %%zmm23 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rbx), %%zmm24 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rbx, %7), %%zmm25 %{%%k1%}                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm22, %%zmm12              \n\t"
                         "vpdpbusd %%zmm16, %%zmm23, %%zmm13              \n\t"
                         "vpdpbusd %%zmm16, %%zmm24, %%zmm14              \n\t"
                         "vpdpbusd %%zmm16, %%zmm25, %%zmm15              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm22, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm23, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm25, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm12, %%zmm26, %%zmm12              \n\t"
                         "vpaddd %%zmm13, %%zmm27, %%zmm13              \n\t"
                         "vpaddd %%zmm14, %%zmm28, %%zmm14              \n\t"
                         "vpaddd %%zmm15, %%zmm29, %%zmm15              \n\t"
#endif

                         ".align 16                                      \n\t"
                         "3:                                             \n\t"
                         "vpunpckhdq %%zmm2, %%zmm0, %%zmm16              \n\t"
                         "vpunpckhdq %%zmm3, %%zmm1, %%zmm17              \n\t"
                         "vpunpckhdq %%zmm6, %%zmm4, %%zmm18              \n\t"
                         "vpunpckhdq %%zmm7, %%zmm5, %%zmm19              \n\t"
                         "vpunpckhdq %%zmm10, %%zmm8,   %%zmm20              \n\t"
                         "vpunpckhdq %%zmm11, %%zmm9, %%zmm21              \n\t"
                         "vpunpckhdq %%zmm14, %%zmm12, %%zmm22              \n\t"
                         "vpunpckhdq %%zmm15, %%zmm13, %%zmm23              \n\t"
                         "vpunpckldq %%zmm2, %%zmm0, %%zmm24              \n\t"
                         "vpunpckldq %%zmm3, %%zmm1, %%zmm25              \n\t"
                         "vpunpckldq %%zmm6, %%zmm4, %%zmm26              \n\t"
                         "vpunpckldq %%zmm7, %%zmm5, %%zmm27              \n\t"
                         "vpaddd %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpaddd %%zmm17, %%zmm25, %%zmm29              \n\t"
                         "vpaddd %%zmm18, %%zmm26, %%zmm30              \n\t"
                         "vpaddd %%zmm19, %%zmm27, %%zmm31              \n\t"
                         "vpunpckldq %%zmm10, %%zmm8, %%zmm24              \n\t"
                         "vpunpckldq %%zmm11, %%zmm9, %%zmm25              \n\t"
                         "vpunpckldq %%zmm14, %%zmm12, %%zmm26              \n\t"
                         "vpunpckldq %%zmm15, %%zmm13, %%zmm27              \n\t"
                         "vpaddd %%zmm20, %%zmm24, %%zmm16              \n\t"
                         "vpaddd %%zmm21, %%zmm25, %%zmm17              \n\t"
                         "vpaddd %%zmm22, %%zmm26, %%zmm18              \n\t"
                         "vpaddd %%zmm23, %%zmm27, %%zmm19              \n\t"
                         "vpunpckhdq %%zmm29, %%zmm28, %%zmm8              \n\t"
                         "vpunpckhdq %%zmm31, %%zmm30, %%zmm9              \n\t"
                         "vpunpckhdq %%zmm17, %%zmm16, %%zmm10              \n\t"
                         "vpunpckhdq %%zmm19, %%zmm18, %%zmm11              \n\t"
                         "vpunpckldq %%zmm29, %%zmm28, %%zmm12              \n\t"
                         "vpunpckldq %%zmm31, %%zmm30, %%zmm13              \n\t"
                         "vpunpckldq %%zmm17, %%zmm16, %%zmm14              \n\t"
                         "vpunpckldq %%zmm19, %%zmm18, %%zmm15              \n\t"
                         "vpaddd %%zmm8, %%zmm12, %%zmm16              \n\t"
                         "vpaddd %%zmm9, %%zmm13, %%zmm17              \n\t"
                         "vpaddd %%zmm10, %%zmm14, %%zmm18              \n\t"
                         "vpaddd %%zmm11, %%zmm15, %%zmm19              \n\t"
                         "vextracti32x8 $0x1, %%zmm16, %%ymm20              \n\t"
                         "vextracti32x8 $0x1, %%zmm17, %%ymm21              \n\t"
                         "vextracti32x8 $0x1, %%zmm18, %%ymm22              \n\t"
                         "vextracti32x8 $0x1, %%zmm19, %%ymm23              \n\t"
                         "vpaddd %%ymm20, %%ymm16, %%ymm24              \n\t"
                         "vpaddd %%ymm21, %%ymm17, %%ymm25              \n\t"
                         "vpaddd %%ymm22, %%ymm18, %%ymm26              \n\t"
                         "vpaddd %%ymm23, %%ymm19, %%ymm27              \n\t"
                         "vextracti32x4 $0x1, %%ymm24, %%xmm28              \n\t"
                         "vextracti32x4 $0x1, %%ymm25, %%xmm29              \n\t"
                         "vextracti32x4 $0x1, %%ymm26, %%xmm30              \n\t"
                         "vextracti32x4 $0x1, %%ymm27, %%xmm31              \n\t"
                         "vpaddd %%xmm28, %%xmm24, %%xmm0              \n\t"
                         "vpaddd %%xmm29, %%xmm25, %%xmm1              \n\t"
                         "vpaddd %%xmm30, %%xmm26, %%xmm2              \n\t"
                         "vpaddd %%xmm31, %%xmm27, %%xmm3              \n\t"

                         "mov %5, %%ebx \n\t"
                         "and $0x1, %%ebx \n\t"
                         "cmp $0x0, %%ebx \n\t"
                         "jne 4f      \n\t"
                         "vbroadcastss (%4), %%xmm4              \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0              \n\t"
                         "vpaddd %%xmm1, %%xmm4, %%xmm1              \n\t"
                         "vpaddd %%xmm2, %%xmm4, %%xmm2              \n\t"
                         "vpaddd %%xmm3, %%xmm4, %%xmm3              \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "cmpq $0x0, %6 \n\t"
                         "je 5f      \n\t"
                         "vbroadcastss (%6), %%xmm5                        \n\t"
                         "vcvtdq2ps %%xmm0, %%xmm0                       \n\t"
                         "vcvtdq2ps %%xmm1, %%xmm1                       \n\t"
                         "vcvtdq2ps %%xmm2, %%xmm2                       \n\t"
                         "vcvtdq2ps %%xmm3, %%xmm3                       \n\t"
                         "vmulps %%xmm0, %%xmm5, %%xmm0                       \n\t"
                         "vmulps %%xmm1, %%xmm5, %%xmm1                       \n\t"
                         "vmulps %%xmm2, %%xmm5, %%xmm2                       \n\t"
                         "vmulps %%xmm3, %%xmm5, %%xmm3                       \n\t"

                         "mov %5, %%ebx \n\t"
                         "and $0x2, %%ebx \n\t"
                         "je 5f      \n\t"
                         "vcvtps2dq %%xmm0, %%xmm0                       \n\t"
                         "vcvtps2dq %%xmm1, %%xmm1                       \n\t"
                         "vcvtps2dq %%xmm2, %%xmm2                       \n\t"
                         "vcvtps2dq %%xmm3, %%xmm3                       \n\t"
                         "mov $128, %%eax \n\t"
                         "vmovd %%eax, %%xmm5                    \n\t"
                         "vbroadcastss %%xmm5, %%xmm4            \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0                       \n\t"
                         "vpaddd %%xmm1, %%xmm4, %%xmm1                       \n\t"
                         "vpaddd %%xmm2, %%xmm4, %%xmm2                       \n\t"
                         "vpaddd %%xmm3, %%xmm4, %%xmm3                       \n\t"
                         "vpmovusdb %%xmm0,  (%9)                             \n\t"
                         "vpmovusdb %%xmm1,  0x4(%9)                             \n\t"
                         "vpmovusdb %%xmm2,  0x8(%9)                             \n\t"
                         "vpmovusdb %%xmm3,  0xC(%9)                             \n\t"
                         "jmp 6f      \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "vmovups %%xmm0, (%2)                           \n\t"
                         "vmovups %%xmm1, 0x10(%2)                       \n\t"
                         "vmovups %%xmm2, 0x20(%2)                       \n\t"
                         "vmovups %%xmm3, 0x30(%2)                       \n\t"
                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         :
                         : "r"(vector), "r"(matrix), "r"(result), "c"(bk), "r"(offsetC), "r"(flags),
                         "r"(scale), "r"((I64)K), "r"(kmask), "r"(u8Result)
                         : "%k1", "%rax", "%rbx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4",
                         "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
                         "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19",
                         "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
                         "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory");
}

void mvm_avx512_8_row(U32 K,
    U32 bk,
    U64 kmask,
    UINT8 *matrix,
    INT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
                         "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
                         "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
                         "vxorps %%zmm4, %%zmm4, %%zmm4                     \n\t"
                         "vxorps %%zmm5, %%zmm5, %%zmm5                     \n\t"
                         "vxorps %%zmm6, %%zmm6, %%zmm6                     \n\t"
                         "vxorps %%zmm7, %%zmm7, %%zmm7                     \n\t"
#ifndef _USE_AVX512_VNNI
                         "mov $1, %%eax \n\t"
                         "vmovd %%eax, %%xmm26                    \n\t"
                         "vpbroadcastw %%xmm26, %%zmm31            \n\t"
#endif
                         "cmp $0x0, %%ecx                         \n\t"
                         "je 2f                                    \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %1, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "vmovups (%0), %%zmm16                     \n\t"
                         "vmovups (%1), %%zmm17                             \n\t"
                         "vmovups (%1, %7), %%zmm18                             \n\t"
                         "vmovups (%%rax), %%zmm19                             \n\t"
                         "vmovups (%%rax, %7), %%zmm20                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0              \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm1              \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm2              \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm18, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm19, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm20, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm29, %%zmm3              \n\t"
#endif

                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "mov %%rax, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "vmovups (%%rax), %%zmm22                             \n\t"
                         "vmovups (%%rax, %7), %%zmm23                             \n\t"
                         "vmovups (%%rbx), %%zmm24                             \n\t"
                         "vmovups (%%rbx, %7), %%zmm25                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm22, %%zmm4              \n\t"
                         "vpdpbusd %%zmm16, %%zmm23, %%zmm5              \n\t"
                         "vpdpbusd %%zmm16, %%zmm24, %%zmm6              \n\t"
                         "vpdpbusd %%zmm16, %%zmm25, %%zmm7              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm22, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm23, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm25, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm4, %%zmm26, %%zmm4              \n\t"
                         "vpaddd %%zmm5, %%zmm27, %%zmm5              \n\t"
                         "vpaddd %%zmm6, %%zmm28, %%zmm6              \n\t"
                         "vpaddd %%zmm7, %%zmm29, %%zmm7              \n\t"
#endif

                         "add $0x40, %0                                  \n\t"
                         "add $0x40, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 1b                                          \n\t"

                         ".align 16                                      \n\t"
                         "2:                                             \n\t"

                         "kmovq %8, %%k1                      \n\t"
                         "cmp $0x0, %8                       \n\t"
                         "je 3f                                  \n\t"

                         "mov %1, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "vxorps %%zmm16, %%zmm16, %%zmm16                     \n\t"
                         "vmovdqu8 (%0), %%zmm16 %{%%k1%}                     \n\t"
                         "vmovdqu8 (%1), %%zmm17 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%1, %7), %%zmm18 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rax), %%zmm19 %{%%k1%}                            \n\t"
                         "vmovdqu8 (%%rax, %7), %%zmm20 %{%%k1%}                            \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0              \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm1              \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm2              \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm18, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm19, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm20, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm29, %%zmm3              \n\t"
#endif

                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "mov %%rax, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "add %7, %%rbx \n\t"
                         "vmovdqu8 (%%rax), %%zmm22 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rax, %7), %%zmm23 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rbx), %%zmm24 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rbx, %7), %%zmm25 %{%%k1%}                            \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm22, %%zmm4              \n\t"
                         "vpdpbusd %%zmm16, %%zmm23, %%zmm5              \n\t"
                         "vpdpbusd %%zmm16, %%zmm24, %%zmm6              \n\t"
                         "vpdpbusd %%zmm16, %%zmm25, %%zmm7              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm22, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm23, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm25, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm4, %%zmm26, %%zmm4              \n\t"
                         "vpaddd %%zmm5, %%zmm27, %%zmm5              \n\t"
                         "vpaddd %%zmm6, %%zmm28, %%zmm6              \n\t"
                         "vpaddd %%zmm7, %%zmm29, %%zmm7              \n\t"
#endif

                         ".align 16                                      \n\t"
                         "3:                                             \n\t"
                         "vpunpckhdq %%zmm2, %%zmm0, %%zmm16              \n\t"
                         "vpunpckhdq %%zmm3, %%zmm1, %%zmm17              \n\t"
                         "vpunpckldq %%zmm2, %%zmm0, %%zmm24              \n\t"
                         "vpunpckldq %%zmm3, %%zmm1, %%zmm25              \n\t"
                         "vpunpckhdq %%zmm6, %%zmm4, %%zmm18              \n\t"
                         "vpunpckhdq %%zmm7, %%zmm5, %%zmm19              \n\t"
                         "vpunpckldq %%zmm6, %%zmm4, %%zmm26              \n\t"
                         "vpunpckldq %%zmm7, %%zmm5, %%zmm27              \n\t"
                         "vpaddd %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpaddd %%zmm17, %%zmm25, %%zmm29              \n\t"
                         "vpaddd %%zmm18, %%zmm26, %%zmm30              \n\t"
                         "vpaddd %%zmm19, %%zmm27, %%zmm31              \n\t"
                         "vpunpckhdq %%zmm29, %%zmm28, %%zmm8              \n\t"
                         "vpunpckhdq %%zmm31, %%zmm30, %%zmm9              \n\t"
                         "vpunpckldq %%zmm29, %%zmm28, %%zmm12              \n\t"
                         "vpunpckldq %%zmm31, %%zmm30, %%zmm13              \n\t"
                         "vpaddd %%zmm8, %%zmm12, %%zmm16              \n\t"
                         "vpaddd %%zmm9, %%zmm13, %%zmm17              \n\t"
                         "vextracti32x8 $0x1, %%zmm16, %%ymm20              \n\t"
                         "vextracti32x8 $0x1, %%zmm17, %%ymm21              \n\t"
                         "vpaddd %%ymm20, %%ymm16, %%ymm24              \n\t"
                         "vpaddd %%ymm21, %%ymm17, %%ymm25              \n\t"
                         "vextracti32x4 $0x1, %%ymm24, %%xmm28              \n\t"
                         "vextracti32x4 $0x1, %%ymm25, %%xmm29              \n\t"
                         "vpaddd %%xmm28, %%xmm24, %%xmm0              \n\t"
                         "vpaddd %%xmm29, %%xmm25, %%xmm1              \n\t"

                         "mov %5, %%ebx \n\t"
                         "and $0x1, %%ebx \n\t"
                         "cmp $0x0, %%ebx \n\t"
                         "jne 4f      \n\t"
                         "vbroadcastss (%4), %%xmm4              \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0              \n\t"
                         "vpaddd %%xmm1, %%xmm4, %%xmm1              \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "cmpq $0x0, %6 \n\t"
                         "je 5f      \n\t"
                         "vbroadcastss (%6), %%xmm5                        \n\t"
                         "vcvtdq2ps %%xmm0, %%xmm0                       \n\t"
                         "vcvtdq2ps %%xmm1, %%xmm1                       \n\t"
                         "vmulps %%xmm0, %%xmm5, %%xmm0                       \n\t"
                         "vmulps %%xmm1, %%xmm5, %%xmm1                       \n\t"

                         "mov %5, %%ebx \n\t"
                         "and $0x2, %%ebx \n\t"
                         "je 5f      \n\t"
                         "vcvtps2dq %%xmm0, %%xmm0                       \n\t"
                         "vcvtps2dq %%xmm1, %%xmm1                       \n\t"
                         "mov $128, %%eax \n\t"
                         "vmovd %%eax, %%xmm5                    \n\t"
                         "vbroadcastss %%xmm5, %%xmm4            \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0                       \n\t"
                         "vpaddd %%xmm1, %%xmm4, %%xmm1                       \n\t"
                         "vpmovusdb %%xmm0, (%9)                             \n\t"
                         "vpmovusdb %%xmm1, 0x4(%9)                             \n\t"
                         "jmp 6f      \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "vmovups %%xmm0, (%2)                           \n\t"
                         "vmovups %%xmm1, 0x10(%2)                       \n\t"
                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         :
                         : "r"(vector), "r"(matrix), "r"(result), "c"(bk), "r"(offsetC), "r"(flags),
                         "r"(scale), "r"((I64)K), "r"((I64)kmask), "r"(u8Result)
                         : "%k1", "%rax", "%rbx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4",
                         "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
                         "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19",
                         "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
                         "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory");
}

void mvm_avx512_4_row(U32 K,
    U32 bk,
    U64 kmask,
    UINT8 *matrix,
    INT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
                         "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
                         "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
#ifndef _USE_AVX512_VNNI
                         "mov $1, %%eax \n\t"
                         "vmovd %%eax, %%xmm26                    \n\t"
                         "vpbroadcastw %%xmm26, %%zmm31            \n\t"
#endif
                         "cmp $0x0, %%ecx                         \n\t"
                         "je 2f                                    \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "mov %1, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "vmovups (%0), %%zmm16                     \n\t"
                         "vmovups (%1), %%zmm17                             \n\t"
                         "vmovups (%1, %7), %%zmm18                             \n\t"
                         "vmovups (%%rax), %%zmm19                             \n\t"
                         "vmovups (%%rax, %7), %%zmm20                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0              \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm1              \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm2              \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm18, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm19, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm20, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm29, %%zmm3              \n\t"
#endif

                         "add $0x40, %0                                  \n\t"
                         "add $0x40, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 1b                                          \n\t"

                         ".align 16                                      \n\t"
                         "2:                                             \n\t"

                         "kmovq %8, %%k1                      \n\t"
                         "cmp $0x0, %8                       \n\t"
                         "je 3f                                  \n\t"

                         "mov %1, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "add %7, %%rax \n\t"
                         "vxorps %%zmm16, %%zmm16, %%zmm16                     \n\t"
                         "vmovdqu8 (%0), %%zmm16 %{%%k1%}                     \n\t"
                         "vmovdqu8 (%1), %%zmm17 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%1, %7), %%zmm18 %{%%k1%}                             \n\t"
                         "vmovdqu8 (%%rax), %%zmm19 %{%%k1%}                            \n\t"
                         "vmovdqu8 (%%rax, %7), %%zmm20 %{%%k1%}                            \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0              \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm1              \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm2              \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm3              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm18, %%zmm27              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm19, %%zmm28              \n\t"
                         "vpmaddubsw %%zmm16, %%zmm20, %%zmm29              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpmaddwd %%zmm27, %%zmm31, %%zmm27              \n\t"
                         "vpmaddwd %%zmm28, %%zmm31, %%zmm28              \n\t"
                         "vpmaddwd %%zmm29, %%zmm31, %%zmm29              \n\t"
                         "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"
                         "vpaddd %%zmm1, %%zmm27, %%zmm1              \n\t"
                         "vpaddd %%zmm2, %%zmm28, %%zmm2              \n\t"
                         "vpaddd %%zmm3, %%zmm29, %%zmm3              \n\t"
#endif

                         ".align 16                                      \n\t"
                         "3:                                             \n\t"
                         "vpunpckhdq %%zmm2, %%zmm0, %%zmm16              \n\t"
                         "vpunpckhdq %%zmm3, %%zmm1, %%zmm17              \n\t"
                         "vpunpckldq %%zmm2, %%zmm0, %%zmm24              \n\t"
                         "vpunpckldq %%zmm3, %%zmm1, %%zmm25              \n\t"
                         "vpaddd %%zmm16, %%zmm24, %%zmm28              \n\t"
                         "vpaddd %%zmm17, %%zmm25, %%zmm29              \n\t"
                         "vpunpckhdq %%zmm29, %%zmm28, %%zmm8              \n\t"
                         "vpunpckldq %%zmm29, %%zmm28, %%zmm12              \n\t"
                         "vpaddd %%zmm8, %%zmm12, %%zmm16              \n\t"
                         "vextracti32x8 $0x1, %%zmm16, %%ymm20              \n\t"
                         "vpaddd %%ymm20, %%ymm16, %%ymm24              \n\t"
                         "vextracti32x4 $0x1, %%ymm24, %%xmm28              \n\t"
                         "vpaddd %%xmm28, %%xmm24, %%xmm0              \n\t"

                         "mov %5, %%ebx \n\t"
                         "and $0x1, %%ebx \n\t"
                         "cmp $0x0, %%ebx \n\t"
                         "jne 4f      \n\t"
                         "vbroadcastss (%4), %%xmm4              \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0              \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "cmpq $0x0, %6 \n\t"
                         "je 5f      \n\t"
                         "vbroadcastss (%6), %%xmm5                        \n\t"
                         "vcvtdq2ps %%xmm0, %%xmm0                       \n\t"
                         "vmulps %%xmm0, %%xmm5, %%xmm0                       \n\t"

                         "mov %5, %%ebx \n\t"
                         "and $0x2, %%ebx \n\t"
                         "je 5f      \n\t"
                         "vcvtps2dq %%xmm0, %%xmm0                       \n\t"
                         "mov $128, %%eax \n\t"
                         "vmovd %%eax, %%xmm5                    \n\t"
                         "vbroadcastss %%xmm5, %%xmm4            \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0                       \n\t"
                         "vpmovusdb %%xmm0, (%9)                             \n\t"
                         "jmp 6f      \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "vmovups %%xmm0, (%2)                           \n\t"
                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         :
                         : "r"(vector), "r"(matrix), "r"(result), "c"(bk), "r"(offsetC), "r"(flags),
                         "r"(scale), "r"((I64)K), "r"(kmask), "r"(u8Result)
                         : "%k1", "%rax", "%rbx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4",
                         "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
                         "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19",
                         "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
                         "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory");
}

void mvm_avx512_1_row(U32 K,
    U32 bk,
    U64 kmask,
    UINT8 *matrix,
    INT8 *vector,
    I32 *result,
    UINT8 *u8Result,
    I32 *offsetC,
    const F32 *scale,
    U32 flags)
{
    __asm__ __volatile__("vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
#ifndef _USE_AVX512_VNNI
                         "mov $1, %%eax \n\t"
                         "vmovd %%eax, %%xmm26                    \n\t"
                         "vpbroadcastw %%xmm26, %%zmm31            \n\t"
#endif
                         "cmp $0x0, %%ecx                         \n\t"
                         "je 2f                                    \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         "vmovups (%0), %%zmm16                     \n\t"
                         "vmovups (%1), %%zmm17                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"
#endif

                         "add $0x40, %0                                  \n\t"
                         "add $0x40, %1                                 \n\t"
                         "dec %%ecx                                      \n\t"
                         "jg 1b                                          \n\t"

                         ".align 16                                      \n\t"
                         "2:                                             \n\t"

                         "kmovq %8, %%k1                      \n\t"
                         "cmp $0x0, %8                       \n\t"
                         "je 3f                                  \n\t"

                         "vxorps %%zmm16, %%zmm16, %%zmm16                     \n\t"
                         "vmovdqu8 (%0), %%zmm16 %{%%k1%}                     \n\t"
                         "vmovdqu8 (%1), %%zmm17 %{%%k1%}                             \n\t"
#ifdef _USE_AVX512_VNNI
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0              \n\t"
#else
                         "vpmaddubsw %%zmm16, %%zmm17, %%zmm26              \n\t"
                         "vpmaddwd %%zmm26, %%zmm31, %%zmm26              \n\t"
                         "vpaddd %%zmm0, %%zmm26, %%zmm0              \n\t"
#endif

                         ".align 16                                      \n\t"
                         "3:                                             \n\t"
                         "vextracti32x8 $0x1, %%zmm0, %%ymm1       \n\t"
                         "vpaddd %%ymm1, %%ymm0, %%ymm0            \n\t"
                         "vextracti32x4 $0x1, %%ymm0, %%xmm1       \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00001011, %%xmm0, %%xmm1    \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00000001, %%xmm0, %%xmm1    \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"

                         "mov %5, %%ebx \n\t"
                         "and $0x1, %%ebx \n\t"
                         "jne 4f      \n\t"
                         "vbroadcastss (%4), %%xmm4              \n\t"
                         "vpaddd %%xmm0, %%xmm4, %%xmm0              \n\t"

                         ".align 16                                      \n\t"
                         "4:                                             \n\t"
                         "cmpq $0x0, %6 \n\t"
                         "je 6f      \n\t"
                         "vbroadcastss (%6), %%xmm5                        \n\t"
                         "vcvtdq2ps %%xmm0, %%xmm0                       \n\t"
                         "vmulps %%xmm0, %%xmm5, %%xmm0                       \n\t"

                         "mov %5, %%ebx \n\t"
                         "and $0x2, %%ebx \n\t"
                         "je 6f      \n\t"
                         "vcvtps2dq %%xmm0, %%xmm0                       \n\t"
                         "vmovd %%xmm0, %%eax                    \n\t"
                         "add $128, %%eax \n\t"
                         "cmp $255, %%eax \n\t"
                         "jl 5f      \n\t"
                         "movb $0xFF, %%al \n\t"

                         ".align 16                                      \n\t"
                         "5:                                             \n\t"
                         "movb %%al, (%9) \n\t"
                         "jmp 7f      \n\t"

                         ".align 16                                      \n\t"
                         "6:                                             \n\t"
                         "vmovss %%xmm0, (%2)                      \n\t"
                         ".align 16                                      \n\t"
                         "7:                                             \n\t"
                         :
                         : "r"(vector), "r"(matrix), "r"(result), "c"(bk), "r"(offsetC), "r"(flags),
                         "r"(scale), "r"((I64)K), "r"(kmask), "r"(u8Result)
                         : "%k1", "%rax", "%rbx", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4",
                         "%zmm5", "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12",
                         "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19",
                         "%zmm20", "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26",
                         "%zmm27", "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory");
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
    U32 num64 = numColumns / 16;
    U64 resMask = pow(2, numColumns % 16) - 1;
    __asm__ __volatile__("vxorps %%zmm0, %%zmm0, %%zmm0            \n\t"
                         "mov %0, %%rdx \n\t"
                         "mov %2, %%ebx \n\t"
                         "cmp $0x0, %%ebx                         \n\t"
                         "je 1f                                    \n\t"
                         ".align 16                                \n\t"
                         "0:                                       \n\t"
                         "vmovups (%%rdx), %%xmm2              \n\t"
                         "vpmovsxbd %%xmm2, %%zmm1              \n\t"
                         "vpaddd %%zmm1, %%zmm0, %%zmm0            \n\t"
                         "add $0x10, %%rdx                            \n\t"
                         "dec %%ebx                                \n\t"
                         "jg 0b                                    \n\t"

                         ".align 16                                \n\t"
                         "1:                                       \n\t"
                         "cmp $0x0, %%eax                         \n\t"
                         "je 2f                                    \n\t"

                         "kmovw %%eax, %%k2                         \n\t"
                         "vxorps %%xmm2, %%xmm2, %%xmm2            \n\t"
                         "vxorps %%zmm1, %%zmm1, %%zmm1            \n\t"
                         "vmovups (%%rdx), %%xmm2 %{%%k2%}              \n\t"
                         "vpmovsxbd %%xmm2, %%zmm1 %{%%k2%}              \n\t"
                         "vpaddd %%zmm1, %%zmm0, %%zmm0            \n\t"
                         ".align 16                                \n\t"
                         "2:                                       \n\t"

                         "vextracti32x8 $0x1, %%zmm0, %%ymm1       \n\t"
                         "vpaddd %%ymm1, %%ymm0, %%ymm0            \n\t"
                         "vextracti32x4 $0x1, %%ymm0, %%xmm1       \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00001011, %%xmm0, %%xmm1    \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vpermilps $0b00000001, %%xmm0, %%xmm1    \n\t"
                         "vpaddd %%xmm1, %%xmm0, %%xmm0            \n\t"
                         "vmovss %%xmm0, (%1)                      \n\t"
                         :
                         : "r"(vector), "r"(&offsetC), "r"(num64), "a"(resMask)
                         : "%k2", "%ebx", "%rdx", "%zmm0", "%zmm1", "%zmm2", "memory", "cc");
    offsetC *= -128;
    if (df == DF_TRANSPOSE) {
        transpose(packB, (UINT8 *)tmp, numRows, numColumns);
        packB = (UINT8 *)tmp;
    }

    U32 blockSizeK = 0, blockSizeN = 0, flags = 0;
    kernel_func kernel[5] = {
        mvm_avx512_1_row, mvm_avx512_4_row, mvm_avx512_8_row, mvm_avx512_8_row, mvm_avx512_16_row};
    U32 unrollSize[5] = {1, 4, 8, 8, 16};
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
        num64 = blockSizeK / 64;
        resMask = 1;
        resMask = (resMask << (blockSizeK % 64)) - 1;
        F32 *useFactor = nullptr;
        if (k == numColumns - blockSizeK) {
            useFactor = factorPtr;
        }
        for (U32 j = 0; j < numRows; j += blockSizeN) {
            blockSizeN = UNI_MIN(UNROLL_N, numRows - j);
            blockSizeN = unrollSize[blockSizeN >> 2];
            UINT8 *curM = packB + j * numColumns;
            kernel[blockSizeN >> 2](numColumns, num64, resMask, curM, vector + k, i32Result + j,
                u8Result + j, &offsetC, useFactor, flags);
        }
    }
    return SUCCESS;
}
