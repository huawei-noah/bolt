// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/blas_fp32.h"
#include "error.h"

#define UNROLL_N 4
#define BOLCK_K_DIM 512

typedef void (*kernel_func)(U32 bk, U32 lda, F32 *matrix, F32 *vector, F32 *result);

void mvm_row_avx_4_32(U32 bk, U32 lda, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                         "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"

                         "mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "mov %1, %%rdx                                     \n\t"
                         "add %%rax, %%rdx                                  \n\t"
                         "mov %%rdx, %%r9                                   \n\t"
                         "add %%rax, %%r9                                   \n\t"
                         "mov %%r9, %%r10                                   \n\t"
                         "add %%rax, %%r10                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $5, %%ecx                                     \n\t"
                         "je .k_loop_32_end                                 \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_32:                                       \n\t"

                         "prefetcht0 0x100(%1)                              \n\t"
                         "prefetcht0 0x140(%1)                              \n\t"
                         "prefetcht0 0x100(%%rdx)                           \n\t"
                         "prefetcht0 0x140(%%rdx)                           \n\t"

                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups 0x20(%2), %%ymm13                         \n\t"
                         "vmovups 0x40(%2), %%ymm14                         \n\t"
                         "vmovups 0x60(%2), %%ymm15                         \n\t"

                         "vmovups (%1), %%ymm4                              \n\t"
                         "vmovups 0x20(%1), %%ymm5                          \n\t"
                         "vmovups 0x40(%1), %%ymm6                          \n\t"
                         "vmovups 0x60(%1), %%ymm7                          \n\t"
                         "vmovups (%%rdx), %%ymm8                           \n\t"
                         "vmovups 0x20(%%rdx), %%ymm9                       \n\t"
                         "vmovups 0x40(%%rdx), %%ymm10                      \n\t"
                         "vmovups 0x60(%%rdx), %%ymm11                      \n\t"
                         "vfmadd231ps %%ymm12, %%ymm4, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm1               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm5, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm9, %%ymm1               \n\t"
                         "vfmadd231ps %%ymm14, %%ymm6, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm14, %%ymm10, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm7, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm1              \n\t"

                         "prefetcht0 0x100(%%r9)                            \n\t"
                         "prefetcht0 0x140(%%r9)                            \n\t"
                         "prefetcht0 0x100(%%r10)                           \n\t"
                         "prefetcht0 0x140(%%r10)                           \n\t"

                         "vmovups (%%r9), %%ymm4                            \n\t"
                         "vmovups 0x20(%%r9), %%ymm5                        \n\t"
                         "vmovups 0x40(%%r9), %%ymm6                        \n\t"
                         "vmovups 0x60(%%r9), %%ymm7                        \n\t"
                         "vmovups (%%r10), %%ymm8                           \n\t"
                         "vmovups 0x20(%%r10), %%ymm9                       \n\t"
                         "vmovups 0x40(%%r10), %%ymm10                      \n\t"
                         "vmovups 0x60(%%r10), %%ymm11                      \n\t"
                         "vfmadd231ps %%ymm12, %%ymm4, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm3               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm5, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm9, %%ymm3               \n\t"
                         "vfmadd231ps %%ymm14, %%ymm6, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm14, %%ymm10, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm7, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"

                         "add $0x80, %1                                     \n\t"
                         "add $0x80, %2                                     \n\t"
                         "add $0x80, %%rdx                                  \n\t"
                         "add $0x80, %%r9                                   \n\t"
                         "add $0x80, %%r10                                  \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_32                                     \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_32_end:                                   \n\t"
                         "mov %0, %%ecx                                     \n\t"
                         "and $0x1F, %%ecx                                  \n\t"
                         "cmp $0x10, %%ecx                                  \n\t"
                         "jl .k_loop_remain_16_end                          \n\t"

                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups 0x20(%2), %%ymm13                         \n\t"

                         "vmovups (%1), %%ymm4                              \n\t"
                         "vmovups 0x20(%1), %%ymm5                          \n\t"
                         "vmovups (%%rdx), %%ymm8                           \n\t"
                         "vmovups 0x20(%%rdx), %%ymm9                       \n\t"
                         "vmovups (%%r9), %%ymm6                            \n\t"
                         "vmovups 0x20(%%r9), %%ymm7                        \n\t"
                         "vmovups (%%r10), %%ymm10                          \n\t"
                         "vmovups 0x20(%%r10), %%ymm11                      \n\t"
                         "vfmadd231ps %%ymm12, %%ymm4, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm1               \n\t"
                         "vfmadd231ps %%ymm12, %%ymm6, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm12, %%ymm10, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm13, %%ymm5, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm9, %%ymm1               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm7, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm11, %%ymm3              \n\t"

                         "add $0x40, %1                                     \n\t"
                         "add $0x40, %2                                     \n\t"
                         "add $0x40, %%rdx                                  \n\t"
                         "add $0x40, %%r9                                   \n\t"
                         "add $0x40, %%r10                                  \n\t"
                         "sub $0x10, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_remain_16_end:                            \n\t"
                         "cmp $0x8, %%ecx                                   \n\t"
                         "jl .k_loop_remain_8_end                           \n\t"
                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups (%1), %%ymm4                              \n\t"
                         "vmovups (%%rdx), %%ymm6                           \n\t"
                         "vmovups (%%r9), %%ymm8                            \n\t"
                         "vmovups (%%r10), %%ymm10                          \n\t"
                         "vfmadd231ps %%ymm4, %%ymm12, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm6, %%ymm12, %%ymm1               \n\t"
                         "vfmadd231ps %%ymm8, %%ymm12, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
                         "add $0x20, %1                                     \n\t"
                         "add $0x20, %2                                     \n\t"
                         "add $0x20, %%rdx                                  \n\t"
                         "add $0x20, %%r9                                   \n\t"
                         "add $0x20, %%r10                                  \n\t"
                         "sub $0x8, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_remain_8_end:                             \n\t"
                         "vperm2f128 $0x1, %%ymm0, %%ymm0, %%ymm12          \n\t"
                         "vperm2f128 $0x1, %%ymm1, %%ymm1, %%ymm13          \n\t"
                         "vperm2f128 $0x1, %%ymm2, %%ymm2, %%ymm14          \n\t"
                         "vperm2f128 $0x1, %%ymm3, %%ymm3, %%ymm15          \n\t"
                         "vaddps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vaddps %%ymm13, %%ymm1, %%ymm1                    \n\t"
                         "vaddps %%ymm14, %%ymm2, %%ymm2                    \n\t"
                         "vaddps %%ymm15, %%ymm3, %%ymm3                    \n\t"

                         "cmp $0x4, %%ecx                                   \n\t"
                         "jl .k_loop_remain_4_end                           \n\t"
                         "vmovups (%2), %%xmm12                             \n\t"
                         "vmovups (%1), %%xmm4                              \n\t"
                         "vmovups (%%rdx), %%xmm6                           \n\t"
                         "vmovups (%%r9), %%xmm8                            \n\t"
                         "vmovups (%%r10), %%xmm10                          \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"
                         "vfmadd231ps %%xmm6, %%xmm12, %%xmm1               \n\t"
                         "vfmadd231ps %%xmm8, %%xmm12, %%xmm2               \n\t"
                         "vfmadd231ps %%xmm10,%%xmm12, %%xmm3               \n\t"
                         "add $0x10, %1                                     \n\t"
                         "add $0x10, %2                                     \n\t"
                         "add $0x10, %%rdx                                  \n\t"
                         "add $0x10, %%r9                                   \n\t"
                         "add $0x10, %%r10                                  \n\t"
                         "sub $0x4, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_remain_4_end:                             \n\t"
                         "vhaddps %%xmm0, %%xmm0, %%xmm0                    \n\t"
                         "vhaddps %%xmm1, %%xmm1, %%xmm1                    \n\t"
                         "vhaddps %%xmm2, %%xmm2, %%xmm2                    \n\t"
                         "vhaddps %%xmm3, %%xmm3, %%xmm3                    \n\t"

                         "cmp $0x2, %%ecx                                   \n\t"
                         "jl .k_loop_remain_2_end                           \n\t"
                         "vxorps %%xmm12, %%xmm12, %%xmm12                  \n\t"
                         "vxorps %%xmm4, %%xmm4, %%xmm4                     \n\t"
                         "vxorps %%xmm6, %%xmm6, %%xmm6                     \n\t"
                         "vxorps %%xmm8, %%xmm8, %%xmm8                     \n\t"
                         "vxorps %%xmm10, %%xmm10, %%xmm10                  \n\t"
                         "vmovsd (%2), %%xmm12                              \n\t"
                         "vmovsd (%1), %%xmm4                               \n\t"
                         "vmovsd (%%rdx), %%xmm6                            \n\t"
                         "vmovsd (%%r9), %%xmm8                             \n\t"
                         "vmovsd (%%r10), %%xmm10                           \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"
                         "vfmadd231ps %%xmm6, %%xmm12, %%xmm1               \n\t"
                         "vfmadd231ps %%xmm8, %%xmm12, %%xmm2               \n\t"
                         "vfmadd231ps %%xmm10, %%xmm12, %%xmm3              \n\t"
                         "add $0x8, %1                                      \n\t"
                         "add $0x8, %2                                      \n\t"
                         "add $0x8, %%rdx                                   \n\t"
                         "add $0x8, %%r9                                    \n\t"
                         "add $0x8, %%r10                                   \n\t"
                         "sub $0x2, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_remain_2_end:                             \n\t"
                         "vhaddps %%xmm0, %%xmm0, %%xmm0                    \n\t"
                         "vhaddps %%xmm1, %%xmm1, %%xmm1                    \n\t"
                         "vhaddps %%xmm2, %%xmm2, %%xmm2                    \n\t"
                         "vhaddps %%xmm3, %%xmm3, %%xmm3                    \n\t"

                         "and $0x1, %%ecx                                   \n\t"
                         "je .k_loop_remain_1_end                           \n\t"
                         "vxorps %%xmm12,%%xmm12, %%xmm12                   \n\t"
                         "vxorps %%xmm4, %%xmm4, %%xmm4                     \n\t"
                         "vxorps %%xmm6, %%xmm6, %%xmm6                     \n\t"
                         "vxorps %%xmm8, %%xmm8, %%xmm8                     \n\t"
                         "vxorps %%xmm10, %%xmm10, %%xmm10                  \n\t"
                         "vmovss (%2), %%xmm12                              \n\t"
                         "vmovss (%1), %%xmm4                               \n\t"
                         "vmovss (%%rdx), %%xmm6                            \n\t"
                         "vmovss (%%r9), %%xmm8                             \n\t"
                         "vmovss (%%r10), %%xmm10                           \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"
                         "vfmadd231ps %%xmm6, %%xmm12, %%xmm1               \n\t"
                         "vfmadd231ps %%xmm8, %%xmm12, %%xmm2               \n\t"
                         "vfmadd231ps %%xmm10, %%xmm12, %%xmm3              \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_remain_1_end:                             \n\t"
                         "vaddps (%3), %%xmm0, %%xmm0                       \n\t"
                         "vmovss %%xmm0, (%3)                               \n\t"
                         "vaddps 0x4(%3), %%xmm1, %%xmm1                    \n\t"
                         "vmovss %%xmm1, 0x4(%3)                            \n\t"
                         "vaddps 0x8(%3), %%xmm2, %%xmm2                    \n\t"
                         "vmovss %%xmm2, 0x8(%3)                            \n\t"
                         "vaddps 0xC(%3), %%xmm3, %%xmm3                    \n\t"
                         "vmovss %%xmm3, 0xC(%3)                            \n\t"
                         :
                         : "r"(bk), "r"(matrix), "r"(vector), "r"(result), "r"(lda)
                         : "%eax", "%rax", "%ecx", "%rdx", "%r9", "%r10", "%ymm0", "%ymm1", "%ymm2",
                         "%ymm3", "%ymm4", "%ymm6", "%ymm8", "%ymm10", "%ymm12", "%ymm13", "%ymm14",
                         "%ymm15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm6", "%xmm8",
                         "%xmm10", "%xmm12", "memory");
}

void mvm_row_avx_2_32(U32 bk, U32 lda, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"

                         "mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "mov %1, %%rdx                                     \n\t"
                         "add %%rax, %%rdx                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $5, %%ecx                                     \n\t"
                         "je .n2_k_loop_32_end                              \n\t"
                         ".align 16                                         \n\t"
                         ".n2_k_loop_32:                                    \n\t"

                         "prefetcht0 0x100(%1)                              \n\t"
                         "prefetcht0 0x140(%1)                              \n\t"
                         "prefetcht0 0x100(%%rdx)                           \n\t"
                         "prefetcht0 0x140(%%rdx)                           \n\t"

                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups 0x20(%2), %%ymm13                         \n\t"
                         "vmovups 0x40(%2), %%ymm14                         \n\t"
                         "vmovups 0x60(%2), %%ymm15                         \n\t"

                         "vmovups (%1), %%ymm4                              \n\t"
                         "vmovups 0x20(%1), %%ymm5                          \n\t"
                         "vmovups 0x40(%1), %%ymm6                          \n\t"
                         "vmovups 0x60(%1), %%ymm7                          \n\t"
                         "vmovups (%%rdx), %%ymm8                           \n\t"
                         "vmovups 0x20(%%rdx), %%ymm9                       \n\t"
                         "vmovups 0x40(%%rdx), %%ymm10                      \n\t"
                         "vmovups 0x60(%%rdx), %%ymm11                      \n\t"
                         "vfmadd231ps %%ymm12, %%ymm4, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm1               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm5, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm9, %%ymm1               \n\t"
                         "vfmadd231ps %%ymm14, %%ymm6, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm14, %%ymm10, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm7, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm1              \n\t"

                         "add $0x80, %1                                     \n\t"
                         "add $0x80, %2                                     \n\t"
                         "add $0x80, %%rdx                                  \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .n2_k_loop_32                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n2_k_loop_32_end:                                \n\t"
                         "mov %0, %%ecx                                     \n\t"
                         "and $0x1F, %%ecx                                  \n\t"
                         "cmp $0x10, %%ecx                                  \n\t"
                         "jl .n2_k_loop_remain_16_end                       \n\t"

                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups 0x20(%2), %%ymm13                         \n\t"

                         "vmovups (%1), %%ymm4                              \n\t"
                         "vmovups 0x20(%1), %%ymm5                          \n\t"
                         "vmovups (%%rdx), %%ymm8                           \n\t"
                         "vmovups 0x20(%%rdx), %%ymm9                       \n\t"
                         "vfmadd231ps %%ymm12, %%ymm4, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm1               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm5, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm9, %%ymm1               \n\t"

                         "add $0x40, %1                                     \n\t"
                         "add $0x40, %2                                     \n\t"
                         "add $0x40, %%rdx                                  \n\t"
                         "sub $0x10, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n2_k_loop_remain_16_end:                         \n\t"
                         "cmp $0x8, %%ecx                                   \n\t"
                         "jl .n2_k_loop_remain_8_end                        \n\t"
                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups (%1), %%ymm4                              \n\t"
                         "vmovups (%%rdx), %%ymm6                           \n\t"
                         "vfmadd231ps %%ymm4, %%ymm12, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm6, %%ymm12, %%ymm1               \n\t"
                         "add $0x20, %1                                     \n\t"
                         "add $0x20, %2                                     \n\t"
                         "add $0x20, %%rdx                                  \n\t"
                         "sub $0x8, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n2_k_loop_remain_8_end:                          \n\t"
                         "vperm2f128 $0x1, %%ymm0, %%ymm0, %%ymm12          \n\t"
                         "vperm2f128 $0x1, %%ymm1, %%ymm1, %%ymm13          \n\t"
                         "vaddps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vaddps %%ymm13, %%ymm1, %%ymm1                    \n\t"

                         "cmp $0x4, %%ecx                                   \n\t"
                         "jl .n2_k_loop_remain_4_end                        \n\t"
                         "vmovups (%2), %%xmm12                             \n\t"
                         "vmovups (%1), %%xmm4                              \n\t"
                         "vmovups (%%rdx), %%xmm6                           \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"
                         "vfmadd231ps %%xmm6, %%xmm12, %%xmm1               \n\t"
                         "add $0x10, %1                                     \n\t"
                         "add $0x10, %2                                     \n\t"
                         "add $0x10, %%rdx                                  \n\t"
                         "sub $0x4, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n2_k_loop_remain_4_end:                          \n\t"
                         "vhaddps %%xmm0, %%xmm0, %%xmm0                    \n\t"
                         "vhaddps %%xmm1, %%xmm1, %%xmm1                    \n\t"

                         "cmp $0x2, %%ecx                                   \n\t"
                         "jl .n2_k_loop_remain_2_end                        \n\t"
                         "vxorps %%xmm12, %%xmm12, %%xmm12                  \n\t"
                         "vxorps %%xmm4, %%xmm4, %%xmm4                     \n\t"
                         "vxorps %%xmm6, %%xmm6, %%xmm6                     \n\t"
                         "vmovsd (%2), %%xmm12                              \n\t"
                         "vmovsd (%1), %%xmm4                               \n\t"
                         "vmovsd (%%rdx), %%xmm6                            \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"
                         "vfmadd231ps %%xmm6, %%xmm12, %%xmm1               \n\t"
                         "add $0x8, %1                                      \n\t"
                         "add $0x8, %2                                      \n\t"
                         "add $0x8, %%rdx                                   \n\t"
                         "sub $0x2, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n2_k_loop_remain_2_end:                          \n\t"
                         "vhaddps %%xmm0, %%xmm0, %%xmm0                    \n\t"
                         "vhaddps %%xmm1, %%xmm1, %%xmm1                    \n\t"
                         "and $1, %%ecx                                     \n\t"
                         "je .n2_k_loop_remain_1_end                        \n\t"
                         "vxorps %%xmm12, %%xmm12, %%xmm12                  \n\t"
                         "vxorps %%xmm4, %%xmm4, %%xmm4                     \n\t"
                         "vxorps %%xmm6, %%xmm6, %%xmm6                     \n\t"
                         "vmovss (%2), %%xmm12                              \n\t"
                         "vmovss (%1), %%xmm4                               \n\t"
                         "vmovss (%%rdx), %%xmm6                            \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"
                         "vfmadd231ps %%xmm6, %%xmm12, %%xmm1               \n\t"

                         ".align 16                                         \n\t"
                         ".n2_k_loop_remain_1_end:                          \n\t"
                         "vaddps (%3), %%xmm0, %%xmm0                       \n\t"
                         "vmovss %%xmm0, (%3)                               \n\t"
                         "vaddps 0x4(%3), %%xmm1, %%xmm1                    \n\t"
                         "vmovss %%xmm1, 0x4(%3)                            \n\t"
                         :
                         : "r"(bk), "r"(matrix), "r"(vector), "r"(result), "r"(lda)
                         : "%eax", "%rax", "%ecx", "%rdx", "%r9", "%r10", "%ymm0", "%ymm1", "%ymm4",
                         "%ymm6", "%ymm12", "%ymm13", "%xmm0", "%xmm1", "%xmm4", "%xmm6", "%xmm12",
                         "%xmm13", "memory");
}

void mvm_row_avx_1_32(U32 bk, U32 lda, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $5, %%ecx                                     \n\t"
                         "je .n1_k_loop_32_end                              \n\t"
                         ".align 16                                         \n\t"
                         ".n1_k_loop_32:                                    \n\t"

                         "prefetcht0 0x100(%1)                              \n\t"
                         "prefetcht0 0x140(%1)                              \n\t"

                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups 0x20(%2), %%ymm13                         \n\t"
                         "vmovups 0x40(%2), %%ymm14                         \n\t"
                         "vmovups 0x60(%2), %%ymm15                         \n\t"

                         "vmovups (%1), %%ymm4                              \n\t"
                         "vmovups 0x20(%1), %%ymm5                          \n\t"
                         "vmovups 0x40(%1), %%ymm6                          \n\t"
                         "vmovups 0x60(%1), %%ymm7                          \n\t"
                         "vfmadd231ps %%ymm12, %%ymm4, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm5, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm14, %%ymm6, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm15, %%ymm7, %%ymm0               \n\t"

                         "add $0x80, %1                                     \n\t"
                         "add $0x80, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .n1_k_loop_32                                  \n\t"
                         ".align 16                                         \n\t"
                         ".n1_k_loop_32_end:                                \n\t"
                         "mov %0, %%ecx                                     \n\t"
                         "and $0x1F, %%ecx                                  \n\t"
                         "cmp $0x10, %%ecx                                  \n\t"
                         "jl .n1_k_loop_remain_16_end                       \n\t"

                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups 0x20(%2), %%ymm13                         \n\t"

                         "vmovups (%1), %%ymm4                              \n\t"
                         "vmovups 0x20(%1), %%ymm5                          \n\t"
                         "vfmadd231ps %%ymm12, %%ymm4, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm13, %%ymm5, %%ymm0               \n\t"

                         "add $0x40, %1                                     \n\t"
                         "add $0x40, %2                                     \n\t"
                         "sub $0x10, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n1_k_loop_remain_16_end:                         \n\t"
                         "cmp $0x8, %%ecx                                   \n\t"
                         "jl .n1_k_loop_remain_8_end                        \n\t"
                         "vmovups (%2), %%ymm12                             \n\t"
                         "vmovups (%1), %%ymm4                              \n\t"
                         "vfmadd231ps %%ymm4, %%ymm12, %%ymm0               \n\t"
                         "add $0x20, %1                                     \n\t"
                         "add $0x20, %2                                     \n\t"
                         "sub $0x8, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n1_k_loop_remain_8_end:                          \n\t"
                         "vperm2f128 $0x1, %%ymm0, %%ymm0, %%ymm13          \n\t"
                         "vaddps %%ymm13, %%ymm0, %%ymm0                    \n\t"

                         "cmp $0x4, %%ecx                                   \n\t"
                         "jl .n1_k_loop_remain_4_end                        \n\t"
                         "vmovups (%2), %%xmm12                             \n\t"
                         "vmovups (%1), %%xmm4                              \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"
                         "add $0x10, %1                                     \n\t"
                         "add $0x10, %2                                     \n\t"
                         "sub $0x4, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n1_k_loop_remain_4_end:                          \n\t"
                         "vhaddps %%xmm0, %%xmm0, %%xmm0                    \n\t"

                         "cmp $0x2, %%ecx                                   \n\t"
                         "jl .n1_k_loop_remain_2_end                        \n\t"
                         "vxorps %%ymm12, %%ymm12, %%ymm12                  \n\t"
                         "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
                         "vmovsd (%2), %%xmm12                              \n\t"
                         "vmovsd (%1), %%xmm4                               \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"
                         "add $0x8, %1                                      \n\t"
                         "add $0x8, %2                                      \n\t"
                         "sub $0x2, %%ecx                                  \n\t"

                         ".align 16                                         \n\t"
                         ".n1_k_loop_remain_2_end:                          \n\t"
                         "vhaddps %%xmm0, %%xmm0, %%xmm0                    \n\t"
                         "and $1, %%ecx                                     \n\t"
                         "je .n1_k_loop_remain_1_end                        \n\t"
                         "vxorps %%ymm12, %%ymm12, %%ymm12                  \n\t"
                         "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
                         "vmovss (%2), %%xmm12                              \n\t"
                         "vmovss (%1), %%xmm4                               \n\t"
                         "vfmadd231ps %%xmm4, %%xmm12, %%xmm0               \n\t"

                         ".align 16                                         \n\t"
                         ".n1_k_loop_remain_1_end:                          \n\t"
                         "vaddps (%3), %%xmm0, %%xmm0                       \n\t"
                         "vmovss %%xmm0, (%3)                               \n\t"
                         :
                         : "r"(bk), "r"(matrix), "r"(vector), "r"(result), "r"(lda)
                         : "%eax", "%rax", "%ecx", "%rdx", "%r9", "%r10", "%ymm0", "%ymm4",
                         "%ymm12", "%ymm13", "%xmm0", "%xmm4", "%xmm12", "memory");
}

void mvm_row_fp32(U32 numRows, U32 numColumns, F32 *matrix, F32 *vector, F32 *result)
{
    // Actual layout is NK, and vector is K
    kernel_func kernel[3] = {mvm_row_avx_1_32, mvm_row_avx_2_32, mvm_row_avx_4_32};
    U32 unrollNSize[3] = {1, 2, 4};
    U32 blockNum = numRows / 4 + (numRows % 4 + 1) / 2;
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
    {
#endif
        U32 private_blockKSize = 0;
        for (U32 bk = 0; bk < numColumns; bk += private_blockKSize) {
            private_blockKSize = UNI_MIN(numColumns - bk, BOLCK_K_DIM);
#ifdef _USE_OPENMP
#pragma omp for
#endif
            for (U32 bIdx = 0; bIdx < blockNum; ++bIdx) {
                U32 bn = bIdx * 4 - ((bIdx * 4) > numRows) * 2;
                U32 blockNSize = UNI_MIN(numRows - bn, UNROLL_N);
                blockNSize = unrollNSize[blockNSize >> 1];
                kernel[blockNSize >> 1](private_blockKSize, numColumns,
                    matrix + bn * numColumns + bk, vector + bk, result + bn);
            }
        }
#ifdef _USE_OPENMP
    }
#endif
}
