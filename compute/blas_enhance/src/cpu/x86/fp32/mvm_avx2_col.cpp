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

#define UNROLL_K 4

typedef void (*kernel_func)(U32 N, F32 *matrix, F32 *vector, F32 *result);

void mvm_col_avx2_4_32(U32 N, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("mov %0, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "mov %1, %%rdx                                     \n\t"
                         "add %%rax, %%rdx                                  \n\t"
                         "mov %%rdx, %%r9                                   \n\t"
                         "add %%rax, %%r9                                   \n\t"
                         "mov %%r9, %%r10                                   \n\t"
                         "add %%rax, %%r10                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "cmp $0x20, %%ecx                                  \n\t"
                         "jl .n_loop_32_end                                 \n\t"
                         ".align 16                                         \n\t"
                         ".n_loop_32:                                       \n\t"
                         "prefetcht0 0x100(%3)                              \n\t"
                         "prefetcht0 0x140(%3)                              \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups 0x20(%3), %%ymm1                          \n\t"
                         "vmovups 0x40(%3), %%ymm2                          \n\t"
                         "vmovups 0x60(%3), %%ymm3                          \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"
                         "prefetcht0 0x180(%1)                              \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vmovups 0x20(%1), %%ymm13                         \n\t"
                         "vmovups 0x40(%1), %%ymm14                         \n\t"
                         "vmovups 0x60(%1), %%ymm11                         \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"

                         "prefetcht0 0x100(%%rdx)                           \n\t"
                         "prefetcht0 0x140(%%rdx)                           \n\t"
                         "vmovups (%%rdx), %%ymm12                          \n\t"
                         "vmovups 0x20(%%rdx), %%ymm13                      \n\t"
                         "vmovups 0x40(%%rdx), %%ymm14                      \n\t"
                         "vmovups 0x60(%%rdx), %%ymm11                      \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"

                         "prefetcht0 0x100(%%r9)                            \n\t"
                         "prefetcht0 0x140(%%r9)                            \n\t"
                         "vmovups (%%r9), %%ymm12                           \n\t"
                         "vmovups 0x20(%%r9), %%ymm13                       \n\t"
                         "vmovups 0x40(%%r9), %%ymm14                       \n\t"
                         "vmovups 0x60(%%r9), %%ymm11                       \n\t"
                         "vbroadcastss 0x8(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"

                         "prefetcht0 0x100(%%r10)                           \n\t"
                         "prefetcht0 0x140(%%r10)                           \n\t"
                         "vmovups (%%r10), %%ymm12                          \n\t"
                         "vmovups 0x20(%%r10), %%ymm13                      \n\t"
                         "vmovups 0x40(%%r10), %%ymm14                      \n\t"
                         "vmovups 0x60(%%r10), %%ymm11                      \n\t"
                         "vbroadcastss 0xC(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"

                         "vmovups %%ymm0, (%3)                              \n\t"
                         "vmovups %%ymm1, 0x20(%3)                          \n\t"
                         "vmovups %%ymm2, 0x40(%3)                          \n\t"
                         "vmovups %%ymm3, 0x60(%3)                          \n\t"

                         "add $0x80, %1                                     \n\t"
                         "add $0x80, %%rdx                                  \n\t"
                         "add $0x80, %%r9                                   \n\t"
                         "add $0x80, %%r10                                  \n\t"
                         "add $0x80, %3                                     \n\t"

                         "sub $0x20, %%ecx                                  \n\t"
                         "cmp $0x20, %%ecx                                  \n\t"
                         "jge .n_loop_32                                    \n\t"

                         ".align 16                                         \n\t"
                         ".n_loop_32_end:                                   \n\t"
                         "cmp $0x10, %%ecx                                  \n\t"
                         "jl .n_loop_remain_16_end                          \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups 0x20(%3), %%ymm1                          \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vmovups 0x20(%1), %%ymm13                         \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vmovups (%%rdx), %%ymm12                          \n\t"
                         "vmovups 0x20(%%rdx), %%ymm13                      \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vmovups (%%r9), %%ymm12                           \n\t"
                         "vmovups 0x20(%%r9), %%ymm13                       \n\t"
                         "vbroadcastss 0x8(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vmovups (%%r10), %%ymm12                          \n\t"
                         "vmovups 0x20(%%r10), %%ymm13                      \n\t"
                         "vbroadcastss 0xC(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1, 0x20(%3)                          \n\t"

                         "add $0x40, %1                                     \n\t"
                         "add $0x40, %%rdx                                  \n\t"
                         "add $0x40, %%r9                                   \n\t"
                         "add $0x40, %%r10                                  \n\t"
                         "add $0x40, %3                                     \n\t"
                         "sub $0x10, %%ecx                                  \n\t"

                         ".n_loop_remain_16_end:                            \n\t"
                         "cmp $0x8, %%ecx                                   \n\t"
                         "jl .n_loop_remain_8_end                           \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovups (%%rdx), %%ymm12                          \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovups (%%r9), %%ymm12                           \n\t"
                         "vbroadcastss 0x8(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovups (%%r10), %%ymm12                          \n\t"
                         "vbroadcastss 0xC(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovups %%ymm0,  (%3)                             \n\t"

                         "add $0x20, %1                                     \n\t"
                         "add $0x20, %%rdx                                  \n\t"
                         "add $0x20, %%r9                                   \n\t"
                         "add $0x20, %%r10                                  \n\t"
                         "add $0x20, %3                                     \n\t"
                         "sub $0x8, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".n_loop_remain_8_end:                             \n\t"
                         "cmp $0x4, %%ecx                                   \n\t"
                         "jl .n_loop_remain_4_end                           \n\t"
                         "vmovups (%3), %%xmm0                              \n\t"
                         "vmovups (%1), %%xmm12                             \n\t"
                         "vbroadcastss 0x0(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovups (%%rdx), %%xmm12                          \n\t"
                         "vbroadcastss 0x4(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovups (%%r9), %%xmm12                           \n\t"
                         "vbroadcastss 0x8(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovups (%%r10), %%xmm12                          \n\t"
                         "vbroadcastss 0xC(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovups %%xmm0,  (%3)                             \n\t"

                         "add $0x10, %1                                     \n\t"
                         "add $0x10, %%rdx                                  \n\t"
                         "add $0x10, %%r9                                   \n\t"
                         "add $0x10, %%r10                                  \n\t"
                         "add $0x10, %3                                     \n\t"
                         "sub $0x4, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".n_loop_remain_4_end:                             \n\t"
                         "cmp $0x2, %%ecx                                   \n\t"
                         "jl .n_loop_remain_2_end                           \n\t"
                         "vmovsd (%3), %%xmm0                               \n\t"
                         "vmovsd (%1), %%xmm12                              \n\t"
                         "vbroadcastss 0x0(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovsd (%%rdx), %%xmm12                           \n\t"
                         "vbroadcastss 0x4(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovsd (%%r9), %%xmm12                            \n\t"
                         "vbroadcastss 0x8(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovsd (%%r10), %%xmm12                           \n\t"
                         "vbroadcastss 0xC(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovsd %%xmm0,  (%3)                              \n\t"

                         "add $0x8, %1                                      \n\t"
                         "add $0x8, %%rdx                                   \n\t"
                         "add $0x8, %%r9                                    \n\t"
                         "add $0x8, %%r10                                   \n\t"
                         "add $0x8, %3                                      \n\t"
                         "sub $0x2, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".n_loop_remain_2_end:                             \n\t"
                         "and $0x1, %%ecx                                   \n\t"
                         "je .n_loop_remain_1_end                           \n\t"
                         "vmovss (%3), %%xmm0                               \n\t"
                         "vmovss (%1), %%xmm12                              \n\t"
                         "vmovss 0x0(%2), %%xmm15                           \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovss (%%rdx), %%xmm12                           \n\t"
                         "vmovss 0x4(%2), %%xmm15                           \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovss (%%r9), %%xmm12                            \n\t"
                         "vmovss 0x8(%2), %%xmm15                           \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovss (%%r10), %%xmm12                           \n\t"
                         "vmovss 0xC(%2), %%xmm15                           \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovss %%xmm0,  (%3)                              \n\t"

                         ".align 16                                         \n\t"
                         ".n_loop_remain_1_end:                             \n\t"
                         :
                         : "r"(N), "r"(matrix), "r"(vector), "r"(result)
                         : "%eax", "%rax", "%ecx", "%rdx", "%r9", "%r10", "%ymm0", "%ymm1", "%ymm2",
                         "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%xmm0", "%xmm12", "%xmm15",
                         "memory");
}

void mvm_col_avx2_2_32(U32 N, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("mov %0, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "mov %1, %%rdx                                     \n\t"
                         "add %%rax, %%rdx                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "cmp $0x20, %%ecx                                  \n\t"
                         "jl .k2_n_loop_32_end                              \n\t"
                         ".align 16                                         \n\t"
                         ".k2_n_loop_32:                                    \n\t"
                         "prefetcht0 0x100(%3)                              \n\t"
                         "prefetcht0 0x140(%3)                              \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups 0x20(%3), %%ymm1                          \n\t"
                         "vmovups 0x40(%3), %%ymm2                          \n\t"
                         "vmovups 0x60(%3), %%ymm3                          \n\t"

                         "prefetcht0 0x100(%1)                              \n\t"
                         "prefetcht0 0x140(%1)                              \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vmovups 0x20(%1), %%ymm13                         \n\t"
                         "vmovups 0x40(%1), %%ymm14                         \n\t"
                         "vmovups 0x60(%1), %%ymm11                         \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"

                         "prefetcht0 0x100(%%rdx)                           \n\t"
                         "prefetcht0 0x140(%%rdx)                           \n\t"
                         "vmovups (%%rdx), %%ymm12                          \n\t"
                         "vmovups 0x20(%%rdx), %%ymm13                      \n\t"
                         "vmovups 0x40(%%rdx), %%ymm14                      \n\t"
                         "vmovups 0x60(%%rdx), %%ymm11                      \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"

                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1, 0x20(%3)                          \n\t"
                         "vmovups %%ymm2, 0x40(%3)                          \n\t"
                         "vmovups %%ymm3, 0x60(%3)                          \n\t"

                         "add $0x80, %1                                     \n\t"
                         "add $0x80, %%rdx                                  \n\t"
                         "add $0x80, %3                                     \n\t"

                         "sub $0x20, %%ecx                                  \n\t"
                         "cmp $0x20, %%ecx                                  \n\t"
                         "jge .k2_n_loop_32                                 \n\t"

                         ".align 16                                         \n\t"
                         ".k2_n_loop_32_end:                                \n\t"
                         "cmp $0x10, %%ecx                                  \n\t"
                         "jl .k2_n_loop_remain_16_end                       \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups 0x20(%3), %%ymm1                          \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vmovups 0x20(%1), %%ymm13                         \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vmovups (%%rdx), %%ymm12                          \n\t"
                         "vmovups 0x20(%%rdx), %%ymm13                      \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1, 0x20(%3)                          \n\t"

                         "add $0x40, %1                                     \n\t"
                         "add $0x40, %%rdx                                  \n\t"
                         "add $0x40, %3                                     \n\t"
                         "sub $0x10, %%ecx                                  \n\t"

                         ".k2_n_loop_remain_16_end:                         \n\t"
                         "cmp $0x8, %%ecx                                   \n\t"
                         "jl .k2_n_loop_remain_8_end                        \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovups (%%rdx), %%ymm12                          \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovups %%ymm0,  (%3)                             \n\t"

                         "add $0x20, %1                                     \n\t"
                         "add $0x20, %%rdx                                  \n\t"
                         "add $0x20, %3                                     \n\t"
                         "sub $0x8, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k2_n_loop_remain_8_end:                          \n\t"
                         "cmp $0x4, %%ecx                                   \n\t"
                         "jl .k2_n_loop_remain_4_end                        \n\t"
                         "vmovups (%3), %%xmm0                              \n\t"
                         "vmovups (%1), %%xmm12                             \n\t"
                         "vbroadcastss 0x0(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovups (%%rdx), %%xmm12                          \n\t"
                         "vbroadcastss 0x4(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"
                         "vmovups %%xmm0,  (%3)                             \n\t"

                         "add $0x10, %1                                     \n\t"
                         "add $0x10, %%rdx                                  \n\t"
                         "add $0x10, %3                                     \n\t"
                         "sub $0x4, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k2_n_loop_remain_4_end:                          \n\t"
                         "cmp $0x2, %%ecx                                   \n\t"
                         "jl .k2_n_loop_remain_2_end                        \n\t"
                         "vmovsd (%3), %%xmm0                               \n\t"
                         "vmovsd (%1), %%xmm12                              \n\t"
                         "vbroadcastss 0x0(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovsd (%%rdx), %%xmm12                           \n\t"
                         "vbroadcastss 0x4(%2), %%xmm15                     \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"
                         "vmovsd %%xmm0,  (%3)                              \n\t"

                         "add $0x8, %1                                      \n\t"
                         "add $0x8, %%rdx                                   \n\t"
                         "add $0x8, %3                                      \n\t"
                         "sub $0x2, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k2_n_loop_remain_2_end:                          \n\t"
                         "and $0x1, %%ecx                                   \n\t"
                         "je .k2_n_loop_remain_1_end                        \n\t"
                         "vmovss (%3), %%xmm0                               \n\t"
                         "vmovss (%1), %%xmm12                              \n\t"
                         "vmovss 0x0(%2), %%xmm15                           \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovss (%%rdx), %%xmm12                           \n\t"
                         "vmovss 0x4(%2), %%xmm15                           \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"
                         "vmovss %%xmm0,  (%3)                              \n\t"

                         ".align 16                                         \n\t"
                         ".k2_n_loop_remain_1_end:                          \n\t"
                         :
                         : "r"(N), "r"(matrix), "r"(vector), "r"(result)
                         : "%eax", "%rax", "%ecx", "%rdx", "%ymm0", "%ymm1", "%ymm2", "%ymm12",
                         "%ymm13", "%ymm14", "%ymm15", "%xmm0", "%xmm12", "%xmm15", "memory");
}

void mvm_col_avx2_1_32(U32 N, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("mov %0, %%ecx                                     \n\t"
                         "cmp $0x20, %%ecx                                  \n\t"
                         "jl .k1_n_loop_32_end                              \n\t"
                         ".align 16                                         \n\t"
                         ".k1_n_loop_32:                                    \n\t"
                         "prefetcht0 0x100(%3)                              \n\t"
                         "prefetcht0 0x140(%3)                              \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups 0x20(%3), %%ymm1                          \n\t"
                         "vmovups 0x40(%3), %%ymm2                          \n\t"
                         "vmovups 0x60(%3), %%ymm3                          \n\t"

                         "prefetcht0 0x100(%1)                              \n\t"
                         "prefetcht0 0x140(%1)                              \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vmovups 0x20(%1), %%ymm13                         \n\t"
                         "vmovups 0x40(%1), %%ymm14                         \n\t"
                         "vmovups 0x60(%1), %%ymm11                         \n\t"
                         "vbroadcastss (%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"

                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1, 0x20(%3)                          \n\t"
                         "vmovups %%ymm2, 0x40(%3)                          \n\t"
                         "vmovups %%ymm3, 0x60(%3)                          \n\t"

                         "add $0x80, %1                                     \n\t"
                         "add $0x80, %3                                     \n\t"

                         "sub $0x20, %%ecx                                  \n\t"
                         "cmp $0x20, %%ecx                                  \n\t"
                         "jge .k1_n_loop_32                                 \n\t"

                         ".align 16                                         \n\t"
                         ".k1_n_loop_32_end:                                \n\t"
                         "cmp $0x10, %%ecx                                  \n\t"
                         "jl .k1_n_loop_remain_16_end                       \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups 0x20(%3), %%ymm1                          \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vmovups 0x20(%1), %%ymm13                         \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1, 0x20(%3)                          \n\t"

                         "add $0x40, %1                                     \n\t"
                         "add $0x40, %3                                     \n\t"
                         "sub $0x10, %%ecx                                  \n\t"

                         ".k1_n_loop_remain_16_end:                         \n\t"
                         "cmp $0x8, %%ecx                                   \n\t"
                         "jl .k1_n_loop_remain_8_end                        \n\t"
                         "vmovups (%3), %%ymm0                              \n\t"
                         "vmovups (%1), %%ymm12                             \n\t"
                         "vbroadcastss (%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovups %%ymm0,  (%3)                             \n\t"

                         "add $0x20, %1                                     \n\t"
                         "add $0x20, %3                                     \n\t"
                         "sub $0x8, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k1_n_loop_remain_8_end:                          \n\t"
                         "cmp $0x4, %%ecx                                   \n\t"
                         "jl .k1_n_loop_remain_4_end                        \n\t"
                         "vmovups (%3), %%xmm0                              \n\t"
                         "vmovups (%1), %%xmm12                             \n\t"
                         "vbroadcastss (%2), %%xmm15                        \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovups %%xmm0,  (%3)                             \n\t"

                         "add $0x10, %1                                     \n\t"
                         "add $0x10, %3                                     \n\t"
                         "sub $0x4, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k1_n_loop_remain_4_end:                          \n\t"
                         "cmp $0x2, %%ecx                                   \n\t"
                         "jl .k1_n_loop_remain_2_end                        \n\t"
                         "vmovsd (%3), %%xmm0                               \n\t"
                         "vmovsd (%1), %%xmm12                              \n\t"
                         "vbroadcastss (%2), %%xmm15                        \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"

                         "vmovsd %%xmm0,  (%3)                              \n\t"

                         "add $0x8, %1                                      \n\t"
                         "add $0x8, %3                                      \n\t"
                         "sub $0x2, %%ecx                                   \n\t"

                         ".align 16                                         \n\t"
                         ".k1_n_loop_remain_2_end:                          \n\t"
                         "and $0x1, %%ecx                                   \n\t"
                         "je .k1_n_loop_remain_1_end                        \n\t"
                         "vmovss (%3), %%xmm0                               \n\t"
                         "vmovss (%1), %%xmm12                              \n\t"
                         "vmovss (%2), %%xmm15                              \n\t"
                         "vfmadd231ps %%xmm15, %%xmm12, %%xmm0              \n\t"
                         "vmovss %%xmm0,  (%3)                              \n\t"

                         ".align 16                                         \n\t"
                         ".k1_n_loop_remain_1_end:                          \n\t"
                         :
                         : "r"(N), "r"(matrix), "r"(vector), "r"(result)
                         : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "%xmm0", "%xmm12", "%xmm15", "memory");
}

void mvm_col_fp32(U32 numRows, U32 numColumns, F32 *matrix, F32 *vector, F32 *result)
{
    // Actual layout is KN, and vector is K
    U32 blockKSize = 0;
    kernel_func kernel[3] = {mvm_col_avx2_1_32, mvm_col_avx2_2_32, mvm_col_avx2_4_32};
    U32 unrollKSize[3] = {1, 2, 4};
    for (U32 bk = 0; bk < numColumns; bk += blockKSize) {
        blockKSize = UNI_MIN(numColumns - bk, 4);
        blockKSize = unrollKSize[blockKSize >> 1];
        kernel[blockKSize >> 1](numRows, matrix + bk * numRows, vector + bk, result);
    }
}
