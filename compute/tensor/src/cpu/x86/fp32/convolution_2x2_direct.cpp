// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"
#include "error.h"
#include "types.h"

#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/fp32/transform_functions_fp32.h"

#define UNROLL_W 3
#define UNROLL_OC_DIM 8
#define BLOCK_OC_DIM 32
#define BLOCK_IC_DIM 32
#define UNROLL_IC_BLOCK_DIM 8
#define BLOCK_HW_DIM 768
#define align_addr(addr, unit) (((uintptr_t)addr + unit - 1) / unit * unit)

typedef void (*kernel_func)(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep);

void avx2_conv_kernel_3x32(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("mov %3, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups (%1), %%ymm1                       \n\t"
                         "vmovups (%1), %%ymm2                       \n\t"
                         "vmovups 0x20(%1), %%ymm3                       \n\t"
                         "vmovups 0x20(%1), %%ymm4                   \n\t"
                         "vmovups 0x20(%1), %%ymm5                   \n\t"
                         "vmovups 0x40(%1), %%ymm6                   \n\t"
                         "vmovups 0x40(%1), %%ymm7                   \n\t"
                         "vmovups 0x40(%1), %%ymm8                   \n\t"
                         "vmovups 0x60(%1), %%ymm9                   \n\t"
                         "vmovups 0x60(%1), %%ymm10                 \n\t"
                         "vmovups 0x60(%1), %%ymm11                 \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %0, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm0                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm1                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm2                     \n\t"
                         "add %2, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm3                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm4                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm5                     \n\t"
                         "add %2, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm6                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm7                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm8                     \n\t"
                         "add %2, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm9                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm10                  \n\t"
                         "vmovups 0x40(%%rax), %%ymm11                  \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store)
                         : "%ecx", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "memory", "cc");

    __asm__ __volatile__(".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %5, %%ebx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "mov %4, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "4:                                                \n\t"

                         "vbroadcastss (%0), %%ymm12                        \n\t"
                         "vbroadcastss (%1), %%ymm13                 \n\t"
                         "vbroadcastss (%2), %%ymm14              \n\t"
                         "vmovups 0x0(%3), %%ymm15                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x20(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovups 0x40(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovups 0x60(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x4(%1), %%ymm13              \n\t"
                         "vbroadcastss 0x4(%2), %%ymm14           \n\t"
                         "vmovups 0x80(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0xA0(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovups 0xC0(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovups 0xE0(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x8(%1), %%ymm13              \n\t"
                         "vbroadcastss 0x8(%2), %%ymm14           \n\t"
                         "vmovups 0x100(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x120(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovups 0x140(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovups 0x160(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0xC(%1), %%ymm13              \n\t"
                         "vbroadcastss 0xC(%2), %%ymm14           \n\t"
                         "vmovups 0x180(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x1A0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovups 0x1C0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovups 0x1E0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x10(%1), %%ymm13             \n\t"
                         "vbroadcastss 0x10(%2), %%ymm14          \n\t"
                         "vmovups 0x200(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x220(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovups 0x240(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovups 0x260(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x14(%1), %%ymm13             \n\t"
                         "vbroadcastss 0x14(%2), %%ymm14          \n\t"
                         "vmovups 0x280(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x2A0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovups 0x2C0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovups 0x2E0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x18(%1), %%ymm13             \n\t"
                         "vbroadcastss 0x18(%2), %%ymm14          \n\t"
                         "vmovups 0x300(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x320(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovups 0x340(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovups 0x360(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm13             \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm14          \n\t"
                         "vmovups 0x380(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x3A0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovups 0x3C0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovups 0x3E0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "add %7, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "add %7, %2                                      \n\t"
                         "add $0x400, %3                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 4b                                             \n\t"

                         "add %6, %0                                     \n\t"
                         "add %6, %1                                     \n\t"
                         "add %6, %2                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %9, %0                                     \n\t"
                         "add %9, %1                                     \n\t"
                         "add %9, %2                                     \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 2b                                             \n\t"

                         :
                         : "r"(curI), "r"(in_1), "r"(in_2), "r"(curW), "r"(fw), "r"(fh),
                         "r"(I64(iStep)), "r"(I64(dw)), "a"(ic / 8), "r"(I64(fStep))
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__(
        // relu
        "and $0x6, %2                                      \n\t"
        "je 5f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
        "vmaxps %%ymm15, %%ymm2, %%ymm2                    \n\t"
        "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
        "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
        "vmaxps %%ymm15, %%ymm5, %%ymm5                    \n\t"
        "vmaxps %%ymm15, %%ymm6, %%ymm6                    \n\t"
        "vmaxps %%ymm15, %%ymm7, %%ymm7                    \n\t"
        "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"
        "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"
        "vmaxps %%ymm15, %%ymm10, %%ymm10                  \n\t"
        "vmaxps %%ymm15, %%ymm11, %%ymm11                  \n\t"

        // relu6
        "and $0x4, %2                                      \n\t"
        "je 5f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
        "vminps %%ymm12, %%ymm2, %%ymm2                    \n\t"
        "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
        "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
        "vminps %%ymm12, %%ymm5, %%ymm5                    \n\t"
        "vminps %%ymm12, %%ymm6, %%ymm6                    \n\t"
        "vminps %%ymm12, %%ymm7, %%ymm7                    \n\t"
        "vminps %%ymm12, %%ymm8, %%ymm8                    \n\t"
        "vminps %%ymm12, %%ymm9, %%ymm9                    \n\t"
        "vminps %%ymm12, %%ymm10, %%ymm10                    \n\t"
        "vminps %%ymm12, %%ymm11, %%ymm11                    \n\t"

        "5:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm1, 0x20(%0)                          \n\t"
        "vmovups %%ymm2, 0x40(%0)                          \n\t"
        "add %1, %0                                     \n\t"
        "vmovups %%ymm3, (%0)                              \n\t"
        "vmovups %%ymm4, 0x20(%0)                          \n\t"
        "vmovups %%ymm5, 0x40(%0)                          \n\t"
        "add %1, %0                                     \n\t"
        "vmovups %%ymm6, (%0)                              \n\t"
        "vmovups %%ymm7, 0x20(%0)                          \n\t"
        "vmovups %%ymm8, 0x40(%0)                          \n\t"
        "add %1, %0                                     \n\t"
        "vmovups %%ymm9, (%0)                              \n\t"
        "vmovups %%ymm10, 0x20(%0)                         \n\t"
        "vmovups %%ymm11, 0x40(%0)                         \n\t"
        : "+r"(curO)
        : "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "memory", "cc");
}

void avx2_conv_kernel_2x32(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups (%8), %%ymm1                       \n\t"
                         "vmovups 0x20(%8), %%ymm3                       \n\t"
                         "vmovups 0x20(%8), %%ymm4                   \n\t"
                         "vmovups 0x40(%8), %%ymm6                   \n\t"
                         "vmovups 0x40(%8), %%ymm7                   \n\t"
                         "vmovups 0x60(%8), %%ymm9                   \n\t"
                         "vmovups 0x60(%8), %%ymm10                 \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %1, %8                                     \n\t"
                         "vmovups (%8), %%ymm0                     \n\t"
                         "vmovups 0x20(%8), %%ymm1                     \n\t"
                         "add %4, %8                                     \n\t"
                         "vmovups (%8), %%ymm3                     \n\t"
                         "vmovups 0x20(%8), %%ymm4                     \n\t"
                         "add %4, %8                                     \n\t"
                         "vmovups (%8), %%ymm6                     \n\t"
                         "vmovups 0x20(%8), %%ymm7                     \n\t"
                         "add %4, %8                                     \n\t"
                         "vmovups (%8), %%ymm9                     \n\t"
                         "vmovups 0x20(%8), %%ymm10                  \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "mov %5, %%ebx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "vbroadcastss (%0), %%ymm12                        \n\t"
                         "vbroadcastss (%10), %%ymm13                 \n\t"
                         "vmovups 0x0(%2), %%ymm15                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x20(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vmovups 0x40(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vmovups 0x60(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

                         "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x4(%10), %%ymm13              \n\t"
                         "vmovups 0x80(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0xA0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vmovups 0xC0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vmovups 0xE0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

                         "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x8(%10), %%ymm13              \n\t"
                         "vmovups 0x100(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x120(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vmovups 0x140(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vmovups 0x160(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

                         "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0xC(%10), %%ymm13              \n\t"
                         "vmovups 0x180(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x1A0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vmovups 0x1C0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vmovups 0x1E0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

                         "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x10(%10), %%ymm13             \n\t"
                         "vmovups 0x200(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x220(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vmovups 0x240(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vmovups 0x260(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

                         "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x14(%10), %%ymm13             \n\t"
                         "vmovups 0x280(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x2A0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vmovups 0x2C0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vmovups 0x2E0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

                         "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x18(%10), %%ymm13             \n\t"
                         "vmovups 0x300(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x320(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vmovups 0x340(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vmovups 0x360(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

                         "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x1C(%10), %%ymm13             \n\t"
                         "vmovups 0x380(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x3A0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vmovups 0x3C0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vmovups 0x3E0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

                         "add %9, %0                                      \n\t"
                         "add %9, %10                                      \n\t"
                         "add $0x400, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %6, %0                                     \n\t"
                         "add %6, %10                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 2b                                             \n\t"

                         "add %12, %0                                     \n\t"
                         "add %12, %10                                      \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
                         "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
                         "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
                         "vmaxps %%ymm15, %%ymm6, %%ymm6                    \n\t"
                         "vmaxps %%ymm15, %%ymm7, %%ymm7                    \n\t"
                         "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"
                         "vmaxps %%ymm15, %%ymm10, %%ymm10                  \n\t"

                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
                         "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
                         "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
                         "vminps %%ymm12, %%ymm6, %%ymm6                    \n\t"
                         "vminps %%ymm12, %%ymm7, %%ymm7                    \n\t"
                         "vminps %%ymm12, %%ymm9, %%ymm9                    \n\t"
                         "vminps %%ymm12, %%ymm10, %%ymm10                    \n\t"

                         "4:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm3, (%1)                              \n\t"
                         "vmovups %%ymm4, 0x20(%1)                          \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm6, (%1)                              \n\t"
                         "vmovups %%ymm7, 0x20(%1)                          \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm9, (%1)                              \n\t"
                         "vmovups %%ymm10, 0x20(%1)                         \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(curW), "r"(fw), "r"(I64(oStep)), "r"(fh),
                         "r"(I64(iStep)), "r"(store), "r"(curB), "r"(I64(dw)), "r"(in_1),
                         "a"(ic / 8), "r"(I64(fStep))
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm3", "%ymm4", "%ymm6", "%ymm7",
                         "%ymm9", "%ymm10", "%ymm12", "%ymm13", "%ymm15", "memory");
}

void avx2_conv_kernel_1x32(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__(
        "mov %7, %%ecx                                  \n\t"
        "and $0x1, %%ecx                                  \n\t"
        "jne 0f                                             \n\t"
        "vmovups (%8), %%ymm0                       \n\t"
        "vmovups 0x20(%8), %%ymm3                       \n\t"
        "vmovups 0x40(%8), %%ymm6                   \n\t"
        "vmovups 0x60(%8), %%ymm9                   \n\t"
        "jmp 1f                                             \n\t"

        ".align 16                                         \n\t"
        "0:                                      \n\t"
        "mov %1, %8                                     \n\t"
        "vmovups (%8), %%ymm0                     \n\t"
        "add %4, %8                                     \n\t"
        "vmovups (%8), %%ymm3                     \n\t"
        "add %4, %8                                     \n\t"
        "vmovups (%8), %%ymm6                     \n\t"
        "add %4, %8                                     \n\t"
        "vmovups (%8), %%ymm9                     \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"

        "mov %5, %%ebx                                     \n\t"
        ".align 16                                         \n\t"
        "2:                                                \n\t"

        "mov %3, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "3:                                                \n\t"

        "vbroadcastss (%0), %%ymm12                        \n\t"
        "vmovups 0x0(%2), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x20(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vmovups 0x40(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vmovups 0x60(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
        "vmovups 0x80(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0xA0(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vmovups 0xC0(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vmovups 0xE0(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
        "vmovups 0x100(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x120(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vmovups 0x140(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vmovups 0x160(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
        "vmovups 0x180(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x1A0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vmovups 0x1C0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vmovups 0x1E0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
        "vmovups 0x200(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x220(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vmovups 0x240(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vmovups 0x260(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
        "vmovups 0x280(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x2A0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vmovups 0x2C0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vmovups 0x2E0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
        "vmovups 0x300(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x320(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vmovups 0x340(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vmovups 0x360(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
        "vmovups 0x380(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x3A0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vmovups 0x3C0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vmovups 0x3E0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"

        "add %9, %0                                     \n\t"
        "add $0x400, %2                                    \n\t"
        "dec %%ecx                                         \n\t"
        "jg 3b                                             \n\t"

        "add %6, %0                                     \n\t"
        "sub $1, %%ebx                                     \n\t"
        "jg 2b                                             \n\t"

        "add %11, %0                                     \n\t"
        "sub $1, %%eax                                     \n\t"
        "jg 1b                                             \n\t"

        // relu
        "and $0x6, %7                                      \n\t"
        "je 4f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
        "vmaxps %%ymm15, %%ymm6, %%ymm6                    \n\t"
        "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"

        // relu6
        "and $0x4, %7                                      \n\t"
        "je 4f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
        "vminps %%ymm12, %%ymm6, %%ymm6                    \n\t"
        "vminps %%ymm12, %%ymm9, %%ymm9                    \n\t"

        "4:                                                \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "add %4, %1                                     \n\t"
        "vmovups %%ymm3, (%1)                              \n\t"
        "add %4, %1                                     \n\t"
        "vmovups %%ymm6, (%1)                              \n\t"
        "add %4, %1                                     \n\t"
        "vmovups %%ymm9, (%1)                              \n\t"
        :
        : "r"(curI), "r"(curO), "r"(curW), "r"(fw), "r"(I64(oStep)), "r"(fh), "r"(I64(iStep)),
        "r"(store), "r"(curB), "r"(I64(dw)), "a"(ic / 8), "r"(I64(fStep))
        : "%ecx", "%ebx", "%ymm0", "%ymm3", "%ymm6", "%ymm9", "%ymm12", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_3x16(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("mov %3, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups (%1), %%ymm1                       \n\t"
                         "vmovups (%1), %%ymm2                       \n\t"
                         "vmovups 0x20(%1), %%ymm3                       \n\t"
                         "vmovups 0x20(%1), %%ymm4                   \n\t"
                         "vmovups 0x20(%1), %%ymm5                   \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %0, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm0                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm1                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm2                     \n\t"
                         "add %2, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm3                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm4                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm5                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store)
                         : "%ecx", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "memory", "cc");

    __asm__ __volatile__(".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %5, %%ebx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "mov %4, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "4:                                                \n\t"

                         "vbroadcastss (%0), %%ymm12                        \n\t"
                         "vbroadcastss (%1), %%ymm13                 \n\t"
                         "vbroadcastss (%2), %%ymm14              \n\t"
                         "vmovups 0x0(%2), %%ymm15                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x20(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

                         "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x4(%1), %%ymm13              \n\t"
                         "vbroadcastss 0x4(%2), %%ymm14           \n\t"
                         "vmovups 0x40(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x60(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

                         "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x8(%1), %%ymm13              \n\t"
                         "vbroadcastss 0x8(%2), %%ymm14           \n\t"
                         "vmovups 0x80(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0xA0(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

                         "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0xC(%1), %%ymm13              \n\t"
                         "vbroadcastss 0xC(%2), %%ymm14           \n\t"
                         "vmovups 0xC0(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0xE0(%3), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

                         "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x10(%1), %%ymm13             \n\t"
                         "vbroadcastss 0x10(%2), %%ymm14          \n\t"
                         "vmovups 0x100(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x120(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

                         "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x14(%1), %%ymm13             \n\t"
                         "vbroadcastss 0x14(%2), %%ymm14          \n\t"
                         "vmovups 0x140(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x160(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

                         "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x18(%1), %%ymm13             \n\t"
                         "vbroadcastss 0x18(%2), %%ymm14          \n\t"
                         "vmovups 0x180(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x1A0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

                         "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm13             \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm14          \n\t"
                         "vmovups 0x1C0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovups 0x1E0(%3), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

                         "add %7, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "add %7, %2                                      \n\t"
                         "add $0x200, %3                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 4b                                             \n\t"

                         "add %6, %0                                     \n\t"
                         "add %6, %1                                     \n\t"
                         "add %6, %2                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %9, %0                                     \n\t"
                         "add %9, %1                                     \n\t"
                         "add %9, %2                                     \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 2b                                             \n\t"

                         :
                         : "r"(curI), "r"(in_1), "r"(in_2), "r"(curW), "r"(fw), "r"(fh),
                         "r"(I64(iStep)), "r"(I64(dw)), "a"(ic / 8), "r"(I64(fStep))
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__(
        // relu
        "and $0x6, %2                                      \n\t"
        "je 5f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
        "vmaxps %%ymm15, %%ymm2, %%ymm2                    \n\t"
        "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
        "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
        "vmaxps %%ymm15, %%ymm5, %%ymm5                    \n\t"

        // relu6
        "and $0x4, %2                                      \n\t"
        "je 5f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
        "vminps %%ymm12, %%ymm2, %%ymm2                    \n\t"
        "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
        "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
        "vminps %%ymm12, %%ymm5, %%ymm5                    \n\t"

        "5:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm1, 0x20(%0)                          \n\t"
        "vmovups %%ymm2, 0x40(%0)                          \n\t"
        "add %1, %0                                     \n\t"
        "vmovups %%ymm3, (%0)                              \n\t"
        "vmovups %%ymm4, 0x20(%0)                          \n\t"
        "vmovups %%ymm5, 0x40(%0)                          \n\t"
        : "+r"(curO)
        : "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "memory", "cc");
}

void avx2_conv_kernel_2x16(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups (%8), %%ymm1                       \n\t"
                         "vmovups 0x20(%8), %%ymm3                       \n\t"
                         "vmovups 0x20(%8), %%ymm4                   \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %1, %8                                     \n\t"
                         "vmovups (%8), %%ymm0                     \n\t"
                         "vmovups 0x20(%8), %%ymm1                     \n\t"
                         "add %4, %8                                     \n\t"
                         "vmovups (%8), %%ymm3                     \n\t"
                         "vmovups 0x20(%8), %%ymm4                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "mov %5, %%ebx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "vbroadcastss (%0), %%ymm12                        \n\t"
                         "vbroadcastss (%10), %%ymm13                 \n\t"
                         "vmovups 0x0(%2), %%ymm15                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x20(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x4(%10), %%ymm13              \n\t"
                         "vmovups 0x40(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x60(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x8(%10), %%ymm13              \n\t"
                         "vmovups 0x80(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0xA0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0xC(%10), %%ymm13              \n\t"
                         "vmovups 0xC0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0xE0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x10(%10), %%ymm13             \n\t"
                         "vmovups 0x100(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x120(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x14(%10), %%ymm13             \n\t"
                         "vmovups 0x140(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x160(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x18(%10), %%ymm13             \n\t"
                         "vmovups 0x180(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x1A0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x1C(%10), %%ymm13             \n\t"
                         "vmovups 0x1C0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vmovups 0x1E0(%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "add %9, %0                                      \n\t"
                         "add %9, %10                                      \n\t"
                         "add $0x200, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %6, %0                                     \n\t"
                         "add %6, %10                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 2b                                             \n\t"

                         "add %12, %0                                     \n\t"
                         "add %12, %10                                      \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
                         "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
                         "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"

                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
                         "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
                         "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"

                         "4:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm3, (%1)                              \n\t"
                         "vmovups %%ymm4, 0x20(%1)                          \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(curW), "r"(fw), "r"(I64(oStep)), "r"(fh),
                         "r"(I64(iStep)), "r"(store), "r"(curB), "r"(I64(dw)), "r"(in_1),
                         "a"(ic / 8), "r"(I64(fStep))
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm3", "%ymm4", "%ymm6", "%ymm7",
                         "%ymm9", "%ymm10", "%ymm12", "%ymm13", "%ymm15", "memory");
}

void avx2_conv_kernel_1x16(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__(
        "mov %7, %%ecx                                  \n\t"
        "and $0x1, %%ecx                                  \n\t"
        "jne 0f                                             \n\t"
        "vmovups (%8), %%ymm0                       \n\t"
        "vmovups 0x20(%8), %%ymm3                       \n\t"
        "jmp 1f                                             \n\t"

        ".align 16                                         \n\t"
        "0:                                      \n\t"
        "mov %1, %8                                     \n\t"
        "vmovups (%8), %%ymm0                     \n\t"
        "add %4, %8                                     \n\t"
        "vmovups (%8), %%ymm3                     \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"

        "mov %5, %%ebx                                     \n\t"
        ".align 16                                         \n\t"
        "2:                                                \n\t"

        "mov %3, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "3:                                                \n\t"

        "vbroadcastss (%0), %%ymm12                        \n\t"
        "vmovups 0x0(%2), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x20(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
        "vmovups 0x40(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x60(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
        "vmovups 0x80(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0xA0(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
        "vmovups 0xC0(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0xE0(%2), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
        "vmovups 0x100(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x120(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
        "vmovups 0x140(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x160(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
        "vmovups 0x180(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x1A0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
        "vmovups 0x1C0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x1E0(%2), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "add %9, %0                                     \n\t"
        "add $0x200, %2                                    \n\t"
        "dec %%ecx                                         \n\t"
        "jg 3b                                             \n\t"

        "add %6, %0                                     \n\t"
        "sub $1, %%ebx                                     \n\t"
        "jg 2b                                             \n\t"

        "add %11, %0                                     \n\t"
        "sub $1, %%eax                                     \n\t"
        "jg 1b                                             \n\t"

        // relu
        "and $0x6, %7                                      \n\t"
        "je 4f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"

        // relu6
        "and $0x4, %7                                      \n\t"
        "je 4f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"

        "4:                                                \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "add %4, %1                                     \n\t"
        "vmovups %%ymm3, (%1)                              \n\t"
        :
        : "r"(curI), "r"(curO), "r"(curW), "r"(fw), "r"(I64(oStep)), "r"(fh), "r"(I64(iStep)),
        "r"(store), "r"(curB), "r"(I64(dw)), "a"(ic / 8), "r"(I64(fStep))
        : "%ecx", "%ebx", "%ymm0", "%ymm3", "%ymm6", "%ymm9", "%ymm12", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_3x8(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups (%8), %%ymm1                       \n\t"
                         "vmovups (%8), %%ymm2                       \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%1), %%ymm0                     \n\t"
                         "vmovups 0x20(%1), %%ymm1                     \n\t"
                         "vmovups 0x40(%1), %%ymm2                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "mov %5, %%ebx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "vbroadcastss (%0), %%ymm12                        \n\t"
                         "vbroadcastss (%10), %%ymm13                 \n\t"
                         "vbroadcastss (%11), %%ymm14              \n\t"
                         "vmovups 0x0(%2), %%ymm15                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

                         "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x4(%10), %%ymm13              \n\t"
                         "vbroadcastss 0x4(%11), %%ymm14           \n\t"
                         "vmovups 0x20(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

                         "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x8(%10), %%ymm13              \n\t"
                         "vbroadcastss 0x8(%11), %%ymm14           \n\t"
                         "vmovups 0x40(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

                         "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0xC(%10), %%ymm13              \n\t"
                         "vbroadcastss 0xC(%11), %%ymm14           \n\t"
                         "vmovups 0x60(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

                         "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x10(%10), %%ymm13             \n\t"
                         "vbroadcastss 0x10(%11), %%ymm14          \n\t"
                         "vmovups 0x80(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

                         "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x14(%10), %%ymm13             \n\t"
                         "vbroadcastss 0x14(%11), %%ymm14          \n\t"
                         "vmovups 0xA0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

                         "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x18(%10), %%ymm13             \n\t"
                         "vbroadcastss 0x18(%11), %%ymm14          \n\t"
                         "vmovups 0xC0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

                         "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x1C(%10), %%ymm13             \n\t"
                         "vbroadcastss 0x1C(%11), %%ymm14          \n\t"
                         "vmovups 0xE0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

                         "add %9, %0                                      \n\t"
                         "add %9, %10                                      \n\t"
                         "add %9, %11                                      \n\t"
                         "add $0x100, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %6, %0                                     \n\t"
                         "add %6, %10                                     \n\t"
                         "add %6, %11                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 2b                                             \n\t"

                         "add %12, %0                                     \n\t"
                         "add %12, %10                                     \n\t"
                         "add %12, %11                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
                         "vmaxps %%ymm15, %%ymm2, %%ymm2                    \n\t"

                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
                         "vminps %%ymm12, %%ymm2, %%ymm2                    \n\t"

                         "4:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         "vmovups %%ymm2, 0x40(%1)                          \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(curW), "r"(fw), "a"(ic / 8), "r"(fh),
                         "r"(I64(iStep)), "r"(store), "r"(curB), "r"(I64(dw)), "r"(in_1), "r"(in_2),
                         "r"(I64(fStep))
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm12", "%ymm13", "%ymm14",
                         "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_2x8(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups (%8), %%ymm1                       \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%1), %%ymm0                     \n\t"
                         "vmovups 0x20(%1), %%ymm1                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "vbroadcastss (%0), %%ymm12                        \n\t"
                         "vbroadcastss (%10), %%ymm13                 \n\t"
                         "vmovups 0x0(%2), %%ymm15                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x4(%10), %%ymm13              \n\t"
                         "vmovups 0x20(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0x8(%10), %%ymm13              \n\t"
                         "vmovups 0x40(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
                         "vbroadcastss 0xC(%10), %%ymm13              \n\t"
                         "vmovups 0x60(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x10(%10), %%ymm13             \n\t"
                         "vmovups 0x80(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x14(%10), %%ymm13             \n\t"
                         "vmovups 0xA0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x18(%10), %%ymm13             \n\t"
                         "vmovups 0xC0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
                         "vbroadcastss 0x1C(%10), %%ymm13             \n\t"
                         "vmovups 0xE0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

                         "add %9, %0                                      \n\t"
                         "add %9, %10                                      \n\t"
                         "add $0x100, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %6, %0                                     \n\t"
                         "add %6, %10                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 2b                                             \n\t"

                         "add %11, %0                                     \n\t"
                         "add %11, %10                                     \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"

                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"

                         "4:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(curW), "r"(fw), "a"(ic / 8), "b"(fh),
                         "r"(I64(iStep)), "r"(store), "r"(curB), "r"(I64(dw)), "r"(in_1),
                         "r"(I64(fStep))
                         : "%ecx", "%ymm0", "%ymm1", "%ymm12", "%ymm13", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_1x8(F32 *curI,
    const F32 *curW,
    F32 *curO,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 store,
    const F32 *curB,
    U32 dw,
    F32 *in_1,
    F32 *in_2,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%1), %%ymm0                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "vbroadcastss (%0), %%ymm12                        \n\t"
                         "vmovups 0x0(%2), %%ymm15                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vbroadcastss 0x4(%0), %%ymm12                     \n\t"
                         "vmovups 0x20(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vbroadcastss 0x8(%0), %%ymm12                     \n\t"
                         "vmovups 0x40(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vbroadcastss 0xC(%0), %%ymm12                     \n\t"
                         "vmovups 0x60(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vbroadcastss 0x10(%0), %%ymm12                    \n\t"
                         "vmovups 0x80(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vbroadcastss 0x14(%0), %%ymm12                    \n\t"
                         "vmovups 0xA0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vbroadcastss 0x18(%0), %%ymm12                    \n\t"
                         "vmovups 0xC0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vbroadcastss 0x1C(%0), %%ymm12                    \n\t"
                         "vmovups 0xE0(%2), %%ymm15                         \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "add %9, %0                                     \n\t"
                         "add $0x100, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %6, %0                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 2b                                             \n\t"

                         "add %11, %0                                     \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"

                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                    \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"

                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(curW), "r"(fw), "r"(I64(oStep)), "b"(fh),
                         "r"(I64(iStep)), "r"(store), "r"(curB), "r"(I64(dw)), "a"(ic / 8),
                         "r"(I64(fStep))
                         : "%ecx", "%ymm0", "%ymm12", "%ymm15", "memory", "cc");
}

EE convolution_2x2_direct(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc)
{
    UNUSED(biasDesc);
    UNUSED(tmpBytes);

    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.padding_top;
    U32 paddingB = convParamSpec.padding_bottom;
    U32 paddingL = convParamSpec.padding_left;
    U32 paddingR = convParamSpec.padding_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;

    if ((fdf != DF_NCHWCxN32) || (idf != DF_NCHWC8) || (ic % 8 != 0)) {
        CHECK_STATUS(NOT_MATCH);
    }

    F32 *curI, *curO, *calI, *calO;
    const F32 *curW, *curB, *calW;
    F32 *ftmp = (F32 *)align_addr(tmp, 32);
    filterArray = (F32 *)align_addr(filterArray, 32);

    U32 icAlignSize = 8;
    U32 icPadding = (ic + icAlignSize - 1) / icAlignSize * icAlignSize;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;

    U32 oStep = oh * ow * UNROLL_OC_DIM * 4;
    U32 iStep = (iw_pad - fw * dilateW + (dilateH - 1) * iw_pad) * UNROLL_IC_BLOCK_DIM * 4;
    U32 fStep = ((ih_pad - fh) * iw_pad) * UNROLL_IC_BLOCK_DIM * 4;
    U32 sw = strideW * UNROLL_IC_BLOCK_DIM * 4;
    U32 dw = dilateW * UNROLL_IC_BLOCK_DIM * 4;
    I32 wSize = 0, store = 0, ocSize = 0, icSize = 0, hwSize = 0;
    I32 ih_idx = 0;
    kernel_func kernel[3][3] = {{avx2_conv_kernel_1x8, avx2_conv_kernel_2x8, avx2_conv_kernel_3x8},
        {avx2_conv_kernel_1x16, avx2_conv_kernel_2x16, avx2_conv_kernel_3x16},
        {avx2_conv_kernel_1x32, avx2_conv_kernel_2x32, avx2_conv_kernel_3x32}};
    U32 ocblocks[3] = {8, 16, 32};

    I32 ohow = oh * ow;

    for (U32 n = 0; n < in; ++n) {
        if (idf == DF_NCHWC8 && paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            ftmp = inArray;
        } else {
            PaddingNCHWC8(inArray, ftmp, inputDesc, convParamSpec);
        }
        store = 0;
        for (U32 icbb = 0; icbb < icPadding; icbb += icSize) {
            icSize = UNI_MIN(BLOCK_IC_DIM, icPadding - icbb);
            store |= (icbb > 0);
            if (icbb == icPadding - icSize) {
                store |= U32(activationDesc.mode) << 1;
            }
            for (I32 hw = 0; hw < ohow; hw += hwSize) {
                hwSize = UNI_MIN(ohow - hw, BLOCK_HW_DIM);
                for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                    curB = biasArray + ocb;
                    ocSize = UNI_MIN(BLOCK_OC_DIM, oc - ocb);
                    ocSize = ocblocks[ocSize >> 4];
                    calW = filterArray + ocb * icPadding * fh * fw + ocSize * icbb * fh * fw;
                    curI = ftmp + icbb * ih_pad * iw_pad;

                    for (I32 ihw = hw; ihw < hw + hwSize; ihw += wSize) {
                        wSize = UNI_MIN(hw + hwSize - ihw, UNROLL_W);
                        U32 in_h_0 = ihw / ow * strideH;
                        U32 in_w_0 = ihw % ow * strideW;
                        U32 in_h_1 = (ihw + 1) / ow * strideH;
                        U32 in_w_1 = (ihw + 1) % ow * strideW;
                        U32 in_h_2 = (ihw + 2) / ow * strideH;
                        U32 in_w_2 = (ihw + 2) % ow * strideW;
                        F32 *out_ptr = outArray + (n * oc + ocb) * ohow + ihw * 8;
                        F32 *in_0 = curI + in_h_0 * iw_pad * 8 + in_w_0 * 8;
                        F32 *in_1 = curI + in_h_1 * iw_pad * 8 + in_w_1 * 8;
                        F32 *in_2 = curI + in_h_2 * iw_pad * 8 + in_w_2 * 8;

                        kernel[ocSize >> 4][wSize - 1](in_0, calW, out_ptr, fw, fh, oStep, iStep,
                            store, curB, dw, in_1, in_2, icSize, fStep);
                    }
                }
            }
        }
        inArray += ic * ih * iw;
        outArray += oc * oh * ow;
    }
    return SUCCESS;
}
