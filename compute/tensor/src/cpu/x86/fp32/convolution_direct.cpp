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

#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/fp32/transform_functions_fp32.h"
#include "cpu/x86/fp32/convolution_functions.h"

#define BLOCK_IC_DIM 16
#define BLOCK_HW_DIM 1024
// #define BLOCK_OC_DIM 32

typedef void (*kernelFunc)(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep);

void Avx2ConvKernel3x32(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep)
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
                         "mov %3, %%ecx                                  \n\t"
                         "and $0x2, %%ecx                                  \n\t"
                         "je 1f                                             \n\t"
                         "mov %4, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm0, %%ymm0                     \n\t"
                         "vaddps 0x20(%%rax), %%ymm1, %%ymm1                     \n\t"
                         "vaddps 0x40(%%rax), %%ymm2, %%ymm2                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm3, %%ymm3                     \n\t"
                         "vaddps 0x20(%%rax), %%ymm4, %%ymm4                     \n\t"
                         "vaddps 0x40(%%rax), %%ymm5, %%ymm5                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm6, %%ymm6                     \n\t"
                         "vaddps 0x20(%%rax), %%ymm7, %%ymm7                     \n\t"
                         "vaddps 0x40(%%rax), %%ymm8, %%ymm8                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm9, %%ymm9                     \n\t"
                         "vaddps 0x20(%%rax), %%ymm10, %%ymm10                  \n\t"
                         "vaddps 0x40(%%rax), %%ymm11, %%ymm11                  \n\t"
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
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store), "r"(curE)
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
                         : "r"(curI[0]), "r"(curI[1]), "r"(curI[2]), "r"(curW), "r"(fw), "r"(fh),
                         "r"(I64(iStep)), "r"(I64(dw)), "a"(ic / 8), "r"(I64(fStep))
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__(  // relu
        "and $0xC, %2                                      \n\t"
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
        "and $0x8, %2                                      \n\t"
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

void Avx2ConvKernel1x32(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep)
{
    __asm__ __volatile__("mov %3, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups 0x20(%1), %%ymm3                       \n\t"
                         "vmovups 0x40(%1), %%ymm6                   \n\t"
                         "vmovups 0x60(%1), %%ymm9                   \n\t"
                         "mov %3, %%ecx                                  \n\t"
                         "and $0x2, %%ecx                                  \n\t"
                         "je 1f                                             \n\t"
                         "mov %4, %%rax                                     \n\t"
                         "vaddps (%%rax), %%ymm0, %%ymm0                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm3, %%ymm3                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm6, %%ymm6                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm9, %%ymm9                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %0, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm0                     \n\t"
                         "add %2, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm3                     \n\t"
                         "add %2, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm6                     \n\t"
                         "add %2, %%rax                                     \n\t"
                         "vmovups (%%rax), %%ymm9                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store), "r"(curE)
                         : "%ecx", "%rax", "%ymm0", "%ymm3", "%ymm6", "%ymm9", "memory", "cc");

    __asm__ __volatile__(
        ".align 16                                         \n\t"
        "2:                                                \n\t"

        "mov %5, %%ebx                                     \n\t"
        ".align 16                                         \n\t"
        "3:                                                \n\t"

        "mov %3, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "4:                                                \n\t"

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

        "add %8, %0                                     \n\t"
        "add $0x400, %2                                    \n\t"
        "dec %%ecx                                         \n\t"
        "jg 4b                                             \n\t"

        "add %6, %0                                     \n\t"
        "dec %%ebx                                     \n\t"
        "jg 3b                                             \n\t"

        "add %10, %0                                     \n\t"
        "dec %%eax                                     \n\t"
        "jg 2b                                             \n\t"

        // relu
        "and $0xC, %7                                      \n\t"
        "je 5f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
        "vmaxps %%ymm15, %%ymm6, %%ymm6                    \n\t"
        "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"

        // relu6
        "and $0x8, %7                                      \n\t"
        "je 5f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
        "vminps %%ymm12, %%ymm6, %%ymm6                    \n\t"
        "vminps %%ymm12, %%ymm9, %%ymm9                    \n\t"

        "5:                                                \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "add %4, %1                                     \n\t"
        "vmovups %%ymm3, (%1)                              \n\t"
        "add %4, %1                                     \n\t"
        "vmovups %%ymm6, (%1)                              \n\t"
        "add %4, %1                                     \n\t"
        "vmovups %%ymm9, (%1)                              \n\t"
        :
        : "r"(curI[0]), "r"(curO), "r"(curW), "r"(fw), "r"(I64(oStep)), "r"(fh), "r"(I64(iStep)),
        "r"(store), "r"(I64(dw)), "a"(ic / 8), "r"(I64(fStep))
        : "%ecx", "%ebx", "%ymm0", "%ymm3", "%ymm6", "%ymm9", "%ymm12", "%ymm15", "memory", "cc");
}

void Avx2ConvKernel4x24(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep)
{
    __asm__ __volatile__("mov %3, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups (%1), %%ymm1                       \n\t"
                         "vmovups (%1), %%ymm2                       \n\t"
                         "vmovups (%1), %%ymm3                       \n\t"
                         "vmovups 0x20(%1), %%ymm4                       \n\t"
                         "vmovups 0x20(%1), %%ymm5                   \n\t"
                         "vmovups 0x20(%1), %%ymm6                   \n\t"
                         "vmovups 0x20(%1), %%ymm7                   \n\t"
                         "vmovups 0x40(%1), %%ymm8                   \n\t"
                         "vmovups 0x40(%1), %%ymm9                   \n\t"
                         "vmovups 0x40(%1), %%ymm10                   \n\t"
                         "vmovups 0x40(%1), %%ymm11                   \n\t"
                         "mov %3, %%ecx                                  \n\t"
                         "and $0x2, %%ecx                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%4), %%ymm0, %%ymm0                     \n\t"
                         "vaddps 0x20(%4), %%ymm1, %%ymm1                     \n\t"
                         "vaddps 0x40(%4), %%ymm2, %%ymm2                     \n\t"
                         "vaddps 0x60(%4), %%ymm3, %%ymm3                     \n\t"
                         "vaddps (%4, %2), %%ymm4, %%ymm4                     \n\t"
                         "vaddps 0x20(%4, %2), %%ymm5, %%ymm5                     \n\t"
                         "vaddps 0x40(%4, %2), %%ymm6, %%ymm6                     \n\t"
                         "vaddps 0x60(%4, %2), %%ymm7, %%ymm7                     \n\t"
                         "vaddps (%4, %2, 2), %%ymm8, %%ymm8                     \n\t"
                         "vaddps 0x20(%4, %2, 2), %%ymm9, %%ymm9                     \n\t"
                         "vaddps 0x40(%4, %2, 2), %%ymm10, %%ymm10                     \n\t"
                         "vaddps 0x60(%4, %2, 2), %%ymm11, %%ymm11                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups 0x20(%0), %%ymm1                     \n\t"
                         "vmovups 0x40(%0), %%ymm2                     \n\t"
                         "vmovups 0x60(%0), %%ymm3                     \n\t"
                         "vmovups (%0, %2), %%ymm4                     \n\t"
                         "vmovups 0x20(%0, %2), %%ymm5                     \n\t"
                         "vmovups 0x40(%0, %2), %%ymm6                     \n\t"
                         "vmovups 0x60(%0, %2), %%ymm7                     \n\t"
                         "vmovups (%0, %2, 2), %%ymm8                     \n\t"
                         "vmovups 0x20(%0, %2, 2), %%ymm9                     \n\t"
                         "vmovups 0x40(%0, %2, 2), %%ymm10                     \n\t"
                         "vmovups 0x60(%0, %2, 2), %%ymm11                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store), "r"(curE)
                         : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                         "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "memory", "cc");

    __asm__ __volatile__(".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %9, %%ebx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "mov %8, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "4:                                                \n\t"

                         "vmovaps 0x0(%4), %%ymm12                          \n\t"
                         "vmovaps 0x20(%4), %%ymm13                         \n\t"
                         "vmovaps 0x40(%4), %%ymm14                         \n\t"
                         "vbroadcastss (%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss (%1), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss (%2), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss (%3), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x60(%4), %%ymm12                          \n\t"
                         "vmovaps 0x80(%4), %%ymm13                         \n\t"
                         "vmovaps 0xA0(%4), %%ymm14                         \n\t"
                         "vbroadcastss 0x4(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x4(%1), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0x4(%3), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0xC0(%4), %%ymm12                          \n\t"
                         "vmovaps 0xE0(%4), %%ymm13                         \n\t"
                         "vmovaps 0x100(%4), %%ymm14                         \n\t"
                         "vbroadcastss 0x8(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x8(%1), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x8(%2), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0x8(%3), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x120(%4), %%ymm12                          \n\t"
                         "vmovaps 0x140(%4), %%ymm13                         \n\t"
                         "vmovaps 0x160(%4), %%ymm14                         \n\t"
                         "vbroadcastss 0xC(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0xC(%1), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0xC(%2), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0xC(%3), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x180(%4), %%ymm12                          \n\t"
                         "vmovaps 0x1A0(%4), %%ymm13                         \n\t"
                         "vmovaps 0x1C0(%4), %%ymm14                         \n\t"
                         "vbroadcastss 0x10(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x10(%1), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x10(%2), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0x10(%3), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x1E0(%4), %%ymm12                          \n\t"
                         "vmovaps 0x200(%4), %%ymm13                         \n\t"
                         "vmovaps 0x220(%4), %%ymm14                         \n\t"
                         "vbroadcastss 0x14(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x14(%1), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x14(%2), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0x14(%3), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x240(%4), %%ymm12                          \n\t"
                         "vmovaps 0x260(%4), %%ymm13                         \n\t"
                         "vmovaps 0x280(%4), %%ymm14                         \n\t"
                         "vbroadcastss 0x18(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x18(%1), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x18(%2), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0x18(%3), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x2A0(%4), %%ymm12                          \n\t"
                         "vmovaps 0x2C0(%4), %%ymm13                         \n\t"
                         "vmovaps 0x2E0(%4), %%ymm14                         \n\t"
                         "vbroadcastss 0x1C(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0x1C(%3), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "add %5, %0                                      \n\t"
                         "add %5, %1                                      \n\t"
                         "add %5, %2                                      \n\t"
                         "add %5, %3                                      \n\t"
                         "add $0x300, %4                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 4b                                             \n\t"

                         "add %6, %0                                     \n\t"
                         "add %6, %1                                     \n\t"
                         "add %6, %2                                     \n\t"
                         "add %6, %3                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %7, %0                                     \n\t"
                         "add %7, %1                                     \n\t"
                         "add %7, %2                                     \n\t"
                         "add %7, %3                                     \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 2b                                             \n\t"
                         :
                         : "r"(curI[0]), "r"(curI[1]), "r"(curI[2]), "r"(curI[3]), "r"(curW),
                         "r"(I64(dw)), "r"(I64(iStep)), "r"(I64(fStep)), "r"(fw), "r"(fh),
                         "a"(ic / 8)
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__(  // relu
        "and $0xC, %2                                      \n\t"
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
        "and $0x8, %2                                      \n\t"
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
        "vmovups %%ymm3, 0x60(%0)                          \n\t"
        "vmovups %%ymm4, (%0, %1)                              \n\t"
        "vmovups %%ymm5, 0x20(%0, %1)                          \n\t"
        "vmovups %%ymm6, 0x40(%0, %1)                          \n\t"
        "vmovups %%ymm7, 0x60(%0, %1)                          \n\t"
        "vmovups %%ymm8, (%0, %1, 2)                              \n\t"
        "vmovups %%ymm9, 0x20(%0, %1, 2)                          \n\t"
        "vmovups %%ymm10, 0x40(%0, %1, 2)                          \n\t"
        "vmovups %%ymm11, 0x60(%0, %1, 2)                          \n\t"
        : "+r"(curO)
        : "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "memory", "cc");
}

void Avx2ConvKernel1x24(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep)
{
    __asm__ __volatile__("mov %3, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups 0x20(%1), %%ymm4                       \n\t"
                         "vmovups 0x40(%1), %%ymm8                   \n\t"
                         "mov %3, %%ecx                                  \n\t"
                         "and $0x2, %%ecx                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%4), %%ymm0, %%ymm0                     \n\t"
                         "vaddps (%4, %2), %%ymm4, %%ymm4                     \n\t"
                         "vaddps (%4, %2, 2), %%ymm8, %%ymm8                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups (%0, %2), %%ymm4                     \n\t"
                         "vmovups (%0, %2, 2), %%ymm8                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store), "r"(curE)
                         : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                         "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "memory", "cc");

    __asm__ __volatile__(".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %6, %%ebx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "mov %5, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "4:                                                \n\t"

                         "vmovaps 0x0(%1), %%ymm12                          \n\t"
                         "vmovaps 0x20(%1), %%ymm13                         \n\t"
                         "vmovaps 0x40(%1), %%ymm14                         \n\t"
                         "vbroadcastss (%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x60(%1), %%ymm12                          \n\t"
                         "vmovaps 0x80(%1), %%ymm13                         \n\t"
                         "vmovaps 0xA0(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x4(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0xC0(%1), %%ymm12                          \n\t"
                         "vmovaps 0xE0(%1), %%ymm13                         \n\t"
                         "vmovaps 0x100(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x8(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x120(%1), %%ymm12                          \n\t"
                         "vmovaps 0x140(%1), %%ymm13                         \n\t"
                         "vmovaps 0x160(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0xC(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x180(%1), %%ymm12                          \n\t"
                         "vmovaps 0x1A0(%1), %%ymm13                         \n\t"
                         "vmovaps 0x1C0(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x10(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x1E0(%1), %%ymm12                          \n\t"
                         "vmovaps 0x200(%1), %%ymm13                         \n\t"
                         "vmovaps 0x220(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x14(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x240(%1), %%ymm12                          \n\t"
                         "vmovaps 0x260(%1), %%ymm13                         \n\t"
                         "vmovaps 0x280(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x18(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x2A0(%1), %%ymm12                          \n\t"
                         "vmovaps 0x2C0(%1), %%ymm13                         \n\t"
                         "vmovaps 0x2E0(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x1C(%0), %%ymm15              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "add %2, %0                                      \n\t"
                         "add $0x300, %1                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 4b                                             \n\t"

                         "add %3, %0                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %4, %0                                     \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 2b                                             \n\t"
                         :
                         : "r"(curI[0]), "r"(curW), "r"(I64(dw)), "r"(I64(iStep)), "r"(I64(fStep)),
                         "r"(fw), "r"(fh), "a"(ic / 8)
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__(  // relu
        "and $0xC, %2                                      \n\t"
        "je 5f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
        "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"

        // relu6
        "and $0x8, %2                                      \n\t"
        "je 5f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
        "vminps %%ymm12, %%ymm8, %%ymm8                    \n\t"

        "5:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm4, (%0, %1)                          \n\t"
        "vmovups %%ymm8, (%0, %1, 2)                          \n\t"
        : "+r"(curO)
        : "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "memory", "cc");
}

void Avx2ConvKernel6x16(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep)
{
    __asm__ __volatile__("mov %3, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups (%1), %%ymm1                       \n\t"
                         "vmovups (%1), %%ymm2                       \n\t"
                         "vmovups (%1), %%ymm3                       \n\t"
                         "vmovups (%1), %%ymm4                       \n\t"
                         "vmovups (%1), %%ymm5                       \n\t"
                         "vmovups 0x20(%1), %%ymm6                       \n\t"
                         "vmovups 0x20(%1), %%ymm7                   \n\t"
                         "vmovups 0x20(%1), %%ymm8                   \n\t"
                         "vmovups 0x20(%1), %%ymm9                   \n\t"
                         "vmovups 0x20(%1), %%ymm10                   \n\t"
                         "vmovups 0x20(%1), %%ymm11                   \n\t"
                         "mov %3, %%ecx                                  \n\t"
                         "and $0x2, %%ecx                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%4), %%ymm0, %%ymm0                     \n\t"
                         "vaddps 0x20(%4), %%ymm1, %%ymm1                     \n\t"
                         "vaddps 0x40(%4), %%ymm2, %%ymm2                     \n\t"
                         "vaddps 0x60(%4), %%ymm3, %%ymm3                     \n\t"
                         "vaddps 0x80(%4), %%ymm4, %%ymm4                     \n\t"
                         "vaddps 0xA0(%4), %%ymm5, %%ymm5                     \n\t"
                         "vaddps (%4, %2), %%ymm6, %%ymm6                     \n\t"
                         "vaddps 0x20(%4, %2), %%ymm7, %%ymm7                     \n\t"
                         "vaddps 0x40(%4, %2), %%ymm8, %%ymm8                     \n\t"
                         "vaddps 0x60(%4, %2), %%ymm9, %%ymm9                     \n\t"
                         "vaddps 0x80(%4, %2), %%ymm10, %%ymm10                     \n\t"
                         "vaddps 0xA0(%4, %2), %%ymm11, %%ymm11                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups 0x20(%0), %%ymm1                     \n\t"
                         "vmovups 0x40(%0), %%ymm2                     \n\t"
                         "vmovups 0x60(%0), %%ymm3                     \n\t"
                         "vmovups 0x80(%0), %%ymm4                     \n\t"
                         "vmovups 0xA0(%0), %%ymm5                     \n\t"
                         "vmovups (%0, %2), %%ymm6                     \n\t"
                         "vmovups 0x20(%0, %2), %%ymm7                     \n\t"
                         "vmovups 0x40(%0, %2), %%ymm8                     \n\t"
                         "vmovups 0x60(%0, %2), %%ymm9                     \n\t"
                         "vmovups 0x80(%0, %2), %%ymm10                     \n\t"
                         "vmovups 0xA0(%0, %2), %%ymm11                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store), "r"(curE)
                         : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                         "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "memory", "cc");

    __asm__ __volatile__(".align 16                                         \n\t"
                         "2:                                                \n\t"

                         "mov %11, %%ebx                                     \n\t"
                         ".align 16                                         \n\t"
                         "3:                                                \n\t"

                         "mov %10, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "4:                                                \n\t"

                         "vbroadcastss (%0), %%ymm12                        \n\t"
                         "vbroadcastss (%1), %%ymm13                 \n\t"
                         "vmovups 0x0(%6), %%ymm15                          \n\t"
                         "vmovups 0x20(%6), %%ymm14                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss (%2), %%ymm12                        \n\t"
                         "vbroadcastss (%3), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vbroadcastss (%4), %%ymm12                        \n\t"
                         "vbroadcastss (%5), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11              \n\t"

                         "vbroadcastss 0x4(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0x4(%1), %%ymm13                 \n\t"
                         "vmovups 0x40(%6), %%ymm15                          \n\t"
                         "vmovups 0x60(%6), %%ymm14                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x4(%2), %%ymm12                        \n\t"
                         "vbroadcastss 0x4(%3), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vbroadcastss 0x4(%4), %%ymm12                        \n\t"
                         "vbroadcastss 0x4(%5), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11              \n\t"

                         "vbroadcastss 0x8(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0x8(%1), %%ymm13                 \n\t"
                         "vmovups 0x80(%6), %%ymm15                          \n\t"
                         "vmovups 0xA0(%6), %%ymm14                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x8(%2), %%ymm12                        \n\t"
                         "vbroadcastss 0x8(%3), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vbroadcastss 0x8(%4), %%ymm12                        \n\t"
                         "vbroadcastss 0x8(%5), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11              \n\t"

                         "vbroadcastss 0xC(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0xC(%1), %%ymm13                 \n\t"
                         "vmovups 0xC0(%6), %%ymm15                          \n\t"
                         "vmovups 0xE0(%6), %%ymm14                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0xC(%2), %%ymm12                        \n\t"
                         "vbroadcastss 0xC(%3), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vbroadcastss 0xC(%4), %%ymm12                        \n\t"
                         "vbroadcastss 0xC(%5), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11              \n\t"

                         "vbroadcastss 0x10(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0x10(%1), %%ymm13                 \n\t"
                         "vmovups 0x100(%6), %%ymm15                          \n\t"
                         "vmovups 0x120(%6), %%ymm14                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x10(%2), %%ymm12                        \n\t"
                         "vbroadcastss 0x10(%3), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vbroadcastss 0x10(%4), %%ymm12                        \n\t"
                         "vbroadcastss 0x10(%5), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11              \n\t"

                         "vbroadcastss 0x14(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0x14(%1), %%ymm13                 \n\t"
                         "vmovups 0x140(%6), %%ymm15                          \n\t"
                         "vmovups 0x160(%6), %%ymm14                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x14(%2), %%ymm12                        \n\t"
                         "vbroadcastss 0x14(%3), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vbroadcastss 0x14(%4), %%ymm12                        \n\t"
                         "vbroadcastss 0x14(%5), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11              \n\t"

                         "vbroadcastss 0x18(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0x18(%1), %%ymm13                 \n\t"
                         "vmovups 0x180(%6), %%ymm15                          \n\t"
                         "vmovups 0x1A0(%6), %%ymm14                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x18(%2), %%ymm12                        \n\t"
                         "vbroadcastss 0x18(%3), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vbroadcastss 0x18(%4), %%ymm12                        \n\t"
                         "vbroadcastss 0x18(%5), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11              \n\t"

                         "vbroadcastss 0x1C(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm13                 \n\t"
                         "vmovups 0x1C0(%6), %%ymm15                          \n\t"
                         "vmovups 0x1E0(%6), %%ymm14                          \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm12                        \n\t"
                         "vbroadcastss 0x1C(%3), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9              \n\t"
                         "vbroadcastss 0x1C(%4), %%ymm12                        \n\t"
                         "vbroadcastss 0x1C(%5), %%ymm13                 \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11              \n\t"

                         "add %7, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "add %7, %2                                      \n\t"
                         "add %7, %3                                      \n\t"
                         "add %7, %4                                      \n\t"
                         "add %7, %5                                      \n\t"
                         "add $0x200, %6                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 4b                                             \n\t"

                         "add %8, %0                                     \n\t"
                         "add %8, %1                                     \n\t"
                         "add %8, %2                                     \n\t"
                         "add %8, %3                                     \n\t"
                         "add %8, %4                                     \n\t"
                         "add %8, %5                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 3b                                             \n\t"

                         "add %9, %0                                     \n\t"
                         "add %9, %1                                     \n\t"
                         "add %9, %2                                     \n\t"
                         "add %9, %3                                     \n\t"
                         "add %9, %4                                     \n\t"
                         "add %9, %5                                     \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 2b                                             \n\t"

                         :
                         : "r"(curI[0]), "r"(curI[1]), "r"(curI[2]), "r"(curI[3]), "r"(curI[4]),
                         "r"(curI[5]), "r"(curW), "r"(I64(dw)), "r"(I64(iStep)), "r"(I64(fStep)),
                         "r"(fw), "r"(fh), "a"(ic / 8)
                         : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__(  // relu
        "and $0xC, %2                                      \n\t"
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
        "vmaxps %%ymm15, %%ymm10, %%ymm10                    \n\t"
        "vmaxps %%ymm15, %%ymm11, %%ymm11                    \n\t"

        // relu6
        "and $0x8, %2                                      \n\t"
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
        "vmovups %%ymm3, 0x60(%0)                          \n\t"
        "vmovups %%ymm4, 0x80(%0)                          \n\t"
        "vmovups %%ymm5, 0xA0(%0)                          \n\t"
        "vmovups %%ymm6, (%0, %1)                              \n\t"
        "vmovups %%ymm7, 0x20(%0, %1)                          \n\t"
        "vmovups %%ymm8, 0x40(%0, %1)                          \n\t"
        "vmovups %%ymm9, 0x60(%0, %1)                          \n\t"
        "vmovups %%ymm10, 0x80(%0, %1)                          \n\t"
        "vmovups %%ymm11, 0xA0(%0, %1)                          \n\t"
        :
        : "r"(curO), "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm15", "memory", "cc");
}

void Avx2ConvKernel1x16(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep)
{
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups 0x20(%8), %%ymm3                       \n\t"
                         "mov %7, %%ecx                                  \n\t"
                         "and $0x2, %%ecx                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%12), %%ymm0, %%ymm0                     \n\t"
                         "vaddps (%12, %4), %%ymm3, %%ymm3                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%1), %%ymm0                     \n\t"
                         "vmovups (%1, %4), %%ymm3                     \n\t"

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
                         "dec %%ebx                                     \n\t"
                         "jg 2b                                             \n\t"

                         "add %11, %0                                     \n\t"
                         "dec %%eax                                     \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"

                         // relu6
                         "and $0x8, %7                                      \n\t"
                         "je 4f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"

                         "4:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm3, (%1, %4)                              \n\t"
                         :
                         : "r"(curI[0]), "r"(curO), "r"(curW), "r"(fw), "r"(I64(oStep)), "r"(fh),
                         "r"(I64(iStep)), "r"(store), "r"(curB), "r"(I64(dw)), "a"(ic / 8),
                         "r"(I64(fStep)), "r"(curE)
                         : "%ecx", "%ebx", "%ymm0", "%ymm3", "%ymm12", "%ymm15", "memory", "cc");
}

void Avx2ConvKernel8x8(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep)
{
    __asm__ __volatile__("mov %2, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups (%1), %%ymm1                       \n\t"
                         "vmovups (%1), %%ymm2                       \n\t"
                         "vmovups (%1), %%ymm3                       \n\t"
                         "vmovups (%1), %%ymm4                       \n\t"
                         "vmovups (%1), %%ymm5                       \n\t"
                         "vmovups (%1), %%ymm6                       \n\t"
                         "vmovups (%1), %%ymm7                       \n\t"
                         "mov %2, %%ecx                                  \n\t"
                         "and $0x2, %%ecx                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%3), %%ymm0, %%ymm0                     \n\t"
                         "vaddps 0x20(%3), %%ymm1, %%ymm1                     \n\t"
                         "vaddps 0x40(%3), %%ymm2, %%ymm2                     \n\t"
                         "vaddps 0x60(%3), %%ymm3, %%ymm3                     \n\t"
                         "vaddps 0x80(%3), %%ymm4, %%ymm4                     \n\t"
                         "vaddps 0xA0(%3), %%ymm5, %%ymm5                     \n\t"
                         "vaddps 0xC0(%3), %%ymm6, %%ymm6                     \n\t"
                         "vaddps 0xE0(%3), %%ymm7, %%ymm7                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups 0x20(%0), %%ymm1                     \n\t"
                         "vmovups 0x40(%0), %%ymm2                     \n\t"
                         "vmovups 0x60(%0), %%ymm3                     \n\t"
                         "vmovups 0x80(%0), %%ymm4                     \n\t"
                         "vmovups 0xA0(%0), %%ymm5                     \n\t"
                         "vmovups 0xC0(%0), %%ymm6                     \n\t"
                         "vmovups 0xE0(%0), %%ymm7                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(store), "r"(curE)
                         : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                         "%ymm7", "memory", "cc");

    __asm__ __volatile__(
        ".align 16                                         \n\t"
        "2:                                                \n\t"

        "pushq %%rbx                                         \n\t"
        ".align 16                                         \n\t"
        "3:                                                \n\t"

        "pushq %%rcx                                         \n\t"
        ".align 16                                         \n\t"
        "4:                                                \n\t"
        "vbroadcastss (%0), %%ymm8                        \n\t"
        "vmovups 0x0(%8), %%ymm12                          \n\t"
        "vbroadcastss (%1), %%ymm9                 \n\t"
        "vbroadcastss (%2), %%ymm10              \n\t"
        "vbroadcastss (%3), %%ymm11              \n\t"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm12, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm12, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm12, %%ymm11, %%ymm3              \n\t"
        "vbroadcastss (%4), %%ymm8              \n\t"
        "vbroadcastss (%5), %%ymm9              \n\t"
        "vbroadcastss (%6), %%ymm10              \n\t"
        "vbroadcastss (%7), %%ymm11              \n\t"
        "vmovups 0x20(%8), %%ymm13                          \n\t"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm4              \n\t"
        "vfmadd231ps %%ymm12, %%ymm9, %%ymm5              \n\t"
        "vfmadd231ps %%ymm12, %%ymm10, %%ymm6              \n\t"
        "vfmadd231ps %%ymm12, %%ymm11, %%ymm7              \n\t"

        "vbroadcastss 0x4(%0), %%ymm8                        \n\t"
        "vbroadcastss 0x4(%1), %%ymm9                 \n\t"
        "vbroadcastss 0x4(%2), %%ymm10              \n\t"
        "vbroadcastss 0x4(%3), %%ymm11              \n\t"
        "vmovups 0x40(%8), %%ymm14                          \n\t"
        "vfmadd231ps %%ymm13, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm13, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm13, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm13, %%ymm11, %%ymm3              \n\t"
        "vbroadcastss 0x4(%4), %%ymm8              \n\t"
        "vbroadcastss 0x4(%5), %%ymm9              \n\t"
        "vbroadcastss 0x4(%6), %%ymm10              \n\t"
        "vbroadcastss 0x4(%7), %%ymm11              \n\t"
        "vmovups 0x60(%8), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm13, %%ymm8, %%ymm4              \n\t"
        "vfmadd231ps %%ymm13, %%ymm9, %%ymm5              \n\t"
        "vfmadd231ps %%ymm13, %%ymm10, %%ymm6              \n\t"
        "vfmadd231ps %%ymm13, %%ymm11, %%ymm7              \n\t"

        "vbroadcastss 0x8(%0), %%ymm8                        \n\t"
        "vbroadcastss 0x8(%1), %%ymm9                 \n\t"
        "vbroadcastss 0x8(%2), %%ymm10              \n\t"
        "vbroadcastss 0x8(%3), %%ymm11              \n\t"
        "vmovups 0x80(%8), %%ymm12                          \n\t"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm14, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm14, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm14, %%ymm11, %%ymm3              \n\t"
        "vbroadcastss 0x8(%4), %%ymm8              \n\t"
        "vbroadcastss 0x8(%5), %%ymm9              \n\t"
        "vbroadcastss 0x8(%6), %%ymm10              \n\t"
        "vbroadcastss 0x8(%7), %%ymm11              \n\t"
        "vmovups 0xA0(%8), %%ymm13                          \n\t"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm4              \n\t"
        "vfmadd231ps %%ymm14, %%ymm9, %%ymm5              \n\t"
        "vfmadd231ps %%ymm14, %%ymm10, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm11, %%ymm7              \n\t"

        "vbroadcastss 0xC(%0), %%ymm8                        \n\t"
        "vbroadcastss 0xC(%1), %%ymm9                 \n\t"
        "vbroadcastss 0xC(%2), %%ymm10              \n\t"
        "vbroadcastss 0xC(%3), %%ymm11              \n\t"
        "vfmadd231ps %%ymm15, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"
        "vbroadcastss 0xC(%4), %%ymm8              \n\t"
        "vbroadcastss 0xC(%5), %%ymm9              \n\t"
        "vbroadcastss 0xC(%6), %%ymm10              \n\t"
        "vbroadcastss 0xC(%7), %%ymm11              \n\t"
        "vfmadd231ps %%ymm15, %%ymm8, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm9, %%ymm5              \n\t"
        "vfmadd231ps %%ymm15, %%ymm10, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm11, %%ymm7              \n\t"

        "vbroadcastss 0x10(%0), %%ymm8                        \n\t"
        "vbroadcastss 0x10(%1), %%ymm9                 \n\t"
        "vbroadcastss 0x10(%2), %%ymm10              \n\t"
        "vbroadcastss 0x10(%3), %%ymm11              \n\t"
        "vmovups 0xC0(%8), %%ymm14                          \n\t"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm12, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm12, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm12, %%ymm11, %%ymm3              \n\t"
        "vbroadcastss 0x10(%4), %%ymm8              \n\t"
        "vbroadcastss 0x10(%5), %%ymm9              \n\t"
        "vbroadcastss 0x10(%6), %%ymm10              \n\t"
        "vbroadcastss 0x10(%7), %%ymm11              \n\t"
        "vmovups 0xE0(%8), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm4              \n\t"
        "vfmadd231ps %%ymm12, %%ymm9, %%ymm5              \n\t"
        "vfmadd231ps %%ymm12, %%ymm10, %%ymm6              \n\t"
        "vfmadd231ps %%ymm12, %%ymm11, %%ymm7              \n\t"

        "vbroadcastss 0x14(%0), %%ymm8                        \n\t"
        "vbroadcastss 0x14(%1), %%ymm9                 \n\t"
        "vbroadcastss 0x14(%2), %%ymm10              \n\t"
        "vbroadcastss 0x14(%3), %%ymm11              \n\t"
        "vfmadd231ps %%ymm13, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm13, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm13, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm13, %%ymm11, %%ymm3              \n\t"
        "vbroadcastss 0x14(%4), %%ymm8              \n\t"
        "vbroadcastss 0x14(%5), %%ymm9              \n\t"
        "vbroadcastss 0x14(%6), %%ymm10              \n\t"
        "vbroadcastss 0x14(%7), %%ymm11              \n\t"
        "vfmadd231ps %%ymm13, %%ymm8, %%ymm4              \n\t"
        "vfmadd231ps %%ymm13, %%ymm9, %%ymm5              \n\t"
        "vfmadd231ps %%ymm13, %%ymm10, %%ymm6              \n\t"
        "vfmadd231ps %%ymm13, %%ymm11, %%ymm7              \n\t"

        "vbroadcastss 0x18(%0), %%ymm8                        \n\t"
        "vbroadcastss 0x18(%1), %%ymm9                 \n\t"
        "vbroadcastss 0x18(%2), %%ymm10              \n\t"
        "vbroadcastss 0x18(%3), %%ymm11              \n\t"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm14, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm14, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm14, %%ymm11, %%ymm3              \n\t"
        "vbroadcastss 0x18(%4), %%ymm8              \n\t"
        "vbroadcastss 0x18(%5), %%ymm9              \n\t"
        "vbroadcastss 0x18(%6), %%ymm10              \n\t"
        "vbroadcastss 0x18(%7), %%ymm11              \n\t"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm4              \n\t"
        "vfmadd231ps %%ymm14, %%ymm9, %%ymm5              \n\t"
        "vfmadd231ps %%ymm14, %%ymm10, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm11, %%ymm7              \n\t"

        "vbroadcastss 0x1C(%0), %%ymm8                        \n\t"
        "vbroadcastss 0x1C(%1), %%ymm9                 \n\t"
        "vbroadcastss 0x1C(%2), %%ymm10              \n\t"
        "vbroadcastss 0x1C(%3), %%ymm11              \n\t"
        "vfmadd231ps %%ymm15, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm15, %%ymm11, %%ymm3              \n\t"
        "vbroadcastss 0x1C(%4), %%ymm8              \n\t"
        "vbroadcastss 0x1C(%5), %%ymm9              \n\t"
        "vbroadcastss 0x1C(%6), %%ymm10              \n\t"
        "vbroadcastss 0x1C(%7), %%ymm11              \n\t"
        "vfmadd231ps %%ymm15, %%ymm8, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm9, %%ymm5              \n\t"
        "vfmadd231ps %%ymm15, %%ymm10, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm11, %%ymm7              \n\t"
        "add %9, %0                                      \n\t"
        "add %9, %1                                      \n\t"
        "add %9, %2                                      \n\t"
        "add %9, %3                                      \n\t"
        "add %9, %4                                      \n\t"
        "add %9, %5                                      \n\t"
        "add %9, %6                                      \n\t"
        "add %9, %7                                      \n\t"
        "add $0x100, %8                                    \n\t"
        "dec %%rcx                                         \n\t"
        "jg 4b                                             \n\t"
        "popq %%rcx                                         \n\t"

        "add %10, %0                                     \n\t"
        "add %10, %1                                     \n\t"
        "add %10, %2                                     \n\t"
        "add %10, %3                                     \n\t"
        "add %10, %4                                     \n\t"
        "add %10, %5                                     \n\t"
        "add %10, %6                                     \n\t"
        "add %10, %7                                     \n\t"
        "dec %%rbx                                         \n\t"
        "jg 3b                                             \n\t"
        "popq %%rbx                                         \n\t"

        "add %11, %0                                     \n\t"
        "add %11, %1                                     \n\t"
        "add %11, %2                                     \n\t"
        "add %11, %3                                     \n\t"
        "add %11, %4                                     \n\t"
        "add %11, %5                                     \n\t"
        "add %11, %6                                     \n\t"
        "add %11, %7                                     \n\t"
        "dec %%rax                                         \n\t"
        "jg 2b                                             \n\t"
        :
        : "r"(curI[0]), "r"(curI[1]), "r"(curI[2]), "r"(curI[3]), "r"(curI[4]), "r"(curI[5]),
        "r"(curI[6]), "r"(curI[7]), "r"(curW), "r"((I64)dw), "r"((I64)iStep), "r"((I64)fStep),
        "a"((I64)(ic / 8)), "b"((I64)fh), "c"((I64)fw)
        : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",
        "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__(  // relu
        "and $0xC, %0                                      \n\t"
        "je 2f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
        "vmaxps %%ymm15, %%ymm2, %%ymm2                    \n\t"
        "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
        "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
        "vmaxps %%ymm15, %%ymm5, %%ymm5                    \n\t"
        "vmaxps %%ymm15, %%ymm6, %%ymm6                    \n\t"
        "vmaxps %%ymm15, %%ymm7, %%ymm7                    \n\t"

        // relu6
        "and $0x8, %0                                      \n\t"
        "je 2f                                             \n\t"
        "mov $0x40C00000, %%eax                            \n\t"
        "vmovd %%eax, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
        "vminps %%ymm12, %%ymm2, %%ymm2                    \n\t"
        "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
        "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
        "vminps %%ymm12, %%ymm5, %%ymm5                    \n\t"
        "vminps %%ymm12, %%ymm6, %%ymm6                    \n\t"
        "vminps %%ymm12, %%ymm7, %%ymm7                    \n\t"

        "2:                                                \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "vmovups %%ymm1, 0x20(%1)                          \n\t"
        "vmovups %%ymm2, 0x40(%1)                          \n\t"
        "vmovups %%ymm3, 0x60(%1)                          \n\t"
        "vmovups %%ymm4, 0x80(%1)                          \n\t"
        "vmovups %%ymm5, 0xA0(%1)                          \n\t"
        "vmovups %%ymm6, 0xC0(%1)                          \n\t"
        "vmovups %%ymm7, 0xE0(%1)                          \n\t"
        :
        : "r"(store), "r"(curO)
        : "%eax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm12",
        "%ymm15", "memory", "cc");
}

void Avx2ConvKernel1x8(F32 **curI,
    const F32 *curW,
    F32 *curO,
    F32 *curE,
    U32 fw,
    U32 fh,
    I32 oStep,
    I32 iStep,
    I32 store,
    const F32 *curB,
    I32 dw,
    I32 ic,
    I32 fStep)
{
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "mov %7, %%ecx                                  \n\t"
                         "and $0x2, %%ecx                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%12), %%ymm0, %%ymm0                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%1), %%ymm0                     \n\t"

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
                         "and $0xC, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"

                         // relu6
                         "and $0x8, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                    \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"

                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         :
                         : "r"(curI[0]), "r"(curO), "r"(curW), "r"(fw), "r"(I64(oStep)), "r"(fh),
                         "r"(I64(iStep)), "r"(store), "r"(curB), "r"(I64(dw)), "a"(ic / 8),
                         "r"(I64(fStep)), "r"(curE)
                         : "%ebx", "%ecx", "%ymm0", "%ymm12", "%ymm15", "memory", "cc");
}

EE convolution_direct(TensorDesc inputDesc,
    F32 *inArray,
    F32 *eltwiseInput,
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

    if ((fdf != DF_NCHWCxN32 && fdf != DF_NCHWCxN24) || (idf != DF_NCHWC8) || (ic % 8 != 0)) {
        CHECK_STATUS(NOT_MATCH);
    }

    // get kernels
    const kernelFunc kernel[4][2] = {{Avx2ConvKernel1x8, Avx2ConvKernel8x8},
        {Avx2ConvKernel1x16, Avx2ConvKernel6x16}, {Avx2ConvKernel1x24, Avx2ConvKernel4x24},
        {Avx2ConvKernel1x32, Avx2ConvKernel3x32}};
    const I32 unrollOcArray[4] = {8, 16, 24, 32};
    const I32 unrollHwArray[4] = {8, 6, 4, 3};

    // get computing params
    I32 strideH = convParamSpec.stride_h;
    I32 strideW = convParamSpec.stride_w;
    I32 paddingT = convParamSpec.pad_top;
    I32 paddingB = convParamSpec.pad_bottom;
    I32 paddingL = convParamSpec.pad_left;
    I32 paddingR = convParamSpec.pad_right;
    I32 dilateH = convParamSpec.dilatedRate_h;
    I32 dilateW = convParamSpec.dilatedRate_w;
    I32 ih_pad = ih + paddingT + paddingB;
    I32 iw_pad = iw + paddingL + paddingR;
    I32 ohow = oh * ow;

    // infer block params
    I32 unrollOc = InferConvDirectUnrollOc(oc);
    I32 unrollHwX = unrollHwArray[(unrollOc >> 3) - 1];
    I32 ocbArray[4] = {0};
    I32 ocBlockNums = InferConvDirectOcBlockNum(oc, ocbArray, unrollOc, unrollOcArray);
    I32 alpha = OMP_NUM_THREADS / gcd<I32>(ocBlockNums, OMP_NUM_THREADS);
    I32 blockHwDim = InferConvBlockHW(ohow, BLOCK_HW_DIM, alpha);
    blockHwDim = (blockHwDim + unrollHwX - 1) / unrollHwX * unrollHwX;
    if ((ohow <= BLOCK_HW_DIM) && (ocBlockNums % OMP_NUM_THREADS == 0)) {
        blockHwDim = ohow;
    }
    I32 hwBlockNums = CeilDivide(ohow, blockHwDim);
    I32 hwocBlockNums = hwBlockNums * ocBlockNums;
    I32 blockIcDim = InferConvDirectBolckIcDim(BLOCK_IC_DIM, unrollOc, blockHwDim, fh, fw);

#ifdef _USE_OPENMP
    OpenMPController ompCtr;
    ompCtr.checkAndSetOpenMP(ohow, BLOCK_HW_DIM, ocBlockNums);
#endif

    // infer kernel params
    I32 oStep = oh * ow * SIMDW * 4;
    I32 iStep = (iw_pad - fw * dilateW + (dilateH - 1) * iw_pad) * SIMDW * 4;
    I32 fStep = ((ih_pad - fh * dilateH) * iw_pad) * SIMDW * 4;
    I32 dw = dilateW * SIMDW * 4;

    F32 *tmpInput = (F32 *)tmp;
    for (U32 n = 0; n < in; ++n) {
        F32 *bInArray = inArray + n * ic * ih * iw;
        if (idf == DF_NCHWC8 && paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            tmpInput = bInArray;
        } else {
            // TODO: optimize the UNI_MEMCPY
            PaddingNCHWC8(bInArray, tmpInput, inputDesc, convParamSpec);
        }

#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (ompCtr.useOmp)
        {
#endif
            I32 flags = 0;
            I32 icSize = 0;
            for (I32 icbb = 0; icbb < (int)ic; icbb += icSize) {
                icSize = UNI_MIN(blockIcDim, (int)ic - icbb);
                flags |= (icbb > 0) + ((eltwiseInput != nullptr) << 1);
                if (icbb == (int)ic - icSize) {
                    flags |= U32(activationDesc.mode) << 2;
                }

#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
                for (I32 bIdx = 0; bIdx < hwocBlockNums; ++bIdx) {
                    FTZ;
                    I32 hw = (bIdx / ocBlockNums) * blockHwDim;
                    I32 hwSize = UNI_MIN(blockHwDim, ohow - hw);
                    I32 ocBlockIdx = bIdx % ocBlockNums;
                    I32 ocb = GetOcIdx(ocBlockIdx, oc, unrollOc, ocbArray);
                    I32 ocSize = UNI_MIN(unrollOc, (int)oc - ocb);
                    I32 unrollHw = unrollHwArray[(ocSize >> 3) - 1];
                    ocSize = unrollOcArray[(ocSize >> 3) - 1];

                    const F32 *curB = biasArray + ocb;
                    const F32 *calW = filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                    F32 *curI = tmpInput + icbb * ih_pad * iw_pad;
                    F32 *in_i[8] = {nullptr};
                    I32 wSize = 0;
                    for (I32 ihw = hw; ihw < (I32)(hw + hwSize); ihw += wSize) {
                        wSize = UNI_MIN(hw + hwSize - ihw, unrollHw);
                        if (wSize < unrollHw) {
                            wSize = 1;
                        }
                        for (I32 ii = 0; ii < wSize; ++ii) {
                            I32 in_h = (ihw + ii) / ow * strideH;
                            I32 in_w = (ihw + ii) % ow * strideW;
                            in_i[ii] = curI + in_h * iw_pad * 8 + in_w * 8;
                        }
                        F32 *out_ptr = outArray + (n * oc + ocb) * ohow + ihw * 8;
                        F32 *eltwise_ptr = eltwiseInput + (n * oc + ocb) * ohow + ihw * 8;
                        kernel[(ocSize >> 3) - 1][wSize > 1](in_i, calW, out_ptr, eltwise_ptr, fw,
                            fh, oStep, iStep, flags, curB, dw, icSize, fStep);
                    }
                }
            }

#ifdef _USE_OPENMP
        }
#endif
    }

    return SUCCESS;
}
