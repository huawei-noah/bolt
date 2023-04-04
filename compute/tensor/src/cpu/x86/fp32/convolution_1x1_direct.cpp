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
#include "cpu/x86/fp32/convolution_functions.h"

#define BLOCK_IC_DIM 128
#define BLOCK_OC_DIM 128
#define BLOCK_HW_DIM 768

typedef void (*kernelFunc)(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep);

inline void Avx2PointwiseKernel3x32(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("shr $3, %%ecx                                     \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%3), %%ymm0                       \n\t"
                         "vmovups (%3), %%ymm1                       \n\t"
                         "vmovups (%3), %%ymm2                       \n\t"
                         "vmovups 0x20(%3), %%ymm3                       \n\t"
                         "vmovups 0x20(%3), %%ymm4                   \n\t"
                         "vmovups 0x20(%3), %%ymm5                   \n\t"
                         "vmovups 0x40(%3), %%ymm6                   \n\t"
                         "vmovups 0x40(%3), %%ymm7                   \n\t"
                         "vmovups 0x40(%3), %%ymm8                   \n\t"
                         "vmovups 0x60(%3), %%ymm9                   \n\t"
                         "vmovups 0x60(%3), %%ymm10                 \n\t"
                         "vmovups 0x60(%3), %%ymm11                 \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x2, %%eax                                  \n\t"
                         "je 1f                                             \n\t"
                         "mov %8, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm0, %%ymm0                     \n\t"
                         "vaddps 0x20(%%rax), %%ymm1, %%ymm1                     \n\t"
                         "vaddps 0x40(%%rax), %%ymm2, %%ymm2                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm3, %%ymm3                     \n\t"
                         "vaddps 0x20(%%rax), %%ymm4, %%ymm4                     \n\t"
                         "vaddps 0x40(%%rax), %%ymm5, %%ymm5                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm6, %%ymm6                     \n\t"
                         "vaddps 0x20(%%rax), %%ymm7, %%ymm7                     \n\t"
                         "vaddps 0x40(%%rax), %%ymm8, %%ymm8                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm9, %%ymm9                     \n\t"
                         "vaddps 0x20(%%rax), %%ymm10, %%ymm10                  \n\t"
                         "vaddps 0x40(%%rax), %%ymm11, %%ymm11                  \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %2, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm0                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm1                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm2                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm3                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm4                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm5                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm6                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm7                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm8                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm9                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm10                  \n\t"
                         "vmovups 0x40(%%rax), %%ymm11                  \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"

                         "vbroadcastss (%1), %%ymm12                     \n\t"
                         "vbroadcastss 0x20(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x40(%1), %%ymm14                     \n\t"
                         "vmovaps (%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovaps 0x20(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovaps 0x40(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovaps 0x60(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x4(%1), %%ymm12                     \n\t"
                         "vbroadcastss 0x24(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x44(%1), %%ymm14                     \n\t"
                         "vmovaps 0x80(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovaps 0xA0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovaps 0xC0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovaps 0xE0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x8(%1), %%ymm12                     \n\t"
                         "vbroadcastss 0x28(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x48(%1), %%ymm14                     \n\t"
                         "vmovaps 0x100(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovaps 0x120(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovaps 0x140(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovaps 0x160(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0xC(%1), %%ymm12                     \n\t"
                         "vbroadcastss 0x2C(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x4C(%1), %%ymm14                     \n\t"
                         "vmovaps 0x180(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovaps 0x1A0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovaps 0x1C0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovaps 0x1E0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x10(%1), %%ymm12                     \n\t"
                         "vbroadcastss 0x30(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x50(%1), %%ymm14                     \n\t"
                         "vmovaps 0x200(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovaps 0x220(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovaps 0x240(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovaps 0x260(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x14(%1), %%ymm12                     \n\t"
                         "vbroadcastss 0x34(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x54(%1), %%ymm14                     \n\t"
                         "vmovaps 0x280(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovaps 0x2A0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovaps 0x2C0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovaps 0x2E0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x18(%1), %%ymm12                     \n\t"
                         "vbroadcastss 0x38(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x58(%1), %%ymm14                     \n\t"
                         "vmovaps 0x300(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovaps 0x320(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovaps 0x340(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovaps 0x360(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vbroadcastss 0x1C(%1), %%ymm12                     \n\t"
                         "vbroadcastss 0x3C(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x5C(%1), %%ymm14                     \n\t"
                         "vmovaps 0x380(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vmovaps 0x3A0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vmovaps 0x3C0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vmovaps 0x3E0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "add $0x400, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %5                                      \n\t"
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
                         "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"
                         "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"
                         "vmaxps %%ymm15, %%ymm10, %%ymm10                  \n\t"
                         "vmaxps %%ymm15, %%ymm11, %%ymm11                  \n\t"

                         // relu6
                         "and $0x8, %5                                      \n\t"
                         "je 2f                                             \n\t"
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

                         "2:                                                \n\t"
                         "vmovups %%ymm0, (%2)                          \n\t"
                         "vmovups %%ymm1, 0x20(%2)                      \n\t"
                         "vmovups %%ymm2, 0x40(%2)                      \n\t"
                         "add %4, %2                                  \n\t"
                         "vmovups %%ymm3, (%2)                      \n\t"
                         "vmovups %%ymm4, 0x20(%2)                          \n\t"
                         "vmovups %%ymm5, 0x40(%2)                      \n\t"
                         "add %4, %2                                  \n\t"
                         "vmovups %%ymm6, (%2)                      \n\t"
                         "vmovups %%ymm7, 0x20(%2)                      \n\t"
                         "vmovups %%ymm8, 0x40(%2)                       \n\t"
                         "add %4, %2                                  \n\t"
                         "vmovups %%ymm9, (%2)                   \n\t"
                         "vmovups %%ymm10, 0x20(%2)                  \n\t"
                         "vmovups %%ymm11, 0x40(%2)                  \n\t"
                         :
                         : "r"(curW), "r"(curI), "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(flags),
                         "c"(ic), "r"(I64(fStep)), "r"(curE)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

inline void Avx2PointwiseKernel4x24(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("shr $3, %%ecx                                     \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%3), %%ymm0                       \n\t"
                         "vmovups (%3), %%ymm1                       \n\t"
                         "vmovups (%3), %%ymm2                       \n\t"
                         "vmovups (%3), %%ymm3                       \n\t"
                         "vmovups 0x20(%3), %%ymm4                   \n\t"
                         "vmovups 0x20(%3), %%ymm5                   \n\t"
                         "vmovups 0x20(%3), %%ymm6                   \n\t"
                         "vmovups 0x20(%3), %%ymm7                   \n\t"
                         "vmovups 0x40(%3), %%ymm8                   \n\t"
                         "vmovups 0x40(%3), %%ymm9                   \n\t"
                         "vmovups 0x40(%3), %%ymm10                 \n\t"
                         "vmovups 0x40(%3), %%ymm11                 \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x2, %%eax                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%8), %%ymm0, %%ymm0                     \n\t"
                         "vaddps 0x20(%8), %%ymm1, %%ymm1                     \n\t"
                         "vaddps 0x40(%8), %%ymm2, %%ymm2                     \n\t"
                         "vaddps 0x60(%8), %%ymm3, %%ymm3                     \n\t"
                         "vaddps (%8, %4), %%ymm4, %%ymm4                     \n\t"
                         "vaddps 0x20(%8, %4), %%ymm5, %%ymm5                     \n\t"
                         "vaddps 0x40(%8, %4), %%ymm6, %%ymm6                     \n\t"
                         "vaddps 0x60(%8, %4), %%ymm7, %%ymm7                     \n\t"
                         "vaddps (%8, %4, 2), %%ymm8, %%ymm8                     \n\t"
                         "vaddps 0x20(%8, %4, 2), %%ymm9, %%ymm9                     \n\t"
                         "vaddps 0x40(%8, %4, 2), %%ymm10, %%ymm10                  \n\t"
                         "vaddps 0x60(%8, %4, 2), %%ymm11, %%ymm11                  \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%2), %%ymm0                     \n\t"
                         "vmovups 0x20(%2), %%ymm1                     \n\t"
                         "vmovups 0x40(%2), %%ymm2                     \n\t"
                         "vmovups 0x60(%2), %%ymm3                     \n\t"
                         "vmovups (%2, %4), %%ymm4                     \n\t"
                         "vmovups 0x20(%2, %4), %%ymm5                     \n\t"
                         "vmovups 0x40(%2, %4), %%ymm6                     \n\t"
                         "vmovups 0x60(%2, %4), %%ymm7                     \n\t"
                         "vmovups (%2, %4, 2), %%ymm8                     \n\t"
                         "vmovups 0x20(%2, %4, 2), %%ymm9                     \n\t"
                         "vmovups 0x40(%2, %4, 2), %%ymm10                  \n\t"
                         "vmovups 0x60(%2, %4, 2), %%ymm11                  \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"

                         "vmovaps (%0), %%ymm12                             \n\t"
                         "vmovaps 0x20(%0), %%ymm13                         \n\t"
                         "vmovaps 0x40(%0), %%ymm14                         \n\t"
                         "vbroadcastss (%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x20(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x40(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0x60(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x60(%0), %%ymm12                         \n\t"
                         "vmovaps 0x80(%0), %%ymm13                         \n\t"
                         "vmovaps 0xA0(%0), %%ymm14                         \n\t"
                         "vbroadcastss 0x4(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x24(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x44(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10             \n\t"
                         "vbroadcastss 0x64(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0xC0(%0), %%ymm12                         \n\t"
                         "vmovaps 0xE0(%0), %%ymm13                         \n\t"
                         "vmovaps 0x100(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x8(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x28(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x48(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10             \n\t"
                         "vbroadcastss 0x68(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x120(%0), %%ymm12                        \n\t"
                         "vmovaps 0x140(%0), %%ymm13                        \n\t"
                         "vmovaps 0x160(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0xC(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x2C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x4C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10             \n\t"
                         "vbroadcastss 0x6C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x180(%0), %%ymm12                             \n\t"
                         "vmovaps 0x1A0(%0), %%ymm13                         \n\t"
                         "vmovaps 0x1C0(%0), %%ymm14                         \n\t"
                         "vbroadcastss 0x10(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x30(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x50(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                         "vbroadcastss 0x70(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x1E0(%0), %%ymm12                         \n\t"
                         "vmovaps 0x200(%0), %%ymm13                         \n\t"
                         "vmovaps 0x220(%0), %%ymm14                         \n\t"
                         "vbroadcastss 0x14(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x34(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x54(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10             \n\t"
                         "vbroadcastss 0x74(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x240(%0), %%ymm12                         \n\t"
                         "vmovaps 0x260(%0), %%ymm13                         \n\t"
                         "vmovaps 0x280(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x18(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x38(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x58(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10             \n\t"
                         "vbroadcastss 0x78(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "vmovaps 0x2A0(%0), %%ymm12                        \n\t"
                         "vmovaps 0x2C0(%0), %%ymm13                        \n\t"
                         "vmovaps 0x2E0(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x3C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                         "vbroadcastss 0x5C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm10             \n\t"
                         "vbroadcastss 0x7C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "add $0x300, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %5                                      \n\t"
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
                         "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"
                         "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"
                         "vmaxps %%ymm15, %%ymm10, %%ymm10                  \n\t"
                         "vmaxps %%ymm15, %%ymm11, %%ymm11                  \n\t"

                         // relu6
                         "and $0x8, %5                                      \n\t"
                         "je 2f                                             \n\t"
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

                         "2:                                                \n\t"
                         "vmovups %%ymm0, (%2)                          \n\t"
                         "vmovups %%ymm1, 0x20(%2)                      \n\t"
                         "vmovups %%ymm2, 0x40(%2)                      \n\t"
                         "vmovups %%ymm3, 0x60(%2)                      \n\t"
                         "vmovups %%ymm4, (%2, %4)                          \n\t"
                         "vmovups %%ymm5, 0x20(%2, %4)                      \n\t"
                         "vmovups %%ymm6, 0x40(%2, %4)                      \n\t"
                         "vmovups %%ymm7, 0x60(%2, %4)                      \n\t"
                         "vmovups %%ymm8, (%2, %4, 2)                       \n\t"
                         "vmovups %%ymm9, 0x20(%2, %4, 2)                   \n\t"
                         "vmovups %%ymm10, 0x40(%2, %4, 2)                  \n\t"
                         "vmovups %%ymm11, 0x60(%2, %4, 2)                  \n\t"
                         :
                         : "r"(curW), "r"(curI), "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(flags),
                         "c"(ic), "r"(I64(fStep)), "r"(curE)
                         : "%eax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                         "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

inline void Avx2PointwiseKernel6x16(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("shr $3, %%ecx                                     \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%3), %%ymm0                       \n\t"
                         "vmovups (%3), %%ymm1                       \n\t"
                         "vmovups (%3), %%ymm2                       \n\t"
                         "vmovups (%3), %%ymm3                       \n\t"
                         "vmovups (%3), %%ymm4                   \n\t"
                         "vmovups (%3), %%ymm5                   \n\t"
                         "vmovups 0x20(%3), %%ymm6                   \n\t"
                         "vmovups 0x20(%3), %%ymm7                   \n\t"
                         "vmovups 0x20(%3), %%ymm8                   \n\t"
                         "vmovups 0x20(%3), %%ymm9                   \n\t"
                         "vmovups 0x20(%3), %%ymm10                   \n\t"
                         "vmovups 0x20(%3), %%ymm11                   \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x2, %%eax                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%8), %%ymm0, %%ymm0                     \n\t"
                         "vaddps 0x20(%8), %%ymm1, %%ymm1                     \n\t"
                         "vaddps 0x40(%8), %%ymm2, %%ymm2                     \n\t"
                         "vaddps 0x60(%8), %%ymm3, %%ymm3                     \n\t"
                         "vaddps 0x80(%8), %%ymm4, %%ymm4                     \n\t"
                         "vaddps 0xA0(%8), %%ymm5, %%ymm5                     \n\t"
                         "vaddps (%8, %4), %%ymm6, %%ymm6                     \n\t"
                         "vaddps 0x20(%8, %4), %%ymm7, %%ymm7                     \n\t"
                         "vaddps 0x40(%8, %4), %%ymm8, %%ymm8                     \n\t"
                         "vaddps 0x60(%8, %4), %%ymm9, %%ymm9                     \n\t"
                         "vaddps 0x80(%8, %4), %%ymm10, %%ymm10                     \n\t"
                         "vaddps 0xA0(%8, %4), %%ymm11, %%ymm11                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%2), %%ymm0                     \n\t"
                         "vmovups 0x20(%2), %%ymm1                     \n\t"
                         "vmovups 0x40(%2), %%ymm2                     \n\t"
                         "vmovups 0x60(%2), %%ymm3                     \n\t"
                         "vmovups 0x80(%2), %%ymm4                     \n\t"
                         "vmovups 0xA0(%2), %%ymm5                     \n\t"
                         "vmovups (%2, %4), %%ymm6                     \n\t"
                         "vmovups 0x20(%2, %4), %%ymm7                     \n\t"
                         "vmovups 0x40(%2, %4), %%ymm8                     \n\t"
                         "vmovups 0x60(%2, %4), %%ymm9                     \n\t"
                         "vmovups 0x80(%2, %4), %%ymm10                     \n\t"
                         "vmovups 0xA0(%2, %4), %%ymm11                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"

                         "vmovaps (%0), %%ymm12                             \n\t"
                         "vmovaps 0x20(%0), %%ymm13                         \n\t"
                         "vbroadcastss (%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x20(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x40(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x60(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9             \n\t"
                         "vbroadcastss 0x80(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0xA0(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11             \n\t"

                         "vmovaps 0x40(%0), %%ymm12                         \n\t"
                         "vmovaps 0x60(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x4(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x24(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x44(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x64(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9             \n\t"
                         "vbroadcastss 0x84(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0xA4(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11             \n\t"

                         "vmovaps 0x80(%0), %%ymm12                         \n\t"
                         "vmovaps 0xA0(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x8(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x28(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x48(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x68(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9             \n\t"
                         "vbroadcastss 0x88(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0xA8(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11             \n\t"

                         "vmovaps 0xC0(%0), %%ymm12                        \n\t"
                         "vmovaps 0xE0(%0), %%ymm13                        \n\t"
                         "vbroadcastss 0xC(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x2C(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x4C(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x6C(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9             \n\t"
                         "vbroadcastss 0x8C(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0xAC(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11             \n\t"

                         "vmovaps 0x100(%0), %%ymm12                             \n\t"
                         "vmovaps 0x120(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x10(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x30(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x50(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x70(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9             \n\t"
                         "vbroadcastss 0x90(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0xB0(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11             \n\t"

                         "vmovaps 0x140(%0), %%ymm12                         \n\t"
                         "vmovaps 0x160(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x14(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x34(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x54(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x74(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9             \n\t"
                         "vbroadcastss 0x94(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0xB4(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11             \n\t"

                         "vmovaps 0x180(%0), %%ymm12                         \n\t"
                         "vmovaps 0x1A0(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x18(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x38(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x58(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x78(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9             \n\t"
                         "vbroadcastss 0x98(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0xB8(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11             \n\t"

                         "vmovaps 0x1C0(%0), %%ymm12                        \n\t"
                         "vmovaps 0x1E0(%0), %%ymm13                        \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x3C(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm7              \n\t"
                         "vbroadcastss 0x5C(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0x7C(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm9             \n\t"
                         "vbroadcastss 0x9C(%1), %%ymm15                     \n\t"
                         "vbroadcastss 0xBC(%1), %%ymm14                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm11             \n\t"

                         "add $0x200, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %5                                      \n\t"
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
                         "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"
                         "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"
                         "vmaxps %%ymm15, %%ymm10, %%ymm10                  \n\t"
                         "vmaxps %%ymm15, %%ymm11, %%ymm11                  \n\t"

                         // relu6
                         "and $0x8, %5                                      \n\t"
                         "je 2f                                             \n\t"
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

                         "2:                                                \n\t"
                         "vmovups %%ymm0, (%2)                          \n\t"
                         "vmovups %%ymm1, 0x20(%2)                      \n\t"
                         "vmovups %%ymm2, 0x40(%2)                      \n\t"
                         "vmovups %%ymm3, 0x60(%2)                      \n\t"
                         "vmovups %%ymm4, 0x80(%2)                          \n\t"
                         "vmovups %%ymm5, 0xA0(%2)                      \n\t"
                         "vmovups %%ymm6, (%2, %4)                      \n\t"
                         "vmovups %%ymm7, 0x20(%2, %4)                      \n\t"
                         "vmovups %%ymm8, 0x40(%2, %4)                      \n\t"
                         "vmovups %%ymm9, 0x60(%2, %4)                      \n\t"
                         "vmovups %%ymm10, 0x80(%2, %4)                      \n\t"
                         "vmovups %%ymm11, 0xA0(%2, %4)                      \n\t"
                         :
                         : "r"(curW), "r"(curI), "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(flags),
                         "c"(ic), "r"(I64(fStep)), "r"(curE)
                         : "%eax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                         "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

inline void Avx2PointwiseKernel12x8(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("shr $3, %%ecx                                     \n\t"
                         "mov %4, %%ebx                                  \n\t"
                         "and $0x1, %%ebx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%3), %%ymm0                       \n\t"
                         "vmovups (%3), %%ymm1                       \n\t"
                         "vmovups (%3), %%ymm2                       \n\t"
                         "vmovups (%3), %%ymm3                       \n\t"
                         "vmovups (%3), %%ymm4                       \n\t"
                         "vmovups (%3), %%ymm5                       \n\t"
                         "vmovups (%3), %%ymm6                       \n\t"
                         "vmovups (%3), %%ymm7                       \n\t"
                         "vmovups (%3), %%ymm8                       \n\t"
                         "vmovups (%3), %%ymm9                       \n\t"
                         "vmovups (%3), %%ymm10                       \n\t"
                         "vmovups (%3), %%ymm11                       \n\t"
                         "mov %4, %%ebx                                  \n\t"
                         "and $0x2, %%ebx                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%7), %%ymm0, %%ymm0                     \n\t"
                         "vaddps 0x20(%7), %%ymm1, %%ymm1                     \n\t"
                         "vaddps 0x40(%7), %%ymm2, %%ymm2                     \n\t"
                         "vaddps 0x60(%7), %%ymm3, %%ymm3                     \n\t"
                         "vaddps 0x80(%7), %%ymm4, %%ymm4                     \n\t"
                         "vaddps 0xA0(%7), %%ymm5, %%ymm5                     \n\t"
                         "vaddps 0xC0(%7), %%ymm6, %%ymm6                     \n\t"
                         "vaddps 0xE0(%7), %%ymm7, %%ymm7                     \n\t"
                         "vaddps 0x100(%7), %%ymm8, %%ymm8                     \n\t"
                         "vaddps 0x120(%7), %%ymm9, %%ymm9                     \n\t"
                         "vaddps 0x140(%7), %%ymm10, %%ymm10                     \n\t"
                         "vaddps 0x160(%7), %%ymm11, %%ymm11                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%2), %%ymm0                     \n\t"
                         "vmovups 0x20(%2), %%ymm1                     \n\t"
                         "vmovups 0x40(%2), %%ymm2                     \n\t"
                         "vmovups 0x60(%2), %%ymm3                     \n\t"
                         "vmovups 0x80(%2), %%ymm4                     \n\t"
                         "vmovups 0xA0(%2), %%ymm5                     \n\t"
                         "vmovups 0xC0(%2), %%ymm6                     \n\t"
                         "vmovups 0xE0(%2), %%ymm7                     \n\t"
                         "vmovups 0x100(%2), %%ymm8                     \n\t"
                         "vmovups 0x120(%2), %%ymm9                     \n\t"
                         "vmovups 0x140(%2), %%ymm10                     \n\t"
                         "vmovups 0x160(%2), %%ymm11                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"

                         "vmovaps (%0), %%ymm12                             \n\t"
                         "vbroadcastss (%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x20(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x40(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vbroadcastss 0x60(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x80(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0xA0(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
                         "vbroadcastss 0xC0(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0xE0(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x100(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
                         "vbroadcastss 0x120(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x140(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x160(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

                         "vmovaps 0x20(%0), %%ymm12                         \n\t"
                         "vbroadcastss 0x4(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x24(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x44(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vbroadcastss 0x64(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x84(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0xA4(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
                         "vbroadcastss 0xC4(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0xE4(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x104(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
                         "vbroadcastss 0x124(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x144(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x164(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

                         "vmovaps 0x40(%0), %%ymm12                         \n\t"
                         "vbroadcastss 0x8(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x28(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x48(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vbroadcastss 0x68(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x88(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0xA8(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
                         "vbroadcastss 0xC8(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0xE8(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x108(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
                         "vbroadcastss 0x128(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x148(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x168(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

                         "vmovaps 0x60(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0xC(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x2C(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x4C(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vbroadcastss 0x6C(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x8C(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0xAC(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
                         "vbroadcastss 0xCC(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0xEC(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x10C(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
                         "vbroadcastss 0x12C(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x14C(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x16C(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

                         "vmovaps 0x80(%0), %%ymm12                             \n\t"
                         "vbroadcastss 0x10(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x30(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x50(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vbroadcastss 0x70(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x90(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0xB0(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
                         "vbroadcastss 0xD0(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0xF0(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x110(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
                         "vbroadcastss 0x130(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x150(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x170(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

                         "vmovaps 0xA0(%0), %%ymm12                         \n\t"
                         "vbroadcastss 0x14(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x34(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x54(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vbroadcastss 0x74(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x94(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0xB4(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
                         "vbroadcastss 0xD4(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0xF4(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x114(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
                         "vbroadcastss 0x134(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x154(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x174(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

                         "vmovaps 0xC0(%0), %%ymm12                         \n\t"
                         "vbroadcastss 0x18(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x38(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x58(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vbroadcastss 0x78(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x98(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0xB8(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
                         "vbroadcastss 0xD8(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0xF8(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x118(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
                         "vbroadcastss 0x138(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x158(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x178(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

                         "vmovaps 0xE0(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x3C(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x5C(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vbroadcastss 0x7C(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x9C(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0xBC(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
                         "vbroadcastss 0xDC(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0xFC(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x11C(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
                         "vbroadcastss 0x13C(%1), %%ymm13                     \n\t"
                         "vbroadcastss 0x15C(%1), %%ymm14                     \n\t"
                         "vbroadcastss 0x17C(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

                         "add $0x100, %0                                      \n\t"
                         "add %6, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %4                                      \n\t"
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
                         "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"
                         "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"
                         "vmaxps %%ymm15, %%ymm10, %%ymm10                  \n\t"
                         "vmaxps %%ymm15, %%ymm11, %%ymm11                  \n\t"

                         // relu6
                         "and $0x8, %4                                      \n\t"
                         "je 2f                                             \n\t"
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

                         "2:                                                \n\t"
                         "vmovups %%ymm0, (%2)                          \n\t"
                         "vmovups %%ymm1, 0x20(%2)                      \n\t"
                         "vmovups %%ymm2, 0x40(%2)                      \n\t"
                         "vmovups %%ymm3, 0x60(%2)                      \n\t"
                         "vmovups %%ymm4, 0x80(%2)                      \n\t"
                         "vmovups %%ymm5, 0xA0(%2)                      \n\t"
                         "vmovups %%ymm6, 0xC0(%2)                      \n\t"
                         "vmovups %%ymm7, 0xE0(%2)                      \n\t"
                         "vmovups %%ymm8, 0x100(%2)                     \n\t"
                         "vmovups %%ymm9, 0x120(%2)                     \n\t"
                         "vmovups %%ymm10, 0x140(%2)                      \n\t"
                         "vmovups %%ymm11, 0x160(%2)                      \n\t"
                         :
                         : "r"(curW), "r"(curI), "r"(curO), "r"(curB), "r"(flags), "c"(ic),
                         "r"(I64(fStep)), "r"(curE)
                         : "%eax", "%rax", "%ebx", "%r9", "%r10", "%ymm0", "%ymm1", "%ymm2",
                         "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
                         "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

inline void Avx2PointwiseKernel1x32(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("shr $3, %%ecx                                     \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%3), %%ymm0                       \n\t"
                         "vmovups 0x20(%3), %%ymm3                       \n\t"
                         "vmovups 0x40(%3), %%ymm6                   \n\t"
                         "vmovups 0x60(%3), %%ymm9                   \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x2, %%eax                                  \n\t"
                         "je 1f                                             \n\t"
                         "mov %8, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm0, %%ymm0                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm3, %%ymm3                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm6, %%ymm6                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vaddps (%%rax), %%ymm9, %%ymm9                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %2, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm0                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm3                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm6                     \n\t"
                         "add %4, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm9                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"

                         "vbroadcastss (%1), %%ymm12                     \n\t"
                         "vmovaps (%0), %%ymm13                             \n\t"
                         "vmovaps 0x20(%0), %%ymm14                             \n\t"
                         "vmovaps 0x40(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vmovaps 0x60(%0), %%ymm14                             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

                         "vbroadcastss 0x4(%1), %%ymm12                     \n\t"
                         "vmovaps 0x80(%0), %%ymm13                             \n\t"
                         "vmovaps 0xA0(%0), %%ymm14                             \n\t"
                         "vmovaps 0xC0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vmovaps 0xE0(%0), %%ymm14                             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

                         "vbroadcastss 0x8(%1), %%ymm12                     \n\t"
                         "vmovaps 0x100(%0), %%ymm13                             \n\t"
                         "vmovaps 0x120(%0), %%ymm14                             \n\t"
                         "vmovaps 0x140(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vmovaps 0x160(%0), %%ymm14                             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

                         "vbroadcastss 0xC(%1), %%ymm12                     \n\t"
                         "vmovaps 0x180(%0), %%ymm13                             \n\t"
                         "vmovaps 0x1A0(%0), %%ymm14                             \n\t"
                         "vmovaps 0x1C0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vmovaps 0x1E0(%0), %%ymm14                             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

                         "vbroadcastss 0x10(%1), %%ymm12                     \n\t"
                         "vmovaps 0x200(%0), %%ymm13                             \n\t"
                         "vmovaps 0x220(%0), %%ymm14                             \n\t"
                         "vmovaps 0x240(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vmovaps 0x260(%0), %%ymm14                             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

                         "vbroadcastss 0x14(%1), %%ymm12                     \n\t"
                         "vmovaps 0x280(%0), %%ymm13                             \n\t"
                         "vmovaps 0x2A0(%0), %%ymm14                             \n\t"
                         "vmovaps 0x2C0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vmovaps 0x2E0(%0), %%ymm14                             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

                         "vbroadcastss 0x18(%1), %%ymm12                     \n\t"
                         "vmovaps 0x300(%0), %%ymm13                             \n\t"
                         "vmovaps 0x320(%0), %%ymm14                             \n\t"
                         "vmovaps 0x340(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vmovaps 0x360(%0), %%ymm14                             \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

                         "vbroadcastss 0x1C(%1), %%ymm12                     \n\t"
                         "vmovaps 0x380(%0), %%ymm13                             \n\t"
                         "vmovaps 0x3A0(%0), %%ymm14                             \n\t"
                         "vmovaps 0x3C0(%0), %%ymm15                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vmovaps 0x3E0(%0), %%ymm13                             \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"

                         "add $0x400, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %5                                      \n\t"
                         "je 2f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
                         "vmaxps %%ymm15, %%ymm6, %%ymm6                    \n\t"
                         "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"

                         // relu6
                         "and $0x8, %5                                      \n\t"
                         "je 2f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
                         "vminps %%ymm12, %%ymm6, %%ymm6                    \n\t"
                         "vminps %%ymm12, %%ymm9, %%ymm9                    \n\t"

                         "2:                                                \n\t"
                         "vmovups %%ymm0, (%2)                          \n\t"
                         "add %4, %2                                  \n\t"
                         "vmovups %%ymm3, (%2)                      \n\t"
                         "add %4, %2                                  \n\t"
                         "vmovups %%ymm6, (%2)                      \n\t"
                         "add %4, %2                                  \n\t"
                         "vmovups %%ymm9, (%2)                   \n\t"
                         :
                         : "r"(curW), "r"(curI), "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(flags),
                         "c"(ic), "r"(I64(fStep)), "r"(curE)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

inline void Avx2PointwiseKernel1x24(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("shr $3, %%ecx                                     \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%3), %%ymm0                       \n\t"
                         "vmovups 0x20(%3), %%ymm4                   \n\t"
                         "vmovups 0x40(%3), %%ymm8                   \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x2, %%eax                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%8), %%ymm0, %%ymm0                     \n\t"
                         "vaddps (%8, %4), %%ymm4, %%ymm4                     \n\t"
                         "vaddps (%8, %4, 2), %%ymm8, %%ymm8                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%2), %%ymm0                     \n\t"
                         "vmovups (%2, %4), %%ymm4                     \n\t"
                         "vmovups (%2, %4, 2), %%ymm8                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"

                         "vmovaps (%0), %%ymm12                             \n\t"
                         "vmovaps 0x20(%0), %%ymm13                        \n\t"
                         "vmovaps 0x40(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x0(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x60(%0), %%ymm12                         \n\t"
                         "vmovaps 0x80(%0), %%ymm13                        \n\t"
                         "vmovaps 0xA0(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x4(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0xC0(%0), %%ymm12                         \n\t"
                         "vmovaps 0xE0(%0), %%ymm13                        \n\t"
                         "vmovaps 0x100(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x8(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x120(%0), %%ymm12                        \n\t"
                         "vmovaps 0x140(%0), %%ymm13                        \n\t"
                         "vmovaps 0x160(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0xC(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x180(%0), %%ymm12                             \n\t"
                         "vmovaps 0x1A0(%0), %%ymm13                        \n\t"
                         "vmovaps 0x1C0(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x10(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x1E0(%0), %%ymm12                         \n\t"
                         "vmovaps 0x200(%0), %%ymm13                        \n\t"
                         "vmovaps 0x220(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x14(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x240(%0), %%ymm12                         \n\t"
                         "vmovaps 0x260(%0), %%ymm13                        \n\t"
                         "vmovaps 0x280(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x18(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "vmovaps 0x2A0(%0), %%ymm12                        \n\t"
                         "vmovaps 0x2C0(%0), %%ymm13                        \n\t"
                         "vmovaps 0x2E0(%0), %%ymm14                        \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                         "add $0x300, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %5                                      \n\t"
                         "je 2f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
                         "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"

                         // relu6
                         "and $0x8, %5                                      \n\t"
                         "je 2f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
                         "vminps %%ymm12, %%ymm8, %%ymm8                    \n\t"

                         "2:                                                \n\t"
                         "vmovups %%ymm0, (%2)                          \n\t"
                         "vmovups %%ymm4, (%2, %4)                          \n\t"
                         "vmovups %%ymm8, (%2, %4, 2)                       \n\t"
                         :
                         : "r"(curW), "r"(curI), "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(flags),
                         "c"(ic), "r"(I64(fStep)), "r"(curE)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

inline void Avx2PointwiseKernel1x16(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("shr $3, %%ecx                                     \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%3), %%ymm0                       \n\t"
                         "vmovups 0x20(%3), %%ymm4                   \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "and $0x2, %%eax                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%8), %%ymm0, %%ymm0                     \n\t"
                         "vaddps (%8, %4), %%ymm4, %%ymm4                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%2), %%ymm0                     \n\t"
                         "vmovups (%2, %4), %%ymm4                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"

                         "vmovaps (%0), %%ymm12                             \n\t"
                         "vmovaps 0x20(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x0(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vmovaps 0x40(%0), %%ymm12                             \n\t"
                         "vmovaps 0x60(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x4(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vmovaps 0x80(%0), %%ymm12                             \n\t"
                         "vmovaps 0xA0(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x8(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vmovaps 0xC0(%0), %%ymm12                             \n\t"
                         "vmovaps 0xE0(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0xC(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vmovaps 0x100(%0), %%ymm12                             \n\t"
                         "vmovaps 0x120(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x10(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vmovaps 0x140(%0), %%ymm12                             \n\t"
                         "vmovaps 0x160(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x14(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vmovaps 0x180(%0), %%ymm12                             \n\t"
                         "vmovaps 0x1A0(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x18(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "vmovaps 0x1C0(%0), %%ymm12                             \n\t"
                         "vmovaps 0x1E0(%0), %%ymm13                         \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                         "add $0x200, %0                                      \n\t"
                         "add %7, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %5                                      \n\t"
                         "je 2f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"

                         // relu6
                         "and $0x8, %5                                      \n\t"
                         "je 2f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"

                         "2:                                                \n\t"
                         "vmovups %%ymm0, (%2)                          \n\t"
                         "vmovups %%ymm4, (%2, %4)                          \n\t"
                         :
                         : "r"(curW), "r"(curI), "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(flags),
                         "c"(ic), "r"(I64(fStep)), "r"(curE)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

inline void Avx2PointwiseKernel1x8(F32 *curI,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    F32 *curE,
    U32 oStep,
    U32 flags,
    U32 ic,
    U32 fStep)
{
    __asm__ __volatile__("shr $3, %%ecx                                     \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%3), %%ymm0                       \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "and $0x2, %%eax                                  \n\t"
                         "je 1f                                             \n\t"
                         "vaddps (%7), %%ymm0, %%ymm0                     \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%2), %%ymm0                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"

                         "vmovaps (%0), %%ymm12                             \n\t"
                         "vbroadcastss 0x0(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovaps 0x20(%0), %%ymm12                         \n\t"
                         "vbroadcastss 0x4(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovaps 0x40(%0), %%ymm12                         \n\t"
                         "vbroadcastss 0x8(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovaps 0x60(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0xC(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovaps 0x80(%0), %%ymm12                             \n\t"
                         "vbroadcastss 0x10(%1), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovaps 0xA0(%0), %%ymm12                         \n\t"
                         "vbroadcastss 0x14(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovaps 0xC0(%0), %%ymm12                         \n\t"
                         "vbroadcastss 0x18(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "vmovaps 0xE0(%0), %%ymm12                        \n\t"
                         "vbroadcastss 0x1C(%1), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                         "add $0x100, %0                                      \n\t"
                         "add %6, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         // relu
                         "and $0xC, %4                                      \n\t"
                         "je 2f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"

                         // relu6
                         "and $0x8, %4                                      \n\t"
                         "je 2f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"

                         "2:                                                \n\t"
                         "vmovups %%ymm0, (%2)                          \n\t"
                         :
                         : "r"(curW), "r"(curI), "r"(curO), "r"(curB), "r"(flags), "c"(ic),
                         "r"(I64(fStep)), "r"(curE)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

EE convolution_1x1_direct(TensorDesc inputDesc,
    F32 *inArray,
    F32 *eltwiseInput,
    TensorDesc filterDesc,
    const F32 *filterArray,
    ConvolutionParamSpec convParamSpec,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec activationDesc)
{
    UNUSED(tmpBytes);
    DataType idt, odt, fdt;
    DataFormat idf, odf, fdf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    U32 ihiw = ih * iw;

    if (idf == DF_NCHW || idf == DF_MTK || idf == DF_NORMAL) {
        TensorDesc inDescCPU = inputDesc;
        inDescCPU.df = DF_NCHWC8;
        F32 *inputCPU = (F32 *)tmp;
        tmp = (U8 *)tmp + tensorNumBytes(inDescCPU);
        __m256i vindex = _mm256_set_epi32(
            ihiw * 7, ihiw * 6, ihiw * 5, ihiw * 4, ihiw * 3, ihiw * 2, ihiw, 0);
        U32 ic8 = ic / 8 * 8;
        U32 icp = (ic + 7) / 8 * 8;
        for (U32 n = 0; n < in; ++n) {
            for (U32 c = 0; c < ic8; c += 8) {
                for (U32 hw = 0; hw < ihiw; ++hw) {
                    _mm256_storeu_ps(inputCPU + (n * icp + c) * ihiw + hw * 8,
                         _mm256_i32gather_ps(inArray + (n * ic + c) * ihiw + hw, vindex, 4));
                }
            }
            if (ic < icp) {
               U32 off[8] = {0};
               for (U32 i = ic8; i < ic; ++i) {
                    off[i - ic8] = ihiw * (i - ic8);
               }
               vindex = _mm256_set_epi32(
                    off[7], off[6], off[5], off[4], off[3], off[2], off[1], off[0]);
               for (U32 hw = 0; hw < ihiw; ++hw) {
                    _mm256_storeu_ps(inputCPU + (n * icp + ic8) * ihiw + hw * 8,
                         _mm256_i32gather_ps(inArray + (n * ic + ic8) * ihiw + hw, vindex, 4));
                }
            }
        }
        inArray = (F32 *)inputCPU;
        idf = DF_NCHWC8;
        ic = icp;
    }

    if ((idf == DF_NCHWC16 || idf == DF_NCHW) && ih == 1 && iw == 1) {
        idf = DF_NCHWC8;
    }

    if ((fdf != DF_NCHWCxN24 && fdf != DF_NCHWCxN32) || (idf != DF_NCHWC8) || (ic % 8 != 0)) {
        CHECK_STATUS(NOT_MATCH);
    }

    // get kernels
    kernelFunc kernel[2][4] = {{Avx2PointwiseKernel1x8, Avx2PointwiseKernel1x16,
                                   Avx2PointwiseKernel1x24, Avx2PointwiseKernel1x32},
        {Avx2PointwiseKernel12x8, Avx2PointwiseKernel6x16, Avx2PointwiseKernel4x24,
            Avx2PointwiseKernel3x32}};
    I32 unrollOcArray[4] = {8, 16, 24, 32};
    I32 unrollHwArray[4] = {12, 6, 4, 3};

    // get computing params
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 phT = (paddingT + strideH - 1) / strideH;
    U32 phB = (paddingB + strideH - 1) / strideH;
    U32 ohow = oh * ow;
    U32 ohowMain = (oh - phT - phB) * ow;
    U32 newIh = (ih + strideH - 1) / strideH;
    U32 newIw = (iw + strideW - 1) / strideW;

    // infer block params
    U32 unrollOc = InferConvPointwiseUnrollOc(oc);
    U32 unrollHwX = unrollHwArray[(unrollOc >> 3) - 1];
    I32 ocbArray[4] = {0};
    U32 ocBlockNums = InferConvDirectOcBlockNum(oc, ocbArray, unrollOc, unrollOcArray);
    U32 ocBBlockNums = BLOCK_OC_DIM / unrollOc;
    U32 alpha = OMP_NUM_THREADS / gcd<U32>(ocBlockNums, OMP_NUM_THREADS);
    U32 blockHwDim = InferConvBlockHW(ohowMain, BLOCK_HW_DIM, alpha);
    blockHwDim = (blockHwDim + unrollHwX - 1) / unrollHwX * unrollHwX;
    U32 hwBlockNums = CeilDivide(ohowMain, blockHwDim);
    if (paddingL != 0 || paddingR != 0) {
        hwBlockNums = oh;
    }


#ifdef _USE_OPENMP
    ocBBlockNums = ocBlockNums;
    OpenMPController ompCtr;
    U32 computeBlock = oc * ic;
    ompCtr.checkAndSetOpenMP(ohowMain, BLOCK_HW_DIM, ocBlockNums, computeBlock, 4096);
#endif

    // infer kernel params
    U32 oStep = ohow * SIMDW * 4;
    U32 fStep = newIh * newIw * SIMDW * 4;

    // activate bias when padding
    F32 *btmp = (F32 *)tmp;
    if ((paddingT != 0) || (paddingB != 0) || (paddingL != 0) || (paddingR != 0)) {
        __m256 zero = _mm256_set1_ps(0.);
        switch (activationDesc.mode) {
            case ACTIVATION_NULL: {
                for (U32 ocb = 0; ocb < oc; ocb += 8) {
                    _mm256_store_ps(btmp + ocb, _mm256_loadu_ps(biasArray + ocb));
                }
                break;
            }
            case ACTIVATION_RELU: {
                for (U32 ocb = 0; ocb < oc; ocb += 8) {
                    _mm256_store_ps(
                        btmp + ocb, _mm256_max_ps(zero, _mm256_loadu_ps(biasArray + ocb)));
                }
                break;
            }
            case ACTIVATION_RELU6: {
                __m256 six = _mm256_set1_ps(6.);
                for (U32 ocb = 0; ocb < oc; ocb += 8) {
                    _mm256_store_ps(btmp + ocb,
                        _mm256_min_ps(six, _mm256_max_ps(zero, _mm256_loadu_ps(biasArray + ocb))));
                }
                break;
            }
            default:
                return NOT_SUPPORTED;
        }
    }

#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (ompCtr.useOmp)
#endif
    {
        F32 *tmpI = inArray;
        for (U32 n = 0; n < in; ++n) {
            F32 *bInArray = inArray + n * ic * ih * iw;
            F32 *bOutArray = outArray + n * oc * oh * ow;
            if (strideH > 1 || strideW > 1) {
                U32 ic8 = ic / 8;
                tmpI = (F32 *)tmp;
#ifdef _USE_OPENMP
#pragma omp for schedule(static) nowait
#endif
                for (U32 hc = 0; hc < oh * ic8; ++hc) {
                    U32 c = hc / oh;
                    U32 h = hc % oh;
                    for (U32 w = 0; w < ow; ++w) {
                        U32 nh = h * strideH;
                        U32 nw = w * strideW;
                        UNI_MEMCPY(tmpI + c * ohow * SIMDW + (h * ow + w) * SIMDW,
                            bInArray + c * ihiw * SIMDW + (nh * iw + nw) * SIMDW,
                            SIMDW * sizeof(F32));
                    }
                }
                paddingT = (paddingT + strideH - 1) / strideH;
                paddingB = (paddingB + strideH - 1) / strideH;
                paddingL = (paddingL + strideW - 1) / strideW;
                paddingR = (paddingR + strideW - 1) / strideW;
            } else {
                tmpI = bInArray;
            }

            U32 ocbSize = 0;
            for (U32 ocbb = 0; ocbb < ocBlockNums; ocbb += ocBBlockNums) {
                ocbSize = UNI_MIN(ocBBlockNums, ocBlockNums - ocbb);
                U32 hwocBlockNums = hwBlockNums * ocbSize;
                U32 icSize = 0;
                for (U32 icb = 0; icb < ic; icb += icSize) {
                    icSize = UNI_MIN(ic - icb, BLOCK_IC_DIM);
                    U32 flags = (icb > 0) | (eltwiseInput != nullptr) << 1;
                    if (icb == ic - icSize) {
                        flags |= U32(activationDesc.mode) << 2;
                    }

                    F32 *curI = tmpI + icb * newIw * newIh;
                    if (phT > 0 || phB > 0) {
                        U32 minUpper = UNI_MIN((ocbb + ocbSize) * unrollOc, oc);
                        for (U32 oci = ocbb * unrollOc; oci < minUpper; oci += SIMDW) {
                            __m256 biasVec = _mm256_load_ps(btmp + oci);
                            for (U32 hw = 0; hw < phT * ow; ++hw) {
                                _mm256_storeu_ps(bOutArray + oci * ohow + hw * SIMDW, biasVec);
                            }
                            for (U32 hw = (oh - phB) * ow; hw < oh * ow; ++hw) {
                                _mm256_storeu_ps(bOutArray + oci * ohow + hw * SIMDW, biasVec);
                            }
                        }
                    }
                    if (paddingL == 0 && paddingR == 0) {
#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
                        for (U32 bIdx = 0; bIdx < hwocBlockNums; ++bIdx) {
                            FTZ;
                            U32 hw = (bIdx / ocbSize) * blockHwDim;
                            U32 hwSize = UNI_MIN(blockHwDim, ohowMain - hw);
                            U32 ocBlockIdx = bIdx % ocbSize + ocbb;
                            U32 ocb = GetOcIdx(ocBlockIdx, oc, unrollOc, ocbArray);
                            U32 ocSize = UNI_MIN(unrollOc, oc - ocb);
                            U32 unrollHw = unrollHwArray[(ocSize >> 3) - 1];
                            ocSize = unrollOcArray[(ocSize >> 3) - 1];

                            const F32 *curB = biasArray + ocb;
                            const F32 *curW = filterArray + ocb * ic + icb * ocSize;
                            F32 *curO = bOutArray + ocb * oh * ow + phT * ow * SIMDW;
                            F32 *curE = eltwiseInput + ocb * oh * ow + phT * ow * SIMDW;
                            U32 ihwSize = 0;
                            for (U32 ihw = hw; ihw < hw + hwSize; ihw += ihwSize) {
                                if ((hw + hwSize - ihw) >= unrollHw) {
                                    ihwSize = unrollHw;
                                } else {
                                    ihwSize = 1;
                                }
                                F32 *calI = curI + ihw * SIMDW;
                                F32 *calO = curO + ihw * SIMDW;
                                F32 *calE = curE + ihw * SIMDW;
                                kernel[ihwSize > 1][(ocSize >> 3) - 1](
                                    calI, curW, calO, curB, calE, oStep, flags, icSize, fStep);
                            }
                        }
                    } else {
#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
                        for (U32 bIdx = 0; bIdx < hwocBlockNums; ++bIdx) {
                            FTZ;
                            U32 h = bIdx / ocbSize + phT;
                            U32 ocBlockIdx = bIdx % ocbSize + ocbb;
                            U32 ocb = GetOcIdx(ocBlockIdx, oc, unrollOc, ocbArray);
                            U32 ocSize = UNI_MIN(unrollOc, oc - ocb);
                            U32 unrollHw = unrollHwArray[(ocSize >> 3) - 1];
                            ocSize = unrollOcArray[(ocSize >> 3) - 1];

                            const F32 *curB = biasArray + ocb;
                            const F32 *curW = filterArray + ocb * ic + icb * ocSize;
                            F32 *curO = bOutArray + ocb * oh * ow;
                            F32 *curE = eltwiseInput + ocb * oh * ow;
                            U32 ihwSize = 0;
                            for (U32 w = 0; w < ow; w += ihwSize) {
                                ihwSize = 1;
                                F32 *calO = curO + (h * ow + w) * SIMDW;
                                // directly store activated bias
                                if ((h < paddingT) || (h >= newIh + paddingT) || (w < paddingL) ||
                                    (w >= paddingL + newIw)) {
                                    for (U32 oci = 0; oci < ocSize; oci += SIMDW) {
                                        _mm256_storeu_ps(
                                            calO + ohow * oci, _mm256_load_ps(btmp + oci + ocb));
                                    }
                                    continue;
                                }
                                if ((newIw - (w - paddingL)) >= unrollHw) {
                                    ihwSize = unrollHw;
                                }
                                F32 *calI = curI + ((h - paddingT) * newIw + w - paddingL) * SIMDW;
                                F32 *calE = curE + (h * ow + w) * SIMDW;
                                kernel[ihwSize > 1][(ocSize >> 3) - 1](
                                    calI, curW, calO, curB, calE, oStep, flags, icSize, fStep);
                            }
                        }
                    }
                }
            }
        }
    }
    return SUCCESS;
}
