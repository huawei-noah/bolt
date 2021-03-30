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

#include "tensor_computing.h"

#include "cpu/x86/fp32/tensor_computing_fp32.h"

#define UNROLL_W 4
#define SIMD_W 8
#define UNROLL_OC_BLOCK_DIM 24

typedef void (*kernel_func)(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 wStep);

void avx2_dw_kernel_4x24(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 wStep)
{
    __asm__ __volatile__("vmovups (%5), %%ymm0                       \n\t"
                         "vmovups (%5), %%ymm1                       \n\t"
                         "vmovups (%5), %%ymm2                       \n\t"
                         "vmovups (%5), %%ymm3                       \n\t"
                         "vmovups 0x20(%5), %%ymm4                   \n\t"
                         "vmovups 0x20(%5), %%ymm5                   \n\t"
                         "vmovups 0x20(%5), %%ymm6                   \n\t"
                         "vmovups 0x20(%5), %%ymm7                   \n\t"
                         "vmovups 0x40(%5), %%ymm8                   \n\t"
                         "vmovups 0x40(%5), %%ymm9                   \n\t"
                         "vmovups 0x40(%5), %%ymm10                 \n\t"
                         "vmovups 0x40(%5), %%ymm11                 \n\t"

                         "cmp $0, %%ecx                                      \n\t"
                         "je 3f                                             \n\t"
                         "cmp $0, %6                                      \n\t"
                         "je 3f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %6, %%eax                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovaps (%4), %%ymm12                         \n\t"
                         "vmovups (%0), %%ymm13                        \n\t"
                         "vmovups (%1), %%ymm14                        \n\t"
                         "vmovups (%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vmovups (%3), %%ymm13                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                         "vmovaps 0x20(%4), %%ymm14                         \n\t"
                         "vmovups (%0, %8), %%ymm15                        \n\t"
                         "vmovups (%1, %8), %%ymm13                        \n\t"
                         "vmovups (%2, %8), %%ymm12                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm13, %%ymm14, %%ymm5              \n\t"
                         "vmovups (%3, %8), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm12, %%ymm14, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm7              \n\t"

                         "vmovaps 0x40(%4), %%ymm13                         \n\t"
                         "vmovups (%0, %8, 2), %%ymm12                        \n\t"
                         "vmovups (%1, %8, 2), %%ymm15                        \n\t"
                         "vmovups (%2, %8, 2), %%ymm14                        \n\t"
                         "vfmadd231ps %%ymm12, %%ymm13, %%ymm8              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm9              \n\t"
                         "vmovups (%3, %8, 2), %%ymm12                        \n\t"
                         "vfmadd231ps %%ymm14, %%ymm13, %%ymm10              \n\t"
                         "vfmadd231ps %%ymm12, %%ymm13, %%ymm11              \n\t"

                         "add %12, %0                                      \n\t"
                         "add %12, %1                                      \n\t"
                         "add %12, %2                                      \n\t"
                         "add %12, %3                                      \n\t"
                         "add $0x60, %4                                    \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %10, %4                                    \n\t"
                         "add %9, %0                                     \n\t"
                         "add %9, %1                                     \n\t"
                         "add %9, %2                                     \n\t"
                         "add %9, %3                                     \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         // relu
                         "mov %11, %%eax                                     \n\t"
                         "and $0x6, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
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
                         "and $0x4, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
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
                         "vminps %%ymm12, %%ymm8, %%ymm8                    \n\t"
                         "vminps %%ymm12, %%ymm9, %%ymm9                    \n\t"
                         "vminps %%ymm12, %%ymm10, %%ymm10                    \n\t"
                         "vminps %%ymm12, %%ymm11, %%ymm11                    \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                 \n\t"
                         :
                         : "r"(in0), "r"(in1), "r"(in2), "r"(in3), "r"(curW), "r"(curB), "r"(fw),
                         "c"(fh), "r"((I64)iStep), "r"((I64)hStep), "r"((I64)wStep), "r"(store),
                         "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__("vmovups %%ymm0, (%0)                              \n\t"
                         "vmovups %%ymm1, 0x20(%0)                          \n\t"
                         "vmovups %%ymm2, 0x40(%0)                          \n\t"
                         "vmovups %%ymm3, 0x60(%0)                              \n\t"
                         "vmovups %%ymm4, (%0, %1)                          \n\t"
                         "vmovups %%ymm5, 0x20(%0, %1)                          \n\t"
                         "vmovups %%ymm6, 0x40(%0, %1)                              \n\t"
                         "vmovups %%ymm7, 0x60(%0, %1)                          \n\t"
                         "vmovups %%ymm8, (%0, %1, 2)                          \n\t"
                         "vmovups %%ymm9, 0x20(%0, %1, 2)                              \n\t"
                         "vmovups %%ymm10, 0x40(%0, %1, 2)                         \n\t"
                         "vmovups %%ymm11, 0x60(%0, %1, 2)                         \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"((I64)oStep)
                         : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
                         "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
                         "%ymm15", "memory", "cc");
}

void avx2_dw_kernel_4x16(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 wStep)
{
    __asm__ __volatile__("vmovups (%5), %%ymm0                       \n\t"
                         "vmovups (%5), %%ymm1                       \n\t"
                         "vmovups (%5), %%ymm2                       \n\t"
                         "vmovups (%5), %%ymm3                       \n\t"
                         "vmovups 0x20(%5), %%ymm4                   \n\t"
                         "vmovups 0x20(%5), %%ymm5                   \n\t"
                         "vmovups 0x20(%5), %%ymm6                   \n\t"
                         "vmovups 0x20(%5), %%ymm7                   \n\t"

                         "cmp $0, %%ecx                                      \n\t"
                         "je 3f                                             \n\t"
                         "cmp $0, %6                                      \n\t"
                         "je 3f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %6, %%eax                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovaps (%4), %%ymm12                         \n\t"
                         "vmovups (%0), %%ymm13                        \n\t"
                         "vmovups (%1), %%ymm14                        \n\t"
                         "vmovups (%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vmovups (%3), %%ymm13                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                         "vmovaps 0x20(%4), %%ymm14                         \n\t"
                         "vmovups (%0, %8), %%ymm15                        \n\t"
                         "vmovups (%1, %8), %%ymm13                        \n\t"
                         "vmovups (%2, %8), %%ymm12                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm13, %%ymm14, %%ymm5              \n\t"
                         "vmovups (%3, %8), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm12, %%ymm14, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm7              \n\t"

                         "add %12, %0                                      \n\t"
                         "add %12, %1                                      \n\t"
                         "add %12, %2                                      \n\t"
                         "add %12, %3                                      \n\t"
                         "add $0x40, %4                                    \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %10, %4                                    \n\t"
                         "add %9, %0                                     \n\t"
                         "add %9, %1                                     \n\t"
                         "add %9, %2                                     \n\t"
                         "add %9, %3                                     \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         // relu
                         "mov %11, %%eax                                     \n\t"
                         "and $0x6, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
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
                         "and $0x4, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
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

                         ".align 16                                         \n\t"
                         "3:                                                 \n\t"
                         :
                         : "r"(in0), "r"(in1), "r"(in2), "r"(in3), "r"(curW), "r"(curB), "r"(fw),
                         "c"(fh), "r"((I64)iStep), "r"((I64)hStep), "r"((I64)wStep), "r"(store),
                         "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");

    __asm__ __volatile__("vmovups %%ymm0, (%0)                              \n\t"
                         "vmovups %%ymm1, 0x20(%0)                          \n\t"
                         "vmovups %%ymm2, 0x40(%0)                          \n\t"
                         "vmovups %%ymm3, 0x60(%0)                              \n\t"
                         "vmovups %%ymm4, (%0, %1)                          \n\t"
                         "vmovups %%ymm5, 0x20(%0, %1)                          \n\t"
                         "vmovups %%ymm6, 0x40(%0, %1)                              \n\t"
                         "vmovups %%ymm7, 0x60(%0, %1)                          \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"((I64)oStep)
                         : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
                         "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
                         "%ymm15", "memory", "cc");
}

void avx2_dw_kernel_4x8(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 wStep)
{
    __asm__ __volatile__("vmovups (%6), %%ymm0                       \n\t"
                         "vmovups (%6), %%ymm1                       \n\t"
                         "vmovups (%6), %%ymm2                       \n\t"
                         "vmovups (%6), %%ymm3                       \n\t"

                         "cmp $0, %%ecx                                      \n\t"
                         "je 3f                                             \n\t"
                         "cmp $0, %6                                      \n\t"
                         "je 3f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %7, %%eax                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovaps (%4), %%ymm12                         \n\t"
                         "vmovups (%0), %%ymm13                        \n\t"
                         "vmovups (%1), %%ymm14                        \n\t"
                         "vmovups (%2), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
                         "vmovups (%3), %%ymm13                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                         "add %13, %0                                      \n\t"
                         "add %13, %1                                      \n\t"
                         "add %13, %2                                      \n\t"
                         "add %13, %3                                      \n\t"
                         "add $0x20, %4                                    \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %9, %4                                    \n\t"
                         "add %11, %0                                     \n\t"
                         "add %11, %1                                     \n\t"
                         "add %11, %2                                     \n\t"
                         "add %11, %3                                     \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         // relu
                         "mov %12, %%eax                                     \n\t"
                         "and $0x6, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
                         "vmaxps %%ymm15, %%ymm2, %%ymm2                    \n\t"
                         "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"

                         // relu6
                         "and $0x4, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
                         "vminps %%ymm12, %%ymm2, %%ymm2                    \n\t"
                         "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%5)                              \n\t"
                         "vmovups %%ymm1, 0x20(%5)                          \n\t"
                         "vmovups %%ymm2, 0x40(%5)                          \n\t"
                         "vmovups %%ymm3, 0x60(%5)                              \n\t"
                         :
                         : "r"(in0), "r"(in1), "r"(in2), "r"(in3), "r"(curW), "r"(curO), "r"(curB),
                         "r"(fw), "c"(fh), "r"((I64)wStep), "r"((I64)oStep), "r"((I64)hStep),
                         "r"(store), "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

void avx2_dw_kernel_1x24(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 wStep)
{
    __asm__ __volatile__("vmovups (%3), %%ymm0                       \n\t"
                         "vmovups 0x20(%3), %%ymm4                   \n\t"
                         "vmovups 0x40(%3), %%ymm8                   \n\t"

                         "cmp $0, %%ecx                                      \n\t"
                         "je 3f                                             \n\t"
                         "cmp $0, %4                                      \n\t"
                         "je 3f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %4, %%eax                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovaps (%1), %%ymm12                         \n\t"
                         "vmovups (%0), %%ymm13                        \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vmovaps 0x20(%1), %%ymm14                         \n\t"
                         "vmovups (%0, %8), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm4              \n\t"

                         "vmovaps 0x40(%1), %%ymm13                         \n\t"
                         "vmovups (%0, %8, 2), %%ymm12                        \n\t"
                         "vfmadd231ps %%ymm12, %%ymm13, %%ymm8              \n\t"

                         "add %11, %0                                      \n\t"
                         "add $0x60, %1                                    \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %6, %1                                    \n\t"
                         "add %9, %0                                     \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         // relu
                         "mov %10, %%eax                                     \n\t"
                         "and $0x6, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
                         "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"

                         // relu6
                         "and $0x4, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
                         "vminps %%ymm12, %%ymm8, %%ymm8                    \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%2)                              \n\t"
                         "vmovups %%ymm4, (%2, %7)                          \n\t"
                         "vmovups %%ymm8, (%2, %7, 2)                          \n\t"
                         :
                         : "r"(in0), "r"(curW), "r"(curO), "r"(curB), "r"(fw), "c"(fh),
                         "r"((I64)wStep), "r"((I64)oStep), "r"((I64)iStep), "r"((I64)hStep),
                         "r"(store), "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

void avx2_dw_kernel_1x16(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 wStep)
{
    __asm__ __volatile__("vmovups (%3), %%ymm0                       \n\t"
                         "vmovups 0x20(%3), %%ymm4                   \n\t"

                         "cmp $0, %%ecx                                      \n\t"
                         "je 3f                                             \n\t"
                         "cmp $0, %4                                      \n\t"
                         "je 3f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %4, %%eax                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovaps (%1), %%ymm12                         \n\t"
                         "vmovups (%0), %%ymm13                        \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
                         "vmovaps 0x20(%1), %%ymm14                         \n\t"
                         "vmovups (%0, %8), %%ymm15                        \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm4              \n\t"

                         "add %11, %0                                      \n\t"
                         "add $0x40, %1                                    \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %6, %1                                    \n\t"
                         "add %9, %0                                     \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         // relu
                         "mov %10, %%eax                                     \n\t"
                         "and $0x6, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"

                         // relu6
                         "and $0x4, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%2)                              \n\t"
                         "vmovups %%ymm4, (%2, %7)                          \n\t"
                         :
                         : "r"(in0), "r"(curW), "r"(curO), "r"(curB), "r"(fw), "c"(fh),
                         "r"((I64)wStep), "r"((I64)oStep), "r"((I64)iStep), "r"((I64)hStep),
                         "r"(store), "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

void avx2_dw_kernel_1x8(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 iStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 wStep)
{
    __asm__ __volatile__("vmovups (%3), %%ymm0                       \n\t"

                         "cmp $0, %%ecx                                      \n\t"
                         "je 3f                                             \n\t"
                         "cmp $0, %4                                      \n\t"
                         "je 3f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %4, %%eax                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovaps (%1), %%ymm12                         \n\t"
                         "vmovups (%0), %%ymm13                        \n\t"
                         "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"

                         "add %11, %0                                      \n\t"
                         "add $0x20, %1                                    \n\t"
                         "dec %%eax                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %6, %1                                    \n\t"
                         "add %9, %0                                     \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         // relu
                         "mov %10, %%eax                                     \n\t"
                         "and $0x6, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"

                         // relu6
                         "and $0x4, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%2)                              \n\t"
                         :
                         : "r"(in0), "r"(curW), "r"(curO), "r"(curB), "r"(fw), "c"(fh),
                         "r"((I64)wStep), "r"((I64)oStep), "r"((I64)iStep), "r"((I64)hStep),
                         "r"(store), "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

EE depthwise_convolution_direct(TensorDesc inputDesc,
    F32 *inArray,
    TensorDesc dwFilterDesc,
    const F32 *dwFilterArray,
    TensorDesc pwFilterDesc,
    const F32 *pwFilterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc dwBiasDesc,
    const F32 *dwBiasArray,
    TensorDesc pwBiasDesc,
    const F32 *pwBiasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    F32 *outArray,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    I32 strideH = convParamSpec.stride_h;
    I32 strideW = convParamSpec.stride_w;
    I32 paddingT = convParamSpec.padding_top;
    I32 paddingB = convParamSpec.padding_bottom;
    I32 paddingL = convParamSpec.padding_left;
    I32 paddingR = convParamSpec.padding_right;
    I32 dilateH = convParamSpec.dilatedRate_h;
    I32 dilateW = convParamSpec.dilatedRate_w;

    if (fdf != DF_NCHWC24 || idf != DF_NCHWC8) {
        CHECK_STATUS(NOT_MATCH);
    }

    F32 *curI, *curO, *calI, *calO;
    const F32 *curW, *curB, *calW;
    F32 *ftmp = inArray;

    U32 icAlignSize = 8;
    U32 icPadding = (ic + icAlignSize - 1) / icAlignSize * icAlignSize;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    F32 *useOutArray = (F32 *)tmp;
    if (pwFilterArray == nullptr) {
        useOutArray = outArray;
    }

    U32 oStep = oh * ow * SIMD_W * 4;
    U32 ocblocking = 0;
    U32 iStep = ih * iw * SIMD_W * 4;
    I32 hStep = (iw - fw * dilateW + (dilateH - 1) * iw) * SIMD_W * 4;
    U32 sw = strideW * SIMD_W * 4;
    U32 dw = dilateW * SIMD_W * 4;
    U32 wSize = 0, store = 0, ocSize = 0;
    U32 ocblocks[3] = {8, 16, 24};

    U32 ohow = oh * ow;

    F32 *curIn[4];
    U32 in_h = 0, in_w = 0, oc_idx = 0;

    kernel_func kernel[2][3] = {{avx2_dw_kernel_1x8, avx2_dw_kernel_1x16, avx2_dw_kernel_1x24},
        {avx2_dw_kernel_4x8, avx2_dw_kernel_4x16, avx2_dw_kernel_4x24}};

    store |= U32(depthwiseActivationParamSpec.mode) << 1;
    for (U32 n = 0; n < in; ++n) {
        for (U32 ocb = 0; ocb < icPadding; ocb += ocSize) {
            curW = dwFilterArray + ocb * fh * fw;
            curB = dwBiasArray + ocb;
            curI = ftmp + ocb * ih * iw;
            curO = useOutArray + (n * icPadding + ocb) * oh * ow;
            ocSize = UNI_MIN(UNROLL_OC_BLOCK_DIM, icPadding - ocb);
            oc_idx = (ocSize >> 3) - 1;
            ocSize = ocblocks[oc_idx];
            if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
                for (U32 hw = 0; hw < ohow; hw += wSize) {
                    wSize = UNI_MIN(ohow - hw, UNROLL_W);
                    if (wSize < 4) {
                        wSize = 1;
                    }
                    U32 in_h_0 = hw / ow * strideH;
                    U32 in_w_0 = hw % ow * strideW;
                    U32 in_h_1 = (hw + 1) / ow * strideH;
                    U32 in_w_1 = (hw + 1) % ow * strideW;
                    U32 in_h_2 = (hw + 2) / ow * strideH;
                    U32 in_w_2 = (hw + 2) % ow * strideW;
                    U32 in_h_3 = (hw + 3) / ow * strideH;
                    U32 in_w_3 = (hw + 3) % ow * strideW;
                    F32 *in_0 = curI + in_h_0 * iw_pad * 8 + in_w_0 * 8;
                    F32 *in_1 = curI + in_h_1 * iw_pad * 8 + in_w_1 * 8;
                    F32 *in_2 = curI + in_h_2 * iw_pad * 8 + in_w_2 * 8;
                    F32 *in_3 = curI + in_h_3 * iw_pad * 8 + in_w_3 * 8;
                    calO = curO + hw * 8;

                    kernel[wSize >> 2][oc_idx](in_0, in_1, in_2, in_3, curW, calO, curB, fw, fh,
                        oStep, iStep, hStep, store, dw, 0);
                }
            } else {
                I32 tfw = fw, tfh = fh;
                I32 in_h = 0, in_w = iw;
                I32 o_h = oh, o_w = ow;
                I32 ow_padding_l = UNI_MIN((paddingL - 1) / strideW + 1, o_w);
                I32 ow_padding_r = UNI_MIN((paddingR - 1) / strideW + 1, o_w - ow_padding_l);
                if (((in_w + paddingL - tfw) / strideW + 1) >= o_w) {
                    ow_padding_r = 0;
                }
                for (I32 h = 0; h < o_h; ++h) {
                    tfh = fh;
                    in_h = h * strideH - paddingT;
                    calW = curW;
                    if (in_h < 0) {
                        tfh = UNI_MIN(fh + in_h, ih);
                        calW = curW + (fh - tfh) * fw * ocSize;
                        in_h = 0;
                    } else if (in_h + fh >= ih) {
                        tfh = ih - in_h;
                    }
                    I32 w = 0;
                    for (; w < ow_padding_l; ++w) {
                        I32 in_w = w * strideW - paddingL;
                        tfw = UNI_MIN(fw + in_w, iw);
                        const F32 *useW = calW;
                        if (in_w < 0) {
                            useW = calW - in_w * (I32)ocSize;
                        }
                        hStep = (iw - tfw * dilateW + (dilateH - 1) * iw) * SIMD_W * 4;
                        F32 *in_0 = curI + in_h * iw * 8;
                        calO = curO + (h * ow + w) * 8;
                        kernel[0][oc_idx](in_0, nullptr, nullptr, nullptr, useW, calO, curB, tfw,
                            tfh, oStep, iStep, hStep, store, dw, (fw - tfw) * ocSize * 4);
                    }
                    for (; w < o_w - ow_padding_r; w += wSize) {
                        hStep = (iw - fw * dilateW + (dilateH - 1) * iw) * SIMD_W * 4;
                        wSize = UNI_MIN(o_w - ow_padding_r - w, UNROLL_W);
                        if (wSize < 4) {
                            wSize = 1;
                        }
                        U32 in_w_0 = w * strideW - paddingL;
                        U32 in_w_1 = (w + 1) * strideW - paddingL;
                        U32 in_w_2 = (w + 2) * strideW - paddingL;
                        U32 in_w_3 = (w + 3) * strideW - paddingL;
                        F32 *in_0 = curI + in_h * iw * 8 + in_w_0 * 8;
                        F32 *in_1 = curI + in_h * iw * 8 + in_w_1 * 8;
                        F32 *in_2 = curI + in_h * iw * 8 + in_w_2 * 8;
                        F32 *in_3 = curI + in_h * iw * 8 + in_w_3 * 8;
                        calO = curO + (h * ow + w) * 8;

                        kernel[wSize >> 2][oc_idx](in_0, in_1, in_2, in_3, calW, calO, curB, fw,
                            tfh, oStep, iStep, hStep, store, dw, 0);
                    }
                    for (; w < o_w; ++w) {
                        I32 in_w = w * strideW - paddingL;
                        tfw = iw - in_w;
                        hStep = (iw - tfw * dilateW + (dilateH - 1) * iw) * SIMD_W * 4;
                        F32 *in_0 = curI + in_h * iw * 8 + in_w * 8;
                        calO = curO + (h * o_w + w) * 8;
                        kernel[0][oc_idx](in_0, nullptr, nullptr, nullptr, calW, calO, curB, tfw,
                            tfh, oStep, iStep, hStep, store, dw, (fw - tfw) * ocSize * 4);
                    }
                }
            }
        }
    }

    if (pwFilterArray != nullptr) {
        TensorDesc pwInputDesc = tensor4df(odt, DF_NCHWC8, 1, ic, oh, ow);
        tmpBytes -= oh * ic * oh * ow + 32;
        tmp = (void *)((F32 *)tmp + oh * ic * oh * ow + 32);
        ConvolutionParamSpec p = createConvolutionParamSpec(
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, fn, Convolution_Pointwise);
        convolution_1x1_direct(pwInputDesc, useOutArray, nullptr, pwFilterDesc, pwFilterArray, p,
            pwBiasArray, tmpBytes, tmp, outputDesc, outArray, pointwiseActivationParamSpec);
    }
    return SUCCESS;
}
