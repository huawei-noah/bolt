// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/fp32/convolution_functions.h"
#include "tensor_transpose.h"

#define UNROLL_W 4
#define UNROLL_OC_BLOCK_DIM 16

typedef void (*kernelFunc)(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep);

typedef void (*kernel33Func)(F32 *in0,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 fh);

void Avx2DwKernel4x24(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep)
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
                         "c"(fh), "r"((I64)iStep), "r"((I64)hStep), "r"((I64)wStep), "r"(flags),
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

void Avx2DwKernel4x16(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep)
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
                         "c"(fh), "r"((I64)iStep), "r"((I64)hStep), "r"((I64)wStep), "r"(flags),
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

void Avx512DwKernel4x16(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep)
{
    __asm__ __volatile__("vmovups (%5), %%zmm0                       \n\t"
                         "vmovups %%zmm0, %%zmm1                       \n\t"
                         "vmovups %%zmm0, %%zmm2                       \n\t"
                         "vmovups %%zmm0, %%zmm3                       \n\t"

                         "cmp $0, %%ecx                                      \n\t"
                         "je 3f                                             \n\t"
                         "cmp $0, %6                                      \n\t"
                         "je 3f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %6, %%eax                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovaps (%4), %%zmm11                         \n\t"
                         "vmovups (%0), %%zmm12                        \n\t"
                         "vmovups (%1), %%zmm13                        \n\t"
                         "vmovups (%2), %%zmm14                        \n\t"
                         "vmovups (%3), %%zmm15                        \n\t"
                         "vfmadd231ps %%zmm12, %%zmm11, %%zmm0              \n\t"
                         "vfmadd231ps %%zmm13, %%zmm11, %%zmm1              \n\t"
                         "prefetcht0 0x40(%4) \n\t"
                         "vfmadd231ps %%zmm14, %%zmm11, %%zmm2              \n\t"
                         "vfmadd231ps %%zmm15, %%zmm11, %%zmm3              \n\t"

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
                         "vxorps %%zmm15, %%zmm15, %%zmm15                  \n\t"
                         "vmaxps %%zmm15, %%zmm0, %%zmm0                    \n\t"
                         "vmaxps %%zmm15, %%zmm1, %%zmm1                    \n\t"
                         "vmaxps %%zmm15, %%zmm2, %%zmm2                    \n\t"
                         "vmaxps %%zmm15, %%zmm3, %%zmm3                    \n\t"
                         "vmaxps %%zmm15, %%zmm4, %%zmm4                    \n\t"
                         "vmaxps %%zmm15, %%zmm5, %%zmm5                    \n\t"
                         "vmaxps %%zmm15, %%zmm6, %%zmm6                    \n\t"
                         "vmaxps %%zmm15, %%zmm7, %%zmm7                    \n\t"

                         // relu6
                         "and $0x4, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%zmm12, %%zmm15, %%zmm12                 \n\t"
                         "vminps %%zmm12, %%zmm0, %%zmm0                    \n\t"
                         "vminps %%zmm12, %%zmm1, %%zmm1                    \n\t"
                         "vminps %%zmm12, %%zmm2, %%zmm2                    \n\t"
                         "vminps %%zmm12, %%zmm3, %%zmm3                    \n\t"
                         "vminps %%zmm12, %%zmm4, %%zmm4                    \n\t"
                         "vminps %%zmm12, %%zmm5, %%zmm5                    \n\t"
                         "vminps %%zmm12, %%zmm6, %%zmm6                    \n\t"
                         "vminps %%zmm12, %%zmm7, %%zmm7                    \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                 \n\t"
                         :
                         : "r"(in0), "r"(in1), "r"(in2), "r"(in3), "r"(curW), "r"(curB), "r"(fw),
                         "c"(fh), "r"((I64)iStep), "r"((I64)hStep), "r"((I64)wStep), "r"(flags),
                         "r"((I64)dw)
                         : "%eax", "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                         "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13",
                         "%zmm14", "%zmm15", "memory", "cc");

    __asm__ __volatile__("vmovups %%zmm0, (%0)                              \n\t"
                         "vmovups %%zmm1, 0x40(%0)                          \n\t"
                         "vmovups %%zmm2, 0x80(%0)                          \n\t"
                         "vmovups %%zmm3, 0xC0(%0)                          \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         :
                         : "r"(curO), "r"((I64)oStep)
                         : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "memory", "cc");
}

void Avx2DwKernel4x8(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep)
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
                         "r"(flags), "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

void Avx2DwKernel1x24(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep)
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
                         "r"(flags), "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

void Avx2DwKernel1x16(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep)
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
                         "r"(flags), "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

void Avx512DwKernel1x16(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep)
{
    __asm__ __volatile__("vmovups (%3), %%zmm0                       \n\t"

                         "cmp $0, %%ecx                                      \n\t"
                         "je 3f                                             \n\t"
                         "cmp $0, %4                                      \n\t"
                         "je 3f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %4, %%eax                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovaps (%1), %%zmm1                         \n\t"
                         "vmovups (%0), %%zmm2                        \n\t"
                         "vfmadd231ps %%zmm2, %%zmm1, %%zmm0              \n\t"

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
                         "vxorps %%zmm3, %%zmm3, %%zmm3                  \n\t"
                         "vmaxps %%zmm3, %%zmm0, %%zmm0                    \n\t"

                         // relu6
                         "and $0x4, %%eax                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm3                              \n\t"
                         "vbroadcastss %%xmm3, %%zmm4                 \n\t"
                         "vminps %%zmm4, %%zmm0, %%zmm0                    \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "vmovups %%zmm0, (%2)                              \n\t"
                         :
                         : "r"(in0), "r"(curW), "r"(curO), "r"(curB), "r"(fw), "c"(fh),
                         "r"((I64)wStep), "r"((I64)oStep), "r"((I64)iStep), "r"((I64)hStep),
                         "r"(flags), "r"((I64)dw)
                         : "%eax", "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "memory", "cc");
}

void Avx2DwKernel1x8(F32 *in0,
    F32 *in1,
    F32 *in2,
    F32 *in3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 fw,
    I32 fh,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 dw,
    I32 wStep)
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
                         "r"(flags), "r"((I64)dw)
                         : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                         "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13",
                         "%ymm14", "%ymm15", "memory", "cc");
}

inline void Avx2DwKernel33s14x24(F32 *in0,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 fh)
{
    __asm__ __volatile__(
        "vmovups (%2), %%ymm0                       \n\t"
        "vmovups (%2), %%ymm1                       \n\t"
        "vmovups (%2), %%ymm2                       \n\t"
        "vmovups (%2), %%ymm3                       \n\t"
        "vmovups 0x20(%2), %%ymm4                   \n\t"
        "vmovups 0x20(%2), %%ymm5                   \n\t"
        "vmovups 0x20(%2), %%ymm6                   \n\t"
        "vmovups 0x20(%2), %%ymm7                   \n\t"
        "vmovups 0x40(%2), %%ymm8                   \n\t"
        "vmovups 0x40(%2), %%ymm9                   \n\t"
        "vmovups 0x40(%2), %%ymm10                 \n\t"
        "vmovups 0x40(%2), %%ymm11                 \n\t"

        "mov %0, %%rax                                      \n\t"

        ".align 16                                         \n\t"
        "0:                                               "
        "mov %%rax, %0                                      \n\t"
        "vmovaps (%1), %%ymm12                         \n\t"
        "vmovups (%0), %%ymm13                        \n\t"
        "vmovups 0x20(%0), %%ymm14                        \n\t"
        "vmovups 0x40(%0), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm1              \n\t"
        "vmovups 0x60(%0), %%ymm13                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

        "vmovaps 0x60(%1), %%ymm12                         \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
        "vmovups 0x80(%0), %%ymm14                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm2              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm3              \n\t"

        "vmovaps 0xC0(%1), %%ymm12                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm1              \n\t"
        "vmovups 0xA0(%0), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm2              \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "add %3, %0                                      \n\t"

        "vmovaps 0x20(%1), %%ymm12                         \n\t"
        "vmovups (%0), %%ymm13                        \n\t"
        "vmovups 0x20(%0), %%ymm14                        \n\t"
        "vmovups 0x40(%0), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm4              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm5              \n\t"
        "vmovups 0x60(%0), %%ymm13                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm7              \n\t"

        "vmovaps 0x80(%1), %%ymm12                         \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm5              \n\t"
        "vmovups 0x80(%0), %%ymm14                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm7              \n\t"

        "vmovaps 0xE0(%1), %%ymm12                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm4              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm5              \n\t"
        "vmovups 0xA0(%0), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm7              \n\t"
        "add %3, %0                                      \n\t"

        "vmovaps 0x40(%1), %%ymm12                         \n\t"
        "vmovups (%0), %%ymm13                        \n\t"
        "vmovups 0x20(%0), %%ymm14                        \n\t"
        "vmovups 0x40(%0), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm8              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"
        "vmovups 0x60(%0), %%ymm13                        \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm10              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm11              \n\t"

        "vmovaps 0xA0(%1), %%ymm12                         \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm8              \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vmovups 0x80(%0), %%ymm14                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm10              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm11              \n\t"

        "vmovaps 0x100(%1), %%ymm12                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm8              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm9              \n\t"
        "vmovups 0xA0(%0), %%ymm15                        \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm10              \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm11              \n\t"

        "add %4, %%rax                                      \n\t"
        "add $0x120, %1                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 0b                                             \n\t"

        // relu
        "mov %5, %%eax                                     \n\t"
        "and $0x6, %%eax                                      \n\t"
        "je 1f                                             \n\t"
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
        "je 1f                                             \n\t"
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
        "1:                                                 \n\t"
        : "+r"(in0)
        : "r"(curW), "r"(curB), "r"((I64)iStep), "r"((I64)hStep), "r"(flags), "c"(fh)
        : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
        "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory",
        "cc");

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
                         :
                         : "r"(curO), "r"((I64)oStep)
                         : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
                         "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
                         "%ymm15", "memory", "cc");
}

inline void Avx512DwKernel33s14x16(F32 *in0,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 fh)
{
    __asm__ __volatile__(
        "vmovups (%2), %%zmm0                       \n\t"
        "vmovups %%zmm0, %%zmm1                       \n\t"
        "vmovups %%zmm0, %%zmm2                       \n\t"
        "vmovups %%zmm0, %%zmm3                       \n\t"

        ".align 16                                         \n\t"
        "0:                                               "

        "vmovaps (%1), %%zmm15                         \n\t"
        "vmovups (%0), %%zmm8                        \n\t"
        "vmovups 0x40(%0), %%zmm9                        \n\t"
        "vmovups 0x80(%0), %%zmm10                        \n\t"
        "vmovups 0xC0(%0), %%zmm11                        \n\t"
        "vfmadd231ps %%zmm8, %%zmm15, %%zmm0              \n\t"
        "vfmadd231ps %%zmm9, %%zmm15, %%zmm1              \n\t"
        "vfmadd231ps %%zmm10, %%zmm15, %%zmm2              \n\t"
        "vfmadd231ps %%zmm11, %%zmm15, %%zmm3              \n\t"

        "vmovaps 0x40(%1), %%zmm15                         \n\t"
        "vmovups 0x100(%0), %%zmm8                        \n\t"
        "vfmadd231ps %%zmm9, %%zmm15, %%zmm0              \n\t"
        "vfmadd231ps %%zmm10, %%zmm15, %%zmm1              \n\t"
        "vfmadd231ps %%zmm11, %%zmm15, %%zmm2              \n\t"
        "vfmadd231ps %%zmm8, %%zmm15, %%zmm3              \n\t"

        "vmovaps 0x80(%1), %%zmm15                         \n\t"
        "vmovups 0x140(%0), %%zmm12                        \n\t"
        "vfmadd231ps %%zmm10, %%zmm15, %%zmm0              \n\t"
        "vfmadd231ps %%zmm11, %%zmm15, %%zmm1              \n\t"
        "vfmadd231ps %%zmm8, %%zmm15, %%zmm2              \n\t"
        "vfmadd231ps %%zmm12, %%zmm15, %%zmm3              \n\t"

        "add %4, %0                                      \n\t"
        "add $0xC0, %1                                      \n\t"

        "dec %%ecx                                         \n\t"
        "jg 0b                                             \n\t"

        // relu
        "mov %5, %%eax                                     \n\t"
        "and $0x6, %%eax                                      \n\t"
        "je 1f                                             \n\t"
        "vxorps %%zmm15, %%zmm15, %%zmm15                  \n\t"
        "vmaxps %%zmm15, %%zmm0, %%zmm0                    \n\t"
        "vmaxps %%zmm15, %%zmm1, %%zmm1                    \n\t"
        "vmaxps %%zmm15, %%zmm2, %%zmm2                    \n\t"
        "vmaxps %%zmm15, %%zmm3, %%zmm3                    \n\t"
        "vmaxps %%zmm15, %%zmm4, %%zmm4                    \n\t"
        "vmaxps %%zmm15, %%zmm5, %%zmm5                    \n\t"
        "vmaxps %%zmm15, %%zmm6, %%zmm6                    \n\t"
        "vmaxps %%zmm15, %%zmm7, %%zmm7                    \n\t"

        // relu6
        "and $0x4, %%eax                                      \n\t"
        "je 1f                                             \n\t"
        "mov $0x40C00000, %%eax                            \n\t"
        "vmovd %%eax, %%xmm12                              \n\t"
        "vpermps %%zmm12, %%zmm15, %%zmm12                 \n\t"
        "vminps %%zmm12, %%zmm0, %%zmm0                    \n\t"
        "vminps %%zmm12, %%zmm1, %%zmm1                    \n\t"
        "vminps %%zmm12, %%zmm2, %%zmm2                    \n\t"
        "vminps %%zmm12, %%zmm3, %%zmm3                    \n\t"
        "vminps %%zmm12, %%zmm4, %%zmm4                    \n\t"
        "vminps %%zmm12, %%zmm5, %%zmm5                    \n\t"
        "vminps %%zmm12, %%zmm6, %%zmm6                    \n\t"
        "vminps %%zmm12, %%zmm7, %%zmm7                    \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        :
        : "r"(in0), "r"(curW), "r"(curB), "r"((I64)iStep), "r"((I64)hStep), "r"(flags), "c"(fh)
        : "%eax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
        "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14", "%zmm15", "memory",
        "cc");

    __asm__ __volatile__("vmovups %%zmm0, (%0)                              \n\t"
                         "vmovups %%zmm1, 0x40(%0)                          \n\t"
                         "vmovups %%zmm2, 0x80(%0)                          \n\t"
                         "vmovups %%zmm3, 0xC0(%0)                              \n\t"
                         :
                         : "r"(curO), "r"((I64)oStep)
                         : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
                         "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                         "%zmm15", "memory", "cc");
}

inline void Avx2DwKernel33s14x16(F32 *in0,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 fh)
{
    __asm__ __volatile__(
        "vmovups (%2), %%ymm0                       \n\t"
        "vmovups (%2), %%ymm1                       \n\t"
        "vmovups (%2), %%ymm2                       \n\t"
        "vmovups (%2), %%ymm3                       \n\t"
        "vmovups 0x20(%2), %%ymm4                   \n\t"
        "vmovups 0x20(%2), %%ymm5                   \n\t"
        "vmovups 0x20(%2), %%ymm6                   \n\t"
        "vmovups 0x20(%2), %%ymm7                   \n\t"

        "mov %0, %%rax                                      \n\t"
        "add %3, %%rax                                      \n\t"

        ".align 16                                         \n\t"
        "0:                                               "

        "vmovaps (%1), %%ymm15                         \n\t"
        "vmovups (%0), %%ymm8                        \n\t"
        "vmovups 0x20(%0), %%ymm9                        \n\t"
        "vmovups 0x40(%0), %%ymm10                        \n\t"
        "vmovups 0x60(%0), %%ymm11                        \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm3              \n\t"

        "vmovaps 0x20(%1), %%ymm15                         \n\t"
        "vmovups (%%rax), %%ymm8                        \n\t"
        "vmovups 0x20(%%rax), %%ymm12                        \n\t"
        "vmovups 0x40(%%rax), %%ymm13                        \n\t"
        "vmovups 0x60(%%rax), %%ymm14                        \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm7              \n\t"

        "vmovaps 0x40(%1), %%ymm15                         \n\t"
        "vmovups 0x80(%0), %%ymm8                        \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm3              \n\t"

        "vmovaps 0x60(%1), %%ymm15                         \n\t"
        "vmovups 0x80(%%rax), %%ymm9                        \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm7              \n\t"

        "vmovaps 0x80(%1), %%ymm15                         \n\t"
        "vmovups 0xA0(%0), %%ymm12                        \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm3              \n\t"

        "vmovaps 0xA0(%1), %%ymm15                         \n\t"
        "vmovups 0xA0(%%rax), %%ymm10                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm7              \n\t"

        "add %4, %0                                      \n\t"
        "add %4, %%rax                                      \n\t"
        "add $0xC0, %1                                      \n\t"

        "dec %%ecx                                         \n\t"
        "jg 0b                                             \n\t"

        // relu
        "mov %5, %%eax                                     \n\t"
        "and $0x6, %%eax                                      \n\t"
        "je 1f                                             \n\t"
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
        "je 1f                                             \n\t"
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
        "1:                                                 \n\t"
        :
        : "r"(in0), "r"(curW), "r"(curB), "r"((I64)iStep), "r"((I64)hStep), "r"(flags), "c"(fh)
        : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
        "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory",
        "cc");

    __asm__ __volatile__("vmovups %%ymm0, (%0)                              \n\t"
                         "vmovups %%ymm1, 0x20(%0)                          \n\t"
                         "vmovups %%ymm2, 0x40(%0)                          \n\t"
                         "vmovups %%ymm3, 0x60(%0)                              \n\t"
                         "vmovups %%ymm4, (%0, %1)                          \n\t"
                         "vmovups %%ymm5, 0x20(%0, %1)                          \n\t"
                         "vmovups %%ymm6, 0x40(%0, %1)                              \n\t"
                         "vmovups %%ymm7, 0x60(%0, %1)                          \n\t"
                         :
                         : "r"(curO), "r"((I64)oStep)
                         : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
                         "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
                         "%ymm15", "memory", "cc");
}

inline void Avx2DwKernel33s18x8(F32 *in0,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 fh)
{
    __asm__ __volatile__(
        "vmovups (%2), %%ymm0                       \n\t"
        "vmovups (%2), %%ymm1                       \n\t"
        "vmovups (%2), %%ymm2                       \n\t"
        "vmovups (%2), %%ymm3                       \n\t"
        "vmovups (%2), %%ymm4                   \n\t"
        "vmovups (%2), %%ymm5                   \n\t"
        "vmovups (%2), %%ymm6                   \n\t"
        "vmovups (%2), %%ymm7                   \n\t"

        ".align 16                                         \n\t"
        "0:                                               "

        "vmovaps (%1), %%ymm15                         \n\t"
        "vmovups (%0), %%ymm8                        \n\t"
        "vmovups 0x20(%0), %%ymm9                        \n\t"
        "vmovups 0x40(%0), %%ymm10                        \n\t"
        "vmovups 0x60(%0), %%ymm11                        \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm3              \n\t"
        "vmovups 0x80(%0), %%ymm12                        \n\t"
        "vmovups 0xA0(%0), %%ymm13                        \n\t"
        "vmovups 0xC0(%0), %%ymm14                        \n\t"
        "vmovups 0xE0(%0), %%ymm8                        \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm7              \n\t"

        "vmovaps 0x20(%1), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm3              \n\t"
        "vmovups 0x100(%0), %%ymm9                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm7              \n\t"

        "vmovaps 0x40(%1), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm3              \n\t"
        "vmovups 0x120(%0), %%ymm10                        \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm7              \n\t"

        "add %4, %0                                      \n\t"
        "add $0x60, %1                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 0b                                             \n\t"

        // relu
        "mov %5, %%eax                                     \n\t"
        "and $0x6, %%eax                                      \n\t"
        "je 1f                                             \n\t"
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
        "je 1f                                             \n\t"
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
        "1:                                                 \n\t"
        :
        : "r"(in0), "r"(curW), "r"(curB), "r"((I64)iStep), "r"((I64)hStep), "r"(flags), "c"(fh)
        : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
        "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory",
        "cc");

    __asm__ __volatile__(
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm1, 0x20(%0)                          \n\t"
        "vmovups %%ymm2, 0x40(%0)                          \n\t"
        "vmovups %%ymm3, 0x60(%0)                              \n\t"
        "vmovups %%ymm4, 0x80(%0)                          \n\t"
        "vmovups %%ymm5, 0xA0(%0)                          \n\t"
        "vmovups %%ymm6, 0xC0(%0)                              \n\t"
        "vmovups %%ymm7, 0xE0(%0)                          \n\t"
        :
        : "r"(curO)
        : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "memory", "cc");
}

inline void Avx2DwKernel33s28x8(F32 *in0,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    I32 oStep,
    I32 iStep,
    I32 hStep,
    I32 flags,
    I32 fh)
{
    __asm__ __volatile__(
        "vmovups (%2), %%ymm0                       \n\t"
        "vmovups (%2), %%ymm1                       \n\t"
        "vmovups (%2), %%ymm2                       \n\t"
        "vmovups (%2), %%ymm3                       \n\t"
        "vmovups (%2), %%ymm4                   \n\t"
        "vmovups (%2), %%ymm5                   \n\t"
        "vmovups (%2), %%ymm6                   \n\t"
        "vmovups (%2), %%ymm7                   \n\t"

        ".align 16                                         \n\t"
        "0:                                               "

        "vmovaps (%1), %%ymm15                         \n\t"
        "vmovups (%0), %%ymm8                        \n\t"
        "vmovups 0x40(%0), %%ymm9                        \n\t"
        "vmovups 0x80(%0), %%ymm10                        \n\t"
        "vmovups 0xC0(%0), %%ymm11                        \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm3              \n\t"
        "vmovups 0x100(%0), %%ymm12                        \n\t"
        "vmovups 0x140(%0), %%ymm13                        \n\t"
        "vmovups 0x180(%0), %%ymm14                        \n\t"
        "vmovups 0x1C0(%0), %%ymm8                        \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm7              \n\t"

        "vmovaps 0x40(%1), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm3              \n\t"
        "vmovups 0x200(%0), %%ymm9                        \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm7              \n\t"

        "vmovaps 0x20(%1), %%ymm15                         \n\t"
        "vmovups 0x20(%0), %%ymm8                        \n\t"
        "vmovups 0x60(%0), %%ymm9                        \n\t"
        "vmovups 0xA0(%0), %%ymm10                        \n\t"
        "vmovups 0xE0(%0), %%ymm11                        \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm0              \n\t"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm1              \n\t"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm2              \n\t"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm3              \n\t"
        "vmovups 0x120(%0), %%ymm12                        \n\t"
        "vmovups 0x160(%0), %%ymm13                        \n\t"
        "vmovups 0x1A0(%0), %%ymm14                        \n\t"
        "vmovups 0x1E0(%0), %%ymm8                        \n\t"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm4              \n\t"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm5              \n\t"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm6              \n\t"
        "vfmadd231ps %%ymm8, %%ymm15, %%ymm7              \n\t"

        "add %4, %0                                      \n\t"
        "add $0x60, %1                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 0b                                             \n\t"

        // relu
        "mov %5, %%eax                                     \n\t"
        "and $0x6, %%eax                                      \n\t"
        "je 1f                                             \n\t"
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
        "je 1f                                             \n\t"
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
        "1:                                                 \n\t"
        :
        : "r"(in0), "r"(curW), "r"(curB), "r"((I64)iStep), "r"((I64)hStep), "r"(flags), "c"(fh)
        : "%eax", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
        "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory",
        "cc");

    __asm__ __volatile__(
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm1, 0x20(%0)                          \n\t"
        "vmovups %%ymm2, 0x40(%0)                          \n\t"
        "vmovups %%ymm3, 0x60(%0)                              \n\t"
        "vmovups %%ymm4, 0x80(%0)                          \n\t"
        "vmovups %%ymm5, 0xA0(%0)                          \n\t"
        "vmovups %%ymm6, 0xC0(%0)                              \n\t"
        "vmovups %%ymm7, 0xE0(%0)                          \n\t"
        :
        : "r"(curO)
        : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "memory", "cc");
}

EE depthwise_convolution_direct(TensorDesc inputDesc,
    F32 *inArray,
    F32 *eltwiseInput,
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
    I32 in, ic, ih, iw;
    I32 fn, fc, fh, fw;
    I32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGetI32(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGetI32(dwFilterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGetI32(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if ((fdf != DF_NCHWC24 && fdf != DF_NCHWC8) || (idf != DF_NCHWC8 && idf != DF_NCHWC16) || (ic % 8 != 0)) {
        CHECK_STATUS(NOT_MATCH);
    }

    if ((idf == DF_NCHWC16) && (ic % 16 != 0)) {
        if (ic % 8 != 0) {
            CHECK_STATUS(NOT_MATCH);
        }
        TensorDesc desc = inputDesc;
        desc.df = DF_NCHWC8;
        transformFormat(inputDesc, inArray, desc, tmp);
        inArray = (F32 *)tmp;
        inputDesc.df = desc.df;
        idf = DF_NCHWC8;
        tmp = (U8 *)tmp + tensorNumBytes(inputDesc);
    }

    // get kernels
    kernelFunc kernel[3][2] = {{Avx2DwKernel1x8, Avx2DwKernel4x8},
                               {Avx2DwKernel1x16, Avx2DwKernel4x16},
                               {Avx2DwKernel1x24, Avx2DwKernel4x24}};
    kernel33Func kernel33[2][3] = {{Avx2DwKernel33s18x8, Avx2DwKernel33s14x16, Avx2DwKernel33s14x24},
        {Avx2DwKernel33s28x8, nullptr, nullptr}};
    kernelFunc kernel512[2] = {Avx512DwKernel1x16, Avx512DwKernel4x16};
    kernel33Func kernel51233[1] = {Avx512DwKernel33s14x16};
    I32 unrollOcArray[3] = {8, 16, 24};
    I32 unrollHw33s1Array[3] = {8, 4, 4};

    // get computing params
    I32 strideH = convParamSpec.stride_h;
    I32 strideW = convParamSpec.stride_w;
    I32 paddingT = convParamSpec.pad_top;
    I32 paddingB = convParamSpec.pad_bottom;
    I32 paddingL = convParamSpec.pad_left;
    I32 paddingR = convParamSpec.pad_right;
    I32 dilateH = convParamSpec.dilatedRate_h;
    I32 dilateW = convParamSpec.dilatedRate_w;
    I32 fhDilated = (fh - 1) * dilateH + 1;
    I32 fwDilated = (fw - 1) * dilateW + 1;
    I32 ohow = oh * ow;

    // infer block params
    I32 unrollOc = UNROLL_OC_BLOCK_DIM;
    I32 cLen = (idf == DF_NCHWC16)? 16: 8;

    // infer kernel params
    I32 oStep = oh * ow * cLen * BYTES;
    I32 iStep = ih * iw * cLen * BYTES;
    I32 hStep33 = iw * cLen * BYTES;
    I32 sw = strideW * cLen * BYTES;
    I32 dw = dilateW * cLen * BYTES;

    // activation flags
    I32 flags = I32(depthwiseActivationParamSpec.mode) << 1;

    bool use3x3 = (fw == 3 && strideW == 1 && dilateW == 1 && dilateH == 1);

    for (I32 n = 0; n < in; ++n) {
        F32 *useOutArray = (F32 *)tmp;
        if (pwFilterArray == nullptr) {
            useOutArray = outArray + n * oc * oh * ow;
        }
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        I32 ocSize = 0;
        for (I32 ocb = 0; ocb < ic; ocb += ocSize) {
            const F32 *curW = dwFilterArray + ocb * fh * fw;
            const F32 *curB = dwBiasArray + ocb;
            F32 *curI = inArray + (n * ic + ocb) * ih * iw;
            F32 *curO = useOutArray + ocb * oh * ow;
            ocSize = UNI_MIN(unrollOc, ic - ocb);
            I32 ocIdx = (ocSize >> 3) - 1;
            ocSize = unrollOcArray[ocIdx];
            kernelFunc *wkernel = kernel[ocIdx];
            kernel33Func wkernel33 = kernel33[0][ocIdx];
            if (idf == DF_NCHWC16) {
                ocSize = 16;
                wkernel = kernel512;
                wkernel33 = kernel51233[0];
            }
            if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
                I32 hStep = (iw - fw * dilateW + (dilateH - 1) * iw) * cLen * BYTES;
                if (use3x3) {
                    I32 unrollHw = unrollHw33s1Array[ocIdx];
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
                    for (I32 h = 0; h < oh; ++h) {
                        I32 wSize = 0;
                        for (I32 w = 0; w < ow; w += wSize) {
                            wSize = UNI_MIN(ow - w, unrollHw);
                            I32 in_h_0 = h * strideH;
                            I32 in_w_0 = w * strideW;
                            F32 *in_0 = curI + in_h_0 * iw * cLen + in_w_0 * cLen;
                            F32 *calO = curO + (h * ow + w) * cLen;

                            if (wSize < unrollHw) {
                                wkernel[0](in_0, nullptr, nullptr, nullptr, curW, calO, curB,
                                    fw, fh, oStep, iStep, hStep, flags, dw, 0);
                                wSize = 1;
                            } else {
                                wkernel33(
                                    in_0, curW, calO, curB, oStep, iStep, hStep33, flags, 3);
                            }
                        }
                    }
                } else {
                    I32 mainNum = ohow / UNROLL_W;
                    I32 ohowNum = mainNum + ohow % UNROLL_W;
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
                    for (I32 hwN = 0; hwN < ohowNum; ++hwN) {
                        I32 hw = 0;
                        if (hwN > mainNum) {
                            hw = mainNum * UNROLL_W + hwN - mainNum;
                        } else {
                            hw = hwN * UNROLL_W;
                        }
                        I32 wSize = UNI_MIN(ohow - hw, UNROLL_W);
                        if (wSize < UNROLL_W) {
                            wSize = 1;
                        }
                        I32 in_h_0 = hw / ow * strideH;
                        I32 in_w_0 = hw % ow * strideW;
                        I32 in_h_1 = (hw + 1) / ow * strideH;
                        I32 in_w_1 = (hw + 1) % ow * strideW;
                        I32 in_h_2 = (hw + 2) / ow * strideH;
                        I32 in_w_2 = (hw + 2) % ow * strideW;
                        I32 in_h_3 = (hw + 3) / ow * strideH;
                        I32 in_w_3 = (hw + 3) % ow * strideW;
                        F32 *in_0 = curI + in_h_0 * iw * cLen + in_w_0 * cLen;
                        F32 *in_1 = curI + in_h_1 * iw * cLen + in_w_1 * cLen;
                        F32 *in_2 = curI + in_h_2 * iw * cLen + in_w_2 * cLen;
                        F32 *in_3 = curI + in_h_3 * iw * cLen + in_w_3 * cLen;

                        wkernel[wSize >> 2](in_0, in_1, in_2, in_3, curW, curO + hw * cLen,
                            curB, fw, fh, oStep, iStep, hStep, flags, dw, 0);
                    }
                }
            } else {
                I32 owPaddingL = UNI_MIN((paddingL - 1) / strideW + 1, ow);
                I32 owPaddingR = UNI_MIN((paddingR - 1) / strideW + 1, ow - owPaddingL);
                if (((iw + paddingL - fwDilated) / strideW + 1) >= ow) {
                    owPaddingR = 0;
                }
#ifdef _USE_OPENMP
#pragma omp for nowait
#endif
                for (I32 h = 0; h < oh; ++h) {
                    I32 hStep = 0;
                    I32 inH = h * strideH - paddingT;
                    I32 tfh = GetNewKernelDilatedPad(ih, inH, fhDilated, dilateH);
                    I32 whJump = JumpToWeightPos(inH, dilateH);
                    I32 ihJump = JumpToInputPos(ih, inH, fhDilated, dilateH);
                    inH = (inH >= 0) ? inH : ihJump;
                    tfh = GetKernelnoDilated(tfh, dilateH);
                    const F32 *calW = curW + whJump * fw * ocSize;
                    I32 w = 0;
                    for (; w < owPaddingL + owPaddingR; ++w) {
                        I32 realW = (w >= owPaddingL) ? (ow - owPaddingR + w - owPaddingL) : w;
                        I32 inW = realW * strideW - paddingL;
                        I32 tfw = GetNewKernelDilatedPad(iw, inW, fwDilated, dilateW);
                        I32 wwJump = JumpToWeightPos(inW, dilateW);
                        I32 iwJump = JumpToInputPos(iw, inW, fwDilated, dilateW);
                        inW = (inW >= 0) ? inW : iwJump;
                        tfw = GetKernelnoDilated(tfw, dilateW);
                        const F32 *useW = calW + wwJump * ocSize;
                        F32 *in_0 = curI + inH * iw * cLen + inW * cLen;
                        F32 *calO = curO + (h * ow + realW) * cLen;
                        hStep = (iw - tfw * dilateW + (dilateH - 1) * iw) * cLen * BYTES;
                        wkernel[0](in_0, nullptr, nullptr, nullptr, useW, calO, curB, tfw,
                            tfh, oStep, iStep, hStep, flags, dw, (fw - tfw) * ocSize * BYTES);
                    }
                    w = owPaddingL;
                    I32 wSize = 0;
                    hStep = (iw - fw * dilateW + (dilateH - 1) * iw) * cLen * BYTES;
                    if (use3x3) {
                        I32 unrollHw = unrollHw33s1Array[ocIdx];
                        for (; w < ow - owPaddingR; w += wSize) {
                            wSize = UNI_MIN(ow - owPaddingR - w, unrollHw);
                            I32 in_w_0 = w * strideW - paddingL;
                            F32 *in_0 = curI + inH * iw * cLen + in_w_0 * cLen;
                            F32 *calO = curO + (h * ow + w) * cLen;
                            if (wSize < unrollHw) {
                                wkernel[0](in_0, nullptr, nullptr, nullptr, calW, calO, curB,
                                    fw, tfh, oStep, iStep, hStep, flags, dw, 0);
                                wSize = 1;
                            } else {
                                wkernel33(
                                    in_0, calW, calO, curB, oStep, iStep, hStep33, flags, tfh);
                            }
                        }
                    } else {
                        for (; w < ow - owPaddingR; w += wSize) {
                            wSize = UNI_MIN(ow - owPaddingR - w, UNROLL_W);
                            if (wSize < UNROLL_W) {
                                wSize = 1;
                            }
                            I32 in_w_0 = w * strideW - paddingL;
                            I32 in_w_1 = (w + 1) * strideW - paddingL;
                            I32 in_w_2 = (w + 2) * strideW - paddingL;
                            I32 in_w_3 = (w + 3) * strideW - paddingL;
                            F32 *in_0 = curI + inH * iw * cLen + in_w_0 * cLen;
                            F32 *in_1 = curI + inH * iw * cLen + in_w_1 * cLen;
                            F32 *in_2 = curI + inH * iw * cLen + in_w_2 * cLen;
                            F32 *in_3 = curI + inH * iw * cLen + in_w_3 * cLen;
                            F32 *calO = curO + (h * ow + w) * cLen;

                            wkernel[wSize >> 2](in_0, in_1, in_2, in_3, calW, calO, curB, fw,
                                tfh, oStep, iStep, hStep, flags, dw, 0);
                        }
                    }
                }
            }
        }
        }

        if (pwFilterArray != nullptr) {
            TensorDesc pwInputDesc = tensor4df(odt, DF_NCHWC8, 1, ic, oh, ow);
            U32 pwTmpBytes = tmpBytes - oh * ic * oh * ow + 32;
            F32 *pwTmpInput = (F32 *)tmp + oh * ic * oh * ow + 32;
            ConvolutionParamSpec p = createConvolutionParamSpec(
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, fn, CONVOLUTION_POINTWISE);
            convolution_1x1_direct(pwInputDesc, useOutArray, eltwiseInput, pwFilterDesc, pwFilterArray,
                p, pwBiasArray, tmpBytes, pwTmpInput, outputDesc, outArray + n * oc * oh * ow,
                pointwiseActivationParamSpec);
        }
    }

    return SUCCESS;
}
