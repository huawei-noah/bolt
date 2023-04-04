// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "uni.h"
#include "cpu/x86/fp32/convolution_functions.h"
#include "cpu/x86/tensor_computing_x86.h"

#define UNROLL_W 4
#define UNROLL_OC_BLOCK_DIM 16
#define SIMDW 16

struct ConvController {
    UINT8 *input;
    const INT8 *filter;
    void *output;
    F32 *eltwise;
    UINT8 *u8Output;
    const I32 *bias;
    I64 ic;
    I64 kw;
    I64 kh;
    I64 *stepC16;
    I64 ostepC16;
    I64 flags;
    I64 fStep;
    I64 hStep;
    I64 stride;
    I64 k4Num;
    void *scale;
};

typedef void (*kernelFunc)(ConvController &c);

void Avx512DepthConvKernel16x16(ConvController &c)
{
    __asm__ __volatile__(
        "prefetcht0 (%[output])                  \n\t"
        "prefetcht0 0x40(%[output])                  \n\t"
        "prefetcht0 0x80(%[output])                  \n\t"
        "prefetcht0 0xC0(%[output])                  \n\t"
        "prefetcht0 0x100(%[output])                  \n\t"
        "prefetcht0 0x140(%[output])                  \n\t"
        "prefetcht0 0x180(%[output])                  \n\t"
        "prefetcht0 0x1C0(%[output])                  \n\t"
        "vmovups (%[bias]), %%zmm0                   \n\t"
        "vmovups %%zmm0, %%zmm1                   \n\t"
        "vmovups %%zmm0, %%zmm2                   \n\t"
        "vmovups %%zmm0, %%zmm3                   \n\t"
        "vmovups %%zmm0, %%zmm4                   \n\t"
        "vmovups %%zmm0, %%zmm5                   \n\t"
        "vmovups %%zmm0, %%zmm6                   \n\t"
        "vmovups %%zmm0, %%zmm7                   \n\t"
        "vmovups %%zmm0, %%zmm8                   \n\t"
        "vmovups %%zmm0, %%zmm9                   \n\t"
        "vmovups %%zmm0, %%zmm10                   \n\t"
        "vmovups %%zmm0, %%zmm11                   \n\t"
        "vmovups %%zmm0, %%zmm12                   \n\t"
        "vmovups %%zmm0, %%zmm13                   \n\t"
        "vmovups %%zmm0, %%zmm14                   \n\t"
        "vmovups %%zmm0, %%zmm15                   \n\t"
        :
        : [bias] "r"(c.bias), [flags] "r"(c.flags), [output] "r"(c.output)
        : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "memory", "cc");

    __asm__ __volatile__(".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "vmovups (%[filter]), %%zmm16     \n\t"
                         "vmovups (%[input]), %%zmm17     \n\t"
                         "vmovups 0x40(%[input]), %%zmm18     \n\t"
                         "vmovups 0x80(%[input]), %%zmm19     \n\t"
                         "vmovups 0xC0(%[input]), %%zmm20     \n\t"
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0          \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm1          \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm2          \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm3          \n\t"
                         "vmovups 0x100(%[input]), %%zmm21     \n\t"
                         "vmovups 0x140(%[input]), %%zmm22     \n\t"
                         "vmovups 0x180(%[input]), %%zmm23     \n\t"
                         "vmovups 0x1C0(%[input]), %%zmm24     \n\t"
                         "vpdpbusd %%zmm16, %%zmm21, %%zmm4          \n\t"
                         "vpdpbusd %%zmm16, %%zmm22, %%zmm5          \n\t"
                         "vpdpbusd %%zmm16, %%zmm23, %%zmm6          \n\t"
                         "vpdpbusd %%zmm16, %%zmm24, %%zmm7          \n\t"
                         "vmovups 0x200(%[input]), %%zmm25     \n\t"
                         "vmovups 0x240(%[input]), %%zmm26     \n\t"
                         "vmovups 0x280(%[input]), %%zmm27     \n\t"
                         "vmovups 0x2C0(%[input]), %%zmm28     \n\t"
                         "vpdpbusd %%zmm16, %%zmm25, %%zmm8         \n\t"
                         "vpdpbusd %%zmm16, %%zmm26, %%zmm9         \n\t"
                         "vpdpbusd %%zmm16, %%zmm27, %%zmm10          \n\t"
                         "vpdpbusd %%zmm16, %%zmm28, %%zmm11          \n\t"
                         "vmovups 0x300(%[input]), %%zmm17     \n\t"
                         "vmovups 0x340(%[input]), %%zmm18     \n\t"
                         "vmovups 0x380(%[input]), %%zmm19     \n\t"
                         "vmovups 0x3C0(%[input]), %%zmm20     \n\t"
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm12          \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm13          \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm14          \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm15          \n\t"
                         "addq $0x40, %[filter]                                    \n\t"
                         "addq %[hStep], %[input]                                         \n\t"
                         "dec %%rcx                                         \n\t"
                         "jg 0b                                             \n\t"
                         : [input] "+r"(c.input), [filter] "+r"(c.filter)
                         : [k4Num] "c"(c.k4Num), [stride] "r"(c.stride), [hStep] "r"(c.hStep)
                         : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
                         "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                         "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21",
                         "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28",
                         "%zmm29", "%zmm30", "%zmm31", "memory", "cc");

    __asm__ __volatile__("cmpq $0x0, %[scale]                                       \n\t"
                         "jne 1f                                                    \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                           \n\t"
                         "je 4f                                                     \n\t"
                         "vpxord %%zmm31, %%zmm31, %%zmm31                             \n\t"
                         "vpmaxsd %%zmm31, %%zmm0, %%zmm0                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm1, %%zmm1                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm2, %%zmm2                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm3, %%zmm3                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm4, %%zmm4                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm5, %%zmm5                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm6, %%zmm6                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm7, %%zmm7                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm8, %%zmm8                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm9, %%zmm9                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm10, %%zmm10                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm11, %%zmm11                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm12, %%zmm12                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm13, %%zmm13                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm14, %%zmm14                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm15, %%zmm15                          \n\t"
                         "jmp 4f                                                    \n\t"

                         ".align 16                                                 \n\t"
                         "1:                                                        \n\t"
                         "vbroadcastss (%[scale]), %%zmm30                          \n\t"
                         "vcvtdq2ps %%zmm30, %%zmm31                                \n\t"
                         "vmulps %%zmm31, %%zmm0, %%zmm0                          \n\t"
                         "vmulps %%zmm31, %%zmm1, %%zmm1                          \n\t"
                         "vmulps %%zmm31, %%zmm2, %%zmm2                          \n\t"
                         "vmulps %%zmm31, %%zmm3, %%zmm3                          \n\t"
                         "vmulps %%zmm31, %%zmm4, %%zmm4                          \n\t"
                         "vmulps %%zmm31, %%zmm5, %%zmm5                          \n\t"
                         "vmulps %%zmm31, %%zmm6, %%zmm6                          \n\t"
                         "vmulps %%zmm31, %%zmm7, %%zmm7                          \n\t"
                         "vmulps %%zmm31, %%zmm8, %%zmm8                          \n\t"
                         "vmulps %%zmm31, %%zmm9, %%zmm9                          \n\t"
                         "vmulps %%zmm31, %%zmm10, %%zmm10                          \n\t"
                         "vmulps %%zmm31, %%zmm11, %%zmm11                          \n\t"
                         "vmulps %%zmm31, %%zmm12, %%zmm12                          \n\t"
                         "vmulps %%zmm31, %%zmm13, %%zmm13                          \n\t"
                         "vmulps %%zmm31, %%zmm14, %%zmm14                          \n\t"
                         "vmulps %%zmm31, %%zmm15, %%zmm15                          \n\t"

                         ".align 16                                                 \n\t"
                         "2:                                                        \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x2, %%rcx                                           \n\t"
                         "je 3f                                                     \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0                       \n\t"
                         "vaddps 0x40(%[eltwise]), %%zmm1, %%zmm1                \n\t"
                         "vaddps 0x80(%[eltwise]), %%zmm2, %%zmm2                \n\t"
                         "vaddps 0xC0(%[eltwise]), %%zmm3, %%zmm3                \n\t"
                         "vaddps 0x100(%[eltwise]), %%zmm4, %%zmm4                \n\t"
                         "vaddps 0x140(%[eltwise]), %%zmm5, %%zmm5                \n\t"
                         "vaddps 0x180(%[eltwise]), %%zmm6, %%zmm6                \n\t"
                         "vaddps 0x1C0(%[eltwise]), %%zmm7, %%zmm7                \n\t"
                         "vaddps 0x200(%[eltwise]), %%zmm8, %%zmm8                \n\t"
                         "vaddps 0x240(%[eltwise]), %%zmm9, %%zmm9                \n\t"
                         "vaddps 0x280(%[eltwise]), %%zmm10, %%zmm10                \n\t"
                         "vaddps 0x2C0(%[eltwise]), %%zmm11, %%zmm11                \n\t"
                         "vaddps 0x300(%[eltwise]), %%zmm12, %%zmm12                \n\t"
                         "vaddps 0x340(%[eltwise]), %%zmm13, %%zmm13                \n\t"
                         "vaddps 0x380(%[eltwise]), %%zmm14, %%zmm14                \n\t"
                         "vaddps 0x3C0(%[eltwise]), %%zmm15, %%zmm15                \n\t"

                         ".align 16                                                 \n\t"
                         "3:                                                        \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                           \n\t"
                         "je 4f                                                     \n\t"
                         "vpxord %%zmm31, %%zmm31, %%zmm31                             \n\t"
                         "vmaxps %%zmm31, %%zmm0, %%zmm0                          \n\t"
                         "vmaxps %%zmm31, %%zmm1, %%zmm1                          \n\t"
                         "vmaxps %%zmm31, %%zmm2, %%zmm2                          \n\t"
                         "vmaxps %%zmm31, %%zmm3, %%zmm3                          \n\t"
                         "vmaxps %%zmm31, %%zmm4, %%zmm4                          \n\t"
                         "vmaxps %%zmm31, %%zmm5, %%zmm5                          \n\t"
                         "vmaxps %%zmm31, %%zmm6, %%zmm6                          \n\t"
                         "vmaxps %%zmm31, %%zmm7, %%zmm7                          \n\t"
                         "vmaxps %%zmm31, %%zmm8, %%zmm8                          \n\t"
                         "vmaxps %%zmm31, %%zmm9, %%zmm9                          \n\t"
                         "vmaxps %%zmm31, %%zmm10, %%zmm10                          \n\t"
                         "vmaxps %%zmm31, %%zmm11, %%zmm11                          \n\t"
                         "vmaxps %%zmm31, %%zmm12, %%zmm12                          \n\t"
                         "vmaxps %%zmm31, %%zmm13, %%zmm13                          \n\t"
                         "vmaxps %%zmm31, %%zmm14, %%zmm14                          \n\t"
                         "vmaxps %%zmm31, %%zmm15, %%zmm15                          \n\t"

                         ".align 16                                                 \n\t"
                         "4:                                                        \n\t"
                         "vmovups %%zmm0, (%[output])                               \n\t"
                         "vmovups %%zmm1, 0x40(%[output])                           \n\t"
                         "vmovups %%zmm2, 0x80(%[output])                           \n\t"
                         "vmovups %%zmm3, 0xC0(%[output])                           \n\t"
                         "vmovups %%zmm4, 0x100(%[output])                          \n\t"
                         "vmovups %%zmm5, 0x140(%[output])                          \n\t"
                         "vmovups %%zmm6, 0x180(%[output])                          \n\t"
                         "vmovups %%zmm7, 0x1C0(%[output])                          \n\t"
                         "vmovups %%zmm8, 0x200(%[output])                          \n\t"
                         "vmovups %%zmm9, 0x240(%[output])                          \n\t"
                         "vmovups %%zmm10, 0x280(%[output])                          \n\t"
                         "vmovups %%zmm11, 0x2C0(%[output])                          \n\t"
                         "vmovups %%zmm12, 0x300(%[output])                          \n\t"
                         "vmovups %%zmm13, 0x340(%[output])                          \n\t"
                         "vmovups %%zmm14, 0x380(%[output])                          \n\t"
                         "vmovups %%zmm15, 0x3C0(%[output])                          \n\t"
                         :
                         : [output] "r"(c.output), [eltwise] "r"(c.eltwise),
                         [ostepC16] "r"(c.ostepC16), [flags] "r"(c.flags), [scale] "r"(c.scale)
                         : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                         "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13",
                         "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20",
                         "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27",
                         "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

void Avx512DepthConvKernel8x16(ConvController &c)
{
    __asm__ __volatile__(
        "prefetcht0 (%[output])                  \n\t"
        "prefetcht0 0x40(%[output])                  \n\t"
        "prefetcht0 0x80(%[output])                  \n\t"
        "prefetcht0 0xC0(%[output])                  \n\t"
        "prefetcht0 0x100(%[output])                  \n\t"
        "prefetcht0 0x140(%[output])                  \n\t"
        "prefetcht0 0x180(%[output])                  \n\t"
        "prefetcht0 0x1C0(%[output])                  \n\t"
        "vmovups (%[bias]), %%zmm0                   \n\t"
        "vmovups %%zmm0, %%zmm1                   \n\t"
        "vmovups %%zmm0, %%zmm2                   \n\t"
        "vmovups %%zmm0, %%zmm3                   \n\t"
        "vmovups %%zmm0, %%zmm4                   \n\t"
        "vmovups %%zmm0, %%zmm5                   \n\t"
        "vmovups %%zmm0, %%zmm6                   \n\t"
        "vmovups %%zmm0, %%zmm7                   \n\t"
        :
        : [bias] "r"(c.bias), [flags] "r"(c.flags), [output] "r"(c.output)
        : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "memory", "cc");

    __asm__ __volatile__(".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "vmovups (%[filter]), %%zmm16     \n\t"
                         "vmovups (%[input]), %%zmm17     \n\t"
                         "vmovups 0x40(%[input]), %%zmm18     \n\t"
                         "vmovups 0x80(%[input]), %%zmm19     \n\t"
                         "vmovups 0xC0(%[input]), %%zmm20     \n\t"
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0          \n\t"
                         "vpdpbusd %%zmm16, %%zmm18, %%zmm1          \n\t"
                         "vpdpbusd %%zmm16, %%zmm19, %%zmm2          \n\t"
                         "vpdpbusd %%zmm16, %%zmm20, %%zmm3          \n\t"
                         "vmovups 0x100(%[input]), %%zmm21     \n\t"
                         "vmovups 0x140(%[input]), %%zmm22     \n\t"
                         "vmovups 0x180(%[input]), %%zmm23     \n\t"
                         "vmovups 0x1C0(%[input]), %%zmm24     \n\t"
                         "vpdpbusd %%zmm16, %%zmm21, %%zmm4          \n\t"
                         "vpdpbusd %%zmm16, %%zmm22, %%zmm5          \n\t"
                         "vpdpbusd %%zmm16, %%zmm23, %%zmm6          \n\t"
                         "vpdpbusd %%zmm16, %%zmm24, %%zmm7          \n\t"
                         "addq $0x40, %[filter]                                    \n\t"
                         "addq %[hStep], %[input]                                         \n\t"
                         "dec %%rcx                                         \n\t"
                         "jg 0b                                             \n\t"
                         : [input] "+r"(c.input), [filter] "+r"(c.filter)
                         : [k4Num] "c"(c.k4Num), [stride] "r"(c.stride), [hStep] "r"(c.hStep)
                         : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
                         "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                         "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21",
                         "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28",
                         "%zmm29", "%zmm30", "%zmm31", "memory", "cc");

    __asm__ __volatile__("cmpq $0x0, %[scale]                                       \n\t"
                         "jne 1f                                                    \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                           \n\t"
                         "je 4f                                                     \n\t"
                         "vpxord %%zmm31, %%zmm31, %%zmm31                             \n\t"
                         "vpmaxsd %%zmm31, %%zmm0, %%zmm0                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm1, %%zmm1                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm2, %%zmm2                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm3, %%zmm3                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm4, %%zmm4                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm5, %%zmm5                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm6, %%zmm6                          \n\t"
                         "vpmaxsd %%zmm31, %%zmm7, %%zmm7                          \n\t"
                         "jmp 4f                                                    \n\t"

                         ".align 16                                                 \n\t"
                         "1:                                                        \n\t"
                         "vbroadcastss (%[scale]), %%zmm30                          \n\t"
                         "vcvtdq2ps %%zmm30, %%zmm31                                \n\t"
                         "vmulps %%zmm31, %%zmm0, %%zmm0                          \n\t"
                         "vmulps %%zmm31, %%zmm1, %%zmm1                          \n\t"
                         "vmulps %%zmm31, %%zmm2, %%zmm2                          \n\t"
                         "vmulps %%zmm31, %%zmm3, %%zmm3                          \n\t"
                         "vmulps %%zmm31, %%zmm4, %%zmm4                          \n\t"
                         "vmulps %%zmm31, %%zmm5, %%zmm5                          \n\t"
                         "vmulps %%zmm31, %%zmm6, %%zmm6                          \n\t"
                         "vmulps %%zmm31, %%zmm7, %%zmm7                          \n\t"

                         ".align 16                                                 \n\t"
                         "2:                                                        \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x2, %%rcx                                           \n\t"
                         "je 3f                                                     \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0                       \n\t"
                         "vaddps 0x40(%[eltwise]), %%zmm1, %%zmm1                \n\t"
                         "vaddps 0x80(%[eltwise]), %%zmm2, %%zmm2                \n\t"
                         "vaddps 0xC0(%[eltwise]), %%zmm3, %%zmm3                \n\t"
                         "vaddps 0x100(%[eltwise]), %%zmm4, %%zmm4                \n\t"
                         "vaddps 0x140(%[eltwise]), %%zmm5, %%zmm5                \n\t"
                         "vaddps 0x180(%[eltwise]), %%zmm6, %%zmm6                \n\t"
                         "vaddps 0x1C0(%[eltwise]), %%zmm7, %%zmm7                \n\t"

                         ".align 16                                                 \n\t"
                         "3:                                                        \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                           \n\t"
                         "je 4f                                                     \n\t"
                         "vpxord %%zmm31, %%zmm31, %%zmm31                             \n\t"
                         "vmaxps %%zmm31, %%zmm0, %%zmm0                          \n\t"
                         "vmaxps %%zmm31, %%zmm1, %%zmm1                          \n\t"
                         "vmaxps %%zmm31, %%zmm2, %%zmm2                          \n\t"
                         "vmaxps %%zmm31, %%zmm3, %%zmm3                          \n\t"
                         "vmaxps %%zmm31, %%zmm4, %%zmm4                          \n\t"
                         "vmaxps %%zmm31, %%zmm5, %%zmm5                          \n\t"
                         "vmaxps %%zmm31, %%zmm6, %%zmm6                          \n\t"
                         "vmaxps %%zmm31, %%zmm7, %%zmm7                          \n\t"

                         ".align 16                                                 \n\t"
                         "4:                                                        \n\t"
                         "vmovups %%zmm0, (%[output])                               \n\t"
                         "vmovups %%zmm1, 0x40(%[output])                           \n\t"
                         "vmovups %%zmm2, 0x80(%[output])                           \n\t"
                         "vmovups %%zmm3, 0xC0(%[output])                           \n\t"
                         "vmovups %%zmm4, 0x100(%[output])                          \n\t"
                         "vmovups %%zmm5, 0x140(%[output])                          \n\t"
                         "vmovups %%zmm6, 0x180(%[output])                          \n\t"
                         "vmovups %%zmm7, 0x1C0(%[output])                          \n\t"
                         :
                         : [output] "r"(c.output), [eltwise] "r"(c.eltwise),
                         [ostepC16] "r"(c.ostepC16), [flags] "r"(c.flags), [scale] "r"(c.scale)
                         : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                         "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13",
                         "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20",
                         "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27",
                         "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

void Avx512DepthConvKernel1x16(ConvController &c)
{
    __asm__ __volatile__(
        "prefetcht0 (%[output])                  \n\t"
        "vmovups (%[bias]), %%zmm0                   \n\t"
        :
        : [bias] "r"(c.bias), [flags] "r"(c.flags), [output] "r"(c.output)
        : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "memory", "cc");

    __asm__ __volatile__(".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "vmovups (%[filter]), %%zmm16     \n\t"
                         "vmovups (%[input]), %%zmm17     \n\t"
                         "vpdpbusd %%zmm16, %%zmm17, %%zmm0          \n\t"
                         "addq $0x40, %[filter]                                    \n\t"
                         "addq %[hStep], %[input]                                         \n\t"
                         "dec %%rcx                                         \n\t"
                         "jg 0b                                             \n\t"
                         : [input] "+r"(c.input), [filter] "+r"(c.filter)
                         : [k4Num] "c"(c.k4Num), [stride] "r"(c.stride), [hStep] "r"(c.hStep)
                         : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7",
                         "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
                         "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21",
                         "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28",
                         "%zmm29", "%zmm30", "%zmm31", "memory", "cc");

    __asm__ __volatile__("cmpq $0x0, %[scale]                                       \n\t"
                         "jne 1f                                                    \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                           \n\t"
                         "je 4f                                                     \n\t"
                         "vpxord %%zmm31, %%zmm31, %%zmm31                             \n\t"
                         "vpmaxsd %%zmm31, %%zmm0, %%zmm0                          \n\t"
                         "jmp 4f                                                    \n\t"

                         ".align 16                                                 \n\t"
                         "1:                                                        \n\t"
                         "vbroadcastss (%[scale]), %%zmm30                          \n\t"
                         "vcvtdq2ps %%zmm30, %%zmm31                                \n\t"
                         "vmulps %%zmm31, %%zmm0, %%zmm0                          \n\t"

                         ".align 16                                                 \n\t"
                         "2:                                                        \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0x2, %%rcx                                           \n\t"
                         "je 3f                                                     \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0                       \n\t"

                         ".align 16                                                 \n\t"
                         "3:                                                        \n\t"
                         "movq %[flags], %%rcx                                      \n\t"
                         "and $0xC, %%rcx                                           \n\t"
                         "je 4f                                                     \n\t"
                         "vpxord %%zmm31, %%zmm31, %%zmm31                             \n\t"
                         "vmaxps %%zmm31, %%zmm0, %%zmm0                          \n\t"

                         ".align 16                                                 \n\t"
                         "4:                                                        \n\t"
                         "vmovups %%zmm0, (%[output])                               \n\t"
                         :
                         : [output] "r"(c.output), [eltwise] "r"(c.eltwise),
                         [ostepC16] "r"(c.ostepC16), [flags] "r"(c.flags), [scale] "r"(c.scale)
                         : "%rax", "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6",
                         "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13",
                         "%zmm14", "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20",
                         "%zmm21", "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27",
                         "%zmm28", "%zmm29", "%zmm30", "%zmm31", "memory", "cc");
}

EE depthwise_pointwise_convolution_int8(TensorDesc inputDesc,
    UINT8 *inArray,
    F32 *eltwiseInput,
    TensorDesc dwFilterDesc,
    const INT8 *dwFilterArray,
    TensorDesc pwFilterDesc,
    const INT8 *pwFilterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc dwBiasDesc,
    const F32 *dwBiasArray,
    TensorDesc pwBiasDesc,
    const F32 *pwBiasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *outArray,
    F32 *scale,
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

    if ((idf != DF_NCHWC16) || (ic % 16 != 0)) {
        CHECK_STATUS(NOT_MATCH);
    }

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
    I32 fhfw = fh * fw;
    I32 iw_pad = iw + paddingL + paddingR;
    I32 ih_pad = ih + paddingT + paddingB;

    // infer kernel params
    ConvController convCtl;
    convCtl.ostepC16 = oh * ow * SIMDW * 4;
    convCtl.fStep = ih_pad * iw_pad * SIMDW;
    convCtl.kw = fw;
    convCtl.kh = fh;
    convCtl.scale = nullptr;
    convCtl.stride = strideW;

    // fuse dw+pw
    F32 *useOutArray = (F32 *)tmp;
    if (pwFilterArray == nullptr) {
        useOutArray = (F32 *)outArray;
    }
    F32 *output = (F32 *)useOutArray;

    const kernelFunc kernel[3] = {
        Avx512DepthConvKernel1x16, Avx512DepthConvKernel8x16, Avx512DepthConvKernel16x16};
    U32 hwSizes[3] = {1, 8, 16};

    // quantization
    F32 *scaleI = scale;
    F32 *scaleO = scale + 1;
    F32 *scaleF = scale + 2;
    if (idt != DT_U8_Q) {
        //quantize to U8_Q
        TensorDesc qDesc = inputDesc;
        qDesc.dt = DT_U8_Q;
        CHECK_STATUS(quantize_x86(inputDesc, (void *)inArray, &qDesc, tmp, scaleI));
        inArray = (UINT8 *)tmp;
        tmp = (void *)((U8 *)tmp + tensorNumBytes(qDesc));
    }
    *scaleO = scaleI[0] * scaleF[0];
    if (odt != DT_F32 && odt != DT_I32) {
        output = (F32 *)tmp;
        tmp = (void *)((U8 *)tmp + tensorNumElements(outputDesc) * bytesOf(DT_I32));
        outputDesc.dt = DT_I32;
    }
    if (eltwiseInput != nullptr) {
        outputDesc.dt = DT_F32;
    }
    F32 *factorPtr = nullptr;
    F32 factor = 0;
    if (scale != nullptr && outputDesc.dt == DT_F32) {
        factor = 1 / (*scaleO);
        factorPtr = &factor;
    }

    I32 *offsetC = (I32 *)tmp;
    tmp = (void *)((U8 *)tmp + oc * bytesOf(DT_I32));
    CHECK_STATUS(quantize_bias_offsetC((const void *)dwBiasArray, dwBiasDesc, DT_I32,
        (const void *)dwFilterArray, dwFilterDesc, scaleO, offsetC));
    dwFilterArray += oc * 4;

    U32 kernelSize = (fh * fw + 3) / 4 * 4;
    convCtl.k4Num = kernelSize / 4;
    UINT8 *tmpInput = (UINT8 *)tmp;

    I64 flags = 0;
    flags |= (eltwiseInput != nullptr) << 1;
    flags |= U32(depthwiseActivationParamSpec.mode) << 2;
    convCtl.scale = factorPtr;
    convCtl.flags = flags;

    for (I32 n = 0; n < in; ++n) {
        I32 ocSize = 16;
        //  Padding
        for (I32 ocb = 0; ocb < oc; ocb += ocSize) {
            convCtl.bias = offsetC + ocb;
            F32 *curO = output + (n * oc + ocb) * oh * ow;
            I32 hwSize = 0;
            UINT8 *curI = inArray + (n * ic + ocb) * ih * iw;
            for (I32 hw = 0; hw < ohow; hw += hwSize) {
                hwSize = UNI_MIN(ohow - hw, 16);
                hwSize = hwSizes[hwSize >> 3];
                I32 h = hw / ow;
                I32 w = hw % ow;
                I32 in_h_0 = h * strideH;
                I32 in_w_0 = w * strideW;

                // TODO: optimize
                for (U32 kk = 0; kk < kernelSize; kk += 4) {
                    for (I32 ii = 0; ii < hwSize; ++ii) {
                        for (I32 jj = 0; jj < SIMDW; ++jj) {
                            for (I32 k4 = 0; k4 < 4; ++k4) {
                                I32 oidx = k4 + jj * 4 + ii * 4 * SIMDW + kk * SIMDW * hwSize;
                                if ((I32)(k4 + kk) < fhfw) {
                                    in_h_0 = (hw + ii) / ow * strideH + (kk + k4) / fw;
                                    in_w_0 = (hw + ii) % ow * strideW + (kk + k4) % fw;
                                    I32 iidx = jj + (in_h_0 * iw + in_w_0) * SIMDW;
                                    tmpInput[oidx] = curI[iidx];
                                } else {
                                    tmpInput[oidx] = 0;
                                }
                            }
                        }
                    }
                }

                convCtl.input = tmpInput;
                convCtl.output = curO + (h * ow + w) * SIMDW;
                convCtl.filter = dwFilterArray + ocb * kernelSize;
                convCtl.hStep = hwSize * SIMDW * 4;
                kernel[hwSize >> 3](convCtl);
            }
        }
    }

    return SUCCESS;
}
