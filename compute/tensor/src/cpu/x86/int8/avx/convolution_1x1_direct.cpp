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
#include "data_type.h"
#include "sys.h"
#include "error.h"
#include "cpu/x86/int8/tensor_computing_int8.h"
#include "cpu/x86/int8/convolution_functions.h"
#include "cpu/x86/tensor_computing_x86.h"

#define SIMDW 8
#define BLOCK_IC_DIM 256
#define BLOCK_HW_DIM 768

// clang-format off
#define convKernel4x24c4(input, off0, off1, off2, i0, i1, i2, i3) \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "vmovups "#off0"(%[filter]), %%ymm13       \n\t" \
    "vmovups "#off1"(%[filter]), %%ymm14       \n\t" \
    "vmovups "#off2"(%[filter]), %%ymm15       \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm12, %%ymm0          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm1          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm2          \n\t" \
    "vpbroadcastd "#i1"("#input"), %%ymm12       \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm12, %%ymm3          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm4          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm5          \n\t" \
    "vpbroadcastd "#i2"("#input"), %%ymm12       \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm12, %%ymm6          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm7          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm8          \n\t" \
    "vpbroadcastd "#i3"("#input"), %%ymm12       \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm12, %%ymm9          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm10          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm11          \n\t" \

#define convKernel2x24c4(input, off0, off1, off2, i0, i1, i2, i3) \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "vmovups "#off0"(%[filter]), %%ymm13       \n\t" \
    "vmovups "#off1"(%[filter]), %%ymm14       \n\t" \
    "vmovups "#off2"(%[filter]), %%ymm15       \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm12, %%ymm0          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm1          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm2          \n\t" \
    "vpbroadcastd "#i1"("#input"), %%ymm12       \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm12, %%ymm3          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm4          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm5          \n\t" \

#define convKernel1x24c4(input, off0, off1, off2, i0, i1, i2, i3) \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "vmovups "#off0"(%[filter]), %%ymm13       \n\t" \
    "vmovups "#off1"(%[filter]), %%ymm14       \n\t" \
    "vmovups "#off2"(%[filter]), %%ymm15       \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm12, %%ymm0          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm1          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm2          \n\t" \


#define convKernelForLoopXx24(rnum, wsize) \
     __asm__ __volatile__("movq %[flags], %%rax                                             \n\t" \
                          "andq $0x1, %%rax                                                 \n\t" \
                          "jne 0f                                                           \n\t" \
                          load3BiasToYmmRegs(rnum, %[bias])             \
                          "cmpq $0x8, %%rcx                                                \n\t" \
                          "jl 4f                                                            \n\t" \
                          "jmp 1f                                                           \n\t" \
                          ".align 16                                                        \n\t" \
                          "0:                                                               \n\t" \
                          clear##rnum##Regs(%%ymm)                                                \
                          "cmpq $0x8, %%rcx                                                \n\t" \
                          "jl 4f                                                            \n\t" \
                          ".align 16                                                        \n\t" \
                          "1:                                                               \n\t" \
                          "movq %[input], %%rax                                             \n\t" \
                          convKernel##wsize##x24c4(%%rax, 0x0, 0x20, 0x40, 0x0, 0x8, 0x10, 0x18) \
                          "addq $0x4, %%rax  \n\t"                                                \
                          convKernel##wsize##x24c4(%%rax, 0x60, 0x80, 0xA0, 0x0, 0x8, 0x10, 0x18) \
                          "addq $0xC0, %[filter]                                           \n\t" \
                          "addq %[fStep], %[input]                                          \n\t" \
                          "subq $0x8, %%rcx                                                \n\t" \
                          "cmpq $0x8, %%rcx                                                \n\t" \
                          "jge 1b                                                           \n\t" \
                          ".align 16                                                        \n\t" \
                          "4:                                                               \n\t" \
                          : "+c" (c.ic),                                                          \
                            [input] "+r" (c.input),                                               \
                            [filter] "+r" (c.filter)                                              \
                          : [bias] "r" (c.bias),                                                  \
                            [kh] "r" (c.kh),                                                      \
                            [kw] "r" (c.kw),                                                      \
                            [fStep] "r" (c.fStep),                                                \
                            [flags] "r" (c.flags)                                                 \
                          : "%rax",                                              \
                            "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",                 \
                            "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",               \
                            "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");                                  \

void Avx512Conv1x1Kernel4x24(ConvController &c) {
     convKernelForLoopXx24(12, 4)

    __asm__ __volatile__("movq %[output], %%rax                                \n\t"
                         "movq %[ostepC16], %%rbx                              \n\t"
                         "movq %[flags], %%rcx                                 \n\t"
                         "and $0x1, %%rcx                                      \n\t"
                         "je 0f                                                \n\t"
                         "vpaddd (%%rax), %%ymm0, %%ymm0                       \n\t"
                         "vpaddd 0x20(%%rax), %%ymm3, %%ymm3                   \n\t"
                         "vpaddd 0x40(%%rax), %%ymm6, %%ymm6                   \n\t"
                         "vpaddd 0x60(%%rax), %%ymm9, %%ymm9                   \n\t"
                         "vpaddd (%%rax, %%rbx), %%ymm1, %%ymm1                \n\t"
                         "vpaddd 0x20(%%rax, %%rbx), %%ymm4, %%ymm4            \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%ymm7, %%ymm7            \n\t"
                         "vpaddd 0x60(%%rax, %%rbx), %%ymm10, %%ymm10          \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%ymm2, %%ymm2             \n\t"
                         "vpaddd 0x20(%%rax, %%rbx, 2), %%ymm5, %%ymm5         \n\t"
                         "vpaddd 0x40(%%rax, %%rbx, 2), %%ymm8, %%ymm8         \n\t"
                         "vpaddd 0x60(%%rax, %%rbx, 2), %%ymm11, %%ymm11       \n\t"

                         ".align 16                                            \n\t"
                         "0:                                                   \n\t"
                         "cmpq $0x0, %[scale]                                  \n\t"
                         "jne 1f                                               \n\t"
                         "movq %[flags], %%rcx                                 \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 4f                                                \n\t"
                         relu12Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                               \n\t"

                         ".align 16                                            \n\t"
                         "1:                                                   \n\t"
                         convert12RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                            \n\t"
                         "2:                                                   \n\t"
                         "movq %[flags], %%rcx                                 \n\t"
                         "and $0x2, %%rcx                                      \n\t"
                         "je 3f                                                \n\t"
                         "vaddps (%[eltwise]), %%ymm0, %%ymm0                  \n\t"
                         "vaddps 0x20(%[eltwise]), %%ymm3, %%ymm3              \n\t"
                         "vaddps 0x40(%[eltwise]), %%ymm6, %%ymm6              \n\t"
                         "vaddps 0x60(%[eltwise]), %%ymm9, %%ymm9              \n\t"
                         "vaddps (%[eltwise], %%rbx), %%ymm1, %%ymm1           \n\t"
                         "vaddps 0x20(%[eltwise], %%rbx), %%ymm4, %%ymm4       \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx), %%ymm7, %%ymm7       \n\t"
                         "vaddps 0x60(%[eltwise], %%rbx), %%ymm10, %%ymm10     \n\t"
                         "vaddps (%[eltwise], %%rbx, 2), %%ymm2, %%ymm2        \n\t"
                         "vaddps 0x20(%[eltwise], %%rbx, 2), %%ymm5, %%ymm5    \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx, 2), %%ymm8, %%ymm8    \n\t"
                         "vaddps 0x60(%[eltwise], %%rbx, 2), %%ymm11, %%ymm11  \n\t"

                         ".align 16                                            \n\t"
                         "3:                                                   \n\t"
                         "movq %[flags], %%rcx                                 \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 4f                                                \n\t"
                         relu12Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                            \n\t"
                         "4:                                                   \n\t"
                         "vmovups %%ymm0, (%%rax)                              \n\t"
                         "vmovups %%ymm3, 0x20(%%rax)                          \n\t"
                         "vmovups %%ymm6, 0x40(%%rax)                          \n\t"
                         "vmovups %%ymm9, 0x60(%%rax)                          \n\t"
                         "vmovups %%ymm1, (%%rax, %%rbx)                       \n\t"
                         "vmovups %%ymm4, 0x20(%%rax, %%rbx)                   \n\t"
                         "vmovups %%ymm7, 0x40(%%rax, %%rbx)                   \n\t"
                         "vmovups %%ymm10, 0x60(%%rax, %%rbx)                  \n\t"
                         "vmovups %%ymm2, (%%rax, %%rbx, 2)                    \n\t"
                         "vmovups %%ymm5, 0x20(%%rax, %%rbx, 2)                \n\t"
                         "vmovups %%ymm8, 0x40(%%rax, %%rbx, 2)                \n\t"
                         "vmovups %%ymm11, 0x60(%%rax, %%rbx, 2)               \n\t"
                         :
                         : [output] "r" (c.output),
                           [eltwise] "r" (c.eltwise),
                           [ostepC16] "r" (c.ostepC16),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void Avx512Conv1x1Kernel2x24(ConvController &c) {
    convKernelForLoopXx24(6, 2)

    __asm__ __volatile__("movq %[output], %%rax                               \n\t"
                         "movq %[ostepC16], %%rbx                             \n\t"
                         "movq %[flags], %%rcx                                \n\t"
                         "and $0x1, %%rcx                                     \n\t"
                         "je 0f                                               \n\t"
                         "vpaddd (%%rax), %%ymm0, %%ymm0                      \n\t"
                         "vpaddd 0x20(%%rax), %%ymm3, %%ymm3                  \n\t"
                         "vpaddd (%%rax, %%rbx), %%ymm1, %%ymm1               \n\t"
                         "vpaddd 0x20(%%rax, %%rbx), %%ymm4, %%ymm4           \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%ymm2, %%ymm2            \n\t"
                         "vpaddd 0x20(%%rax, %%rbx, 2), %%ymm5, %%ymm5        \n\t"

                         ".align 16                                           \n\t"
                         "0:                                                  \n\t"
                         "cmpq $0x0, %[scale]                                 \n\t"
                         "jne 1f                                              \n\t"
                         "movq %[flags], %%rcx                                \n\t"
                         "and $0xC, %%rcx                                     \n\t"
                         "je 4f                                               \n\t"
                         relu6Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                              \n\t"

                         ".align 16                                           \n\t"
                         "1:                                                  \n\t"
                         convert6RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                           \n\t"
                         "2:                                                  \n\t"
                         "movq %[flags], %%rcx                                \n\t"
                         "and $0x2, %%rcx                                     \n\t"
                         "je 3f                                               \n\t"
                         "vaddps (%[eltwise]), %%ymm0, %%ymm0                 \n\t"
                         "vaddps 0x20(%[eltwise]), %%ymm3, %%ymm3             \n\t"
                         "vaddps (%[eltwise], %%rbx), %%ymm1, %%ymm1          \n\t"
                         "vaddps 0x20(%[eltwise], %%rbx), %%ymm4, %%ymm4      \n\t"
                         "vaddps (%[eltwise], %%rbx, 2), %%ymm2, %%ymm2       \n\t"
                         "vaddps 0x20(%[eltwise], %%rbx, 2), %%ymm5, %%ymm5   \n\t"

                         ".align 16                                           \n\t"
                         "3:                                                  \n\t"
                         "movq %[flags], %%rcx                                \n\t"
                         "and $0xC, %%rcx                                     \n\t"
                         "je 4f                                               \n\t"
                         relu6Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                           \n\t"
                         "4:                                                  \n\t"
                         "vmovups %%ymm0, (%%rax)                             \n\t"
                         "vmovups %%ymm3, 0x20(%%rax)                         \n\t"
                         "vmovups %%ymm1, (%%rax, %%rbx)                      \n\t"
                         "vmovups %%ymm4, 0x20(%%rax, %%rbx)                  \n\t"
                         "vmovups %%ymm2, (%%rax, %%rbx, 2)                   \n\t"
                         "vmovups %%ymm5, 0x20(%%rax, %%rbx, 2)               \n\t"
                         :
                         : [output] "r" (c.output),
                           [eltwise] "r" (c.eltwise),
                           [ostepC16] "r" (c.ostepC16),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");

}

void Avx512Conv1x1Kernel1x24(ConvController &c) {
    convKernelForLoopXx24(3, 1)

    __asm__ __volatile__("movq %[output], %%rax                         \n\t"
                         "movq %[ostepC16], %%rbx                       \n\t"
                         "movq %[flags], %%rcx                          \n\t"
                         "and $0x1, %%rcx                               \n\t"
                         "je 0f                                         \n\t"
                         "vpaddd (%%rax), %%ymm0, %%ymm0                \n\t"
                         "vpaddd (%%rax, %%rbx), %%ymm1, %%ymm1         \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%ymm2, %%ymm2      \n\t"

                         ".align 16                                     \n\t"
                         "0:                                            \n\t"
                         "cmpq $0x0, %[scale]                           \n\t"
                         "jne 1f                                        \n\t"
                         "movq %[flags], %%rcx                          \n\t"
                         "and $0xC, %%rcx                               \n\t"
                         "je 4f                                         \n\t"
                         relu3Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                        \n\t"

                         ".align 16                                     \n\t"
                         "1:                                            \n\t"
                         convert3RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                     \n\t"
                         "2:                                            \n\t"
                         "movq %[flags], %%rcx                          \n\t"
                         "and $0x2, %%rcx                               \n\t"
                         "je 3f                                         \n\t"
                         "vaddps (%[eltwise]), %%ymm0, %%ymm0           \n\t"
                         "vaddps (%[eltwise], %%rbx), %%ymm1, %%ymm1    \n\t"
                         "vaddps (%[eltwise], %%rbx, 2), %%ymm2, %%ymm2 \n\t"

                         ".align 16                                     \n\t"
                         "3:                                            \n\t"
                         "movq %[flags], %%rcx                          \n\t"
                         "and $0xC, %%rcx                               \n\t"
                         "je 4f                                         \n\t"
                         relu3Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                     \n\t"
                         "4:                                            \n\t"
                         "vmovups %%ymm0, (%%rax)                       \n\t"
                         "vmovups %%ymm1, (%%rax, %%rbx)                \n\t"
                         "vmovups %%ymm2, (%%rax, %%rbx, 2)             \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm2",
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

#define convKernel6x16c4(input, off0, off1, i0, i1, i2, i3, i4, i5) \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "vpbroadcastd "#i1"("#input"), %%ymm13       \n\t" \
    "vmovups "#off0"(%[filter]), %%ymm14       \n\t" \
    "vmovups "#off1"(%[filter]), %%ymm15       \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm0          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm1          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm13, %%ymm2          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm13, %%ymm3          \n\t" \
    "vpbroadcastd "#i2"("#input"), %%ymm12       \n\t" \
    "vpbroadcastd "#i3"("#input"), %%ymm13       \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm4          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm5          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm13, %%ymm6          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm13, %%ymm7          \n\t" \
    "vpbroadcastd "#i4"("#input"), %%ymm12       \n\t" \
    "vpbroadcastd "#i5"("#input"), %%ymm13       \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm8          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm9          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm13, %%ymm10          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm13, %%ymm11          \n\t" \

#define convKernel3x16c4(input, off0, off1, i0, i1, i2, i3, i4, i5) \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "vpbroadcastd "#i1"("#input"), %%ymm13       \n\t" \
    "vmovups "#off0"(%[filter]), %%ymm14       \n\t" \
    "vmovups "#off1"(%[filter]), %%ymm15       \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm0          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm1          \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm13, %%ymm2          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm13, %%ymm3          \n\t" \
    "vpbroadcastd "#i2"("#input"), %%ymm12       \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm4          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm5          \n\t" \

#define convKernel1x16c4(input, off0, off1, i0, i1, i2, i3, i4, i5) \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "vmovups "#off0"(%[filter]), %%ymm14       \n\t" \
    "vmovups "#off1"(%[filter]), %%ymm15       \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm12, %%ymm0          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm1          \n\t" \

#define convKernelForLoopXx16(rnum, wsize) \
     __asm__ __volatile__("movq %[flags], %%rax                                          \n\t" \
                          "andq $0x1, %%rax                                              \n\t" \
                          "jne 0f                                                        \n\t" \
                          load2BiasToYmmRegs(rnum, %[bias])             \
                          "cmpq $0x8, %%rcx                                             \n\t" \
                          "jl 4f                                                         \n\t" \
                          "jmp 1f                                                        \n\t" \
                          ".align 16                                                     \n\t" \
                          "0:                                                            \n\t" \
                          clear##rnum##Regs(%%ymm)                                             \
                          "cmpq $0x8, %%rcx                                             \n\t" \
                          "jl 4f                                                         \n\t" \
                          ".align 16                                                     \n\t" \
                          "1:                                                            \n\t" \
                          "movq %[input], %%rax                                          \n\t" \
                          convKernel##wsize##x16c4(%%rax, 0x0, 0x20, 0x0, 0x8, 0x10, 0x18, 0x20, 0x28)  \
                          "addq $0x4, %%rax                                              \n\t" \
                          convKernel##wsize##x16c4(%%rax, 0x40, 0x60, 0x0, 0x8, 0x10, 0x18, 0x20, 0x28)  \
                          "addq $0x80, %[filter]                                        \n\t" \
                          "addq %[fStep], %[input]                                       \n\t" \
                          "subq $0x8, %%rcx                                             \n\t" \
                          "cmpq $0x8, %%rcx                                             \n\t" \
                          "jge 1b                                                        \n\t" \
                          ".align 16                                                     \n\t" \
                          "4:                                                            \n\t" \
                          : "+c" (c.ic),                                                       \
                            [input] "+r" (c.input),                                            \
                            [filter] "+r" (c.filter)                                           \
                          : [bias] "r" (c.bias),                                               \
                            [fStep] "r" (c.fStep),                                             \
                            [flags] "r" (c.flags)                                              \
                          : "%rax",                                           \
                            "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",              \
                            "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",            \
                            "%ymm12", "%ymm13", "%ymm14", "%ymm15","memory", "cc");                               \

void Avx512Conv1x1Kernel6x16(ConvController &c) {
    convKernelForLoopXx16(12, 6)

    __asm__ __volatile__("movq %[output], %%rax                             \n\t"
                         "movq %[ostepC16], %%rbx                           \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0x1, %%rcx                                   \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%ymm0, %%ymm0                    \n\t"
                         "vpaddd 0x20(%%rax), %%ymm2, %%ymm2                \n\t"
                         "vpaddd 0x40(%%rax), %%ymm4, %%ymm4                \n\t"
                         "vpaddd 0x60(%%rax), %%ymm6, %%ymm6                \n\t"
                         "vpaddd 0x80(%%rax), %%ymm8, %%ymm8               \n\t"
                         "vpaddd 0xA0(%%rax), %%ymm10, %%ymm10             \n\t"
                         "vpaddd (%%rax, %%rbx), %%ymm1, %%ymm1             \n\t"
                         "vpaddd 0x20(%%rax, %%rbx), %%ymm3, %%ymm3         \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%ymm5, %%ymm5         \n\t"
                         "vpaddd 0x60(%%rax, %%rbx), %%ymm7, %%ymm7         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%ymm9, %%ymm9        \n\t"
                         "vpaddd 0xA0(%%rax, %%rbx), %%ymm11, %%ymm11      \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0xC, %%rcx                                   \n\t"
                         "je 4f                                             \n\t"
                         relu12Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                            \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         convert12RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0x2, %%rcx                                   \n\t"
                         "je 3f                                             \n\t"
                         "vaddps (%[eltwise]), %%ymm0, %%ymm0               \n\t"
                         "vaddps 0x20(%[eltwise]), %%ymm2, %%ymm2           \n\t"
                         "vaddps 0x40(%[eltwise]), %%ymm4, %%ymm4           \n\t"
                         "vaddps 0x60(%[eltwise]), %%ymm6, %%ymm6           \n\t"
                         "vaddps 0x80(%[eltwise]), %%ymm8, %%ymm8          \n\t"
                         "vaddps 0xA0(%[eltwise]), %%ymm10, %%ymm10        \n\t"
                         "vaddps (%[eltwise], %%rbx), %%ymm1, %%ymm1        \n\t"
                         "vaddps 0x20(%[eltwise], %%rbx), %%ymm3, %%ymm3    \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx), %%ymm5, %%ymm5    \n\t"
                         "vaddps 0x60(%[eltwise], %%rbx), %%ymm7, %%ymm7    \n\t"
                         "vaddps 0x80(%[eltwise], %%rbx), %%ymm9, %%ymm9   \n\t"
                         "vaddps 0xA0(%[eltwise], %%rbx), %%ymm11, %%ymm11 \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0xC, %%rcx                                   \n\t"
                         "je 4f                                             \n\t"
                         relu12Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                         \n\t"
                         "4:                                                \n\t"
                         "vmovups %%ymm0, (%%rax)                           \n\t"
                         "vmovups %%ymm2, 0x20(%%rax)                       \n\t"
                         "vmovups %%ymm4, 0x40(%%rax)                       \n\t"
                         "vmovups %%ymm6, 0x60(%%rax)                       \n\t"
                         "vmovups %%ymm8, 0x80(%%rax)                      \n\t"
                         "vmovups %%ymm10, 0xA0(%%rax)                     \n\t"
                         "vmovups %%ymm1, (%%rax, %%rbx)                    \n\t"
                         "vmovups %%ymm3, 0x20(%%rax, %%rbx)                \n\t"
                         "vmovups %%ymm5, 0x40(%%rax, %%rbx)                \n\t"
                         "vmovups %%ymm7, 0x60(%%rax, %%rbx)                \n\t"
                         "vmovups %%ymm9, 0x80(%%rax, %%rbx)               \n\t"
                         "vmovups %%ymm11, 0xA0(%%rax, %%rbx)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", 
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void Avx512Conv1x1Kernel3x16(ConvController &c) {
    convKernelForLoopXx16(6, 3)

    __asm__ __volatile__("movq %[output], %%rax                             \n\t"
                         "movq %[ostepC16], %%rbx                           \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0x1, %%rcx                                   \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%ymm0, %%ymm0                    \n\t"
                         "vpaddd 0x20(%%rax), %%ymm2, %%ymm2                \n\t"
                         "vpaddd 0x40(%%rax), %%ymm4, %%ymm4                \n\t"
                         "vpaddd (%%rax, %%rbx), %%ymm1, %%ymm1             \n\t"
                         "vpaddd 0x20(%%rax, %%rbx), %%ymm3, %%ymm3         \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%ymm5, %%ymm5         \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0xC, %%rcx                                   \n\t"
                         "je 4f                                             \n\t"
                         relu6Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                            \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         convert6RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0x2, %%rcx                                   \n\t"
                         "je 3f                                             \n\t"
                         "vaddps (%[eltwise]), %%ymm0, %%ymm0               \n\t"
                         "vaddps 0x20(%[eltwise]), %%ymm2, %%ymm2           \n\t"
                         "vaddps 0x40(%[eltwise]), %%ymm4, %%ymm4           \n\t"
                         "vaddps (%[eltwise], %%rbx), %%ymm1, %%ymm1        \n\t"
                         "vaddps 0x20(%[eltwise], %%rbx), %%ymm3, %%ymm3    \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx), %%ymm5, %%ymm5    \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0xC, %%rcx                                   \n\t"
                         "je 4f                                             \n\t"
                         relu6Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                         \n\t"
                         "4:                                                \n\t"
                         "vmovups %%ymm0, (%%rax)                           \n\t"
                         "vmovups %%ymm2, 0x20(%%rax)                       \n\t"
                         "vmovups %%ymm4, 0x40(%%rax)                       \n\t"
                         "vmovups %%ymm1, (%%rax, %%rbx)                    \n\t"
                         "vmovups %%ymm3, 0x20(%%rax, %%rbx)                \n\t"
                         "vmovups %%ymm5, 0x40(%%rax, %%rbx)                \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                           "memory", "cc");
}

void Avx512Conv1x1Kernel1x16(ConvController &c) {
    convKernelForLoopXx16(2, 1)

    __asm__ __volatile__("movq %[output], %%rax                      \n\t"
                         "movq %[ostepC16], %%rbx                    \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x1, %%rcx                            \n\t"
                         "je 0f                                      \n\t"
                         "vpaddd (%%rax), %%ymm0, %%ymm0             \n\t"
                         "vpaddd (%%rax, %%rbx), %%ymm1, %%ymm1      \n\t"

                         ".align 16                                  \n\t"
                         "0:                                         \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu2Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                     \n\t"

                         ".align 16                                  \n\t"
                         "1:                                         \n\t"
                         convert2RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                  \n\t"
                         "2:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x2, %%rcx                            \n\t"
                         "je 3f                                      \n\t"
                         "vaddps (%[eltwise]), %%ymm0, %%ymm0        \n\t"
                         "vaddps (%[eltwise], %%rbx), %%ymm1, %%ymm1 \n\t"

                         ".align 16                                  \n\t"
                         "3:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu2Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                  \n\t"
                         "4:                                         \n\t"
                         "vmovups %%ymm0, (%%rax)                    \n\t"
                         "vmovups %%ymm1, (%%rax, %%rbx)             \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm12", "%ymm13",
                           "%ymm14", "%ymm15", "memory", "cc");
}

#define convKernel8x8c4(input, off0, i0, i1, i2, i3, i4, i5, i6, i7) \
    "vmovups "#off0"(%[filter]), %%ymm15       \n\t" \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "vpbroadcastd "#i1"("#input"), %%ymm13       \n\t" \
    "vpbroadcastd "#i2"("#input"), %%ymm14       \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm0          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm13, %%ymm1          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm14, %%ymm2          \n\t" \
    "vpbroadcastd "#i3"("#input"), %%ymm12       \n\t" \
    "vpbroadcastd "#i4"("#input"), %%ymm13       \n\t" \
    "vpbroadcastd "#i5"("#input"), %%ymm14       \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm3          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm13, %%ymm4          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm14, %%ymm5          \n\t" \
    "vpbroadcastd "#i6"("#input"), %%ymm12       \n\t" \
    "vpbroadcastd "#i7"("#input"), %%ymm13       \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm6          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm13, %%ymm7          \n\t" \

#define convKernel4x8c4(input, off0, i0, i1, i2, i3, i4, i5, i6, i7) \
    "vmovups "#off0"(%[filter]), %%ymm15       \n\t" \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "vpbroadcastd "#i1"("#input"), %%ymm13       \n\t" \
    "vpbroadcastd "#i2"("#input"), %%ymm14       \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm0          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm13, %%ymm1          \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm14, %%ymm2          \n\t" \
    "vpbroadcastd "#i3"("#input"), %%ymm12       \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm3          \n\t" \

#define convKernel1x8c4(input, off0, i0, i1, i2, i3, i4, i5, i6, i7) \
    "vmovups "#off0"(%[filter]), %%ymm15       \n\t" \
    "vpbroadcastd "#i0"("#input"), %%ymm12       \n\t" \
    "%{vex%} vpdpbusd %%ymm15, %%ymm12, %%ymm0          \n\t" \

#define convKernelForLoopXx8(rnum, wsize) \
     __asm__ __volatile__("movq %[flags], %%rax                                                \n\t" \
                          "andq $0x1, %%rax                                                    \n\t" \
                          "jne 0f                                                              \n\t" \
                          load1BiasToRegs(rnum, %[bias], %%ymm)             \
                          "cmpq $0x8, %%rcx                                                   \n\t" \
                          "jl 4f                                                               \n\t" \
                          "jmp 1f                                                              \n\t" \
                          ".align 16                                                           \n\t" \
                          "0:                                                                  \n\t" \
                          clear##rnum##Regs(%%ymm)                                                   \
                          "cmpq $0x8, %%rcx                                                   \n\t" \
                          "jl 4f                                                               \n\t" \
                          ".align 16                                                           \n\t" \
                          "1:                                                                  \n\t" \
                          "movq %[input], %%rax                                                \n\t" \
                          convKernel##wsize##x8c4(%%rax, 0x0, 0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38) \
                          "addq $0x4, %%rax                                                    \n\t" \
                          convKernel##wsize##x8c4(%%rax, 0x20, 0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38) \
                          "addq $0x40, %[filter]                                            \n\t" \
                          "addq %[fStep], %[input]                                             \n\t" \
                          "subq $0x8, %%rcx                                                   \n\t" \
                          "cmpq $0x8, %%rcx                                                   \n\t" \
                          "jge 1b                                                              \n\t" \
                          ".align 16                                                           \n\t" \
                          "4:                                                                  \n\t" \
                          : "+c" (c.ic),                                                             \
                            [input] "+r" (c.input),                                                  \
                            [filter] "+r" (c.filter)                                                 \
                          : [bias] "r" (c.bias),                                                     \
                            [kh] "r" (c.kh),                                                         \
                            [kw] "r" (c.kw),                                                         \
                            [fStep] "r" (c.fStep),                                                   \
                            [flags] "r" (c.flags)                                                    \
                          : "%rax",                                                   \
                            "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",                    \
                            "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",                  \
                            "%ymm12", "%ymm13", "%ymm14", "%ymm15",  "memory", "cc");                                     \

void Avx512Conv1x1Kernel8x8(ConvController &c) {
    convKernelForLoopXx8(8, 8)

    __asm__ __volatile__("movq %[output], %%rax                      \n\t"
                         "movq %[ostepC16], %%rbx                    \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x1, %%rcx                            \n\t"
                         "je 0f                                      \n\t"
                         "vpaddd (%%rax),      %%ymm0, %%ymm0        \n\t"
                         "vpaddd 0x20(%%rax),  %%ymm1, %%ymm1        \n\t"
                         "vpaddd 0x40(%%rax),  %%ymm2, %%ymm2        \n\t"
                         "vpaddd 0x60(%%rax),  %%ymm3, %%ymm3        \n\t"
                         "vpaddd 0x80(%%rax), %%ymm4, %%ymm4        \n\t"
                         "vpaddd 0xA0(%%rax), %%ymm5, %%ymm5        \n\t"
                         "vpaddd 0xC0(%%rax), %%ymm6, %%ymm6        \n\t"
                         "vpaddd 0xE0(%%rax), %%ymm7, %%ymm7        \n\t"
                         
                         ".align 16                                  \n\t"
                         "0:                                         \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu8Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                     \n\t"

                         ".align 16                                  \n\t"
                         "1:                                         \n\t"
                         convert8RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                  \n\t"
                         "2:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x2, %%rcx                            \n\t"
                         "je 3f                                      \n\t"
                         "vaddps (%[eltwise]),      %%ymm0, %%ymm0   \n\t"
                         "vaddps 0x20(%[eltwise]),  %%ymm1, %%ymm1   \n\t"
                         "vaddps 0x40(%[eltwise]),  %%ymm2, %%ymm2   \n\t"
                         "vaddps 0x60(%[eltwise]),  %%ymm3, %%ymm3   \n\t"
                         "vaddps 0x80(%[eltwise]), %%ymm4, %%ymm4   \n\t"
                         "vaddps 0xA0(%[eltwise]), %%ymm5, %%ymm5   \n\t"
                         "vaddps 0xC0(%[eltwise]), %%ymm6, %%ymm6   \n\t"
                         "vaddps 0xE0(%[eltwise]), %%ymm7, %%ymm7   \n\t"

                         ".align 16                                  \n\t"
                         "3:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu8Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                  \n\t"
                         "4:                                         \n\t"
                         "vmovups %%ymm0,  (%%rax)                   \n\t"
                         "vmovups %%ymm1,  0x20(%%rax)               \n\t"
                         "vmovups %%ymm2,  0x40(%%rax)               \n\t"
                         "vmovups %%ymm3,  0x60(%%rax)               \n\t"
                         "vmovups %%ymm4,  0x80(%%rax)              \n\t"
                         "vmovups %%ymm5,  0xA0(%%rax)              \n\t"
                         "vmovups %%ymm6,  0xC0(%%rax)              \n\t"
                         "vmovups %%ymm7,  0xE0(%%rax)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void Avx512Conv1x1Kernel4x8(ConvController &c) {
     convKernelForLoopXx8(4, 4)

    __asm__ __volatile__("movq %[output], %%rax                      \n\t"
                         "movq %[ostepC16], %%rbx                    \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x1, %%rcx                            \n\t"
                         "je 0f                                      \n\t"
                         "vpaddd (%%rax),      %%ymm0, %%ymm0        \n\t"
                         "vpaddd 0x20(%%rax),  %%ymm1, %%ymm1        \n\t"
                         "vpaddd 0x40(%%rax),  %%ymm2, %%ymm2        \n\t"
                         "vpaddd 0x60(%%rax),  %%ymm3, %%ymm3        \n\t"

                         ".align 16                                  \n\t"
                         "0:                                         \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu4Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                     \n\t"

                         ".align 16                                  \n\t"
                         "1:                                         \n\t"
                         convert4RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                  \n\t"
                         "2:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x2, %%rcx                            \n\t"
                         "je 3f                                      \n\t"
                         "vaddps (%[eltwise]),      %%ymm0, %%ymm0   \n\t"
                         "vaddps 0x20(%[eltwise]),  %%ymm1, %%ymm1   \n\t"
                         "vaddps 0x40(%[eltwise]),  %%ymm2, %%ymm2   \n\t"
                         "vaddps 0x60(%[eltwise]),  %%ymm3, %%ymm3   \n\t"

                         ".align 16                                  \n\t"
                         "3:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu4Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                  \n\t"
                         "4:                                         \n\t"
                         "vmovups %%ymm0,  (%%rax)                   \n\t"
                         "vmovups %%ymm1,  0x20(%%rax)               \n\t"
                         "vmovups %%ymm2,  0x40(%%rax)               \n\t"
                         "vmovups %%ymm3,  0x60(%%rax)               \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm12", "%ymm13",
                           "%ymm14","%ymm15", "memory", "cc");
}

void Avx512Conv1x1Kernel1x8(ConvController &c) {
    convKernelForLoopXx8(1, 1)

    __asm__ __volatile__("movq %[output], %%rax                 \n\t"
                         "movq %[flags], %%rcx                  \n\t"
                         "and $0x1, %%rcx                       \n\t"
                         "je 0f                                 \n\t"
                         "vpaddd (%%rax),      %%ymm0, %%ymm0   \n\t"

                         ".align 16                             \n\t"
                         "0:                                    \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                  \n\t"
                         "and $0xC, %%rcx                       \n\t"
                         "je 4f                                 \n\t"
                         relu1Regs(%%ymm, vpxor, vpmaxsd)
                         "jmp 4f                                \n\t"

                         ".align 16                             \n\t"
                         "1:                                    \n\t"
                         convert1RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                             \n\t"
                         "2:                                    \n\t"
                         "movq %[flags], %%rcx                  \n\t"
                         "and $0x2, %%rcx                       \n\t"
                         "je 3f                                 \n\t"
                         "vaddps (%[eltwise]), %%ymm0, %%ymm0   \n\t"

                         ".align 16                             \n\t"
                         "3:                                    \n\t"
                         "movq %[flags], %%rcx                  \n\t"
                         "and $0xC, %%rcx                       \n\t"
                         "je 4f                                 \n\t"
                         relu1Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                             \n\t"
                         "4:                                    \n\t"
                         "vmovups %%ymm0,  (%%rax)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}


template <typename T1>
EE activateBias(const T1 *biasArray, T1 *activatedArray, U32 len, ActivationMode mode) {
    switch (mode) {
        case ACTIVATION_RELU: {
            for (U32 ocb = 0; ocb < len; ++ocb) {
                activatedArray[ocb] = (biasArray[ocb] <= 0)? 0: biasArray[ocb];
            }
            break;
        }
        case ACTIVATION_RELU6: {
            for (U32 ocb = 0; ocb < len; ++ocb) {
                activatedArray[ocb] =
                    (biasArray[ocb] <= 0)? 0: ((biasArray[ocb] >= 6)? 6: biasArray[ocb]);
            }
            break;
        }
        default:
            return NOT_SUPPORTED;
    }
    return SUCCESS;
}

inline void getActivatedBiasForPadding(
    const F32 *biasArray, TensorDesc biasDesc, DataType targetType, void *activatedBias, ActivationMode mode, F32 scaleB)
{
    if (targetType == DT_I32) {
        CHECK_STATUS(quantize_bias_offsetC((const void *)biasArray, biasDesc, DT_I32,
            nullptr, biasDesc, &scaleB, activatedBias));
        CHECK_STATUS(activateBias<I32>((const I32 *)activatedBias,
            (I32 *)activatedBias, tensorNumElements(biasDesc), mode));
    } else if (targetType == DT_F32) {
        CHECK_STATUS(activateBias<F32>((const F32 *)biasArray,
            (F32 *)activatedBias, tensorNumElements(biasDesc), mode));
    } else {
        CHECK_STATUS(NOT_MATCH);
    } 
}

// clang-format on
EE convolution_1x1_direct(TensorDesc inputDesc,
    UINT8 *inArray,
    F32 *eltwiseInput,
    TensorDesc filterDesc,
    const INT8 *filterArray,
    ConvolutionParamSpec convParamSpec,
    TensorDesc biasDesc,
    const F32 *biasArray,
    U32 tmpBytes,
    void *tmp,
    TensorDesc outputDesc,
    void *outArray,
    F32 *scale,
    ActivationParamSpec activationDesc)
{
    DataType idt, fdt, odt;
    DataFormat idf, fdf, odf;
    U32 in, ic, ih, iw;
    U32 fn, fc, fh, fw;
    U32 on, oc, oh, ow;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw));
    CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));

    if (idf == DF_MTK) {
        idf = DF_NCHW;
    }

    if (fdf != DF_NCHWC2NxC4 || (idf != DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }

    // get kernels
    U32 ocSizeArray[3] = {8, 16, 24};
    U32 wSizeArray[3] = {8, 6, 4};
    const kernelFunc kernel[3][3] = {
        {Avx512Conv1x1Kernel1x8, Avx512Conv1x1Kernel4x8, Avx512Conv1x1Kernel8x8},
        {Avx512Conv1x1Kernel1x16, Avx512Conv1x1Kernel3x16, Avx512Conv1x1Kernel6x16},
        {Avx512Conv1x1Kernel1x24, Avx512Conv1x1Kernel2x24, Avx512Conv1x1Kernel4x24}};

    // get computing params
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;
    U32 dilateH = convParamSpec.dilatedRate_h;
    U32 dilateW = convParamSpec.dilatedRate_w;
    U32 ih_stride = (ih + strideH - 1) / strideH;
    U32 iw_stride = (iw + strideW - 1) / strideW;
    U32 ohow = oh * ow;
    UINT8 *output = (UINT8 *)outArray;

    // infer block params

    // infer kernel params
    ConvController convCtl;
    convCtl.ostepC16 = oh * ow * 8 * 4;
    convCtl.dilateW = dilateW * SIMDW;
    convCtl.dilateH = (iw_stride - fw * dilateW + (dilateH - 1) * iw_stride) * SIMDW;
    convCtl.fStep = ih_stride * iw_stride * SIMDW;
    convCtl.kw = fw;
    convCtl.kh = fh;
    convCtl.scale = nullptr;
    U32 unrollOc = 24;
    // if (fn % 32 == 0 && fn % 24 != 0) {
    //     unrollOc = 32;
    // }

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
    if ((odt != DT_F32) && (odt != DT_I32)) {
        output = (UINT8 *)tmp;
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
    CHECK_STATUS(quantize_bias_offsetC((const void *)biasArray, biasDesc, DT_I32,
        (const void *)filterArray, filterDesc, scaleO, offsetC));
    filterArray += oc * 4;

    // for (U32 i = 0; i < oc; ++i) {
    //     offsetC[i] = 0;
    // }

    F32 *activatedBias = (F32 *)tmp;
    if (paddingT > 0 || paddingB > 0 || paddingL > 0 || paddingR > 0) {
        getActivatedBiasForPadding(
            biasArray, biasDesc, outputDesc.dt, activatedBias, activationDesc.mode, *scaleO);
        tmp = (void *)((U8 *)tmp + oc * bytesOf(DT_F32));
    }

    U32 oBytes = bytesOf(outputDesc.dt);
    UINT8 *tmpInput = (UINT8 *)tmp;
    if (idf != DF_NCHWC8) {
        tmp = (void *)((U8 *)tmp + ic * ih * iw);
    }
    UINT8 *useInput = (UINT8 *)tmp;

    for (U32 n = 0; n < in; ++n) {
        UINT8 *bInArray = inArray + n * ic * ih * iw;
        if (idf == DF_NCHWC8) {
            tmpInput = bInArray;
        } else {
            return NOT_SUPPORTED;
            // PaddingNCHW2NCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
        }

        if (strideH > 1 || strideW > 1) {
            U32 ic8 = ic / 8;
            for (U32 hc = 0; hc < ih_stride * ic8; ++hc) {
                U32 c = hc / ih_stride;
                U32 h = hc % ih_stride;
                for (U32 w = 0; w < iw_stride; ++w) {
                    U32 nh = h * strideH;
                    U32 nw = w * strideW;
                    UNI_MEMCPY(
                        useInput + c * ih_stride * iw_stride * SIMDW + (h * iw_stride + w) * SIMDW,
                        tmpInput + c * ih * iw * SIMDW + (nh * iw + nw) * SIMDW, SIMDW);
                }
            }
        } else {
            useInput = tmpInput;
        }

        U32 flags = 0;
        U32 icSize = 0;
        for (U32 icbb = 0; icbb < ic; icbb += icSize) {
            icSize = UNI_MIN(BLOCK_IC_DIM, ic - icbb);
            flags |= (icbb > 0);
            if (icbb == (int)ic - icSize) {
                flags |= (eltwiseInput != nullptr) << 1;
                flags |= U32(activationDesc.mode) << 2;
                convCtl.scale = factorPtr;
            }
            convCtl.flags = flags;
            if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
                U32 hwSize = 0;
                for (U32 hw = 0; hw < ohow; hw += hwSize) {
                    U32 ocSize = 0;
                    hwSize = UNI_MIN(BLOCK_HW_DIM, ohow - hw);
                    for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                        ocSize = UNI_MIN(unrollOc, oc - ocb);
                        U32 oIdx = (ocSize >> 3) - 1;
                        ocSize = ocSizeArray[oIdx];
                        convCtl.bias = offsetC + ocb;
                        UINT8 *curI = useInput + icbb * ih_stride * iw_stride;
                        U32 wSize = 8;
                        U32 unrollW = wSizeArray[oIdx];
                        for (U32 ihw = hw; ihw < hw + hwSize; ihw += wSize) {
                            wSize = UNI_MIN(hw + hwSize - ihw, unrollW);
                            U32 idx = wSize * 2 / unrollW;
                            wSize = UNI_MAX(idx * unrollW / 2, 1);
                            U32 in_h = ihw / ow;
                            U32 in_w = ihw % ow;
                            convCtl.input = curI + in_h * iw_stride * SIMDW + in_w * SIMDW;
                            convCtl.output =
                                output + ((n * oc + ocb) * ohow + ihw * SIMDW) * oBytes;
                            convCtl.eltwise = eltwiseInput + (n * oc + ocb) * ohow + ihw * SIMDW;
                            convCtl.filter =
                                filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                            convCtl.ic = icSize;
                            kernel[oIdx][idx](convCtl);
                        }
                    }
                }
            } else {
                for (U32 h = 0; h < oh; ++h) {
                    U32 ocSize = 0;
                    for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                        ocSize = UNI_MIN(unrollOc, oc - ocb);
                        U32 oIdx = (ocSize >> 3) - 1;
                        ocSize = ocSizeArray[oIdx];
                        convCtl.bias = offsetC + ocb;
                        UINT8 *curI = useInput + icbb * ih_stride * iw_stride;
                        U32 wSize = 8;
                        U32 unrollW = wSizeArray[oIdx];
                        for (U32 w = 0; w < ow; w += wSize) {
                            wSize = 1;
                            convCtl.output =
                                output + ((n * oc + ocb) * ohow + (h * ow + w) * SIMDW) * oBytes;
                            convCtl.eltwise = eltwiseInput +
                                ((n * oc + ocb) * ohow + (h * ow + w) * SIMDW) * oBytes;
                            // directly store activated bias
                            if ((h < paddingT) || (h >= ih_stride + paddingT) || (w < paddingL) ||
                                (w >= paddingL + iw_stride)) {
                                if (!(flags & 0x2) && (icbb == (int)ic - icSize)) {
                                    int oci = 0;
                                    for (oci = 0; oci < (int)ocSize + 1 - SIMDW; oci += SIMDW) {
                                        UNI_MEMCPY(((U8 *)convCtl.output) + ohow * oci * oBytes,
                                            activatedBias + oci + ocb, SIMDW * oBytes);
                                    }
                                    for (; oci < (int)ocSize; oci += 8) {
                                        UNI_MEMCPY(((U8 *)convCtl.output) + ohow * oci * oBytes,
                                            activatedBias + oci + ocb, 8 * oBytes);
                                    }
                                }
                                continue;
                            }
                            wSize = UNI_MIN(iw_stride - (w - paddingL), unrollW);
                            U32 idx = wSize * 2 / unrollW;
                            wSize = UNI_MAX(idx * unrollW / 2, 1);

                            convCtl.input =
                                curI + (h - paddingT) * iw_stride * SIMDW + (w - paddingL) * SIMDW;
                            convCtl.filter =
                                filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                            convCtl.ic = icSize;
                            kernel[oIdx][idx](convCtl);
                        }
                    }
                }
            }
        }
    }

    // quantization
    if (odt == DT_U8_Q) {
        F32 scales[2] = {-1, scaleO[0]};
        TensorDesc qDesc = outputDesc;
        qDesc.dt = DT_U8_Q;
        CHECK_STATUS(quantize_x86(outputDesc, (void *)output, &qDesc, (void *)outArray, scales));
        *scaleO = scales[0];
    }

    return SUCCESS;
}
