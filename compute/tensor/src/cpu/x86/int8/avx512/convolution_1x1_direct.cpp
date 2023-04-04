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
#include "cpu/x86/int8/transform_functions_int8.h"
#include "cpu/x86/int8/tensor_computing_int8.h"
#include "cpu/x86/int8/convolution_functions.h"
#include "cpu/x86/tensor_computing_x86.h"

#define SIMDW 16
#define BLOCK_IC_DIM 256
#define BLOCK_HW_DIM 768

// clang-format off
#ifdef _USE_AVX512_VNNI
#define convKernel8x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                         i0, i1, i2, i3, i4, i5, i6, i7) \
    "vpbroadcastd "#i0"("#input"), %%zmm30       \n\t" \
    "vpbroadcastd "#i1"("#input"), %%zmm31       \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1          \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2          \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"        \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm3          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm4          \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm5          \n\t" \
    "vpbroadcastd "#i2"("#input"), %%zmm30       \n\t" \
    "vpbroadcastd "#i3"("#input"), %%zmm31       \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm6          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm7          \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm8          \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"        \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm9          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm10         \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm11         \n\t" \
    "vpbroadcastd "#i4"("#input"), %%zmm30       \n\t" \
    "vpbroadcastd "#i5"("#input"), %%zmm31       \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12         \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13         \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm14         \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"        \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm15         \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm16         \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm17         \n\t" \
    "vpbroadcastd "#i6"("#input"), %%zmm30       \n\t" \
    "vpbroadcastd "#i7"("#input"), %%zmm31       \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm18         \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm19         \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm20         \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm21         \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm22         \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm23         \n\t"

#define convKernel4x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                         i0, i1, i2, i3, i4, i5, i6, i7) \
    "vpbroadcastd "#i0"("#input"), %%zmm30       \n\t" \
    "vpbroadcastd "#i1"("#input"), %%zmm31       \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1          \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"        \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm3          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm4          \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm5          \n\t" \
    "vpbroadcastd "#i2"("#input"), %%zmm30       \n\t" \
    "vpbroadcastd "#i3"("#input"), %%zmm31       \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm6          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm7          \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"        \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm8          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm9          \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"        \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm10         \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm11         \n\t"

#define convKernel1x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                         i0, i1, i2, i3, i4, i5, i6, i7) \
    "vpbroadcastd ("#input"), %%zmm30            \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"        \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"        \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"        \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1          \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2          \n\t"
#else
#define convKernel8x48c4_3(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                           i0, i1, i2, i3, i4, i5, i6, i7) \
    "vpbroadcastd "#i0"("#input"), %%zmm30       \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i1"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0             \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1             \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2             \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i2"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm3, "#preg0", %%zmm3             \n\t" \
    "vpaddd %%zmm4, "#preg1", %%zmm4             \n\t" \
    "vpaddd %%zmm5, "#preg2", %%zmm5             \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i3"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm6, "#preg0", %%zmm6             \n\t" \
    "vpaddd %%zmm7, "#preg1", %%zmm7             \n\t" \
    "vpaddd %%zmm8, "#preg2", %%zmm8             \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i4"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm9, "#preg0", %%zmm9             \n\t" \
    "vpaddd %%zmm10, "#preg1", %%zmm10           \n\t" \
    "vpaddd %%zmm11, "#preg2", %%zmm11           \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i5"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm12, "#preg0", %%zmm12           \n\t" \
    "vpaddd %%zmm13, "#preg1", %%zmm13           \n\t" \
    "vpaddd %%zmm14, "#preg2", %%zmm14           \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i6"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm15, "#preg0", %%zmm15           \n\t" \
    "vpaddd %%zmm16, "#preg1", %%zmm16           \n\t" \
    "vpaddd %%zmm17, "#preg2", %%zmm17           \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i7"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm18, "#preg0", %%zmm18           \n\t" \
    "vpaddd %%zmm19, "#preg1", %%zmm19           \n\t" \
    "vpaddd %%zmm20, "#preg2", %%zmm20           \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vmovups "#off2"(%[filter]), "#freg2"        \n\t" \
    "vpaddd %%zmm21, "#preg0", %%zmm21           \n\t" \
    "vpaddd %%zmm22, "#preg1", %%zmm22           \n\t" \
    "vpaddd %%zmm23, "#preg2", %%zmm23           \n\t"

#define convKernel4x48c4_3(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                           i0, i1, i2, i3, i4, i5, i6, i7) \
    "vpbroadcastd "#i0"("#input"), %%zmm30       \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i1"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0             \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1             \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2             \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i2"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm3, "#preg0", %%zmm3             \n\t" \
    "vpaddd %%zmm4, "#preg1", %%zmm4             \n\t" \
    "vpaddd %%zmm5, "#preg2", %%zmm5             \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vpbroadcastd "#i3"("#input"), %%zmm30       \n\t" \
    "vpaddd %%zmm6, "#preg0", %%zmm6             \n\t" \
    "vpaddd %%zmm7, "#preg1", %%zmm7             \n\t" \
    "vpaddd %%zmm8, "#preg2", %%zmm8             \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vmovups "#off2"(%[filter]), "#freg2"        \n\t" \
    "vpaddd %%zmm9, "#preg0", %%zmm9             \n\t" \
    "vpaddd %%zmm10, "#preg1", %%zmm10           \n\t" \
    "vpaddd %%zmm11, "#preg2", %%zmm11           \n\t"

#define convKernel1x48c4_3(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                           i0, i1, i2, i3, i4, i5, i6, i7) \
    "vpbroadcastd ("#input"), %%zmm30            \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"      \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"      \n\t" \
    "vpmaddubsw "#freg2", %%zmm30, "#preg2"      \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"        \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"        \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"        \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"        \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"        \n\t" \
    "vmovups "#off2"(%[filter]), "#freg2"        \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0             \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1             \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2             \n\t"

#define convKernel8x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                         i0, i1, i2, i3, i4, i5, i6, i7)                                    \
    convKernel8x48c4_3(input, %%zmm24, %%zmm25, %%zmm26, off0, off1, off2,                  \
                       %%zmm27, %%zmm28, %%zmm29,                                           \
                       i0, i1, i2, i3, i4, i5, i6, i7)

#define convKernel4x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                         i0, i1, i2, i3, i4, i5, i6, i7)                                    \
    convKernel4x48c4_3(input, %%zmm24, %%zmm25, %%zmm26, off0, off1, off2,                  \
                       %%zmm27, %%zmm28, %%zmm29,                                           \
                       i0, i1, i2, i3, i4, i5, i6, i7)

#define convKernel1x48c4(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2, \
                         i0, i1, i2, i3, i4, i5, i6, i7)                                    \
    convKernel1x48c4_3(input, %%zmm24, %%zmm25, %%zmm26, off0, off1, off2,                  \
                       %%zmm27, %%zmm28, %%zmm29,                                           \
                       i0, i1, i2, i3, i4, i5, i6, i7)
#endif

#define convKernelForLoopXx48(rnum, wsize) \
     __asm__ __volatile__("vmovups (%[filter]), %%zmm24                                     \n\t" \
                          "vmovups 0x40(%[filter]), %%zmm25                                 \n\t" \
                          "vmovups 0x80(%[filter]), %%zmm26                                 \n\t" \
                          "addq $0xC0, %[filter]                                            \n\t" \
                          "mov $1, %%eax                                                    \n\t" \
                          "vmovd %%eax, %%xmm0                                              \n\t" \
                          "vpbroadcastw %%xmm0, %%zmm31                                     \n\t" \
                          "movq %[flags], %%rax                                             \n\t" \
                          "andq $0x1, %%rax                                                 \n\t" \
                          "jne 0f                                                           \n\t" \
                          load3BiasToZmmRegs(rnum, %[bias])                                       \
                          "cmpq $0x10, %%rcx                                                \n\t" \
                          "jl 4f                                                            \n\t" \
                          "jmp 1f                                                           \n\t" \
                          ".align 16                                                        \n\t" \
                          "0:                                                               \n\t" \
                          clear##rnum##Regs(%%zmm)                                                \
                          "cmpq $0x10, %%rcx                                                \n\t" \
                          "jl 4f                                                            \n\t" \
                          ".align 16                                                        \n\t" \
                          "1:                                                               \n\t" \
                          "movq %[input], %%rax                                             \n\t" \
                          convKernel##wsize##x48c4(%%rax, %%zmm24, %%zmm25, %%zmm26,              \
                                                   0x0, 0x40, 0x80, %%zmm27, %%zmm28, %%zmm29,    \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70) \
                          "addq $0x4, %%rax  \n\t"                                                \
                          convKernel##wsize##x48c4(%%rax, %%zmm27, %%zmm28, %%zmm29,              \
                                                   0xC0, 0x100, 0x140, %%zmm24, %%zmm25, %%zmm26, \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70) \
                          "addq $0x4, %%rax  \n\t"                                                \
                          convKernel##wsize##x48c4(%%rax, %%zmm24, %%zmm25, %%zmm26,              \
                                                   0x180, 0x1C0, 0x200,                           \
                                                   %%zmm27, %%zmm28, %%zmm29,                     \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70) \
                          "addq $0x4, %%rax  \n\t"                                                \
                          convKernel##wsize##x48c4(%%rax, %%zmm27, %%zmm28, %%zmm29,              \
                                                   0x240, 0x280, 0x2C0,                           \
                                                   %%zmm24, %%zmm25, %%zmm26,                     \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70) \
                          "addq $0x300, %[filter]                                           \n\t" \
                          "addq %[fStep], %[input]                                          \n\t" \
                          "subq $0x10, %%rcx                                                \n\t" \
                          "cmpq $0x10, %%rcx                                                \n\t" \
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
                          : "%rax", "%rbx", "%r9",                                                \
                            "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",                 \
                            "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",               \
                            "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",           \
                            "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",           \
                            "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",           \
                            "%zmm30", "%zmm31", "memory", "cc");                                  \

void Avx512Conv1x1Kernel8x48(ConvController &c) {
     convKernelForLoopXx48(24, 8)

    __asm__ __volatile__("movq %[output], %%rax                                \n\t"
                         "movq %[ostepC16], %%rbx                              \n\t"
                         "movq %[flags], %%rcx                                 \n\t"
                         "and $0x1, %%rcx                                      \n\t"
                         "je 0f                                                \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                       \n\t"
                         "vpaddd 0x40(%%rax), %%zmm3, %%zmm3                   \n\t"
                         "vpaddd 0x80(%%rax), %%zmm6, %%zmm6                   \n\t"
                         "vpaddd 0xC0(%%rax), %%zmm9, %%zmm9                   \n\t"
                         "vpaddd 0x100(%%rax), %%zmm12, %%zmm12                \n\t"
                         "vpaddd 0x140(%%rax), %%zmm15, %%zmm15                \n\t"
                         "vpaddd 0x180(%%rax), %%zmm18, %%zmm18                \n\t"
                         "vpaddd 0x1C0(%%rax), %%zmm21, %%zmm21                \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1                \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%zmm4, %%zmm4            \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%zmm7, %%zmm7            \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx), %%zmm10, %%zmm10          \n\t"
                         "vpaddd 0x100(%%rax, %%rbx), %%zmm13, %%zmm13         \n\t"
                         "vpaddd 0x140(%%rax, %%rbx), %%zmm16, %%zmm16         \n\t"
                         "vpaddd 0x180(%%rax, %%rbx), %%zmm19, %%zmm19         \n\t"
                         "vpaddd 0x1C0(%%rax, %%rbx), %%zmm22, %%zmm22         \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%zmm2, %%zmm2             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx, 2), %%zmm5, %%zmm5         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx, 2), %%zmm8, %%zmm8         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx, 2), %%zmm11, %%zmm11       \n\t"
                         "vpaddd 0x100(%%rax, %%rbx, 2), %%zmm14, %%zmm14      \n\t"
                         "vpaddd 0x140(%%rax, %%rbx, 2), %%zmm17, %%zmm17      \n\t"
                         "vpaddd 0x180(%%rax, %%rbx, 2), %%zmm20, %%zmm20      \n\t"
                         "vpaddd 0x1C0(%%rax, %%rbx, 2), %%zmm23, %%zmm23      \n\t"

                         ".align 16                                            \n\t"
                         "0:                                                   \n\t"
                         "cmpq $0x0, %[scale]                                  \n\t"
                         "jne 1f                                               \n\t"
                         "movq %[flags], %%rcx                                 \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 4f                                                \n\t"
                         relu24Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                               \n\t"

                         ".align 16                                            \n\t"
                         "1:                                                   \n\t"
                         convert24RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                            \n\t"
                         "2:                                                   \n\t"
                         "movq %[flags], %%rcx                                 \n\t"
                         "and $0x2, %%rcx                                      \n\t"
                         "je 3f                                                \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0                  \n\t"
                         "vaddps 0x40(%[eltwise]), %%zmm3, %%zmm3              \n\t"
                         "vaddps 0x80(%[eltwise]), %%zmm6, %%zmm6              \n\t"
                         "vaddps 0xC0(%[eltwise]), %%zmm9, %%zmm9              \n\t"
                         "vaddps 0x100(%[eltwise]), %%zmm12, %%zmm12           \n\t"
                         "vaddps 0x140(%[eltwise]), %%zmm15, %%zmm15           \n\t"
                         "vaddps 0x180(%[eltwise]), %%zmm18, %%zmm18           \n\t"
                         "vaddps 0x1C0(%[eltwise]), %%zmm21, %%zmm21           \n\t"
                         "vaddps (%[eltwise], %%rbx), %%zmm1, %%zmm1           \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx), %%zmm4, %%zmm4       \n\t"
                         "vaddps 0x80(%[eltwise], %%rbx), %%zmm7, %%zmm7       \n\t"
                         "vaddps 0xC0(%[eltwise], %%rbx), %%zmm10, %%zmm10     \n\t"
                         "vaddps 0x100(%[eltwise], %%rbx), %%zmm13, %%zmm13    \n\t"
                         "vaddps 0x140(%[eltwise], %%rbx), %%zmm16, %%zmm16    \n\t"
                         "vaddps 0x180(%[eltwise], %%rbx), %%zmm19, %%zmm19    \n\t"
                         "vaddps 0x1C0(%[eltwise], %%rbx), %%zmm22, %%zmm22    \n\t"
                         "vaddps (%[eltwise], %%rbx, 2), %%zmm2, %%zmm2        \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx, 2), %%zmm5, %%zmm5    \n\t"
                         "vaddps 0x80(%[eltwise], %%rbx, 2), %%zmm8, %%zmm8    \n\t"
                         "vaddps 0xC0(%[eltwise], %%rbx, 2), %%zmm11, %%zmm11  \n\t"
                         "vaddps 0x100(%[eltwise], %%rbx, 2), %%zmm14, %%zmm14 \n\t"
                         "vaddps 0x140(%[eltwise], %%rbx, 2), %%zmm17, %%zmm17 \n\t"
                         "vaddps 0x180(%[eltwise], %%rbx, 2), %%zmm20, %%zmm20 \n\t"
                         "vaddps 0x1C0(%[eltwise], %%rbx, 2), %%zmm23, %%zmm23 \n\t"

                         ".align 16                                            \n\t"
                         "3:                                                   \n\t"
                         "movq %[flags], %%rcx                                 \n\t"
                         "and $0xC, %%rcx                                      \n\t"
                         "je 4f                                                \n\t"
                         relu24Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                                            \n\t"
                         "4:                                                   \n\t"
                         "vmovups %%zmm0, (%%rax)                              \n\t"
                         "vmovups %%zmm3, 0x40(%%rax)                          \n\t"
                         "vmovups %%zmm6, 0x80(%%rax)                          \n\t"
                         "vmovups %%zmm9, 0xC0(%%rax)                          \n\t"
                         "vmovups %%zmm12, 0x100(%%rax)                        \n\t"
                         "vmovups %%zmm15, 0x140(%%rax)                        \n\t"
                         "vmovups %%zmm18, 0x180(%%rax)                        \n\t"
                         "vmovups %%zmm21, 0x1C0(%%rax)                        \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                       \n\t"
                         "vmovups %%zmm4, 0x40(%%rax, %%rbx)                   \n\t"
                         "vmovups %%zmm7, 0x80(%%rax, %%rbx)                   \n\t"
                         "vmovups %%zmm10, 0xC0(%%rax, %%rbx)                  \n\t"
                         "vmovups %%zmm13, 0x100(%%rax, %%rbx)                 \n\t"
                         "vmovups %%zmm16, 0x140(%%rax, %%rbx)                 \n\t"
                         "vmovups %%zmm19, 0x180(%%rax, %%rbx)                 \n\t"
                         "vmovups %%zmm22, 0x1C0(%%rax, %%rbx)                 \n\t"
                         "vmovups %%zmm2, (%%rax, %%rbx, 2)                    \n\t"
                         "vmovups %%zmm5, 0x40(%%rax, %%rbx, 2)                \n\t"
                         "vmovups %%zmm8, 0x80(%%rax, %%rbx, 2)                \n\t"
                         "vmovups %%zmm11, 0xC0(%%rax, %%rbx, 2)               \n\t"
                         "vmovups %%zmm14, 0x100(%%rax, %%rbx, 2)              \n\t"
                         "vmovups %%zmm17, 0x140(%%rax, %%rbx, 2)              \n\t"
                         "vmovups %%zmm20, 0x180(%%rax, %%rbx, 2)              \n\t"
                         "vmovups %%zmm23, 0x1C0(%%rax, %%rbx, 2)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [eltwise] "r" (c.eltwise),
                           [ostepC16] "r" (c.ostepC16),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",
                           "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",
                           "%zmm30", "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel4x48(ConvController &c) {
    convKernelForLoopXx48(12, 4)

    __asm__ __volatile__("movq %[output], %%rax                               \n\t"
                         "movq %[ostepC16], %%rbx                             \n\t"
                         "movq %[flags], %%rcx                                \n\t"
                         "and $0x1, %%rcx                                     \n\t"
                         "je 0f                                               \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                      \n\t"
                         "vpaddd 0x40(%%rax), %%zmm3, %%zmm3                  \n\t"
                         "vpaddd 0x80(%%rax), %%zmm6, %%zmm6                  \n\t"
                         "vpaddd 0xC0(%%rax), %%zmm9, %%zmm9                  \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1               \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%zmm4, %%zmm4           \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%zmm7, %%zmm7           \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx), %%zmm10, %%zmm10         \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%zmm2, %%zmm2            \n\t"
                         "vpaddd 0x40(%%rax, %%rbx, 2), %%zmm5, %%zmm5        \n\t"
                         "vpaddd 0x80(%%rax, %%rbx, 2), %%zmm8, %%zmm8        \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx, 2), %%zmm11, %%zmm11      \n\t"

                         ".align 16                                           \n\t"
                         "0:                                                  \n\t"
                         "cmpq $0x0, %[scale]                                 \n\t"
                         "jne 1f                                              \n\t"
                         "movq %[flags], %%rcx                                \n\t"
                         "and $0xC, %%rcx                                     \n\t"
                         "je 4f                                               \n\t"
                         relu12Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                              \n\t"

                         ".align 16                                           \n\t"
                         "1:                                                  \n\t"
                         convert12RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                           \n\t"
                         "2:                                                  \n\t"
                         "movq %[flags], %%rcx                                \n\t"
                         "and $0x2, %%rcx                                     \n\t"
                         "je 3f                                               \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0                 \n\t"
                         "vaddps 0x40(%[eltwise]), %%zmm3, %%zmm3             \n\t"
                         "vaddps 0x80(%[eltwise]), %%zmm6, %%zmm6             \n\t"
                         "vaddps 0xC0(%[eltwise]), %%zmm9, %%zmm9             \n\t"
                         "vaddps (%[eltwise], %%rbx), %%zmm1, %%zmm1          \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx), %%zmm4, %%zmm4      \n\t"
                         "vaddps 0x80(%[eltwise], %%rbx), %%zmm7, %%zmm7      \n\t"
                         "vaddps 0xC0(%[eltwise], %%rbx), %%zmm10, %%zmm10    \n\t"
                         "vaddps (%[eltwise], %%rbx, 2), %%zmm2, %%zmm2       \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx, 2), %%zmm5, %%zmm5   \n\t"
                         "vaddps 0x80(%[eltwise], %%rbx, 2), %%zmm8, %%zmm8   \n\t"
                         "vaddps 0xC0(%[eltwise], %%rbx, 2), %%zmm11, %%zmm11 \n\t"

                         ".align 16                                           \n\t"
                         "3:                                                  \n\t"
                         "movq %[flags], %%rcx                                \n\t"
                         "and $0xC, %%rcx                                     \n\t"
                         "je 4f                                               \n\t"
                         relu12Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                                           \n\t"
                         "4:                                                  \n\t"
                         "vmovups %%zmm0, (%%rax)                             \n\t"
                         "vmovups %%zmm3, 0x40(%%rax)                         \n\t"
                         "vmovups %%zmm6, 0x80(%%rax)                         \n\t"
                         "vmovups %%zmm9, 0xC0(%%rax)                         \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                      \n\t"
                         "vmovups %%zmm4, 0x40(%%rax, %%rbx)                  \n\t"
                         "vmovups %%zmm7, 0x80(%%rax, %%rbx)                  \n\t"
                         "vmovups %%zmm10, 0xC0(%%rax, %%rbx)                 \n\t"
                         "vmovups %%zmm2, (%%rax, %%rbx, 2)                   \n\t"
                         "vmovups %%zmm5, 0x40(%%rax, %%rbx, 2)               \n\t"
                         "vmovups %%zmm8, 0x80(%%rax, %%rbx, 2)               \n\t"
                         "vmovups %%zmm11, 0xC0(%%rax, %%rbx, 2)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [eltwise] "r" (c.eltwise),
                           [ostepC16] "r" (c.ostepC16),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",
                           "%zmm24", "%zmm31", "memory", "cc");

}

void Avx512Conv1x1Kernel1x48(ConvController &c) {
    convKernelForLoopXx48(3, 1)

    __asm__ __volatile__("movq %[output], %%rax                         \n\t"
                         "movq %[ostepC16], %%rbx                       \n\t"
                         "movq %[flags], %%rcx                          \n\t"
                         "and $0x1, %%rcx                               \n\t"
                         "je 0f                                         \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1         \n\t"
                         "vpaddd (%%rax, %%rbx, 2), %%zmm2, %%zmm2      \n\t"

                         ".align 16                                     \n\t"
                         "0:                                            \n\t"
                         "cmpq $0x0, %[scale]                           \n\t"
                         "jne 1f                                        \n\t"
                         "movq %[flags], %%rcx                          \n\t"
                         "and $0xC, %%rcx                               \n\t"
                         "je 4f                                         \n\t"
                         relu3Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                        \n\t"

                         ".align 16                                     \n\t"
                         "1:                                            \n\t"
                         convert3RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                     \n\t"
                         "2:                                            \n\t"
                         "movq %[flags], %%rcx                          \n\t"
                         "and $0x2, %%rcx                               \n\t"
                         "je 3f                                         \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0           \n\t"
                         "vaddps (%[eltwise], %%rbx), %%zmm1, %%zmm1    \n\t"
                         "vaddps (%[eltwise], %%rbx, 2), %%zmm2, %%zmm2 \n\t"

                         ".align 16                                     \n\t"
                         "3:                                            \n\t"
                         "movq %[flags], %%rcx                          \n\t"
                         "and $0xC, %%rcx                               \n\t"
                         "je 4f                                         \n\t"
                         relu3Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                                     \n\t"
                         "4:                                            \n\t"
                         "vmovups %%zmm0, (%%rax)                       \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                \n\t"
                         "vmovups %%zmm2, (%%rax, %%rbx, 2)             \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%zmm0", "%zmm1", "%zmm2",
                           "%zmm24", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define convKernel12x32c4(input, freg0, freg1, off0, off1, preg0, preg1, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    "vpbroadcastd "#i0"("#input"), %%zmm28           \n\t" \
    "vpbroadcastd "#i1"("#input"), %%zmm29           \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1              \n\t" \
    "vpbroadcastd "#i2"("#input"), %%zmm30           \n\t" \
    "vpbroadcastd "#i3"("#input"), %%zmm31           \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2              \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3              \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5              \n\t" \
    "vpbroadcastd "#i4"("#input"), %%zmm28           \n\t" \
    "vpbroadcastd "#i5"("#input"), %%zmm29           \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7              \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9              \n\t" \
    "vpbroadcastd "#i6"("#input"), %%zmm30           \n\t" \
    "vpbroadcastd "#i7"("#input"), %%zmm31           \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10             \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11             \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"            \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12             \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13             \n\t" \
    "vpbroadcastd "#i8"("#input"), %%zmm28           \n\t" \
    "vpbroadcastd "#i9"("#input"), %%zmm29           \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14             \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15             \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16             \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17             \n\t" \
    "vpbroadcastd "#i10"("#input"), %%zmm30          \n\t" \
    "vpbroadcastd "#i11"("#input"), %%zmm31          \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18             \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19             \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm20             \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm21             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm22             \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm23             \n\t"

#define convKernel6x32c4(input, freg0, freg1, off0, off1, preg0, preg1, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    "vpbroadcastd "#i0"("#input"), %%zmm28           \n\t" \
    "vpbroadcastd "#i1"("#input"), %%zmm29           \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1              \n\t" \
    "vpbroadcastd "#i2"("#input"), %%zmm30           \n\t" \
    "vpbroadcastd "#i3"("#input"), %%zmm31           \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2              \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3              \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4              \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5              \n\t" \
    "vpbroadcastd "#i4"("#input"), %%zmm28           \n\t" \
    "vpbroadcastd "#i5"("#input"), %%zmm29           \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6              \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7              \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"            \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9              \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10             \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11             \n\t"

#define convKernel1x32c4(input, freg0, freg1, off0, off1, preg0, preg1, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    "vpbroadcastd ("#input"), %%zmm28                \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"            \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0              \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1              \n\t"
#else
#define convKernel12x32c4_3(input, freg0, freg1, off0, off1, preg0, preg1, preg2, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    "vpbroadcastd "#i0"("#input"), %%zmm29           \n\t" \
    "vpbroadcastd "#i1"("#input"), %%zmm30           \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg0"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg1"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg2"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i2"("#input"), %%zmm29           \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0                 \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1                 \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2                 \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg0"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg1"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg2"          \n\t" \
    "vpbroadcastd "#i3"("#input"), %%zmm30           \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i4"("#input"), %%zmm29           \n\t" \
    "vpaddd %%zmm3, "#preg0", %%zmm3                 \n\t" \
    "vpaddd %%zmm4, "#preg1", %%zmm4                 \n\t" \
    "vpaddd %%zmm5, "#preg2", %%zmm5                 \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg2"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i5"("#input"), %%zmm30           \n\t" \
    "vpaddd %%zmm6, "#preg0", %%zmm6                 \n\t" \
    "vpaddd %%zmm7, "#preg1", %%zmm7                 \n\t" \
    "vpaddd %%zmm8, "#preg2", %%zmm8                 \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg0"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg1"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg2"          \n\t" \
    "vpbroadcastd "#i6"("#input"), %%zmm29           \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i7"("#input"), %%zmm30           \n\t" \
    "vpaddd %%zmm9, "#preg0", %%zmm9                 \n\t" \
    "vpaddd %%zmm10, "#preg1", %%zmm10               \n\t" \
    "vpaddd %%zmm11, "#preg2", %%zmm11               \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg0"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg1"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg2"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i8"("#input"), %%zmm29           \n\t" \
    "vpaddd %%zmm12, "#preg0", %%zmm12               \n\t" \
    "vpaddd %%zmm13, "#preg1", %%zmm13               \n\t" \
    "vpaddd %%zmm14, "#preg2", %%zmm14               \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg0"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg1"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg2"          \n\t" \
    "vpbroadcastd "#i9"("#input"), %%zmm30           \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i10"("#input"), %%zmm29          \n\t" \
    "vpaddd %%zmm15, "#preg0", %%zmm15               \n\t" \
    "vpaddd %%zmm16, "#preg1", %%zmm16               \n\t" \
    "vpaddd %%zmm17, "#preg2", %%zmm17               \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg2"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i11"("#input"), %%zmm30          \n\t" \
    "vpaddd %%zmm18, "#preg0", %%zmm18               \n\t" \
    "vpaddd %%zmm19, "#preg1", %%zmm19               \n\t" \
    "vpaddd %%zmm20, "#preg2", %%zmm20               \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg0"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg1"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg2"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"            \n\t" \
    "vpaddd %%zmm21, "#preg0", %%zmm21               \n\t" \
    "vpaddd %%zmm22, "#preg1", %%zmm22               \n\t" \
    "vpaddd %%zmm23, "#preg2", %%zmm23               \n\t"

#define convKernel6x32c4_3(input, freg0, freg1, off0, off1, preg0, preg1, preg2, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    "vpbroadcastd "#i0"("#input"), %%zmm29           \n\t" \
    "vpbroadcastd "#i1"("#input"), %%zmm30           \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg0"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg1"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg2"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i2"("#input"), %%zmm29           \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0                 \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1                 \n\t" \
    "vpaddd %%zmm2, "#preg2", %%zmm2                 \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg0"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg1"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg2"          \n\t" \
    "vpbroadcastd "#i3"("#input"), %%zmm30           \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i4"("#input"), %%zmm29           \n\t" \
    "vpaddd %%zmm3, "#preg0", %%zmm3                 \n\t" \
    "vpaddd %%zmm4, "#preg1", %%zmm4                 \n\t" \
    "vpaddd %%zmm5, "#preg2", %%zmm5                 \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg0"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg1"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg2"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vpbroadcastd "#i5"("#input"), %%zmm30           \n\t" \
    "vpaddd %%zmm6, "#preg0", %%zmm6                 \n\t" \
    "vpaddd %%zmm7, "#preg1", %%zmm7                 \n\t" \
    "vpaddd %%zmm8, "#preg2", %%zmm8                 \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg0"          \n\t" \
    "vpmaddubsw "#freg0", %%zmm30, "#preg1"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm30, "#preg2"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"            \n\t" \
    "vpmaddwd "#preg2", %%zmm31, "#preg2"            \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"            \n\t" \
    "vpaddd %%zmm9, "#preg0", %%zmm9                 \n\t" \
    "vpaddd %%zmm10, "#preg1", %%zmm10               \n\t" \
    "vpaddd %%zmm11, "#preg2", %%zmm11               \n\t"

#define convKernel1x32c4_3(input, freg0, freg1, off0, off1, preg0, preg1, preg2, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    "vpbroadcastd ("#input"), %%zmm29                \n\t" \
    "vpmaddubsw "#freg0", %%zmm29, "#preg0"          \n\t" \
    "vpmaddubsw "#freg1", %%zmm29, "#preg1"          \n\t" \
    "vpmaddwd "#preg0", %%zmm31, "#preg0"            \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"            \n\t" \
    "vpmaddwd "#preg1", %%zmm31, "#preg1"            \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"            \n\t" \
    "vpaddd %%zmm0, "#preg0", %%zmm0                 \n\t" \
    "vpaddd %%zmm1, "#preg1", %%zmm1                 \n\t"

#define convKernel12x32c4(input, freg0, freg1, off0, off1, preg0, preg1, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    convKernel12x32c4_3(input, %%zmm24, %%zmm25, off0, off1, \
                        %%zmm26, %%zmm27, %%zmm28, \
                        i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11)

#define convKernel6x32c4(input, freg0, freg1, off0, off1, preg0, preg1, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    convKernel6x32c4_3(input, %%zmm24, %%zmm25, off0, off1, \
                       %%zmm26, %%zmm27, %%zmm28, \
                       i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11)

#define convKernel1x32c4(input, freg0, freg1, off0, off1, preg0, preg1, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) \
    convKernel1x32c4_3(input, %%zmm24, %%zmm25, off0, off1, \
                       %%zmm26, %%zmm27, %%zmm28, \
                       i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11)
#endif

#define convKernelForLoopXx32(rnum, wsize) \
     __asm__ __volatile__("vmovups (%[filter]), %%zmm24                                  \n\t" \
                          "vmovups 0x40(%[filter]), %%zmm25                              \n\t" \
                          "addq $0x80, %[filter]                                         \n\t" \
                          "mov $1, %%eax                                                 \n\t" \
                          "vmovd %%eax, %%xmm0                                           \n\t" \
                          "vpbroadcastw %%xmm0, %%zmm31                                  \n\t" \
                          "movq %[flags], %%rax                                          \n\t" \
                          "andq $0x1, %%rax                                              \n\t" \
                          "jne 0f                                                        \n\t" \
                          load2BiasToZmmRegs(rnum, %[bias])                                    \
                          "cmpq $0x10, %%rcx                                             \n\t" \
                          "jl 4f                                                         \n\t" \
                          "jmp 1f                                                        \n\t" \
                          ".align 16                                                     \n\t" \
                          "0:                                                            \n\t" \
                          clear##rnum##Regs(%%zmm)                                             \
                          "cmpq $0x10, %%rcx                                             \n\t" \
                          "jl 4f                                                         \n\t" \
                          ".align 16                                                     \n\t" \
                          "1:                                                            \n\t" \
                          "movq %[input], %%rax                                          \n\t" \
                          convKernel##wsize##x32c4(%%rax, %%zmm24, %%zmm25, 0x0, 0x40,         \
                                                   %%zmm26, %%zmm27,                           \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50,          \
                                                   0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0)         \
                          "addq $0x4, %%rax                                              \n\t" \
                          convKernel##wsize##x32c4(%%rax, %%zmm26, %%zmm27, 0x80, 0xC0,        \
                                                   %%zmm24, %%zmm25,                           \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50,          \
                                                   0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0)         \
                          "addq $0x4, %%rax                                              \n\t" \
                          convKernel##wsize##x32c4(%%rax, %%zmm24, %%zmm25, 0x100, 0x140,      \
                                                   %%zmm26, %%zmm27,                           \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50,          \
                                                   0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0)         \
                          "addq $0x4, %%rax                                              \n\t" \
                          convKernel##wsize##x32c4(%%rax, %%zmm26, %%zmm27, 0x180, 0x1C0,      \
                                                   %%zmm24, %%zmm25,                           \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50,          \
                                                   0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0)         \
                          "addq $0x200, %[filter]                                        \n\t" \
                          "addq %[fStep], %[input]                                       \n\t" \
                          "subq $0x10, %%rcx                                             \n\t" \
                          "cmpq $0x10, %%rcx                                             \n\t" \
                          "jge 1b                                                        \n\t" \
                          ".align 16                                                     \n\t" \
                          "4:                                                            \n\t" \
                          : "+c" (c.ic),                                                       \
                            [input] "+r" (c.input),                                            \
                            [filter] "+r" (c.filter)                                           \
                          : [bias] "r" (c.bias),                                               \
                            [dilateW] "r" (c.dilateW),                                         \
                            [dilateH] "r" (c.dilateH),                                         \
                            [fStep] "r" (c.fStep),                                             \
                            [flags] "r" (c.flags)                                              \
                          : "%rax", "%rbx", "%r9",                                             \
                            "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",              \
                            "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",            \
                            "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",        \
                            "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",        \
                            "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",        \
                            "%zmm30", "%zmm31", "memory", "cc");                               \

void Avx512Conv1x1Kernel12x32(ConvController &c) {
    convKernelForLoopXx32(24, 12)

    __asm__ __volatile__("movq %[output], %%rax                             \n\t"
                         "movq %[ostepC16], %%rbx                           \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0x1, %%rcx                                   \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                    \n\t"
                         "vpaddd 0x40(%%rax), %%zmm2, %%zmm2                \n\t"
                         "vpaddd 0x80(%%rax), %%zmm4, %%zmm4                \n\t"
                         "vpaddd 0xC0(%%rax), %%zmm6, %%zmm6                \n\t"
                         "vpaddd 0x100(%%rax), %%zmm8, %%zmm8               \n\t"
                         "vpaddd 0x140(%%rax), %%zmm10, %%zmm10             \n\t"
                         "vpaddd 0x180(%%rax), %%zmm12, %%zmm12             \n\t"
                         "vpaddd 0x1C0(%%rax), %%zmm14, %%zmm14             \n\t"
                         "vpaddd 0x200(%%rax), %%zmm16, %%zmm16             \n\t"
                         "vpaddd 0x240(%%rax), %%zmm18, %%zmm18             \n\t"
                         "vpaddd 0x280(%%rax), %%zmm20, %%zmm20             \n\t"
                         "vpaddd 0x2C0(%%rax), %%zmm22, %%zmm22             \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%zmm3, %%zmm3         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%zmm5, %%zmm5         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx), %%zmm7, %%zmm7         \n\t"
                         "vpaddd 0x100(%%rax, %%rbx), %%zmm9, %%zmm9        \n\t"
                         "vpaddd 0x140(%%rax, %%rbx), %%zmm11, %%zmm11      \n\t"
                         "vpaddd 0x180(%%rax, %%rbx), %%zmm13, %%zmm13      \n\t"
                         "vpaddd 0x1C0(%%rax, %%rbx), %%zmm15, %%zmm15      \n\t"
                         "vpaddd 0x200(%%rax, %%rbx), %%zmm17, %%zmm17      \n\t"
                         "vpaddd 0x240(%%rax, %%rbx), %%zmm19, %%zmm19      \n\t"
                         "vpaddd 0x280(%%rax, %%rbx), %%zmm21, %%zmm21      \n\t"
                         "vpaddd 0x2C0(%%rax, %%rbx), %%zmm23, %%zmm23      \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0xC, %%rcx                                   \n\t"
                         "je 4f                                             \n\t"
                         relu24Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                            \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         convert24RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0x2, %%rcx                                   \n\t"
                         "je 3f                                             \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0               \n\t"
                         "vaddps 0x40(%[eltwise]), %%zmm2, %%zmm2           \n\t"
                         "vaddps 0x80(%[eltwise]), %%zmm4, %%zmm4           \n\t"
                         "vaddps 0xC0(%[eltwise]), %%zmm6, %%zmm6           \n\t"
                         "vaddps 0x100(%[eltwise]), %%zmm8, %%zmm8          \n\t"
                         "vaddps 0x140(%[eltwise]), %%zmm10, %%zmm10        \n\t"
                         "vaddps 0x180(%[eltwise]), %%zmm12, %%zmm12        \n\t"
                         "vaddps 0x1C0(%[eltwise]), %%zmm14, %%zmm14        \n\t"
                         "vaddps 0x200(%[eltwise]), %%zmm16, %%zmm16        \n\t"
                         "vaddps 0x240(%[eltwise]), %%zmm18, %%zmm18        \n\t"
                         "vaddps 0x280(%[eltwise]), %%zmm20, %%zmm20        \n\t"
                         "vaddps 0x2C0(%[eltwise]), %%zmm22, %%zmm22        \n\t"
                         "vaddps (%[eltwise], %%rbx), %%zmm1, %%zmm1        \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx), %%zmm3, %%zmm3    \n\t"
                         "vaddps 0x80(%[eltwise], %%rbx), %%zmm5, %%zmm5    \n\t"
                         "vaddps 0xC0(%[eltwise], %%rbx), %%zmm7, %%zmm7    \n\t"
                         "vaddps 0x100(%[eltwise], %%rbx), %%zmm9, %%zmm9   \n\t"
                         "vaddps 0x140(%[eltwise], %%rbx), %%zmm11, %%zmm11 \n\t"
                         "vaddps 0x180(%[eltwise], %%rbx), %%zmm13, %%zmm13 \n\t"
                         "vaddps 0x1C0(%[eltwise], %%rbx), %%zmm15, %%zmm15 \n\t"
                         "vaddps 0x200(%[eltwise], %%rbx), %%zmm17, %%zmm17 \n\t"
                         "vaddps 0x240(%[eltwise], %%rbx), %%zmm19, %%zmm19 \n\t"
                         "vaddps 0x280(%[eltwise], %%rbx), %%zmm21, %%zmm21 \n\t"
                         "vaddps 0x2C0(%[eltwise], %%rbx), %%zmm23, %%zmm23 \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0xC, %%rcx                                   \n\t"
                         "je 4f                                             \n\t"
                         relu24Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                                         \n\t"
                         "4:                                                \n\t"
                         "vmovups %%zmm0, (%%rax)                           \n\t"
                         "vmovups %%zmm2, 0x40(%%rax)                       \n\t"
                         "vmovups %%zmm4, 0x80(%%rax)                       \n\t"
                         "vmovups %%zmm6, 0xC0(%%rax)                       \n\t"
                         "vmovups %%zmm8, 0x100(%%rax)                      \n\t"
                         "vmovups %%zmm10, 0x140(%%rax)                     \n\t"
                         "vmovups %%zmm12, 0x180(%%rax)                     \n\t"
                         "vmovups %%zmm14, 0x1C0(%%rax)                     \n\t"
                         "vmovups %%zmm16, 0x200(%%rax)                     \n\t"
                         "vmovups %%zmm18, 0x240(%%rax)                     \n\t"
                         "vmovups %%zmm20, 0x280(%%rax)                     \n\t"
                         "vmovups %%zmm22, 0x2C0(%%rax)                     \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                    \n\t"
                         "vmovups %%zmm3, 0x40(%%rax, %%rbx)                \n\t"
                         "vmovups %%zmm5, 0x80(%%rax, %%rbx)                \n\t"
                         "vmovups %%zmm7, 0xC0(%%rax, %%rbx)                \n\t"
                         "vmovups %%zmm9, 0x100(%%rax, %%rbx)               \n\t"
                         "vmovups %%zmm11, 0x140(%%rax, %%rbx)              \n\t"
                         "vmovups %%zmm13, 0x180(%%rax, %%rbx)              \n\t"
                         "vmovups %%zmm15, 0x1C0(%%rax, %%rbx)              \n\t"
                         "vmovups %%zmm17, 0x200(%%rax, %%rbx)              \n\t"
                         "vmovups %%zmm19, 0x240(%%rax, %%rbx)              \n\t"
                         "vmovups %%zmm21, 0x280(%%rax, %%rbx)              \n\t"
                         "vmovups %%zmm23, 0x2C0(%%rax, %%rbx)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx", 
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",
                           "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",
                           "%zmm30", "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel6x32(ConvController &c) {
    convKernelForLoopXx32(12, 6)

    __asm__ __volatile__("movq %[output], %%rax                             \n\t"
                         "movq %[ostepC16], %%rbx                           \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0x1, %%rcx                                   \n\t"
                         "je 0f                                             \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0                    \n\t"
                         "vpaddd 0x40(%%rax), %%zmm2, %%zmm2                \n\t"
                         "vpaddd 0x80(%%rax), %%zmm4, %%zmm4                \n\t"
                         "vpaddd 0xC0(%%rax), %%zmm6, %%zmm6                \n\t"
                         "vpaddd 0x100(%%rax), %%zmm8, %%zmm8               \n\t"
                         "vpaddd 0x140(%%rax), %%zmm10, %%zmm10             \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1             \n\t"
                         "vpaddd 0x40(%%rax, %%rbx), %%zmm3, %%zmm3         \n\t"
                         "vpaddd 0x80(%%rax, %%rbx), %%zmm5, %%zmm5         \n\t"
                         "vpaddd 0xC0(%%rax, %%rbx), %%zmm7, %%zmm7         \n\t"
                         "vpaddd 0x100(%%rax, %%rbx), %%zmm9, %%zmm9        \n\t"
                         "vpaddd 0x140(%%rax, %%rbx), %%zmm11, %%zmm11      \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0xC, %%rcx                                   \n\t"
                         "je 4f                                             \n\t"
                         relu12Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                            \n\t"

                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         convert12RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0x2, %%rcx                                   \n\t"
                         "je 3f                                             \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0               \n\t"
                         "vaddps 0x40(%[eltwise]), %%zmm2, %%zmm2           \n\t"
                         "vaddps 0x80(%[eltwise]), %%zmm4, %%zmm4           \n\t"
                         "vaddps 0xC0(%[eltwise]), %%zmm6, %%zmm6           \n\t"
                         "vaddps 0x100(%[eltwise]), %%zmm8, %%zmm8          \n\t"
                         "vaddps 0x140(%[eltwise]), %%zmm10, %%zmm10        \n\t"
                         "vaddps (%[eltwise], %%rbx), %%zmm1, %%zmm1        \n\t"
                         "vaddps 0x40(%[eltwise], %%rbx), %%zmm3, %%zmm3    \n\t"
                         "vaddps 0x80(%[eltwise], %%rbx), %%zmm5, %%zmm5    \n\t"
                         "vaddps 0xC0(%[eltwise], %%rbx), %%zmm7, %%zmm7    \n\t"
                         "vaddps 0x100(%[eltwise], %%rbx), %%zmm9, %%zmm9   \n\t"
                         "vaddps 0x140(%[eltwise], %%rbx), %%zmm11, %%zmm11 \n\t"

                         ".align 16                                         \n\t"
                         "3:                                                \n\t"
                         "movq %[flags], %%rcx                              \n\t"
                         "and $0xC, %%rcx                                   \n\t"
                         "je 4f                                             \n\t"
                         relu12Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                                         \n\t"
                         "4:                                                \n\t"
                         "vmovups %%zmm0, (%%rax)                           \n\t"
                         "vmovups %%zmm2, 0x40(%%rax)                       \n\t"
                         "vmovups %%zmm4, 0x80(%%rax)                       \n\t"
                         "vmovups %%zmm6, 0xC0(%%rax)                       \n\t"
                         "vmovups %%zmm8, 0x100(%%rax)                      \n\t"
                         "vmovups %%zmm10, 0x140(%%rax)                     \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)                    \n\t"
                         "vmovups %%zmm3, 0x40(%%rax, %%rbx)                \n\t"
                         "vmovups %%zmm5, 0x80(%%rax, %%rbx)                \n\t"
                         "vmovups %%zmm7, 0xC0(%%rax, %%rbx)                \n\t"
                         "vmovups %%zmm9, 0x100(%%rax, %%rbx)               \n\t"
                         "vmovups %%zmm11, 0x140(%%rax, %%rbx)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",
                           "%zmm24", "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel1x32(ConvController &c) {
    convKernelForLoopXx32(2, 1)

    __asm__ __volatile__("movq %[output], %%rax                      \n\t"
                         "movq %[ostepC16], %%rbx                    \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x1, %%rcx                            \n\t"
                         "je 0f                                      \n\t"
                         "vpaddd (%%rax), %%zmm0, %%zmm0             \n\t"
                         "vpaddd (%%rax, %%rbx), %%zmm1, %%zmm1      \n\t"

                         ".align 16                                  \n\t"
                         "0:                                         \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu2Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                     \n\t"

                         ".align 16                                  \n\t"
                         "1:                                         \n\t"
                         convert2RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                  \n\t"
                         "2:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x2, %%rcx                            \n\t"
                         "je 3f                                      \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0        \n\t"
                         "vaddps (%[eltwise], %%rbx), %%zmm1, %%zmm1 \n\t"

                         ".align 16                                  \n\t"
                         "3:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu2Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                                  \n\t"
                         "4:                                         \n\t"
                         "vmovups %%zmm0, (%%rax)                    \n\t"
                         "vmovups %%zmm1, (%%rax, %%rbx)             \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%zmm0", "%zmm1", "%zmm24", "%zmm31", "memory", "cc");
}

#ifdef _USE_AVX512_VNNI
#define convKernel24x16c4(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    "vpbroadcastd "#i0"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i1"("#input"), "#rtype"27                       \n\t" \
    "vpbroadcastd "#i2"("#input"), "#rtype"28                       \n\t" \
    "vpbroadcastd "#i3"("#input"), "#rtype"29                       \n\t" \
    "vpbroadcastd "#i4"("#input"), "#rtype"30                       \n\t" \
    "vpbroadcastd "#i5"("#input"), "#rtype"31                       \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                           \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"5                       \n\t" \
    "vpbroadcastd "#i6"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i7"("#input"), "#rtype"27                       \n\t" \
    "vpbroadcastd "#i8"("#input"), "#rtype"28                       \n\t" \
    "vpbroadcastd "#i9"("#input"), "#rtype"29                       \n\t" \
    "vpbroadcastd "#i10"("#input"), "#rtype"30                      \n\t" \
    "vpbroadcastd "#i11"("#input"), "#rtype"31                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"8                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"11                      \n\t" \
    "vpbroadcastd "#i12"("#input"), "#rtype"26                      \n\t" \
    "vpbroadcastd "#i13"("#input"), "#rtype"27                      \n\t" \
    "vpbroadcastd "#i14"("#input"), "#rtype"28                      \n\t" \
    "vpbroadcastd "#i15"("#input"), "#rtype"29                      \n\t" \
    "vpbroadcastd "#i16"("#input"), "#rtype"30                      \n\t" \
    "vpbroadcastd "#i17"("#input"), "#rtype"31                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"13                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"14                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"16                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"17                      \n\t" \
    "vpbroadcastd "#i18"("#input"), "#rtype"26                      \n\t" \
    "vpbroadcastd "#i19"("#input"), "#rtype"27                      \n\t" \
    "vpbroadcastd "#i20"("#input"), "#rtype"28                      \n\t" \
    "vpbroadcastd "#i21"("#input"), "#rtype"29                      \n\t" \
    "vpbroadcastd "#i22"("#input"), "#rtype"30                      \n\t" \
    "vpbroadcastd "#i23"("#input"), "#rtype"31                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"19                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"20                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"21                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"22                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"23                      \n\t"

#define convKernel12x16c4(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    "vpbroadcastd "#i0"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i1"("#input"), "#rtype"27                       \n\t" \
    "vpbroadcastd "#i2"("#input"), "#rtype"28                       \n\t" \
    "vpbroadcastd "#i3"("#input"), "#rtype"29                       \n\t" \
    "vpbroadcastd "#i4"("#input"), "#rtype"30                       \n\t" \
    "vpbroadcastd "#i5"("#input"), "#rtype"31                       \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                           \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"5                       \n\t" \
    "vpbroadcastd "#i6"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i7"("#input"), "#rtype"27                       \n\t" \
    "vpbroadcastd "#i8"("#input"), "#rtype"28                       \n\t" \
    "vpbroadcastd "#i9"("#input"), "#rtype"29                       \n\t" \
    "vpbroadcastd "#i10"("#input"), "#rtype"30                      \n\t" \
    "vpbroadcastd "#i11"("#input"), "#rtype"31                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"8                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9                       \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10                      \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"11                      \n\t"

#define convKernel1x16c4(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    "vpbroadcastd ("#input"), "#rtype"26                            \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"                           \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0                       \n\t"
#else

#define convKernel24x16c4_3(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    "vpbroadcastd "#i0"("#input"), "#rtype"25                       \n\t" \
    "vpbroadcastd "#i1"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i2"("#input"), "#rtype"27                       \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i3"("#input"), "#rtype"25                       \n\t" \
    "vpbroadcastd "#i4"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i5"("#input"), "#rtype"27                       \n\t" \
    "vpaddd "#rtype"0, "#rtype"28, "#rtype"0                        \n\t" \
    "vpaddd "#rtype"1, "#rtype"29, "#rtype"1                        \n\t" \
    "vpaddd "#rtype"2, "#rtype"30, "#rtype"2                        \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i6"("#input"), "#rtype"25                       \n\t" \
    "vpbroadcastd "#i7"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i8"("#input"), "#rtype"27                       \n\t" \
    "vpaddd "#rtype"3, "#rtype"28, "#rtype"3                        \n\t" \
    "vpaddd "#rtype"4, "#rtype"29, "#rtype"4                        \n\t" \
    "vpaddd "#rtype"5, "#rtype"30, "#rtype"5                        \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i9"("#input"), "#rtype"25                       \n\t" \
    "vpbroadcastd "#i10"("#input"), "#rtype"26                      \n\t" \
    "vpbroadcastd "#i11"("#input"), "#rtype"27                      \n\t" \
    "vpaddd "#rtype"6, "#rtype"28, "#rtype"6                        \n\t" \
    "vpaddd "#rtype"7, "#rtype"29, "#rtype"7                        \n\t" \
    "vpaddd "#rtype"8, "#rtype"30, "#rtype"8                        \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i12"("#input"), "#rtype"25                      \n\t" \
    "vpbroadcastd "#i13"("#input"), "#rtype"26                      \n\t" \
    "vpbroadcastd "#i14"("#input"), "#rtype"27                      \n\t" \
    "vpaddd "#rtype"9, "#rtype"28, "#rtype"9                        \n\t" \
    "vpaddd "#rtype"10, "#rtype"29, "#rtype"10                      \n\t" \
    "vpaddd "#rtype"11, "#rtype"30, "#rtype"11                      \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i15"("#input"), "#rtype"25                      \n\t" \
    "vpbroadcastd "#i16"("#input"), "#rtype"26                      \n\t" \
    "vpbroadcastd "#i17"("#input"), "#rtype"27                      \n\t" \
    "vpaddd "#rtype"12, "#rtype"28, "#rtype"12                      \n\t" \
    "vpaddd "#rtype"13, "#rtype"29, "#rtype"13                      \n\t" \
    "vpaddd "#rtype"14, "#rtype"30, "#rtype"14                      \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i18"("#input"), "#rtype"25                      \n\t" \
    "vpbroadcastd "#i19"("#input"), "#rtype"26                      \n\t" \
    "vpbroadcastd "#i20"("#input"), "#rtype"27                      \n\t" \
    "vpaddd "#rtype"15, "#rtype"28, "#rtype"15                      \n\t" \
    "vpaddd "#rtype"16, "#rtype"29, "#rtype"16                      \n\t" \
    "vpaddd "#rtype"17, "#rtype"30, "#rtype"17                      \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i21"("#input"), "#rtype"25                      \n\t" \
    "vpbroadcastd "#i22"("#input"), "#rtype"26                      \n\t" \
    "vpbroadcastd "#i23"("#input"), "#rtype"27                      \n\t" \
    "vpaddd "#rtype"18, "#rtype"28, "#rtype"18                      \n\t" \
    "vpaddd "#rtype"19, "#rtype"29, "#rtype"19                      \n\t" \
    "vpaddd "#rtype"20, "#rtype"30, "#rtype"20                      \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                           \n\t" \
    "vpaddd "#rtype"21, "#rtype"28, "#rtype"21                      \n\t" \
    "vpaddd "#rtype"22, "#rtype"29, "#rtype"22                      \n\t" \
    "vpaddd "#rtype"23, "#rtype"30, "#rtype"23                      \n\t"

#define convKernel12x16c4_3(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    "vpbroadcastd "#i0"("#input"), "#rtype"25                       \n\t" \
    "vpbroadcastd "#i1"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i2"("#input"), "#rtype"27                       \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i3"("#input"), "#rtype"25                       \n\t" \
    "vpbroadcastd "#i4"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i5"("#input"), "#rtype"27                       \n\t" \
    "vpaddd "#rtype"0, "#rtype"28, "#rtype"0                        \n\t" \
    "vpaddd "#rtype"1, "#rtype"29, "#rtype"1                        \n\t" \
    "vpaddd "#rtype"2, "#rtype"30, "#rtype"2                        \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i6"("#input"), "#rtype"25                       \n\t" \
    "vpbroadcastd "#i7"("#input"), "#rtype"26                       \n\t" \
    "vpbroadcastd "#i8"("#input"), "#rtype"27                       \n\t" \
    "vpaddd "#rtype"3, "#rtype"28, "#rtype"3                        \n\t" \
    "vpaddd "#rtype"4, "#rtype"29, "#rtype"4                        \n\t" \
    "vpaddd "#rtype"5, "#rtype"30, "#rtype"5                        \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vpbroadcastd "#i9"("#input"), "#rtype"25                       \n\t" \
    "vpbroadcastd "#i10"("#input"), "#rtype"26                      \n\t" \
    "vpbroadcastd "#i11"("#input"), "#rtype"27                      \n\t" \
    "vpaddd "#rtype"6, "#rtype"28, "#rtype"6                        \n\t" \
    "vpaddd "#rtype"7, "#rtype"29, "#rtype"7                        \n\t" \
    "vpaddd "#rtype"8, "#rtype"30, "#rtype"8                        \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"26, "#rtype"29                    \n\t" \
    "vpmaddubsw "#freg0", "#rtype"27, "#rtype"30                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"29, "#rtype"31, "#rtype"29                    \n\t" \
    "vpmaddwd "#rtype"30, "#rtype"31, "#rtype"30                    \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                           \n\t" \
    "vpaddd "#rtype"9, "#rtype"28, "#rtype"9                        \n\t" \
    "vpaddd "#rtype"10, "#rtype"29, "#rtype"10                      \n\t" \
    "vpaddd "#rtype"11, "#rtype"30, "#rtype"11                      \n\t"

#define convKernel1x16c4_3(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    "vpbroadcastd ("#input"), "#rtype"25                            \n\t" \
    "vpmaddubsw "#freg0", "#rtype"25, "#rtype"28                    \n\t" \
    "vpmaddwd "#rtype"28, "#rtype"31, "#rtype"28                    \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"                           \n\t" \
    "vpaddd "#rtype"0, "#rtype"28, "#rtype"0                        \n\t"

#define convKernel24x16c4(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    convKernel24x16c4_3(input, rtype##24, off0, rtype##25, rtype, \
                        i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                        i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23)

#define convKernel12x16c4(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    convKernel12x16c4_3(input, rtype##24, off0, rtype##25, rtype, \
                        i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                        i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23)

#define convKernel1x16c4(input, freg0, off0, preg0, rtype, \
                          i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                          i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23) \
    convKernel1x16c4_3(input, rtype##24, off0, rtype##25, rtype, \
                       i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, \
                       i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23)
#endif

#define convKernelForLoopXx16(rnum, wsize, rtype, off0, off1, off2, off3, off4) \
     __asm__ __volatile__("vmovups (%[filter]), "#rtype"24                                     \n\t" \
                          "addq $"#off1", %[filter]                                            \n\t" \
                          "mov $1, %%eax                                                       \n\t" \
                          "vmovd %%eax, %%xmm0                                                 \n\t" \
                          "vpbroadcastw %%xmm0, "#rtype"31                                     \n\t" \
                          "movq %[flags], %%rax                                                \n\t" \
                          "andq $0x1, %%rax                                                    \n\t" \
                          "jne 0f                                                              \n\t" \
                          load1BiasToRegs(rnum, %[bias], rtype)                                      \
                          "cmpq $0x10, %%rcx                                                   \n\t" \
                          "jl 4f                                                               \n\t" \
                          "jmp 1f                                                              \n\t" \
                          ".align 16                                                           \n\t" \
                          "0:                                                                  \n\t" \
                          clear##rnum##Regs(rtype)                                                   \
                          "cmpq $0x10, %%rcx                                                   \n\t" \
                          "jl 4f                                                               \n\t" \
                          ".align 16                                                           \n\t" \
                          "1:                                                                  \n\t" \
                          "movq %[input], %%rax                                                \n\t" \
                          convKernel##wsize##x16c4(%%rax, rtype##24, off0, rtype##25, rtype,         \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50,                \
                                                   0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0,               \
                                                   0xC0, 0xD0, 0xE0, 0xF0, 0x100, 0x110,             \
                                                   0x120, 0x130, 0x140, 0x150, 0x160, 0x170)         \
                          "addq $0x4, %%rax                                                    \n\t" \
                          convKernel##wsize##x16c4(%%rax, rtype##25, off1, rtype##24, rtype,         \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50,                \
                                                   0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0,               \
                                                   0xC0, 0xD0, 0xE0, 0xF0, 0x100, 0x110,             \
                                                   0x120, 0x130, 0x140, 0x150, 0x160, 0x170)         \
                          "addq $0x4, %%rax                                                    \n\t" \
                          convKernel##wsize##x16c4(%%rax, rtype##24, off2, rtype##25, rtype,         \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50,                \
                                                   0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0,               \
                                                   0xC0, 0xD0, 0xE0, 0xF0, 0x100, 0x110,             \
                                                   0x120, 0x130, 0x140, 0x150, 0x160, 0x170)         \
                          "addq $0x4, %%rax                                                    \n\t" \
                          convKernel##wsize##x16c4(%%rax, rtype##25, off3, rtype##24, rtype,         \
                                                   0x0, 0x10, 0x20, 0x30, 0x40, 0x50,                \
                                                   0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0,               \
                                                   0xC0, 0xD0, 0xE0, 0xF0, 0x100, 0x110,             \
                                                   0x120, 0x130, 0x140, 0x150, 0x160, 0x170)         \
                          "addq $"#off4", %[filter]                                            \n\t" \
                          "addq %[fStep], %[input]                                             \n\t" \
                          "subq $0x10, %%rcx                                                   \n\t" \
                          "cmpq $0x10, %%rcx                                                   \n\t" \
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
                          : "%rax", "%rbx", "%r9",                                                   \
                            "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",                    \
                            "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",                  \
                            "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",              \
                            "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",              \
                            "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",              \
                            "%zmm30", "%zmm31", "memory", "cc");                                     \

void Avx512Conv1x1Kernel24x16(ConvController &c) {
    convKernelForLoopXx16(24, 24, %%zmm, 0x0, 0x40, 0x80, 0xC0, 0x100)

    __asm__ __volatile__("movq %[output], %%rax                      \n\t"
                         "movq %[ostepC16], %%rbx                    \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x1, %%rcx                            \n\t"
                         "je 0f                                      \n\t"
                         "vpaddd (%%rax),      %%zmm0, %%zmm0        \n\t"
                         "vpaddd 0x40(%%rax),  %%zmm1, %%zmm1        \n\t"
                         "vpaddd 0x80(%%rax),  %%zmm2, %%zmm2        \n\t"
                         "vpaddd 0xC0(%%rax),  %%zmm3, %%zmm3        \n\t"
                         "vpaddd 0x100(%%rax), %%zmm4, %%zmm4        \n\t"
                         "vpaddd 0x140(%%rax), %%zmm5, %%zmm5        \n\t"
                         "vpaddd 0x180(%%rax), %%zmm6, %%zmm6        \n\t"
                         "vpaddd 0x1C0(%%rax), %%zmm7, %%zmm7        \n\t"
                         "vpaddd 0x200(%%rax), %%zmm8, %%zmm8        \n\t"
                         "vpaddd 0x240(%%rax), %%zmm9, %%zmm9        \n\t"
                         "vpaddd 0x280(%%rax), %%zmm10, %%zmm10      \n\t"
                         "vpaddd 0x2C0(%%rax), %%zmm11, %%zmm11      \n\t"
                         "vpaddd 0x300(%%rax), %%zmm12, %%zmm12      \n\t"
                         "vpaddd 0x340(%%rax), %%zmm13, %%zmm13      \n\t"
                         "vpaddd 0x380(%%rax), %%zmm14, %%zmm14      \n\t"
                         "vpaddd 0x3C0(%%rax), %%zmm15, %%zmm15      \n\t"
                         "vpaddd 0x400(%%rax), %%zmm16, %%zmm16      \n\t"
                         "vpaddd 0x440(%%rax), %%zmm17, %%zmm17      \n\t"
                         "vpaddd 0x480(%%rax), %%zmm18, %%zmm18      \n\t"
                         "vpaddd 0x4C0(%%rax), %%zmm19, %%zmm19      \n\t"
                         "vpaddd 0x500(%%rax), %%zmm20, %%zmm20      \n\t"
                         "vpaddd 0x540(%%rax), %%zmm21, %%zmm21      \n\t"
                         "vpaddd 0x580(%%rax), %%zmm22, %%zmm22      \n\t"
                         "vpaddd 0x5C0(%%rax), %%zmm23, %%zmm23      \n\t"

                         ".align 16                                  \n\t"
                         "0:                                         \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu24Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                     \n\t"

                         ".align 16                                  \n\t"
                         "1:                                         \n\t"
                         convert24RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                  \n\t"
                         "2:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x2, %%rcx                            \n\t"
                         "je 3f                                      \n\t"
                         "vaddps (%[eltwise]),      %%zmm0, %%zmm0   \n\t"
                         "vaddps 0x40(%[eltwise]),  %%zmm1, %%zmm1   \n\t"
                         "vaddps 0x80(%[eltwise]),  %%zmm2, %%zmm2   \n\t"
                         "vaddps 0xC0(%[eltwise]),  %%zmm3, %%zmm3   \n\t"
                         "vaddps 0x100(%[eltwise]), %%zmm4, %%zmm4   \n\t"
                         "vaddps 0x140(%[eltwise]), %%zmm5, %%zmm5   \n\t"
                         "vaddps 0x180(%[eltwise]), %%zmm6, %%zmm6   \n\t"
                         "vaddps 0x1C0(%[eltwise]), %%zmm7, %%zmm7   \n\t"
                         "vaddps 0x200(%[eltwise]), %%zmm8, %%zmm8   \n\t"
                         "vaddps 0x240(%[eltwise]), %%zmm9, %%zmm9   \n\t"
                         "vaddps 0x280(%[eltwise]), %%zmm10, %%zmm10 \n\t"
                         "vaddps 0x2C0(%[eltwise]), %%zmm11, %%zmm11 \n\t"
                         "vaddps 0x300(%[eltwise]), %%zmm12, %%zmm12 \n\t"
                         "vaddps 0x340(%[eltwise]), %%zmm13, %%zmm13 \n\t"
                         "vaddps 0x380(%[eltwise]), %%zmm14, %%zmm14 \n\t"
                         "vaddps 0x3C0(%[eltwise]), %%zmm15, %%zmm15 \n\t"
                         "vaddps 0x400(%[eltwise]), %%zmm16, %%zmm16 \n\t"
                         "vaddps 0x440(%[eltwise]), %%zmm17, %%zmm17 \n\t"
                         "vaddps 0x480(%[eltwise]), %%zmm18, %%zmm18 \n\t"
                         "vaddps 0x4C0(%[eltwise]), %%zmm19, %%zmm19 \n\t"
                         "vaddps 0x500(%[eltwise]), %%zmm20, %%zmm20 \n\t"
                         "vaddps 0x540(%[eltwise]), %%zmm21, %%zmm21 \n\t"
                         "vaddps 0x580(%[eltwise]), %%zmm22, %%zmm22 \n\t"
                         "vaddps 0x5C0(%[eltwise]), %%zmm23, %%zmm23 \n\t"

                         ".align 16                                  \n\t"
                         "3:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu24Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                                  \n\t"
                         "4:                                         \n\t"
                         "vmovups %%zmm0,  (%%rax)                   \n\t"
                         "vmovups %%zmm1,  0x40(%%rax)               \n\t"
                         "vmovups %%zmm2,  0x80(%%rax)               \n\t"
                         "vmovups %%zmm3,  0xC0(%%rax)               \n\t"
                         "vmovups %%zmm4,  0x100(%%rax)              \n\t"
                         "vmovups %%zmm5,  0x140(%%rax)              \n\t"
                         "vmovups %%zmm6,  0x180(%%rax)              \n\t"
                         "vmovups %%zmm7,  0x1C0(%%rax)              \n\t"
                         "vmovups %%zmm8,  0x200(%%rax)              \n\t"
                         "vmovups %%zmm9,  0x240(%%rax)              \n\t"
                         "vmovups %%zmm10, 0x280(%%rax)              \n\t"
                         "vmovups %%zmm11, 0x2C0(%%rax)              \n\t"
                         "vmovups %%zmm12, 0x300(%%rax)              \n\t"
                         "vmovups %%zmm13, 0x340(%%rax)              \n\t"
                         "vmovups %%zmm14, 0x380(%%rax)              \n\t"
                         "vmovups %%zmm15, 0x3C0(%%rax)              \n\t"
                         "vmovups %%zmm16, 0x400(%%rax)              \n\t"
                         "vmovups %%zmm17, 0x440(%%rax)              \n\t"
                         "vmovups %%zmm18, 0x480(%%rax)              \n\t"
                         "vmovups %%zmm19, 0x4C0(%%rax)              \n\t"
                         "vmovups %%zmm20, 0x500(%%rax)              \n\t"
                         "vmovups %%zmm21, 0x540(%%rax)              \n\t"
                         "vmovups %%zmm22, 0x580(%%rax)              \n\t"
                         "vmovups %%zmm23, 0x5C0(%%rax)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",
                           "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",
                           "%zmm30","%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel12x16(ConvController &c) {
     convKernelForLoopXx16(12, 12, %%zmm, 0x0, 0x40, 0x80, 0xC0, 0x100)

    __asm__ __volatile__("movq %[output], %%rax                      \n\t"
                         "movq %[ostepC16], %%rbx                    \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x1, %%rcx                            \n\t"
                         "je 0f                                      \n\t"
                         "vpaddd (%%rax),      %%zmm0, %%zmm0        \n\t"
                         "vpaddd 0x40(%%rax),  %%zmm1, %%zmm1        \n\t"
                         "vpaddd 0x80(%%rax),  %%zmm2, %%zmm2        \n\t"
                         "vpaddd 0xC0(%%rax),  %%zmm3, %%zmm3        \n\t"
                         "vpaddd 0x100(%%rax), %%zmm4, %%zmm4        \n\t"
                         "vpaddd 0x140(%%rax), %%zmm5, %%zmm5        \n\t"
                         "vpaddd 0x180(%%rax), %%zmm6, %%zmm6        \n\t"
                         "vpaddd 0x1C0(%%rax), %%zmm7, %%zmm7        \n\t"
                         "vpaddd 0x200(%%rax), %%zmm8, %%zmm8        \n\t"
                         "vpaddd 0x240(%%rax), %%zmm9, %%zmm9        \n\t"
                         "vpaddd 0x280(%%rax), %%zmm10, %%zmm10      \n\t"
                         "vpaddd 0x2C0(%%rax), %%zmm11, %%zmm11      \n\t"

                         ".align 16                                  \n\t"
                         "0:                                         \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu12Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                     \n\t"

                         ".align 16                                  \n\t"
                         "1:                                         \n\t"
                         convert12RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                                  \n\t"
                         "2:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x2, %%rcx                            \n\t"
                         "je 3f                                      \n\t"
                         "vaddps (%[eltwise]),      %%zmm0, %%zmm0   \n\t"
                         "vaddps 0x40(%[eltwise]),  %%zmm1, %%zmm1   \n\t"
                         "vaddps 0x80(%[eltwise]),  %%zmm2, %%zmm2   \n\t"
                         "vaddps 0xC0(%[eltwise]),  %%zmm3, %%zmm3   \n\t"
                         "vaddps 0x100(%[eltwise]), %%zmm4, %%zmm4   \n\t"
                         "vaddps 0x140(%[eltwise]), %%zmm5, %%zmm5   \n\t"
                         "vaddps 0x180(%[eltwise]), %%zmm6, %%zmm6   \n\t"
                         "vaddps 0x1C0(%[eltwise]), %%zmm7, %%zmm7   \n\t"
                         "vaddps 0x200(%[eltwise]), %%zmm8, %%zmm8   \n\t"
                         "vaddps 0x240(%[eltwise]), %%zmm9, %%zmm9   \n\t"
                         "vaddps 0x280(%[eltwise]), %%zmm10, %%zmm10 \n\t"
                         "vaddps 0x2C0(%[eltwise]), %%zmm11, %%zmm11 \n\t"

                         ".align 16                                  \n\t"
                         "3:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu12Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                                  \n\t"
                         "4:                                         \n\t"
                         "vmovups %%zmm0,  (%%rax)                   \n\t"
                         "vmovups %%zmm1,  0x40(%%rax)               \n\t"
                         "vmovups %%zmm2,  0x80(%%rax)               \n\t"
                         "vmovups %%zmm3,  0xC0(%%rax)               \n\t"
                         "vmovups %%zmm4,  0x100(%%rax)              \n\t"
                         "vmovups %%zmm5,  0x140(%%rax)              \n\t"
                         "vmovups %%zmm6,  0x180(%%rax)              \n\t"
                         "vmovups %%zmm7,  0x1C0(%%rax)              \n\t"
                         "vmovups %%zmm8,  0x200(%%rax)              \n\t"
                         "vmovups %%zmm9,  0x240(%%rax)              \n\t"
                         "vmovups %%zmm10, 0x280(%%rax)              \n\t"
                         "vmovups %%zmm11, 0x2C0(%%rax)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",
                           "%zmm6","%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",
                           "%zmm24", "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel1x16(ConvController &c) {
    convKernelForLoopXx16(1, 1, %%zmm, 0x0, 0x40, 0x80, 0xC0, 0x100)

    __asm__ __volatile__("movq %[output], %%rax                 \n\t"
                         "movq %[flags], %%rcx                  \n\t"
                         "and $0x1, %%rcx                       \n\t"
                         "je 0f                                 \n\t"
                         "vpaddd (%%rax),      %%zmm0, %%zmm0   \n\t"
                         "vpaddd 0x40(%%rax),  %%zmm1, %%zmm1   \n\t"

                         ".align 16                             \n\t"
                         "0:                                    \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                  \n\t"
                         "and $0xC, %%rcx                       \n\t"
                         "je 4f                                 \n\t"
                         relu1Regs(%%zmm, vpxord, vpmaxsd)
                         "jmp 4f                                \n\t"

                         ".align 16                             \n\t"
                         "1:                                    \n\t"
                         convert1RegsI32ToF32(%[scale], %%zmm)

                         ".align 16                             \n\t"
                         "2:                                    \n\t"
                         "movq %[flags], %%rcx                  \n\t"
                         "and $0x2, %%rcx                       \n\t"
                         "je 3f                                 \n\t"
                         "vaddps (%[eltwise]), %%zmm0, %%zmm0   \n\t"

                         ".align 16                             \n\t"
                         "3:                                    \n\t"
                         "movq %[flags], %%rcx                  \n\t"
                         "and $0xC, %%rcx                       \n\t"
                         "je 4f                                 \n\t"
                         relu1Regs(%%zmm, vxorps, vmaxps)

                         ".align 16                             \n\t"
                         "4:                                    \n\t"
                         "vmovups %%zmm0,  (%%rax)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%zmm0","%zmm24", "%zmm31", "memory", "cc");
}

void Avx512Conv1x1Kernel24x8(ConvController &c) {
    convKernelForLoopXx16(24, 24, %%ymm, 0x0, 0x20, 0x40, 0x60, 0x80)

    __asm__ __volatile__("movq %[output], %%rax                      \n\t"
                         "movq %[ostepC16], %%rbx                    \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x1, %%rcx                            \n\t"
                         "je 0f                                      \n\t"
                         "vpaddd (%%rax),      %%ymm0,  %%ymm0       \n\t"
                         "vpaddd 0x20(%%rax),  %%ymm1,  %%ymm1       \n\t"
                         "vpaddd 0x40(%%rax),  %%ymm2,  %%ymm2       \n\t"
                         "vpaddd 0x60(%%rax),  %%ymm3,  %%ymm3       \n\t"
                         "vpaddd 0x80(%%rax),  %%ymm4,  %%ymm4       \n\t"
                         "vpaddd 0xA0(%%rax),  %%ymm5,  %%ymm5       \n\t"
                         "vpaddd 0xC0(%%rax),  %%ymm6,  %%ymm6       \n\t"
                         "vpaddd 0xE0(%%rax),  %%ymm7,  %%ymm7       \n\t"
                         "vpaddd 0x100(%%rax), %%ymm8,  %%ymm8       \n\t"
                         "vpaddd 0x120(%%rax), %%ymm9,  %%ymm9       \n\t"
                         "vpaddd 0x140(%%rax), %%ymm10, %%ymm10      \n\t"
                         "vpaddd 0x160(%%rax), %%ymm11, %%ymm11      \n\t"
                         "vpaddd 0x180(%%rax), %%ymm12, %%ymm12      \n\t"
                         "vpaddd 0x1A0(%%rax), %%ymm13, %%ymm13      \n\t"
                         "vpaddd 0x1C0(%%rax), %%ymm14, %%ymm14      \n\t"
                         "vpaddd 0x1E0(%%rax), %%ymm15, %%ymm15      \n\t"
                         "vpaddd 0x200(%%rax), %%ymm16, %%ymm16      \n\t"
                         "vpaddd 0x220(%%rax), %%ymm17, %%ymm17      \n\t"
                         "vpaddd 0x240(%%rax), %%ymm18, %%ymm18      \n\t"
                         "vpaddd 0x260(%%rax), %%ymm19, %%ymm19      \n\t"
                         "vpaddd 0x280(%%rax), %%ymm20, %%ymm20      \n\t"
                         "vpaddd 0x2A0(%%rax), %%ymm21, %%ymm21      \n\t"
                         "vpaddd 0x2C0(%%rax), %%ymm22, %%ymm22      \n\t"
                         "vpaddd 0x2E0(%%rax), %%ymm23, %%ymm23      \n\t"

                         ".align 16                                  \n\t"
                         "0:                                         \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu24Regs(%%ymm, vpxord, vpmaxsd)
                         "jmp 4f                                     \n\t"

                         ".align 16                                  \n\t"
                         "1:                                         \n\t"
                         convert24RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                  \n\t"
                         "2:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x2, %%rcx                            \n\t"
                         "je 3f                                      \n\t"
                         "vaddps (%[eltwise]),      %%ymm0, %%ymm0   \n\t"
                         "vaddps 0x20(%[eltwise]),  %%ymm1, %%ymm1   \n\t"
                         "vaddps 0x40(%[eltwise]),  %%ymm2, %%ymm2   \n\t"
                         "vaddps 0x60(%[eltwise]),  %%ymm3, %%ymm3   \n\t"
                         "vaddps 0x80(%[eltwise]),  %%ymm4, %%ymm4   \n\t"
                         "vaddps 0xA0(%[eltwise]),  %%ymm5, %%ymm5   \n\t"
                         "vaddps 0xC0(%[eltwise]),  %%ymm6, %%ymm6   \n\t"
                         "vaddps 0xE0(%[eltwise]),  %%ymm7, %%ymm7   \n\t"
                         "vaddps 0x100(%[eltwise]), %%ymm8, %%ymm8   \n\t"
                         "vaddps 0x120(%[eltwise]), %%ymm9, %%ymm9   \n\t"
                         "vaddps 0x140(%[eltwise]), %%ymm10, %%ymm10 \n\t"
                         "vaddps 0x160(%[eltwise]), %%ymm11, %%ymm11 \n\t"
                         "vaddps 0x180(%[eltwise]), %%ymm12, %%ymm12 \n\t"
                         "vaddps 0x1A0(%[eltwise]), %%ymm13, %%ymm13 \n\t"
                         "vaddps 0x1C0(%[eltwise]), %%ymm14, %%ymm14 \n\t"
                         "vaddps 0x1E0(%[eltwise]), %%ymm15, %%ymm15 \n\t"
                         "vaddps 0x200(%[eltwise]), %%ymm16, %%ymm16 \n\t"
                         "vaddps 0x220(%[eltwise]), %%ymm17, %%ymm17 \n\t"
                         "vaddps 0x240(%[eltwise]), %%ymm18, %%ymm18 \n\t"
                         "vaddps 0x260(%[eltwise]), %%ymm19, %%ymm19 \n\t"
                         "vaddps 0x280(%[eltwise]), %%ymm20, %%ymm20 \n\t"
                         "vaddps 0x2A0(%[eltwise]), %%ymm21, %%ymm21 \n\t"
                         "vaddps 0x2C0(%[eltwise]), %%ymm22, %%ymm22 \n\t"
                         "vaddps 0x2E0(%[eltwise]), %%ymm23, %%ymm23 \n\t"

                         ".align 16                                  \n\t"
                         "3:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu24Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                  \n\t"
                         "4:                                         \n\t"
                         "vmovups %%ymm0,  (%%rax)                   \n\t"
                         "vmovups %%ymm1,  0x20(%%rax)               \n\t"
                         "vmovups %%ymm2,  0x40(%%rax)               \n\t"
                         "vmovups %%ymm3,  0x60(%%rax)               \n\t"
                         "vmovups %%ymm4,  0x80(%%rax)               \n\t"
                         "vmovups %%ymm5,  0xA0(%%rax)               \n\t"
                         "vmovups %%ymm6,  0xC0(%%rax)               \n\t"
                         "vmovups %%ymm7,  0xE0(%%rax)               \n\t"
                         "vmovups %%ymm8,  0x100(%%rax)              \n\t"
                         "vmovups %%ymm9,  0x120(%%rax)              \n\t"
                         "vmovups %%ymm10, 0x140(%%rax)              \n\t"
                         "vmovups %%ymm11, 0x160(%%rax)              \n\t"
                         "vmovups %%ymm12, 0x180(%%rax)              \n\t"
                         "vmovups %%ymm13, 0x1A0(%%rax)              \n\t"
                         "vmovups %%ymm14, 0x1C0(%%rax)              \n\t"
                         "vmovups %%ymm15, 0x1E0(%%rax)              \n\t"
                         "vmovups %%ymm16, 0x200(%%rax)              \n\t"
                         "vmovups %%ymm17, 0x220(%%rax)              \n\t"
                         "vmovups %%ymm18, 0x240(%%rax)              \n\t"
                         "vmovups %%ymm19, 0x260(%%rax)              \n\t"
                         "vmovups %%ymm20, 0x280(%%rax)              \n\t"
                         "vmovups %%ymm21, 0x2A0(%%rax)              \n\t"
                         "vmovups %%ymm22, 0x2C0(%%rax)              \n\t"
                         "vmovups %%ymm23, 0x2E0(%%rax)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "%ymm16", "%ymm17",
                           "%ymm18", "%ymm19", "%ymm20", "%ymm21", "%ymm22", "%ymm23",
                           "%ymm24", "%ymm25", "%ymm26", "%ymm27", "%ymm28", "%ymm29",
                           "%ymm30", "%ymm31", "memory", "cc");
}

void Avx512Conv1x1Kernel12x8(ConvController &c) {
    convKernelForLoopXx16(12, 12, %%ymm, 0x0, 0x20, 0x40, 0x60, 0x80)

    __asm__ __volatile__("movq %[output], %%rax                      \n\t"
                         "movq %[ostepC16], %%rbx                    \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x1, %%rcx                            \n\t"
                         "je 0f                                      \n\t"
                         "vpaddd (%%rax),      %%ymm0,  %%ymm0       \n\t"
                         "vpaddd 0x20(%%rax),  %%ymm1,  %%ymm1       \n\t"
                         "vpaddd 0x40(%%rax),  %%ymm2,  %%ymm2       \n\t"
                         "vpaddd 0x60(%%rax),  %%ymm3,  %%ymm3       \n\t"
                         "vpaddd 0x80(%%rax),  %%ymm4,  %%ymm4       \n\t"
                         "vpaddd 0xA0(%%rax),  %%ymm5,  %%ymm5       \n\t"
                         "vpaddd 0xC0(%%rax),  %%ymm6,  %%ymm6       \n\t"
                         "vpaddd 0xE0(%%rax),  %%ymm7,  %%ymm7       \n\t"
                         "vpaddd 0x100(%%rax), %%ymm8,  %%ymm8       \n\t"
                         "vpaddd 0x120(%%rax), %%ymm9,  %%ymm9       \n\t"
                         "vpaddd 0x140(%%rax), %%ymm10, %%ymm10      \n\t"
                         "vpaddd 0x160(%%rax), %%ymm11, %%ymm11      \n\t"

                         ".align 16                                  \n\t"
                         "0:                                         \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu12Regs(%%ymm, vpxord, vpmaxsd)
                         "jmp 4f                                     \n\t"

                         ".align 16                                  \n\t"
                         "1:                                         \n\t"
                         convert12RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                  \n\t"
                         "2:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0x2, %%rcx                            \n\t"
                         "je 3f                                      \n\t"
                         "vaddps (%[eltwise]),      %%ymm0, %%ymm0   \n\t"
                         "vaddps 0x20(%[eltwise]),  %%ymm1, %%ymm1   \n\t"
                         "vaddps 0x40(%[eltwise]),  %%ymm2, %%ymm2   \n\t"
                         "vaddps 0x60(%[eltwise]),  %%ymm3, %%ymm3   \n\t"
                         "vaddps 0x80(%[eltwise]),  %%ymm4, %%ymm4   \n\t"
                         "vaddps 0xA0(%[eltwise]),  %%ymm5, %%ymm5   \n\t"
                         "vaddps 0xC0(%[eltwise]),  %%ymm6, %%ymm6   \n\t"
                         "vaddps 0xE0(%[eltwise]),  %%ymm7, %%ymm7   \n\t"
                         "vaddps 0x100(%[eltwise]), %%ymm8, %%ymm8   \n\t"
                         "vaddps 0x120(%[eltwise]), %%ymm9, %%ymm9   \n\t"
                         "vaddps 0x140(%[eltwise]), %%ymm10, %%ymm10 \n\t"
                         "vaddps 0x160(%[eltwise]), %%ymm11, %%ymm11 \n\t"

                         ".align 16                                  \n\t"
                         "3:                                         \n\t"
                         "movq %[flags], %%rcx                       \n\t"
                         "and $0xC, %%rcx                            \n\t"
                         "je 4f                                      \n\t"
                         relu12Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                  \n\t"
                         "4:                                         \n\t"
                         "vmovups %%ymm0,  (%%rax)                   \n\t"
                         "vmovups %%ymm1,  0x20(%%rax)               \n\t"
                         "vmovups %%ymm2,  0x40(%%rax)               \n\t"
                         "vmovups %%ymm3,  0x60(%%rax)               \n\t"
                         "vmovups %%ymm4,  0x80(%%rax)               \n\t"
                         "vmovups %%ymm5,  0xA0(%%rax)               \n\t"
                         "vmovups %%ymm6,  0xC0(%%rax)               \n\t"
                         "vmovups %%ymm7,  0xE0(%%rax)               \n\t"
                         "vmovups %%ymm8,  0x100(%%rax)              \n\t"
                         "vmovups %%ymm9,  0x120(%%rax)              \n\t"
                         "vmovups %%ymm10, 0x140(%%rax)              \n\t"
                         "vmovups %%ymm11, 0x160(%%rax)              \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",
                           "%ymm24", "%ymm31", "memory", "cc");
}

void Avx512Conv1x1Kernel1x8(ConvController &c) {
    convKernelForLoopXx16(1, 1, %%ymm, 0x0, 0x20, 0x40, 0x60, 0x80)

    __asm__ __volatile__("movq %[output], %%rax                    \n\t"
                         "movq %[ostepC16], %%rbx                  \n\t"
                         "movq %[flags], %%rcx                     \n\t"
                         "and $0x1, %%rcx                          \n\t"
                         "je 0f                                    \n\t"
                         "vpaddd (%%rax), %%ymm0,  %%ymm0          \n\t"

                         ".align 16                                \n\t"
                         "0:                                       \n\t"
                         "cmpq $0x0, %[scale] \n\t"
                         "jne 1f      \n\t"
                         "movq %[flags], %%rcx                     \n\t"
                         "and $0xC, %%rcx                          \n\t"
                         "je 4f                                    \n\t"
                         relu1Regs(%%ymm, vpxord, vpmaxsd)
                         "jmp 4f                                   \n\t"

                         ".align 16                                \n\t"
                         "1:                                       \n\t"
                         convert1RegsI32ToF32(%[scale], %%ymm)

                         ".align 16                                \n\t"
                         "2:                                       \n\t"
                         "movq %[flags], %%rcx                     \n\t"
                         "and $0x2, %%rcx                          \n\t"
                         "je 3f                                    \n\t"
                         "vaddps (%[eltwise]), %%ymm0, %%ymm0      \n\t"

                         ".align 16                                \n\t"
                         "3:                                       \n\t"
                         "movq %[flags], %%rcx                     \n\t"
                         "and $0xC, %%rcx                          \n\t"
                         "je 4f                                    \n\t"
                         relu1Regs(%%ymm, vxorps, vmaxps)

                         ".align 16                                \n\t"
                         "4:                                       \n\t"
                         "vmovups %%ymm0,  (%%rax)                    \n\t"
                         :
                         : [output] "r" (c.output),
                           [ostepC16] "r" (c.ostepC16),
                           [eltwise] "r" (c.eltwise),
                           [flags] "r" (c.flags),
                           [scale] "r" (c.scale)
                         : "%rax", "%rbx", "%rcx",
                           "%ymm0", "%ymm24", "%ymm31", "memory", "cc");
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

    if (fdf != DF_NCHWC2NxC4 || (idf != DF_NCHWC16 && idf != DF_NCHW && idf != DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }

    // get kernels
    U32 ocSizeArray[4] = {8, 16, 32, 48};
    U32 wSizeArray[4] = {24, 24, 12, 8};
    const kernelFunc kernel[4][3] = {
        {Avx512Conv1x1Kernel1x8, Avx512Conv1x1Kernel12x8, Avx512Conv1x1Kernel24x8},
        {Avx512Conv1x1Kernel1x16, Avx512Conv1x1Kernel12x16, Avx512Conv1x1Kernel24x16},
        {Avx512Conv1x1Kernel1x32, Avx512Conv1x1Kernel6x32, Avx512Conv1x1Kernel12x32},
        {Avx512Conv1x1Kernel1x48, Avx512Conv1x1Kernel4x48, Avx512Conv1x1Kernel8x48}};

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
    convCtl.ostepC16 = oh * ow * 16 * 4;
    convCtl.dilateW = dilateW * SIMDW;
    convCtl.dilateH = (iw_stride - fw * dilateW + (dilateH - 1) * iw_stride) * SIMDW;
    convCtl.fStep = ih_stride * iw_stride * SIMDW;
    convCtl.kw = fw;
    convCtl.kh = fh;
    convCtl.scale = nullptr;
    U32 unrollOc = 48;
    if (fn % 32 == 0 && fn % 48 != 0) {
        unrollOc = 32;
    }

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

    F32 *activatedBias = (F32 *)tmp;
    if (paddingT > 0 || paddingB > 0 || paddingL > 0 || paddingR > 0) {
        getActivatedBiasForPadding(
            biasArray, biasDesc, outputDesc.dt, activatedBias, activationDesc.mode, *scaleO);
        tmp = (void *)((U8 *)tmp + oc * bytesOf(DT_F32));
    }

    U32 oBytes = bytesOf(outputDesc.dt);
    UINT8 *tmpInput = (UINT8 *)tmp;
    if (idf != DF_NCHWC16) {
        tmp = (void *)((U8 *)tmp + ic * ih * iw);
    }
    UINT8 *useInput = (UINT8 *)tmp;

    for (U32 n = 0; n < in; ++n) {
        UINT8 *bInArray = inArray + n * ic * ih * iw;
        if (idf == DF_NCHWC16) {
            if (ic % SIMDW != 0) {
                PaddingChannelNCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
            } else {
                tmpInput = bInArray;
            }
        } else if (idf == DF_NCHWC8) {
            PaddingNCHWC8ToNCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
        } else {
            PaddingNCHW2NCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
        }

        ic = UNI_ALIGN(ic, SIMDW);

        if (strideH > 1 || strideW > 1) {
            U32 ic16 = ic / 16;
            for (U32 hc = 0; hc < ih_stride * ic16; ++hc) {
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
            U32 simdOc = SIMDW;
            if (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
                U32 hwSize = 0;
                for (U32 hw = 0; hw < ohow; hw += hwSize) {
                    U32 ocSize = 0;
                    hwSize = UNI_MIN(BLOCK_HW_DIM, ohow - hw);
                    for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                        ocSize = UNI_MIN(unrollOc, oc - ocb);
                        ocSize = ocSizeArray[ocSize >> 4];
                        simdOc = UNI_MIN(SIMDW, ocSize);
                        convCtl.bias = offsetC + ocb;
                        UINT8 *curI = useInput + icbb * ih_stride * iw_stride;
                        U32 wSize = 8;
                        U32 unrollW = wSizeArray[ocSize >> 4];
                        for (U32 ihw = hw; ihw < hw + hwSize; ihw += wSize) {
                            wSize = UNI_MIN(hw + hwSize - ihw, unrollW);
                            U32 idx = wSize * 2 / unrollW;
                            wSize = UNI_MAX(idx * unrollW / 2, 1);
                            U32 in_h = ihw / ow;
                            U32 in_w = ihw % ow;
                            convCtl.input = curI + in_h * iw_stride * SIMDW + in_w * SIMDW;
                            convCtl.output =
                                output + ((n * oc + ocb) * ohow + ihw * simdOc) * oBytes;
                            convCtl.eltwise = eltwiseInput + (n * oc + ocb) * ohow + ihw * simdOc;
                            convCtl.filter =
                                filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                            convCtl.ic = icSize;
                            kernel[ocSize >> 4][idx](convCtl);
                        }
                    }
                }
            } else {
                for (U32 h = 0; h < oh; ++h) {
                    U32 ocSize = 0;
                    for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                        ocSize = UNI_MIN(unrollOc, oc - ocb);
                        ocSize = ocSizeArray[ocSize >> 4];
                        simdOc = UNI_MIN(SIMDW, ocSize);
                        convCtl.bias = offsetC + ocb;
                        UINT8 *curI = useInput + icbb * ih_stride * iw_stride;
                        U32 wSize = 8;
                        U32 unrollW = wSizeArray[ocSize >> 4];
                        for (U32 w = 0; w < ow; w += wSize) {
                            wSize = 1;
                            convCtl.output =
                                output + ((n * oc + ocb) * ohow + (h * ow + w) * simdOc) * oBytes;
                            convCtl.eltwise = eltwiseInput +
                                ((n * oc + ocb) * ohow + (h * ow + w) * simdOc) * oBytes;
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
                            kernel[ocSize >> 4][idx](convCtl);
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
