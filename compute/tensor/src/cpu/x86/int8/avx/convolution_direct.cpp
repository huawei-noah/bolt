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
#include "cpu/tensor_computing_cpu.h"

#define SIMDW 16
#define BLOCK_IC_DIM 128
#define BLOCK_HW_DIM 96

struct ConvControllerAVX {
    UINT8 *input[12];
    const INT8 *filter;
    void *output;
    F32 *eltwise;
    UINT8 *u8Output;
    const I32 *bias;
    I64 ic;
    I64 kw;
    I64 kh;
    I64 dilateW;
    I64 dilateH;
    I64 ostepC8;
    I64 flags;
    I64 fStep;
    I64 stride;
    void *scale;
};

typedef void (*kernelFuncAVX)(ConvControllerAVX &c);

// clang-format off

#define convKernel1x24c4(input0, input1, input2, input3, freg0, freg1, freg2, off, off0, off1, off2) \
    "vpbroadcastd "#off"("#input0"), %%ymm15          \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"      \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"      \n\t" \
    "vmovups "#off2"(%[filter]), "#freg2"      \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm0        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm15, %%ymm1        \n\t" \
    "%{vex%} vpdpbusd "#freg2", %%ymm15, %%ymm2        \n\t" \

#define convKernel2x24c4(input0, input1, input2, input3, freg0, freg1, freg2, off, off0, off1, off2) \
    convKernel1x24c4(input0, input1, input2, input3, freg0, freg1, freg2, off, off0, off1, off2) \
    "vpbroadcastd "#off"("#input1"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm3        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm15, %%ymm4        \n\t" \
    "%{vex%} vpdpbusd "#freg2", %%ymm15, %%ymm5        \n\t" \

#define convKernel3x24c4(input0, input1, input2, input3, freg0, freg1, freg2, off, off0, off1, off2) \
    convKernel2x24c4(input0, input1, input2, input3, freg0, freg1, freg2, off, off0, off1, off2) \
    "vpbroadcastd "#off"("#input2"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm6        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm15, %%ymm7        \n\t" \
    "%{vex%} vpdpbusd "#freg2", %%ymm15, %%ymm8        \n\t" \

#define convKernel4x24c4(input0, input1, input2, input3, freg0, freg1, freg2, off, off0, off1, off2) \
    convKernel3x24c4(input0, input1, input2, input3, freg0, freg1, freg2, off, off0, off1, off2) \
    "vpbroadcastd "#off"("#input3"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm9        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm15, %%ymm10        \n\t" \
    "%{vex%} vpdpbusd "#freg2", %%ymm15, %%ymm11        \n\t" \

#define convKernel1x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    "vpbroadcastd "#off"("#input0"), %%ymm15          \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"      \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"      \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm0        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm15, %%ymm1        \n\t" \

#define convKernel2x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    "vpbroadcastd "#off"("#input0"), %%ymm14          \n\t" \
    "vpbroadcastd "#off"("#input1"), %%ymm15          \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"      \n\t" \
    "vmovups "#off1"(%[filter]), "#freg1"      \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm0        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm14, %%ymm1        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm2        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm15, %%ymm3        \n\t" \

#define convKernel3x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    convKernel2x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    "vpbroadcastd "#off"("#input2"), %%ymm14          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm4        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm14, %%ymm5        \n\t" \

#define convKernel4x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    convKernel2x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    "vpbroadcastd "#off"("#input2"), %%ymm14          \n\t" \
    "vpbroadcastd "#off"("#input3"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm4        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm14, %%ymm5        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm6        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm15, %%ymm7        \n\t" \

#define convKernel5x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    convKernel4x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    "vpbroadcastd "#off"("#input4"), %%ymm14          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm8        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm14, %%ymm9        \n\t" \

#define convKernel6x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    convKernel4x16c4(input0, input1, input2, input3, input4, input5, freg0, freg1, off, off0, off1) \
    "vpbroadcastd "#off"("#input4"), %%ymm14          \n\t" \
    "vpbroadcastd "#off"("#input5"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm8        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm14, %%ymm9        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm10        \n\t" \
    "%{vex%} vpdpbusd "#freg1", %%ymm15, %%ymm11        \n\t" \

#define convKernel1x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    "vpbroadcastd "#off"("#input0"), %%ymm15          \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"      \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm0       \n\t" \

#define convKernel2x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    "vpbroadcastd "#off"("#input0"), %%ymm15          \n\t" \
    "vpbroadcastd "#off"("#input1"), %%ymm14          \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"      \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm0        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm1        \n\t" \

#define convKernel3x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    "vpbroadcastd "#off"("#input0"), %%ymm15          \n\t" \
    "vpbroadcastd "#off"("#input1"), %%ymm14          \n\t" \
    "vpbroadcastd "#off"("#input2"), %%ymm13          \n\t" \
    "vmovups "#off0"(%[filter]), "#freg0"      \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm0        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm1        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm13, %%ymm2        \n\t" \

#define convKernel4x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    convKernel2x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    "vpbroadcastd "#off"("#input2"), %%ymm13          \n\t" \
    "vpbroadcastd "#off"("#input3"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm13, %%ymm2        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm3        \n\t" \

#define convKernel5x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    convKernel3x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    "vpbroadcastd "#off"("#input3"), %%ymm15          \n\t" \
    "vpbroadcastd "#off"("#input4"), %%ymm14          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm3        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm4        \n\t" \

#define convKernel6x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    convKernel3x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    "vpbroadcastd "#off"("#input3"), %%ymm15          \n\t" \
    "vpbroadcastd "#off"("#input4"), %%ymm14          \n\t" \
    "vpbroadcastd "#off"("#input5"), %%ymm13          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm3        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm4        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm13, %%ymm5        \n\t" \

#define convKernel7x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    convKernel4x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    "vpbroadcastd "#off"("#input4"), %%ymm14          \n\t" \
    "vpbroadcastd "#off"("#input5"), %%ymm13          \n\t" \
    "vpbroadcastd "#off"("#input6"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm4        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm13, %%ymm5        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm6        \n\t" \

#define convKernel8x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    convKernel6x8c4(input0, input1, input2, input3, input4, input5, input6, input7, freg0, off, off0) \
    "vpbroadcastd "#off"("#input6"), %%ymm15          \n\t" \
    "vpbroadcastd "#off"("#input7"), %%ymm14          \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm15, %%ymm6        \n\t" \
    "%{vex%} vpdpbusd "#freg0", %%ymm14, %%ymm7        \n\t" \

#define add_1x8(add, output, step) \
    ""#add" ("#output"), %%ymm0, %%ymm0                    \n\t" \

#define add_2x8(add, output, step) \
    add_1x8(add, output, step) \
    ""#add" 0x20("#output"), %%ymm1, %%ymm1                    \n\t" \

#define add_3x8(add, output, step) \
    add_2x8(add, output, step) \
    ""#add" 0x40("#output"),  %%ymm2, %%ymm2        \n\t"

#define add_4x8(add, output, step) \
    add_3x8(add, output, step) \
    ""#add" 0x60("#output"),  %%ymm3, %%ymm3        \n\t"

#define add_5x8(add, output, step) \
    add_4x8(add, output, step) \
    ""#add" 0x80("#output"), %%ymm4, %%ymm4        \n\t"

#define add_6x8(add, output, step) \
    add_5x8(add, output, step) \
    ""#add" 0xA0("#output"), %%ymm5, %%ymm5        \n\t"

#define add_7x8(add, output, step) \
    add_6x8(add, output, step) \
    ""#add" 0xC0("#output"), %%ymm6, %%ymm6        \n\t"

#define add_8x8(add, output, step) \
    add_7x8(add, output, step) \
    ""#add" 0xE0("#output"), %%ymm7, %%ymm7        \n\t"

#define add_1x16(add, output, step) \
    ""#add" ("#output"), %%ymm0, %%ymm0                    \n\t" \
    ""#add" ("#output", "#step"), %%ymm1, %%ymm1             \n\t"

#define add_2x16(add, output, step) \
    add_1x16(add, output, step) \
    ""#add" 0x20("#output"), %%ymm2, %%ymm2                \n\t" \
    ""#add" 0x20("#output", "#step"), %%ymm3, %%ymm3         \n\t"

#define add_3x16(add, output, step) \
    add_2x16(add, output, step) \
    ""#add" 0x40("#output"), %%ymm4, %%ymm4                \n\t" \
    ""#add" 0x40("#output", "#step"), %%ymm5, %%ymm5         \n\t"

#define add_4x16(add, output, step) \
    add_3x16(add, output, step) \
    ""#add" 0x60("#output"), %%ymm6, %%ymm6                \n\t" \
    ""#add" 0x60("#output", "#step"), %%ymm7, %%ymm7         \n\t"

#define add_5x16(add, output, step) \
    add_4x16(add, output, step) \
    ""#add" 0x80("#output"), %%ymm8, %%ymm8               \n\t" \
    ""#add" 0x80("#output", "#step"), %%ymm9, %%ymm9        \n\t"

#define add_6x16(add, output, step) \
    add_5x16(add, output, step) \
    ""#add" 0xA0("#output"), %%ymm10, %%ymm10             \n\t" \
    ""#add" 0xA0("#output", "#step"), %%ymm11, %%ymm11      \n\t"

#define add_1x24(add, output, step) \
    ""#add" ("#output"), %%ymm0, %%ymm0                    \n\t" \
    ""#add" ("#output", "#step"), %%ymm1, %%ymm1             \n\t" \
    ""#add" ("#output", "#step", 2), %%ymm2, %%ymm2         \n\t"

#define add_2x24(add, output, step) \
    add_1x24(add, output, step) \
    ""#add" 0x20("#output"), %%ymm3, %%ymm3                    \n\t" \
    ""#add" 0x20("#output", "#step"), %%ymm4, %%ymm4             \n\t" \
    ""#add" 0x20("#output", "#step", 2), %%ymm5, %%ymm5         \n\t"

#define add_3x24(add, output, step) \
    add_2x24(add, output, step) \
    ""#add" 0x40("#output"), %%ymm6, %%ymm6                    \n\t" \
    ""#add" 0x40("#output", "#step"), %%ymm7, %%ymm7             \n\t" \
    ""#add" 0x40("#output", "#step", 2), %%ymm8, %%ymm8         \n\t"

#define add_4x24(add, output, step) \
    add_3x24(add, output, step) \
    ""#add" 0x60("#output"), %%ymm9, %%ymm9                    \n\t" \
    ""#add" 0x60("#output", "#step"), %%ymm10, %%ymm10             \n\t" \
    ""#add" 0x60("#output", "#step", 2), %%ymm11, %%ymm11         \n\t"

#define store_1x8(output, step) \
    "vmovups %%ymm0, ("#output")                    \n\t"

#define store_2x8(output, step) \
    store_1x8(output, step) \
    "vmovups %%ymm1,  0x20("#output")               \n\t"

#define store_3x8(output, step) \
    store_2x8(output, step) \
    "vmovups %%ymm2,  0x40("#output")               \n\t"

#define store_4x8(output, step) \
    store_3x8(output, step) \
    "vmovups %%ymm3,  0x60("#output")               \n\t"

#define store_5x8(output, step) \
    store_4x8(output, step) \
    "vmovups %%ymm4,  0x80("#output")              \n\t"

#define store_6x8(output, step) \
    store_5x8(output, step) \
    "vmovups %%ymm5,  0xA0("#output")              \n\t"

#define store_7x8(output, step) \
    store_6x8(output, step) \
    "vmovups %%ymm6,  0xC0("#output")              \n\t"

#define store_8x8(output, step) \
    store_7x8(output, step) \
    "vmovups %%ymm7,  0xE0("#output")              \n\t"

#define store_1x16(output, step) \
    "vmovups %%ymm0, ("#output")                           \n\t" \
    "vmovups %%ymm1, ("#output", "#step")                    \n\t"

#define store_2x16(output, step) \
    store_1x16(output, step) \
    "vmovups %%ymm2, 0x20("#output")                       \n\t" \
    "vmovups %%ymm3, 0x20("#output", "#step")                \n\t"

#define store_3x16(output, step) \
    store_2x16(output, step) \
    "vmovups %%ymm4, 0x40("#output")                       \n\t" \
    "vmovups %%ymm5, 0x40("#output", "#step")                \n\t"

#define store_4x16(output, step) \
    store_3x16(output, step) \
    "vmovups %%ymm6, 0x60("#output")                       \n\t" \
    "vmovups %%ymm7, 0x60("#output", "#step")                \n\t"

#define store_5x16(output, step) \
    store_4x16(output, step) \
    "vmovups %%ymm8, 0x80("#output")                      \n\t" \
    "vmovups %%ymm9, 0x80("#output", "#step")               \n\t"

#define store_6x16(output, step) \
    store_5x16(output, step) \
    "vmovups %%ymm10, 0xA0("#output")                     \n\t" \
    "vmovups %%ymm11, 0xA0("#output", "#step")              \n\t"

#define store_1x24(output, step) \
    "vmovups %%ymm0, ("#output")                              \n\t" \
    "vmovups %%ymm1, ("#output", "#step")                       \n\t" \
    "vmovups %%ymm2, ("#output", "#step", 2)                    \n\t"

#define store_2x24(output, step) \
    store_1x24(output, step) \
    "vmovups %%ymm3, 0x20("#output")                          \n\t" \
    "vmovups %%ymm4, 0x20("#output", "#step")                   \n\t" \
    "vmovups %%ymm5, 0x20("#output", "#step", 2)                \n\t"

#define store_3x24(output, step) \
    store_2x24(output, step) \
    "vmovups %%ymm6, 0x40("#output")                          \n\t" \
    "vmovups %%ymm7, 0x40("#output", "#step")                   \n\t" \
    "vmovups %%ymm8, 0x40("#output", "#step", 2)                \n\t"

#define store_4x24(output, step) \
    store_3x24(output, step) \
    "vmovups %%ymm9,  0x60("#output")                          \n\t" \
    "vmovups %%ymm10, 0x60("#output", "#step")                  \n\t" \
    "vmovups %%ymm11, 0x60("#output", "#step", 2)               \n\t"


#define convKernelForLoopXx24(rnum, wsize) \
    __asm__ __volatile__("movq %[flags], %%rax                   \n\t" \
                         "andq $0x1, %%rax                       \n\t" \
                         "jne 0f                                 \n\t" \
                         load3BiasToYmmRegs(rnum, %[bias])             \
                         "jmp 1f                                 \n\t" \
                         ".align 16                              \n\t" \
                         "0:                                     \n\t" \
                         clear##rnum##Regs(%%ymm)                      \
                         ".align 16                              \n\t" \
                         "1:                                     \n\t" \
                         : [filter] "+r" (c.filter)                    \
                         : [bias] "r" (c.bias),                        \
                           [flags] "r" (c.flags)                       \
                         : "%rax",                                     \
                           "%ymm0", "%ymm1","%ymm2", "%ymm3", "%ymm4", "%ymm5",        \
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",     \
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");              \
    __asm__ __volatile__(".align 16                                                   \n\t" \
                         "1:                                                          \n\t" \
                         "mov %[kh], %%rbx                                            \n\t" \
                         ".align 16                                                   \n\t" \
                         "2:                                                          \n\t" \
                         "mov %[kw], %%rax                                             \n\t" \
                         ".align 16                                                   \n\t" \
                         "3:                                                          \n\t" \
                         convKernel##wsize##x24c4(%[input0], %[input1], %[input2], %[input3], \
                            %%ymm12, %%ymm13, %%ymm14, 0x0, 0x0, 0x20, 0x40)                     \
                         convKernel##wsize##x24c4(%[input0], %[input1], %[input2], %[input3], \
                            %%ymm12, %%ymm13, %%ymm14, 0x4, 0x60, 0x80, 0xA0)                     \
                         "addq $0xC0, %[filter]                                      \n\t" \
                         "addq %[dilateW], %[input0]                                   \n\t" \
                         "addq %[dilateW], %[input1]                                   \n\t" \
                         "addq %[dilateW], %[input2]                                   \n\t" \
                         "addq %[dilateW], %[input3]                                   \n\t" \
                         "dec %%rax                                                    \n\t" \
                         "jg 3b                                                       \n\t" \
                         "addq %[dilateH], %[input0]                                   \n\t" \
                         "addq %[dilateH], %[input1]                                   \n\t" \
                         "addq %[dilateH], %[input2]                                   \n\t" \
                         "addq %[dilateH], %[input3]                                   \n\t" \
                         "dec %%rbx                                                   \n\t" \
                         "jg 2b                                                       \n\t" \
                         "addq %[fStep], %[input0]                                     \n\t" \
                         "addq %[fStep], %[input1]                                     \n\t" \
                         "addq %[fStep], %[input2]                                     \n\t" \
                         "addq %[fStep], %[input3]                                     \n\t" \
                         "dec %%rcx                                           \n\t" \
                         "jg 1b                                                      \n\t" \
                         ".align 16                                                   \n\t" \
                         "4:                                                          \n\t" \
                         : "+c" (c.ic),                                                     \
                           [input0] "+r" (c.input[0]),                                          \
                           [input1] "+r" (c.input[1]),                                          \
                           [input2] "+r" (c.input[2]),                                          \
                           [input3] "+r" (c.input[3]),                                          \
                           [filter] "+r" (c.filter)                                         \
                         : [kh] "r" (c.kh),                                                 \
                           [kw] "r" (c.kw),                                                 \
                           [dilateW] "r" (c.dilateW),                                       \
                           [dilateH] "r" (c.dilateH),                                       \
                           [fStep] "r" (c.fStep)                                           \
                         : "%rax", "%rbx",                                    \
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",            \
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",          \
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");                             \
    __asm__ __volatile__("movq %[output], %%rax                             \n\t" \
                         "movq %[ostepC8], %%rbx                           \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x1, %%rcx                                   \n\t" \
                         "je 0f                                             \n\t" \
                         add_##wsize##x24(vpaddd, %%rax, %%rbx) \
                         ".align 16                                         \n\t" \
                         "0:                                                \n\t" \
                         "cmpq $0x0, %[scale]                               \n\t" \
                         "jne 1f                                            \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%ymm, vpxor, vpmaxsd) \
                         "jmp 4f                                            \n\t" \
                         ".align 16                                         \n\t" \
                         "1:                                                \n\t" \
                         convert##rnum##RegsI32ToF32(%[scale], %%ymm) \
                         ".align 16                                         \n\t" \
                         "2:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x2, %%rcx                                   \n\t" \
                         "je 3f                                             \n\t" \
                         add_##wsize##x24(vaddps, %[eltwise], %%rbx) \
                         ".align 16                                         \n\t" \
                         "3:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%ymm, vxorps, vmaxps) \
                         ".align 16                                         \n\t" \
                         "4:                                                \n\t" \
                         store_##wsize##x24(%%rax, %%rbx) \
                         : \
                         : [output] "r" (c.output), \
                           [eltwise] "r" (c.eltwise), \
                           [ostepC8] "r" (c.ostepC8), \
                           [flags] "r" (c.flags), \
                           [scale] "r" (c.scale) \
                         : "%rax", "%rbx", "%rcx", \
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", \
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", \
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");


#define convKernelForLoopXx16(rnum, wsize) \
     __asm__ __volatile__("movq %[flags], %%rax                 \n\t" \
                          "andq $0x1, %%rax                     \n\t" \
                          "jne 0f                               \n\t" \
                          load2BiasToYmmRegs(rnum, %[bias])             \
                          "jmp 1f                               \n\t" \
                          ".align 16                            \n\t" \
                          "0:                                   \n\t" \
                          clear##rnum##Regs(%%ymm)                    \
                          ".align 16                            \n\t" \
                          "1:                                   \n\t" \
                          : [filter] "+r" (c.filter)                  \
                          : [bias] "r" (c.bias),                      \
                            [flags] "r" (c.flags)                     \
                          : "%rax",                                   \
                            "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",       \
                            "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",     \
                            "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc"); \
     __asm__ __volatile__(".align 16                                           \n\t" \
                          "1:                                                  \n\t" \
                          "mov %[kh], %%rbx                                    \n\t" \
                          ".align 16                                           \n\t" \
                          "2:                                                  \n\t" \
                          "mov %[kw], %%rax                                     \n\t" \
                          ".align 16                                           \n\t" \
                          "3:                                                  \n\t" \
                          convKernel##wsize##x16c4(%[input0], %[input1], %[input2], %[input3], \
                            %[input4], %[input5], %%ymm12, %%ymm13, 0x0, 0x0, 0x20)                     \
                          convKernel##wsize##x16c4(%[input0], %[input1], %[input2], %[input3], \
                            %[input4], %[input5], %%ymm12, %%ymm13, 0x4, 0x40, 0x60)                     \
                          "addq $0x80, %[filter]                              \n\t" \
                          "addq %[dilateW], %[input0]                           \n\t" \
                          "addq %[dilateW], %[input1]                           \n\t" \
                          "addq %[dilateW], %[input2]                           \n\t" \
                          "addq %[dilateW], %[input3]                           \n\t" \
                          "addq %[dilateW], %[input4]                           \n\t" \
                          "addq %[dilateW], %[input5]                           \n\t" \
                          "dec %%rax                                            \n\t" \
                          "jg 3b                                               \n\t" \
                          "addq %[dilateH], %[input0]                           \n\t" \
                          "addq %[dilateH], %[input1]                           \n\t" \
                          "addq %[dilateH], %[input2]                           \n\t" \
                          "addq %[dilateH], %[input3]                           \n\t" \
                          "addq %[dilateH], %[input4]                           \n\t" \
                          "addq %[dilateH], %[input5]                           \n\t" \
                          "dec %%rbx                                           \n\t" \
                          "jg 2b                                               \n\t" \
                          "addq %[fStep], %[input0]                             \n\t" \
                          "addq %[fStep], %[input1]                             \n\t" \
                          "addq %[fStep], %[input2]                             \n\t" \
                          "addq %[fStep], %[input3]                             \n\t" \
                          "addq %[fStep], %[input4]                             \n\t" \
                          "addq %[fStep], %[input5]                             \n\t" \
                          "dec %%rcx                                   \n\t" \
                          "jg 1b                                              \n\t" \
                          ".align 16                                           \n\t" \
                          "4:                                                  \n\t" \
                          : "+c" (c.ic),               \
                            [input0] "+r" (c.input[0]),    \
                            [input1] "+r" (c.input[1]),    \
                            [input2] "+r" (c.input[2]),    \
                            [input3] "+r" (c.input[3]),    \
                            [input4] "+r" (c.input[4]),    \
                            [input5] "+r" (c.input[5]),    \
                            [filter] "+r" (c.filter)   \
                          : [kh] "r" (c.kh),           \
                            [kw] "r" (c.kw),           \
                            [dilateW] "r" (c.dilateW), \
                            [dilateH] "r" (c.dilateH), \
                            [fStep] "r" (c.fStep)     \
                          : "%rax", "%rbx",                             \
                            "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",       \
                            "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",     \
                            "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");                        \
    __asm__ __volatile__("movq %[output], %%rax                             \n\t" \
                         "movq %[ostepC8], %%rbx                           \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x1, %%rcx                                   \n\t" \
                         "je 0f                                             \n\t" \
                         add_##wsize##x16(vpaddd, %%rax, %%rbx) \
                         ".align 16                                         \n\t" \
                         "0:                                                \n\t" \
                         "cmpq $0x0, %[scale]                               \n\t" \
                         "jne 1f                                            \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%ymm, vpxor, vpmaxsd) \
                         "jmp 4f                                            \n\t" \
                         ".align 16                                         \n\t" \
                         "1:                                                \n\t" \
                         convert##rnum##RegsI32ToF32(%[scale], %%ymm) \
                         ".align 16                                         \n\t" \
                         "2:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x2, %%rcx                                   \n\t" \
                         "je 3f                                             \n\t" \
                         add_##wsize##x16(vaddps, %[eltwise], %%rbx) \
                         ".align 16                                         \n\t" \
                         "3:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%ymm, vxorps, vmaxps) \
                         ".align 16                                         \n\t" \
                         "4:                                                \n\t" \
                         store_##wsize##x16(%%rax, %%rbx) \
                         : \
                         : [output] "r" (c.output), \
                           [eltwise] "r" (c.eltwise), \
                           [ostepC8] "r" (c.ostepC8), \
                           [flags] "r" (c.flags), \
                           [scale] "r" (c.scale) \
                         : "%rax", "%rbx", "%rcx", \
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", \
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", \
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");

                         

#define convKernelForLoopXx8(rnum, wsize) \
    __asm__ __volatile__("movq %[flags], %%rax                         \n\t" \
                         "andq $0x1, %%rax                             \n\t" \
                         "jne 0f                                       \n\t" \
                         load1BiasToRegs(rnum, %[bias], %%ymm)             \
                         "jmp 1f                                       \n\t" \
                         ".align 16                                    \n\t" \
                         "0:                                           \n\t" \
                         clear##rnum##Regs(%%ymm)                            \
                         ".align 16                                    \n\t" \
                         "1:                                           \n\t" \
                         : [filter] "+r" (c.filter)                          \
                         : [bias] "r" (c.bias),                              \
                           [flags] "r" (c.flags)                             \
                         : "%rax",                                           \
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",       \
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",     \
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc"); \
    __asm__ __volatile__(".align 16                             \n\t" \
                         "1:                                    \n\t" \
                         "pushq %%rbx                                         \n\t" \
                         ".align 16                             \n\t" \
                         "2:                                    \n\t" \
                         "pushq %%rax                                         \n\t" \
                         ".align 16                             \n\t" \
                         "3:                                    \n\t" \
                        convKernel##wsize##x8c4(%[input0], %[input1], %[input2], %[input3], \
                            %[input4], %[input5], %[input6], %[input7], %%ymm12, 0x0, 0x0)                     \
                          convKernel##wsize##x8c4(%[input0], %[input1], %[input2], %[input3], \
                            %[input4], %[input5], %[input6], %[input7], %%ymm12, 0x4, 0x20)                     \
                         "addq $0x40, %[filter]              \n\t" \
                         "addq %[dilateW], %[input0]             \n\t" \
                         "addq %[dilateW], %[input1]             \n\t" \
                         "addq %[dilateW], %[input2]             \n\t" \
                         "addq %[dilateW], %[input3]             \n\t" \
                         "addq %[dilateW], %[input4]             \n\t" \
                         "addq %[dilateW], %[input5]             \n\t" \
                         "addq %[dilateW], %[input6]             \n\t" \
                         "addq %[dilateW], %[input7]             \n\t" \
                         "dec %%rax                              \n\t" \
                         "jg 3b                                 \n\t" \
                         "popq %%rax                                         \n\t"  \
                         "addq %[dilateH], %[input0]             \n\t" \
                         "addq %[dilateH], %[input1]             \n\t" \
                         "addq %[dilateH], %[input2]             \n\t" \
                         "addq %[dilateH], %[input3]             \n\t" \
                         "addq %[dilateH], %[input4]             \n\t" \
                         "addq %[dilateH], %[input5]             \n\t" \
                         "addq %[dilateH], %[input6]             \n\t" \
                         "addq %[dilateH], %[input7]             \n\t" \
                         "dec %%rbx                             \n\t" \
                         "jg 2b                                 \n\t" \
                         "popq %%rbx                                         \n\t" \
                         "addq %[fStep], %[input0]               \n\t" \
                         "addq %[fStep], %[input1]               \n\t" \
                         "addq %[fStep], %[input2]               \n\t" \
                         "addq %[fStep], %[input3]               \n\t" \
                         "addq %[fStep], %[input4]               \n\t" \
                         "addq %[fStep], %[input5]               \n\t" \
                         "addq %[fStep], %[input6]               \n\t" \
                         "addq %[fStep], %[input7]               \n\t" \
                         "dec %%rcx                     \n\t" \
                         "jg 1b                                \n\t" \
                         ".align 16                             \n\t" \
                         "4:                                    \n\t" \
                         : "+c" (c.ic),                               \
                           [input0] "+r" (c.input[0]),                    \
                           [input1] "+r" (c.input[1]),                    \
                           [input2] "+r" (c.input[2]),                    \
                           [input3] "+r" (c.input[3]),                    \
                           [input4] "+r" (c.input[4]),                    \
                           [input5] "+r" (c.input[5]),                    \
                           [input6] "+r" (c.input[6]),                    \
                           [input7] "+r" (c.input[7]),                    \
                           [filter] "+r" (c.filter),                   \
                           [kh] "+b" (c.kh),                           \
                           [kw] "+a" (c.kw)                           \
                         : [dilateW] "r" (c.dilateW),                 \
                           [dilateH] "r" (c.dilateH),                 \
                           [fStep] "r" (c.fStep)                     \
                         : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",       \
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",     \
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc"); \
    __asm__ __volatile__("movq %[output], %%rax                             \n\t" \
                         "movq %[ostepC8], %%rbx                           \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x1, %%rcx                                   \n\t" \
                         "je 0f                                             \n\t" \
                         add_##wsize##x8(vpaddd, %%rax, %%rbx) \
                         ".align 16                                         \n\t" \
                         "0:                                                \n\t" \
                         "cmpq $0x0, %[scale]                               \n\t" \
                         "jne 1f                                            \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%ymm, vpxor, vpmaxsd) \
                         "jmp 4f                                            \n\t" \
                         ".align 16                                         \n\t" \
                         "1:                                                \n\t" \
                         convert##rnum##RegsI32ToF32(%[scale], %%ymm) \
                         ".align 16                                         \n\t" \
                         "2:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x2, %%rcx                                   \n\t" \
                         "je 3f                                             \n\t" \
                         add_##wsize##x8(vaddps, %[eltwise], %%rbx) \
                         ".align 16                                         \n\t" \
                         "3:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%ymm, vxorps, vmaxps) \
                         ".align 16                                         \n\t" \
                         "4:                                                \n\t" \
                         store_##wsize##x8(%%rax, %%rbx) \
                         : \
                         : [output] "r" (c.output), \
                           [eltwise] "r" (c.eltwise), \
                           [ostepC8] "r" (c.ostepC8), \
                           [flags] "r" (c.flags), \
                           [scale] "r" (c.scale) \
                         : "%rax", "%rbx", "%rcx", \
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", \
                           "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", \
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");

#define convKernelx24(nReg, wSize) \
    void Avx512ConvKernel##wSize##x24(ConvControllerAVX &c) { \
        convKernelForLoopXx24(nReg, wSize) \
    }

#define convKernelx16(nReg, wSize) \
    void Avx512ConvKernel##wSize##x16(ConvControllerAVX &c) { \
        convKernelForLoopXx16(nReg, wSize) \
    }

#define convKernelx8(nReg, wSize) \
    void Avx512ConvKernel##wSize##x8(ConvControllerAVX &c) { \
        convKernelForLoopXx8(nReg, wSize) \
    }

convKernelx24(12, 4)
convKernelx24(9, 3)
convKernelx24(6, 2)
convKernelx24(3, 1)
convKernelx16(12, 6)
convKernelx16(10, 5)
convKernelx16(8, 4)
convKernelx16(6, 3)
convKernelx16(4, 2)
convKernelx16(2, 1)
convKernelx8(8, 8)
convKernelx8(7, 7)
convKernelx8(6, 6)
convKernelx8(5, 5)
convKernelx8(4, 4)
convKernelx8(3, 3)
convKernelx8(2, 2)
convKernelx8(1, 1)

// clang-format on
EE convolution_direct(TensorDesc inputDesc,
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

    const kernelFuncAVX kernel[8][3] = {
        {Avx512ConvKernel1x8, Avx512ConvKernel1x16, Avx512ConvKernel1x24}, 
        {Avx512ConvKernel2x8, Avx512ConvKernel2x16, Avx512ConvKernel2x24}, 
        {Avx512ConvKernel3x8, Avx512ConvKernel3x16, Avx512ConvKernel3x24}, 
        {Avx512ConvKernel4x8, Avx512ConvKernel4x16, Avx512ConvKernel4x24}, 
        {Avx512ConvKernel5x8, Avx512ConvKernel5x16, Avx512ConvKernel4x24}, 
        {Avx512ConvKernel6x8, Avx512ConvKernel6x16, Avx512ConvKernel4x24}, 
        {Avx512ConvKernel7x8, Avx512ConvKernel6x16, Avx512ConvKernel4x24}, 
        {Avx512ConvKernel8x8, Avx512ConvKernel6x16, Avx512ConvKernel4x24}};

    // get computing params
    U32 strideH = convParamSpec.stride_h;
    U32 strideW = convParamSpec.stride_w;
    U32 paddingT = convParamSpec.pad_top;
    U32 paddingB = convParamSpec.pad_bottom;
    U32 paddingL = convParamSpec.pad_left;
    U32 paddingR = convParamSpec.pad_right;
    I32 dilateH = convParamSpec.dilatedRate_h;
    I32 dilateW = convParamSpec.dilatedRate_w;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    U32 ohow = oh * ow;
    UINT8 *output = (UINT8 *)outArray;

    // infer kernel params
    ConvControllerAVX convCtl;
    convCtl.ostepC8 = oh * ow * 8 * 4;
    convCtl.dilateW = dilateW * 8;
    convCtl.kw = fw;
    convCtl.kh = fh;
    convCtl.dilateH = ((I32)iw_pad - convCtl.kw * dilateW + (dilateH - 1) * (I32)iw_pad) * 8;
    convCtl.fStep = (((I32)ih_pad - convCtl.kh * dilateH) * (I32)iw_pad) * 8;
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
    if (odt != DT_F32) {
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

    U32 oBytes = bytesOf(outputDesc.dt);
    UINT8 *tmpInput = (UINT8 *)tmp;
    for (U32 n = 0; n < in; ++n) {
        UINT8 *bInArray = inArray + n * ic * ih * iw;
        if (idf == DF_NCHWC8 && paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            tmpInput = bInArray;
        } else {
            if (idf == DF_NCHWC8) {
                PaddingNCHWCx(bInArray, tmpInput, inputDesc, convParamSpec);
            } else {
                return NOT_SUPPORTED;
            }
        }

        U32 flags = 0;
        U32 icSize = 0;
        for (U32 icbb = 0; icbb < ic; icbb += icSize) {
            icSize = UNI_MIN(BLOCK_IC_DIM, ic - icbb);
            flags |= (icbb > 0);
            if (icbb == ic - icSize) {
                flags |= (eltwiseInput != nullptr) << 1;
                flags |= U32(activationDesc.mode) << 2;
                convCtl.scale = factorPtr;
            }
            convCtl.flags = flags;
            U32 simdOc = 8;

            U32 hwSize = 0;
            for (U32 hw = 0; hw < oh * ow; hw += hwSize) {
                hwSize = UNI_MIN(BLOCK_HW_DIM, oh * ow - hw);
                U32 ocSize = 0;
                for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                    ocSize = UNI_MIN(unrollOc, oc - ocb);
                    ocSize = ocSizeArray[(ocSize >> 3) - 1];
                    convCtl.bias = offsetC + ocb;
                    UINT8 *curI = tmpInput + icbb * ih_pad * iw_pad;
                    U32 wSize = 8;
                    U32 unrollW = wSizeArray[(ocSize >> 3) - 1];
                    for (U32 ihw = hw; ihw < hw + hwSize; ihw += wSize) {
                        wSize = UNI_MIN(hw + hwSize - ihw, unrollW);
                        for (U32 wi = 0; wi < wSize; ++wi) {
                            U32 in_h = (ihw + wi) / ow * strideH;
                            U32 in_w = (ihw + wi) % ow * strideW;
                            convCtl.input[wi] = curI + in_h * iw_pad * simdOc + in_w * simdOc;
                        }
                        convCtl.output = output + ((n * oc + ocb) * ohow + ihw * simdOc) * oBytes;
                        convCtl.eltwise = eltwiseInput + (n * oc + ocb) * ohow + ihw * simdOc;
                        convCtl.filter = filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                        convCtl.ic = icSize / 8;
                        convCtl.kw = fw;
                        convCtl.kh = fh;

                        kernel[wSize - 1][(ocSize >> 3) - 1](convCtl);
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
