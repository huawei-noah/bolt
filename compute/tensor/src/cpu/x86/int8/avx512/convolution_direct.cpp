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

// clang-format off

#define convKernel2x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "movq (%[stepC16]), %%r10                  \n\t" \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31   \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0        \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1        \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"      \n\t" \
    "addq %%r10, "#input"                      \n\t" \
    "addq 0x8(%[stepC16]), "#input"            \n\t" \
    "movq 0x10(%[stepC16]), %%r10             \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2        \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm3        \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm4        \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm5        \n\t" \

#define convKernel3x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel2x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30         \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31  \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm6       \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm7       \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm8       \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"     \n\t" \

#define convKernel5x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel3x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                     \n\t" \
    "addq 0x18(%[stepC16]), "#input"          \n\t" \
    "movq 0x20(%[stepC16]), %%r10             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm9       \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm10      \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm11      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30         \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31  \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12      \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13      \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm14      \n\t" \

#define convKernel8x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel5x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                     \n\t" \
    "addq 0x28(%[stepC16]), "#input"          \n\t" \
    "movq 0x30(%[stepC16]), %%r10             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm15      \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm16      \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm17      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30         \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31  \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm18      \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm19      \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm20      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm21      \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm22      \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm23      \n\t"

#define convKernel7x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel5x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                     \n\t" \
    "addq 0x28(%[stepC16]), "#input"          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm15      \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm16      \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm17      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm18      \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm19      \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm20      \n\t" \

#define convKernel6x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel5x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpdpbusd "#freg0", %%zmm31, %%zmm15      \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm16      \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm17      \n\t" \

#define convKernel5x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel3x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                     \n\t" \
    "addq 0x18(%[stepC16]), "#input"          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm9       \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm10      \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm11      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12      \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13      \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm14      \n\t" \

#define convKernel4x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel3x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpdpbusd "#freg0", %%zmm31, %%zmm9       \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm10      \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm11      \n\t" \

#define convKernel3x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel2x48c4Core_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"      \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm6        \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm7        \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm8        \n\t" \

#define convKernel2x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "movq (%[stepC16]), %%r10                  \n\t" \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31   \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"      \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0        \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"      \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1        \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2        \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm3        \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm4        \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm5        \n\t" \

#define convKernel1x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"      \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"      \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"      \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0        \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1        \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2        \n\t"

#define convKernel2x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31   \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0        \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1        \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"      \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2        \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm3        \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm4        \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"      \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm5        \n\t" \

#define convKernel4x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel2x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                      \n\t" \
    "addq %%r10, "#input"                      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31   \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm6        \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm7        \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm8        \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm9        \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm10       \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm11       \n\t" \

#define convKernel6x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel4x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                      \n\t" \
    "addq %%r10, "#input"                      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31   \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12       \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13       \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm14       \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm15       \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm16       \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm17       \n\t" \

#define convKernel8x48c4_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel6x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                      \n\t" \
    "addq %%r10, "#input"                      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31   \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm18       \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm19       \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm20       \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm21       \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm22       \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm23       \n\t"

#define convKernel7x48c4_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel6x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                      \n\t" \
    "addq %%r10, "#input"                      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm18       \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm19       \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm20       \n\t" \

#define convKernel6x48c4_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel6x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2)

#define convKernel5x48c4_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel4x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                      \n\t" \
    "addq %%r10, "#input"                      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12       \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13       \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm14       \n\t" \

#define convKernel4x48c4_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel4x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2)

#define convKernel3x48c4_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel2x48c4Core_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "addq %%r10, "#input"                      \n\t" \
    "addq %%r10, "#input"                      \n\t" \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"      \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm6        \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm7        \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm8        \n\t" \

#define convKernel2x48c4_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    "vpbroadcastd ("#input"), %%zmm30          \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31   \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"      \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm0        \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm1        \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"      \n\t" \
    "vpdpbusd "#freg2", %%zmm30, %%zmm2        \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm3        \n\t" \
    "vmovups "#off2"(%[filter]), "#preg2"      \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm4        \n\t" \
    "vpdpbusd "#freg2", %%zmm31, %%zmm5        \n\t" \

#define convKernel1x48c4_0(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2) \
    convKernel1x48c4_1(input, freg0, freg1, freg2, off0, off1, off2, preg0, preg1, preg2)

#define convKernel2x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "movq (%[stepC16]), %%r10                     \n\t" \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "addq 0x8(%[stepC16]), "#input"               \n\t" \
    "movq 0x10(%[stepC16]), %%r10                 \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t" \

#define convKernel3x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel2x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "addq %%r10, "#input"                         \n\t" \
    "addq 0x18(%[stepC16]), "#input"              \n\t" \
    "movq 0x20(%[stepC16]), %%r10                 \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4           \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5           \n\t" \

#define convKernel4x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel3x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t" \

#define convKernel5x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel4x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "addq %%r10, "#input"                         \n\t" \
    "addq 0x28(%[stepC16]), "#input"              \n\t" \
    "movq 0x30(%[stepC16]), %%r10                 \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9           \n\t" \

#define convKernel6x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel5x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t" \

#define convKernel7x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel6x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "addq 0x38(%[stepC16]), "#input"              \n\t" \
    "movq 0x40(%[stepC16]), %%r10                 \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13          \n\t" \

#define convKernel8x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel7x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t" \

#define convKernel9x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel8x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "addq %%r10, "#input"                         \n\t" \
    "addq 0x48(%[stepC16]), "#input"              \n\t" \
    "movq 0x50(%[stepC16]), %%r10                 \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16          \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17          \n\t" \

#define convKernel12x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel9x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm20          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm21          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm22          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm23          \n\t"

#define convKernel11x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel9x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm20          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm21          \n\t" \

#define convKernel10x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel8x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16          \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17          \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t" \

#define convKernel9x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel7x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16          \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17          \n\t" \

#define convKernel8x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel6x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "addq 0x38(%[stepC16]), "#input"              \n\t" \
    "movq 0x40(%[stepC16]), %%r10                 \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t" \

#define convKernel7x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel5x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13          \n\t" \

#define convKernel6x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel4x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9           \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t"

#define convKernel5x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel3x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9           \n\t"

#define convKernel4x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel2x32_core(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4           \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5           \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t"

#define convKernel3x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "movq (%[stepC16]), %%r10                     \n\t" \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "addq 0x8(%[stepC16]), "#input"               \n\t" \
    "movq 0x10(%[stepC16]), %%r10                 \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4           \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5           \n\t"

#define convKernel2x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "movq (%[stepC16]), %%r10                     \n\t" \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t"

#define convKernel1x32c4_2(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t"

#define convKernel2x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t" \

#define convKernel3x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel2x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4           \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5           \n\t" \

#define convKernel4x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel3x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t" \

#define convKernel5x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel4x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "addq %%r10, "#input"                         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9           \n\t" \

#define convKernel6x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel5x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
     "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t" \

#define convKernel7x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel6x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13          \n\t" \

#define convKernel8x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel7x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t" \

#define convKernel9x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel8x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "addq %%r10, "#input"                         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16          \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17          \n\t" \

#define convKernel12x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel9x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm20          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm21          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm22          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm23          \n\t"

#define convKernel11x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel9x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm20          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm21          \n\t" \

#define convKernel10x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel8x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16          \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17          \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t"

#define convKernel9x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel7x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16          \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17          \n\t" \

#define convKernel8x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel6x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t"

#define convKernel7x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel5x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13          \n\t"

#define convKernel6x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel4x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9           \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t"

#define convKernel5x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel3x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9           \n\t"

#define convKernel4x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel2x32Core_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4           \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t"

#define convKernel3x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "addq %%r10, "#input"                         \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \
    "vpbroadcastd ("#input"), %%zmm30             \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4           \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5           \n\t"

#define convKernel2x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd ("#input", %%r10), %%zmm29      \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t"

#define convKernel1x32c4_1(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t"

#define convKernel12x32c4Core_0_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \

#define convKernel12x32c4Core_1_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x10("#input"), %%zmm29      \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \

#define convKernel12x32c4Core_2_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x20("#input"), %%zmm30             \n\t" \

#define convKernel12x32c4Core_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x30("#input"), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4           \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5           \n\t" \

#define convKernel12x32c4Core_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x40("#input"), %%zmm28             \n\t" \

#define convKernel12x32c4Core_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x50("#input"), %%zmm29      \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9           \n\t" \

#define convKernel12x32c4Core_6_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x60("#input"), %%zmm30             \n\t" \

#define convKernel12x32c4Core_7_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x70("#input"), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13          \n\t" \

#define convKernel12x32c4Core_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x80("#input"), %%zmm28         \n\t" \

#define convKernel12x32c4Core_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0x90("#input"), %%zmm29         \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16          \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17          \n\t" \

#define convKernel12x32c4Core_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0xA0("#input"), %%zmm30             \n\t" \

#define convKernel12x32c4Core_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpbroadcastd 0xB0("#input"), %%zmm31      \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm20          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm21          \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm22          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm23          \n\t"

#define convKernel12x32c4Core_6_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_6_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_7_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4Core_0_1_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_1_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \

#define convKernel12x32c4Core_0_2_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_1_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_2_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \

#define convKernel12x32c4Core_0_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_2_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \

#define convKernel12x32c4Core_0_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \

#define convKernel12x32c4Core_0_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4Core_0_6_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_6_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4Core_0_7_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_6_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_7_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4Core_0_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_7_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4Core_0_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4Core_0_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_6_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_1(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_1_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_2_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_6_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_2(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_1_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_2_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_6_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_3(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_2_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_6_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_4(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_6_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_5(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_6_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_6(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_6_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_7(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_6_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_7_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_8(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_7_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_9(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_10(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0_11(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    convKernel12x32c4Core_0_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "addq "#off2", "#input"                      \n\t" \
    convKernel12x32c4Core_11_0(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel12x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4_0_##idx(input, freg0, freg1, off0, off1, preg0, preg1, off2)

#define convKernel11x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4Core_0_10_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm20          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm21          \n\t"

#define convKernel10x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4Core_0_9_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpdpbusd "#freg0", %%zmm29, %%zmm18          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm19          \n\t"

#define convKernel9x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4Core_0_8_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm16          \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm17          \n\t"

#define convKernel8x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4Core_0_7_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpdpbusd "#freg0", %%zmm31, %%zmm14          \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm15          \n\t"

#define convKernel7x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4Core_0_6_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm12          \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm13          \n\t" \

#define convKernel6x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4Core_0_5_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm10          \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm11          \n\t" \

#define convKernel5x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4Core_0_4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm8           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm9           \n\t" \

#define convKernel4x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    convKernel12x32c4Core_0_3_0(input, freg0, freg1, off0, off1, preg0, preg1, off2) \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm31, %%zmm6           \n\t" \
    "vpdpbusd "#freg1", %%zmm31, %%zmm7           \n\t" \

#define convKernel3x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm29      \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpbroadcastd 0x20("#input"), %%zmm30             \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm30, %%zmm4           \n\t" \
    "vpdpbusd "#freg1", %%zmm30, %%zmm5           \n\t"

#define convKernel2x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vpbroadcastd 0x10("#input"), %%zmm29      \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t" \
    "vpdpbusd "#freg0", %%zmm29, %%zmm2           \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg1", %%zmm29, %%zmm3           \n\t" \

#define convKernel1x32c4_0(input, freg0, freg1, off0, off1, preg0, preg1, off2, idx) \
    "vpbroadcastd ("#input"), %%zmm28             \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"         \n\t" \
    "vmovups "#off1"(%[filter]), "#preg1"         \n\t" \
    "vpdpbusd "#freg0", %%zmm28, %%zmm0           \n\t" \
    "vpdpbusd "#freg1", %%zmm28, %%zmm1           \n\t"

#define convKernel3x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "movq (%[stepC16]), %%r10                        \n\t" \
    "movq 0x8(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2        \n\t" \

#define convKernel6x16c4Core_1(input, freg0, off0, preg0, rtype) \
    convKernel3x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x10(%[stepC16]), "#input"                  \n\t" \
    "movq 0x18(%[stepC16]), %%r10                        \n\t" \
    "movq 0x20(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"31      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4        \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"5        \n\t" \

#define convKernel9x16c4Core_1(input, freg0, off0, preg0, rtype) \
    convKernel6x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x28(%[stepC16]), "#input"                  \n\t" \
    "movq 0x30(%[stepC16]), %%r10                        \n\t" \
    "movq 0x38(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"8        \n\t" \

#define convKernel12x16c4Core_1(input, freg0, off0, preg0, rtype) \
    convKernel9x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x40(%[stepC16]), "#input"                  \n\t" \
    "movq 0x48(%[stepC16]), %%r10                        \n\t" \
    "movq 0x50(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"31      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10        \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"11        \n\t" \

#define convKernel15x16c4Core_1(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x58(%[stepC16]), "#input"                  \n\t" \
    "movq 0x60(%[stepC16]), %%r10                        \n\t" \
    "movq 0x68(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"13        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"14        \n\t" \

#define convKernel18x16c4Core_1(input, freg0, off0, preg0, rtype) \
    convKernel15x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x70(%[stepC16]), "#input"                  \n\t" \
    "movq 0x78(%[stepC16]), %%r10                        \n\t" \
    "movq 0x80(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"31      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"16        \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"17        \n\t" \

#define convKernel21x16c4Core_1(input, freg0, off0, preg0, rtype) \
    convKernel18x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x88(%[stepC16]), "#input"                  \n\t" \
    "movq 0x90(%[stepC16]), %%r10                        \n\t" \
    "movq 0x98(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"19        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"20        \n\t" \

#define convKernel24x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel21x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0xA0(%[stepC16]), "#input"                  \n\t" \
    "movq 0xA8(%[stepC16]), %%r10                        \n\t" \
    "movq 0xB0(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"31      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"21        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"22        \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"23        \n\t" \

#define convKernel23x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel21x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0xA0(%[stepC16]), "#input"                  \n\t" \
    "movq 0xA8(%[stepC16]), %%r10                        \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"21        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"22        \n\t" \

#define convKernel22x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel18x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x88(%[stepC16]), "#input"                  \n\t" \
    "movq 0x90(%[stepC16]), %%r10                        \n\t" \
    "movq 0x98(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18        \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0xA0(%[stepC16]), "#input"                  \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"19        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"20        \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"21        \n\t" \

#define convKernel21x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel21x16c4Core_1(input, freg0, off0, preg0, rtype) \

#define convKernel20x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel18x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x88(%[stepC16]), "#input"                  \n\t" \
    "movq 0x90(%[stepC16]), %%r10                        \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"19        \n\t" \

#define convKernel19x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel15x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x70(%[stepC16]), "#input"                  \n\t" \
    "movq 0x78(%[stepC16]), %%r10                        \n\t" \
    "movq 0x80(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"31      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15        \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x88(%[stepC16]), "#input"                  \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"16        \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"17        \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18        \n\t" \

#define convKernel18x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel18x16c4Core_1(input, freg0, off0, preg0, rtype) \

#define convKernel17x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel15x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x70(%[stepC16]), "#input"                  \n\t" \
    "movq 0x78(%[stepC16]), %%r10                        \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"16        \n\t" \

#define convKernel16x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x58(%[stepC16]), "#input"                  \n\t" \
    "movq 0x60(%[stepC16]), %%r10                        \n\t" \
    "movq 0x68(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12        \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x70(%[stepC16]), "#input"                  \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"13        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"14        \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15        \n\t" \

#define convKernel15x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel15x16c4Core_1(input, freg0, off0, preg0, rtype) \

#define convKernel14x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x58(%[stepC16]), "#input"                  \n\t" \
    "movq 0x60(%[stepC16]), %%r10                        \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"13        \n\t" \

#define convKernel13x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel9x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x40(%[stepC16]), "#input"                  \n\t" \
    "movq 0x48(%[stepC16]), %%r10                        \n\t" \
    "movq 0x50(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"31      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9        \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x58(%[stepC16]), "#input"                  \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10        \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"11        \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12        \n\t" \

#define convKernel12x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4Core_1(input, freg0, off0, preg0, rtype) \

#define convKernel11x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel9x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x40(%[stepC16]), "#input"                  \n\t" \
    "movq 0x48(%[stepC16]), %%r10                        \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10        \n\t" \

#define convKernel10x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel6x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x28(%[stepC16]), "#input"                  \n\t" \
    "movq 0x30(%[stepC16]), %%r10                        \n\t" \
    "movq 0x38(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x40(%[stepC16]), "#input"                  \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"8        \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9        \n\t" \

#define convKernel9x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel9x16c4Core_1(input, freg0, off0, preg0, rtype)

#define convKernel8x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel6x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x28(%[stepC16]), "#input"                  \n\t" \
    "movq 0x30(%[stepC16]), %%r10                        \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7        \n\t" \

#define convKernel7x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel3x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x10(%[stepC16]), "#input"                  \n\t" \
    "movq 0x18(%[stepC16]), %%r10                        \n\t" \
    "movq 0x20(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"31      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3        \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x28(%[stepC16]), "#input"                  \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4        \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"5        \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6        \n\t" \

#define convKernel6x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel6x16c4Core_1(input, freg0, off0, preg0, rtype) \
    
#define convKernel5x16c4_1(input, freg0, off0, preg0, rtype) \
    convKernel3x16c4Core_1(input, freg0, off0, preg0, rtype) \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x10(%[stepC16]), "#input"                  \n\t" \
    "movq 0x18(%[stepC16]), %%r10                        \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30             \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4        \n\t" \

#define convKernel4x16c4_1(input, freg0, off0, preg0, rtype) \
    "movq (%[stepC16]), %%r10                        \n\t" \
    "movq 0x8(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "addq 0x10(%[stepC16]), "#input"                  \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29      \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2        \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3        \n\t" \

#define convKernel3x16c4_1(input, freg0, off0, preg0, rtype) \
    "movq (%[stepC16]), %%r10                        \n\t" \
    "movq 0x8(%[stepC16]), %%r11                        \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r11), "#rtype"28      \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2        \n\t" \
    
#define convKernel2x16c4_1(input, freg0, off0, preg0, rtype) \
    "movq (%[stepC16]), %%r10                        \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1        \n\t" \

#define convKernel1x16c4_1(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t"

#define convKernel3x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "movq %%r10, %%r11                            \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "addq %%r10, %%r11                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"28   \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2        \n\t" \

#define convKernel6x16c4Core_0(input, freg0, off0, preg0, rtype) \
    convKernel3x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"31   \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4        \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"5        \n\t" \

#define convKernel9x16c4Core_0(input, freg0, off0, preg0, rtype) \
    convKernel6x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"28   \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"8        \n\t" \

#define convKernel12x16c4Core_0(input, freg0, off0, preg0, rtype) \
    convKernel9x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"31   \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10       \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"11       \n\t" \

#define convKernel15x16c4Core_0(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"28   \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12       \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"13       \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"14       \n\t" \

#define convKernel18x16c4Core_0(input, freg0, off0, preg0, rtype) \
    convKernel15x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"31   \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15       \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"16       \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"17       \n\t" \

#define convKernel21x16c4Core_0(input, freg0, off0, preg0, rtype) \
    convKernel18x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"28   \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18       \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"19       \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"20       \n\t" \

#define convKernel24x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel21x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"31   \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"21       \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"22       \n\t" \
    "vpdpbusd "#freg0", "#rtype"31, "#rtype"23       \n\t"

#define convKernel23x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel21x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"21       \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"22       \n\t" \

#define convKernel22x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel21x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"21       \n\t" \

#define convKernel21x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel21x16c4Core_0(input, freg0, off0, preg0, rtype) \

#define convKernel20x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel18x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18       \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"19       \n\t" \

#define convKernel19x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel18x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"18       \n\t" \

#define convKernel18x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel18x16c4Core_0(input, freg0, off0, preg0, rtype) \

#define convKernel17x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel15x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15       \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"16       \n\t" \

#define convKernel16x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel15x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"15       \n\t" \

#define convKernel15x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel15x16c4Core_0(input, freg0, off0, preg0, rtype) \

#define convKernel14x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12       \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"13       \n\t" \

#define convKernel13x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"12       \n\t" \

#define convKernel12x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel12x16c4Core_0(input, freg0, off0, preg0, rtype) \

#define convKernel11x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel9x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"10       \n\t" \

#define convKernel10x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel9x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"9        \n\t" \

#define convKernel9x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel9x16c4Core_0(input, freg0, off0, preg0, rtype) \

#define convKernel8x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel6x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"7        \n\t" \

#define convKernel7x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel6x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"6        \n\t" \

#define convKernel6x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel6x16c4Core_0(input, freg0, off0, preg0, rtype) \

#define convKernel5x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel3x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"30      \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3        \n\t" \
    "vpdpbusd "#freg0", "#rtype"30, "#rtype"4        \n\t" \

#define convKernel4x16c4_0(input, freg0, off0, preg0, rtype) \
    convKernel3x16c4Core_0(input, freg0, off0, preg0, rtype) \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "addq %%r11, "#input"                            \n\t" \
    "vpbroadcastd ("#input"), "#rtype"29             \n\t" \
    "vpdpbusd "#freg0", "#rtype"29, "#rtype"3        \n\t" \

#define convKernel3x16c4_0(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vpbroadcastd ("#input", %%r10, 2), "#rtype"28   \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1        \n\t" \
    "vpdpbusd "#freg0", "#rtype"28, "#rtype"2        \n\t" \

#define convKernel2x16c4_0(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vpbroadcastd ("#input", %%r10), "#rtype"27      \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t" \
    "vpdpbusd "#freg0", "#rtype"27, "#rtype"1        \n\t" \

#define convKernel1x16c4_0(input, freg0, off0, preg0, rtype) \
    "vpbroadcastd ("#input"), "#rtype"26             \n\t" \
    "vmovups "#off0"(%[filter]), "#preg0"            \n\t" \
    "vpdpbusd "#freg0", "#rtype"26, "#rtype"0        \n\t"

#define add_1x16(add, output, step) \
    ""#add" ("#output"), %%zmm0, %%zmm0                    \n\t" \

#define add_2x16(add, output, step) \
    add_1x16(add, output, step) \
    ""#add" 0x40("#output"), %%zmm1, %%zmm1                    \n\t" \

#define add_3x16(add, output, step) \
    add_2x16(add, output, step) \
    ""#add" 0x80("#output"),  %%zmm2, %%zmm2        \n\t"

#define add_4x16(add, output, step) \
    add_3x16(add, output, step) \
    ""#add" 0xC0("#output"),  %%zmm3, %%zmm3        \n\t"

#define add_5x16(add, output, step) \
    add_4x16(add, output, step) \
    ""#add" 0x100("#output"), %%zmm4, %%zmm4        \n\t"

#define add_6x16(add, output, step) \
    add_5x16(add, output, step) \
    ""#add" 0x140("#output"), %%zmm5, %%zmm5        \n\t"

#define add_7x16(add, output, step) \
    add_6x16(add, output, step) \
    ""#add" 0x180("#output"), %%zmm6, %%zmm6        \n\t"

#define add_8x16(add, output, step) \
    add_7x16(add, output, step) \
    ""#add" 0x1C0("#output"), %%zmm7, %%zmm7        \n\t"

#define add_9x16(add, output, step) \
    add_8x16(add, output, step) \
    ""#add" 0x200("#output"), %%zmm8, %%zmm8        \n\t"

#define add_10x16(add, output, step) \
    add_9x16(add, output, step) \
    ""#add" 0x240("#output"), %%zmm9, %%zmm9        \n\t"

#define add_11x16(add, output, step) \
    add_10x16(add, output, step) \
    ""#add" 0x280("#output"), %%zmm10, %%zmm10      \n\t"

#define add_12x16(add, output, step) \
    add_11x16(add, output, step) \
    ""#add" 0x2C0("#output"), %%zmm11, %%zmm11      \n\t"

#define add_13x16(add, output, step) \
    add_12x16(add, output, step) \
    ""#add" 0x300("#output"), %%zmm12, %%zmm12      \n\t"

#define add_14x16(add, output, step) \
    add_13x16(add, output, step) \
    ""#add" 0x340("#output"), %%zmm13, %%zmm13      \n\t"

#define add_15x16(add, output, step) \
    add_14x16(add, output, step) \
    ""#add" 0x380("#output"), %%zmm14, %%zmm14      \n\t"

#define add_16x16(add, output, step) \
    add_15x16(add, output, step) \
    ""#add" 0x3C0("#output"), %%zmm15, %%zmm15      \n\t"

#define add_17x16(add, output, step) \
    add_16x16(add, output, step) \
    ""#add" 0x400("#output"), %%zmm16, %%zmm16      \n\t"

#define add_18x16(add, output, step) \
    add_17x16(add, output, step) \
    ""#add" 0x440("#output"), %%zmm17, %%zmm17      \n\t"

#define add_19x16(add, output, step) \
    add_18x16(add, output, step) \
    ""#add" 0x480("#output"), %%zmm18, %%zmm18      \n\t"

#define add_20x16(add, output, step) \
    add_19x16(add, output, step) \
    ""#add" 0x4C0("#output"), %%zmm19, %%zmm19      \n\t"

#define add_21x16(add, output, step) \
    add_20x16(add, output, step) \
    ""#add" 0x500("#output"), %%zmm20, %%zmm20      \n\t"

#define add_22x16(add, output, step) \
    add_21x16(add, output, step) \
    ""#add" 0x540("#output"), %%zmm21, %%zmm21      \n\t"

#define add_23x16(add, output, step) \
    add_22x16(add, output, step) \
    ""#add" 0x580("#output"), %%zmm22, %%zmm22      \n\t"

#define add_24x16(add, output, step) \
    add_23x16(add, output, step) \
    ""#add" 0x5C0("#output"), %%zmm23, %%zmm23      \n\t"

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

#define add_9x8(add, output, step) \
    add_8x8(add, output, step) \
    ""#add" 0x100("#output"), %%ymm8, %%ymm8        \n\t"

#define add_10x8(add, output, step) \
    add_9x8(add, output, step) \
    ""#add" 0x120("#output"), %%ymm9, %%ymm9        \n\t"

#define add_11x8(add, output, step) \
    add_10x8(add, output, step) \
    ""#add" 0x140("#output"), %%ymm10, %%ymm10      \n\t"

#define add_12x8(add, output, step) \
    add_11x8(add, output, step) \
    ""#add" 0x160("#output"), %%ymm11, %%ymm11      \n\t"

#define add_13x8(add, output, step) \
    add_12x8(add, output, step) \
    ""#add" 0x180("#output"), %%ymm12, %%ymm12      \n\t"

#define add_14x8(add, output, step) \
    add_13x8(add, output, step) \
    ""#add" 0x1A0("#output"), %%ymm13, %%ymm13      \n\t"

#define add_15x8(add, output, step) \
    add_14x8(add, output, step) \
    ""#add" 0x1C0("#output"), %%ymm14, %%ymm14      \n\t"

#define add_16x8(add, output, step) \
    add_15x8(add, output, step) \
    ""#add" 0x1E0("#output"), %%ymm15, %%ymm15      \n\t"

#define add_17x8(add, output, step) \
    add_16x8(add, output, step) \
    ""#add" 0x200("#output"), %%ymm16, %%ymm16      \n\t"

#define add_18x8(add, output, step) \
    add_17x8(add, output, step) \
    ""#add" 0x220("#output"), %%ymm17, %%ymm17      \n\t"

#define add_19x8(add, output, step) \
    add_18x8(add, output, step) \
    ""#add" 0x240("#output"), %%ymm18, %%ymm18      \n\t"

#define add_20x8(add, output, step) \
    add_19x8(add, output, step) \
    ""#add" 0x260("#output"), %%ymm19, %%ymm19      \n\t"

#define add_21x8(add, output, step) \
    add_20x8(add, output, step) \
    ""#add" 0x280("#output"), %%ymm20, %%ymm20      \n\t"

#define add_22x8(add, output, step) \
    add_21x8(add, output, step) \
    ""#add" 0x2A0("#output"), %%ymm21, %%ymm21      \n\t"

#define add_23x8(add, output, step) \
    add_22x8(add, output, step) \
    ""#add" 0x2C0("#output"), %%ymm22, %%ymm22      \n\t"

#define add_24x8(add, output, step) \
    add_23x8(add, output, step) \
    ""#add" 0x2E0("#output"), %%ymm23, %%ymm23      \n\t"

#define add_1x32(add, output, step) \
    add_1x16(add, output, step) \
    ""#add" ("#output", "#step"), %%zmm1, %%zmm1             \n\t"

#define add_2x32(add, output, step) \
    add_1x32(add, output, step) \
    ""#add" 0x40("#output"), %%zmm2, %%zmm2                \n\t" \
    ""#add" 0x40("#output", "#step"), %%zmm3, %%zmm3         \n\t"

#define add_3x32(add, output, step) \
    add_2x32(add, output, step) \
    ""#add" 0x80("#output"), %%zmm4, %%zmm4                \n\t" \
    ""#add" 0x80("#output", "#step"), %%zmm5, %%zmm5         \n\t"

#define add_4x32(add, output, step) \
    add_3x32(add, output, step) \
    ""#add" 0xC0("#output"), %%zmm6, %%zmm6                \n\t" \
    ""#add" 0xC0("#output", "#step"), %%zmm7, %%zmm7         \n\t"

#define add_5x32(add, output, step) \
    add_4x32(add, output, step) \
    ""#add" 0x100("#output"), %%zmm8, %%zmm8               \n\t" \
    ""#add" 0x100("#output", "#step"), %%zmm9, %%zmm9        \n\t"

#define add_6x32(add, output, step) \
    add_5x32(add, output, step) \
    ""#add" 0x140("#output"), %%zmm10, %%zmm10             \n\t" \
    ""#add" 0x140("#output", "#step"), %%zmm11, %%zmm11      \n\t"

#define add_7x32(add, output, step) \
    add_6x32(add, output, step) \
    ""#add" 0x180("#output"), %%zmm12, %%zmm12             \n\t" \
    ""#add" 0x180("#output", "#step"), %%zmm13, %%zmm13      \n\t"

#define add_8x32(add, output, step) \
    add_7x32(add, output, step) \
    ""#add" 0x1C0("#output"), %%zmm14, %%zmm14             \n\t" \
    ""#add" 0x1C0("#output", "#step"), %%zmm15, %%zmm15      \n\t"

#define add_9x32(add, output, step) \
    add_8x32(add, output, step) \
    ""#add" 0x200("#output"), %%zmm16, %%zmm16             \n\t" \
    ""#add" 0x200("#output", "#step"), %%zmm17, %%zmm17      \n\t"

#define add_10x32(add, output, step) \
    add_9x32(add, output, step) \
    ""#add" 0x240("#output"), %%zmm18, %%zmm18             \n\t" \
    ""#add" 0x240("#output", "#step"), %%zmm19, %%zmm19      \n\t"

#define add_11x32(add, output, step) \
    add_10x32(add, output, step) \
    ""#add" 0x280("#output"), %%zmm20, %%zmm20             \n\t" \
    ""#add" 0x280("#output", "#step"), %%zmm21, %%zmm21      \n\t"

#define add_12x32(add, output, step) \
    add_11x32(add, output, step) \
    ""#add" 0x2C0("#output"), %%zmm22, %%zmm22             \n\t" \
    ""#add" 0x2C0("#output", "#step"), %%zmm23, %%zmm23      \n\t"

#define add_1x48(add, output, step) \
    add_1x32(add, output, step) \
    ""#add" ("#output", "#step", 2), %%zmm2, %%zmm2         \n\t"

#define add_2x48(add, output, step) \
    add_1x48(add, output, step) \
    ""#add" 0x40("#output"), %%zmm3, %%zmm3                    \n\t" \
    ""#add" 0x40("#output", "#step"), %%zmm4, %%zmm4             \n\t" \
    ""#add" 0x40("#output", "#step", 2), %%zmm5, %%zmm5         \n\t"

#define add_3x48(add, output, step) \
    add_2x48(add, output, step) \
    ""#add" 0x80("#output"), %%zmm6, %%zmm6                    \n\t" \
    ""#add" 0x80("#output", "#step"), %%zmm7, %%zmm7             \n\t" \
    ""#add" 0x80("#output", "#step", 2), %%zmm8, %%zmm8         \n\t"

#define add_4x48(add, output, step) \
    add_3x48(add, output, step) \
    ""#add" 0xC0("#output"), %%zmm9, %%zmm9                    \n\t" \
    ""#add" 0xC0("#output", "#step"), %%zmm10, %%zmm10             \n\t" \
    ""#add" 0xC0("#output", "#step", 2), %%zmm11, %%zmm11         \n\t"

#define add_5x48(add, output, step) \
    add_4x48(add, output, step) \
    ""#add" 0x100("#output"), %%zmm12, %%zmm12                    \n\t" \
    ""#add" 0x100("#output", "#step"), %%zmm13, %%zmm13             \n\t" \
    ""#add" 0x100("#output", "#step", 2), %%zmm14, %%zmm14         \n\t"

#define add_6x48(add, output, step) \
    add_5x48(add, output, step) \
    ""#add" 0x140("#output"), %%zmm15, %%zmm15                    \n\t" \
    ""#add" 0x140("#output", "#step"), %%zmm16, %%zmm16             \n\t" \
    ""#add" 0x140("#output", "#step", 2), %%zmm17, %%zmm17         \n\t"

#define add_7x48(add, output, step) \
    add_6x48(add, output, step) \
    ""#add" 0x180("#output"), %%zmm18, %%zmm18                    \n\t" \
    ""#add" 0x180("#output", "#step"), %%zmm19, %%zmm19             \n\t" \
    ""#add" 0x180("#output", "#step", 2), %%zmm20, %%zmm20         \n\t"

#define add_8x48(add, output, step) \
    add_7x48(add, output, step) \
    ""#add" 0x1C0("#output"), %%zmm21, %%zmm21                    \n\t" \
    ""#add" 0x1C0("#output", "#step"), %%zmm22, %%zmm22             \n\t" \
    ""#add" 0x1C0("#output", "#step", 2), %%zmm23, %%zmm23         \n\t"

#define store_1x16(output, step) \
    "vmovups %%zmm0, ("#output")                    \n\t"

#define store_2x16(output, step) \
    store_1x16(output, step) \
    "vmovups %%zmm1,  0x40("#output")               \n\t"

#define store_3x16(output, step) \
    store_2x16(output, step) \
    "vmovups %%zmm2,  0x80("#output")               \n\t"

#define store_4x16(output, step) \
    store_3x16(output, step) \
    "vmovups %%zmm3,  0xC0("#output")               \n\t"

#define store_5x16(output, step) \
    store_4x16(output, step) \
    "vmovups %%zmm4,  0x100("#output")              \n\t"

#define store_6x16(output, step) \
    store_5x16(output, step) \
    "vmovups %%zmm5,  0x140("#output")              \n\t"

#define store_7x16(output, step) \
    store_6x16(output, step) \
    "vmovups %%zmm6,  0x180("#output")              \n\t"

#define store_8x16(output, step) \
    store_7x16(output, step) \
    "vmovups %%zmm7,  0x1C0("#output")              \n\t"

#define store_9x16(output, step) \
    store_8x16(output, step) \
    "vmovups %%zmm8,  0x200("#output")              \n\t"

#define store_10x16(output, step) \
    store_9x16(output, step) \
    "vmovups %%zmm9,  0x240("#output")              \n\t"

#define store_11x16(output, step) \
    store_10x16(output, step) \
    "vmovups %%zmm10, 0x280("#output")              \n\t"

#define store_12x16(output, step) \
    store_11x16(output, step) \
    "vmovups %%zmm11, 0x2C0("#output")              \n\t"

#define store_13x16(output, step) \
    store_12x16(output, step) \
    "vmovups %%zmm12, 0x300("#output")              \n\t"

#define store_14x16(output, step) \
    store_13x16(output, step) \
    "vmovups %%zmm13, 0x340("#output")              \n\t"

#define store_15x16(output, step) \
    store_14x16(output, step) \
    "vmovups %%zmm14, 0x380("#output")              \n\t"

#define store_16x16(output, step) \
    store_15x16(output, step) \
    "vmovups %%zmm15, 0x3C0("#output")              \n\t"

#define store_17x16(output, step) \
    store_16x16(output, step) \
    "vmovups %%zmm16, 0x400("#output")              \n\t"

#define store_18x16(output, step) \
    store_17x16(output, step) \
    "vmovups %%zmm17, 0x440("#output")              \n\t"

#define store_19x16(output, step) \
    store_18x16(output, step) \
    "vmovups %%zmm18, 0x480("#output")              \n\t"

#define store_20x16(output, step) \
    store_19x16(output, step) \
    "vmovups %%zmm19, 0x4C0("#output")              \n\t"

#define store_21x16(output, step) \
    store_20x16(output, step) \
    "vmovups %%zmm20, 0x500("#output")              \n\t"

#define store_22x16(output, step) \
    store_21x16(output, step) \
    "vmovups %%zmm21, 0x540("#output")              \n\t"

#define store_23x16(output, step) \
    store_22x16(output, step) \
    "vmovups %%zmm22, 0x580("#output")              \n\t"

#define store_24x16(output, step) \
    store_23x16(output, step) \
    "vmovups %%zmm23, 0x5C0("#output")              \n\t"

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

#define store_9x8(output, step) \
    store_8x8(output, step) \
    "vmovups %%ymm8,  0x100("#output")              \n\t"

#define store_10x8(output, step) \
    store_9x8(output, step) \
    "vmovups %%ymm9,  0x120("#output")              \n\t"

#define store_11x8(output, step) \
    store_10x8(output, step) \
    "vmovups %%ymm10, 0x140("#output")              \n\t"

#define store_12x8(output, step) \
    store_11x8(output, step) \
    "vmovups %%ymm11, 0x160("#output")              \n\t"

#define store_13x8(output, step) \
    store_12x8(output, step) \
    "vmovups %%ymm12, 0x180("#output")              \n\t"

#define store_14x8(output, step) \
    store_13x8(output, step) \
    "vmovups %%ymm13, 0x1A0("#output")              \n\t"

#define store_15x8(output, step) \
    store_14x8(output, step) \
    "vmovups %%ymm14, 0x1C0("#output")              \n\t"

#define store_16x8(output, step) \
    store_15x8(output, step) \
    "vmovups %%ymm15, 0x1E0("#output")              \n\t"

#define store_17x8(output, step) \
    store_16x8(output, step) \
    "vmovups %%ymm16, 0x200("#output")              \n\t"

#define store_18x8(output, step) \
    store_17x8(output, step) \
    "vmovups %%ymm17, 0x220("#output")              \n\t"

#define store_19x8(output, step) \
    store_18x8(output, step) \
    "vmovups %%ymm18, 0x240("#output")              \n\t"

#define store_20x8(output, step) \
    store_19x8(output, step) \
    "vmovups %%ymm19, 0x260("#output")              \n\t"

#define store_21x8(output, step) \
    store_20x8(output, step) \
    "vmovups %%ymm20, 0x280("#output")              \n\t"

#define store_22x8(output, step) \
    store_21x8(output, step) \
    "vmovups %%ymm21, 0x2A0("#output")              \n\t"

#define store_23x8(output, step) \
    store_22x8(output, step) \
    "vmovups %%ymm22, 0x2C0("#output")              \n\t"

#define store_24x8(output, step) \
    store_23x8(output, step) \
    "vmovups %%ymm23, 0x2E0("#output")              \n\t"

#define store_1x32(output, step) \
    "vmovups %%zmm0, ("#output")                           \n\t" \
    "vmovups %%zmm1, ("#output", "#step")                    \n\t"

#define store_2x32(output, step) \
    store_1x32(output, step) \
    "vmovups %%zmm2, 0x40("#output")                       \n\t" \
    "vmovups %%zmm3, 0x40("#output", "#step")                \n\t"

#define store_3x32(output, step) \
    store_2x32(output, step) \
    "vmovups %%zmm4, 0x80("#output")                       \n\t" \
    "vmovups %%zmm5, 0x80("#output", "#step")                \n\t"

#define store_4x32(output, step) \
    store_3x32(output, step) \
    "vmovups %%zmm6, 0xC0("#output")                       \n\t" \
    "vmovups %%zmm7, 0xC0("#output", "#step")                \n\t"

#define store_5x32(output, step) \
    store_4x32(output, step) \
    "vmovups %%zmm8, 0x100("#output")                      \n\t" \
    "vmovups %%zmm9, 0x100("#output", "#step")               \n\t"

#define store_6x32(output, step) \
    store_5x32(output, step) \
    "vmovups %%zmm10, 0x140("#output")                     \n\t" \
    "vmovups %%zmm11, 0x140("#output", "#step")              \n\t"

#define store_7x32(output, step) \
    store_6x32(output, step) \
    "vmovups %%zmm12, 0x180("#output")                     \n\t" \
    "vmovups %%zmm13, 0x180("#output", "#step")              \n\t"

#define store_8x32(output, step) \
    store_7x32(output, step) \
    "vmovups %%zmm14, 0x1C0("#output")                     \n\t" \
    "vmovups %%zmm15, 0x1C0("#output", "#step")              \n\t"

#define store_9x32(output, step) \
    store_8x32(output, step) \
    "vmovups %%zmm16, 0x200("#output")                     \n\t" \
    "vmovups %%zmm17, 0x200("#output", "#step")              \n\t"

#define store_10x32(output, step) \
    store_9x32(output, step) \
    "vmovups %%zmm18, 0x240("#output")                     \n\t" \
    "vmovups %%zmm19, 0x240("#output", "#step")              \n\t"

#define store_11x32(output, step) \
    store_10x32(output, step) \
    "vmovups %%zmm20, 0x280("#output")                     \n\t" \
    "vmovups %%zmm21, 0x280("#output", "#step")              \n\t"

#define store_12x32(output, step) \
    store_11x32(output, step) \
    "vmovups %%zmm22, 0x2C0("#output")                     \n\t" \
    "vmovups %%zmm23, 0x2C0("#output", "#step")              \n\t"

#define store_1x48(output, step) \
    "vmovups %%zmm0, ("#output")                              \n\t" \
    "vmovups %%zmm1, ("#output", "#step")                       \n\t" \
    "vmovups %%zmm2, ("#output", "#step", 2)                    \n\t"

#define store_2x48(output, step) \
    store_1x48(output, step) \
    "vmovups %%zmm3, 0x40("#output")                          \n\t" \
    "vmovups %%zmm4, 0x40("#output", "#step")                   \n\t" \
    "vmovups %%zmm5, 0x40("#output", "#step", 2)                \n\t"

#define store_3x48(output, step) \
    store_2x48(output, step) \
    "vmovups %%zmm6, 0x80("#output")                          \n\t" \
    "vmovups %%zmm7, 0x80("#output", "#step")                   \n\t" \
    "vmovups %%zmm8, 0x80("#output", "#step", 2)                \n\t"

#define store_4x48(output, step) \
    store_3x48(output, step) \
    "vmovups %%zmm9, 0xC0("#output")                          \n\t" \
    "vmovups %%zmm10, 0xC0("#output", "#step")                  \n\t" \
    "vmovups %%zmm11, 0xC0("#output", "#step", 2)               \n\t"

#define store_5x48(output, step) \
    store_4x48(output, step) \
    "vmovups %%zmm12, 0x100("#output")                        \n\t" \
    "vmovups %%zmm13, 0x100("#output", "#step")                 \n\t" \
    "vmovups %%zmm14, 0x100("#output", "#step", 2)              \n\t"

#define store_6x48(output, step) \
    store_5x48(output, step) \
    "vmovups %%zmm15, 0x140("#output")                        \n\t" \
    "vmovups %%zmm16, 0x140("#output", "#step")                 \n\t" \
    "vmovups %%zmm17, 0x140("#output", "#step", 2)              \n\t"

#define store_7x48(output, step) \
    store_6x48(output, step) \
    "vmovups %%zmm18, 0x180("#output")                        \n\t" \
    "vmovups %%zmm19, 0x180("#output", "#step")                 \n\t" \
    "vmovups %%zmm20, 0x180("#output", "#step", 2)              \n\t"

#define store_8x48(output, step) \
    store_7x48(output, step) \
    "vmovups %%zmm21, 0x1C0("#output")                        \n\t" \
    "vmovups %%zmm22, 0x1C0("#output", "#step")                 \n\t" \
    "vmovups %%zmm23, 0x1C0("#output", "#step", 2)              \n\t"

#define convKernelForLoopXx48(rnum, wsize, idx, cross) \
    __asm__ __volatile__("vmovups (%[filter]), %%zmm24           \n\t" \
                         "vmovups 0x40(%[filter]), %%zmm25       \n\t" \
                         "vmovups 0x80(%[filter]), %%zmm26       \n\t" \
                         "addq $0xC0, %[filter]                  \n\t" \
                         "mov $1, %%eax                          \n\t" \
                         "vmovd %%eax, %%xmm0                    \n\t" \
                         "vpbroadcastw %%xmm0, %%zmm31           \n\t" \
                         "movq %[flags], %%rax                   \n\t" \
                         "andq $0x1, %%rax                       \n\t" \
                         "jne 0f                                 \n\t" \
                         load3BiasToZmmRegs(rnum, %[bias])             \
                         "jmp 1f                                 \n\t" \
                         ".align 16                              \n\t" \
                         "0:                                     \n\t" \
                         clear##rnum##Regs(%%zmm)                      \
                         ".align 16                              \n\t" \
                         "1:                                     \n\t" \
                         : [filter] "+r" (c.filter)                    \
                         : [bias] "r" (c.bias),                        \
                           [flags] "r" (c.flags)                       \
                         : "%rax",                                     \
                           "%zmm0", "%zmm1","%zmm2", "%zmm3", "%zmm4", "%zmm5",        \
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",     \
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", \
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                           "%zmm24", "%zmm25", "%zmm26", "memory", "cc");              \
    __asm__ __volatile__("movq (%[stepC16]), %%r10                                    \n\t" \
                         ".align 16                                                   \n\t" \
                         "1:                                                          \n\t" \
                         "mov %[kh], %%rbx                                            \n\t" \
                         ".align 16                                                   \n\t" \
                         "2:                                                          \n\t" \
                         "mov %[kw], %%r9                                             \n\t" \
                         ".align 16                                                   \n\t" \
                         "3:                                                          \n\t" \
                         "movq %[input], %%rax                                        \n\t" \
                         convKernel##wsize##x48c4_##cross(%%rax, %%zmm24, %%zmm25, %%zmm26, \
                            0x0, 0x40, 0x80, %%zmm27, %%zmm28, %%zmm29)                     \
                         "movq %[input], %%rax                                        \n\t" \
                         "addq $0x4, %%rax                                            \n\t" \
                         convKernel##wsize##x48c4_##cross(%%rax, %%zmm27, %%zmm28, %%zmm29, \
                            0xC0, 0x100, 0x140, %%zmm24, %%zmm25, %%zmm26)                  \
                         "movq %[input], %%rax                                        \n\t" \
                         "addq $0x8, %%rax                                            \n\t" \
                         convKernel##wsize##x48c4_##cross(%%rax, %%zmm24, %%zmm25, %%zmm26, \
                            0x180, 0x1C0, 0x200, %%zmm27, %%zmm28, %%zmm29)                 \
                         "movq %[input], %%rax                                        \n\t" \
                         "addq $0xC, %%rax                                            \n\t" \
                         convKernel##wsize##x48c4_##cross(%%rax, %%zmm27, %%zmm28, %%zmm29, \
                            0x240, 0x280, 0x2C0, %%zmm24, %%zmm25, %%zmm26)                 \
                         "addq $0x300, %[filter]                                      \n\t" \
                         "addq %[dilateW], %[input]                                   \n\t" \
                         "dec %%r9                                                    \n\t" \
                         "jg 3b                                                       \n\t" \
                         "addq %[dilateH], %[input]                                   \n\t" \
                         "dec %%rbx                                                   \n\t" \
                         "jg 2b                                                       \n\t" \
                         "addq %[fStep], %[input]                                     \n\t" \
                         "subq $0x10, %%rcx                                           \n\t" \
                         "cmpq $0x10, %%rcx                                           \n\t" \
                         "jge 1b                                                      \n\t" \
                         "subq %[fStep], %[input]                                     \n\t" \
                         "addq %[f8Step], %[input]                                    \n\t" \
                         ".align 16                                                   \n\t" \
                         "4:                                                          \n\t" \
                         : "+c" (c.ic),                                                     \
                           [input] "+r" (c.input),                                          \
                           [filter] "+r" (c.filter)                                         \
                         : [kh] "r" (c.kh),                                                 \
                           [kw] "r" (c.kw),                                                 \
                           [stepC16] "r" (c.stepC16),                                       \
                           [dilateW] "r" (c.dilateW),                                       \
                           [dilateH] "r" (c.dilateH),                                       \
                           [fStep] "r" (c.fStep),                                           \
                           [f8Step] "r" (c.f8Step)                                          \
                         : "%rax", "%rbx", "%r9", "%r10",                                   \
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",            \
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",          \
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",      \
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",      \
                           "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",      \
                           "%zmm30", "%zmm31", "memory", "cc");                             \
    __asm__ __volatile__("movq %[output], %%rax                             \n\t" \
                         "movq %[ostepC16], %%rbx                           \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x1, %%rcx                                   \n\t" \
                         "je 0f                                             \n\t" \
                         add_##wsize##x48(vpaddd, %%rax, %%rbx) \
                         ".align 16                                         \n\t" \
                         "0:                                                \n\t" \
                         "cmpq $0x0, %[scale]                               \n\t" \
                         "jne 1f                                            \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%zmm, vpxord, vpmaxsd) \
                         "jmp 4f                                            \n\t" \
                         ".align 16                                         \n\t" \
                         "1:                                                \n\t" \
                         convert##rnum##RegsI32ToF32(%[scale], %%zmm) \
                         ".align 16                                         \n\t" \
                         "2:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x2, %%rcx                                   \n\t" \
                         "je 3f                                             \n\t" \
                         add_##wsize##x48(vaddps, %[eltwise], %%rbx) \
                         ".align 16                                         \n\t" \
                         "3:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%zmm, vxorps, vmaxps) \
                         ".align 16                                         \n\t" \
                         "4:                                                \n\t" \
                         store_##wsize##x48(%%rax, %%rbx) \
                         : \
                         : [output] "r" (c.output), \
                           [eltwise] "r" (c.eltwise), \
                           [ostepC16] "r" (c.ostepC16), \
                           [flags] "r" (c.flags), \
                           [scale] "r" (c.scale) \
                         : "%rax", "%rbx", "%rcx", \
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", \
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", \
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", \
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                           "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", \
                           "%zmm30", "%zmm31", "memory", "cc");

#define convKernelForLoopXx32(rnum, wsize, idx, cross) \
     __asm__ __volatile__("vmovups (%[filter]), %%zmm24         \n\t" \
                          "vmovups 0x40(%[filter]), %%zmm25     \n\t" \
                          "addq $0x80, %[filter]                \n\t" \
                          "mov $1, %%eax                        \n\t" \
                          "vmovd %%eax, %%xmm0                  \n\t" \
                          "vpbroadcastw %%xmm0, %%zmm31         \n\t" \
                          "movq %[flags], %%rax                 \n\t" \
                          "andq $0x1, %%rax                     \n\t" \
                          "jne 0f                               \n\t" \
                          load2BiasToZmmRegs(rnum, %[bias])             \
                          "jmp 1f                               \n\t" \
                          ".align 16                            \n\t" \
                          "0:                                   \n\t" \
                          clear##rnum##Regs(%%zmm)                    \
                          ".align 16                            \n\t" \
                          "1:                                   \n\t" \
                          : [filter] "+r" (c.filter)                  \
                          : [bias] "r" (c.bias),                      \
                            [flags] "r" (c.flags)                     \
                          : "%rax",                                   \
                            "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",       \
                            "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",     \
                            "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", \
                            "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                            "memory", "cc"); \
     __asm__ __volatile__("cmpq $0, %%rcx                                   \n\t" \
                          "je 4f                                              \n\t" \
                          "movq (%[stepC16]), %%r10                            \n\t" \
                          ".align 16                                           \n\t" \
                          "1:                                                  \n\t" \
                          "mov %[kh], %%rbx                                    \n\t" \
                          ".align 16                                           \n\t" \
                          "2:                                                  \n\t" \
                          "mov %[kw], %%r9                                     \n\t" \
                          ".align 16                                           \n\t" \
                          "3:                                                  \n\t" \
                          "movq %[input], %%rax                                \n\t" \
                          convKernel##wsize##x32c4_##cross(                          \
                            %%rax, %%zmm24, %%zmm25, 0x0, 0x40, %%zmm26, %%zmm27, %[off], idx)    \
                          "movq %[input], %%rax                                \n\t" \
                          "addq $0x4, %%rax                                    \n\t" \
                          convKernel##wsize##x32c4_##cross(                          \
                            %%rax, %%zmm26, %%zmm27, 0x80, 0xC0, %%zmm24, %%zmm25, %[off], idx)   \
                          "movq %[input], %%rax                                \n\t" \
                          "addq $0x8, %%rax                                    \n\t" \
                          convKernel##wsize##x32c4_##cross(                          \
                            %%rax, %%zmm24, %%zmm25, 0x100, 0x140, %%zmm26, %%zmm27, %[off], idx) \
                          "movq %[input], %%rax                                \n\t" \
                          "addq $0xC, %%rax                                    \n\t" \
                          convKernel##wsize##x32c4_##cross(                          \
                            %%rax, %%zmm26, %%zmm27, 0x180, 0x1C0, %%zmm24, %%zmm25, %[off], idx) \
                          "addq $0x200, %[filter]                              \n\t" \
                          "addq %[dilateW], %[input]                           \n\t" \
                          "dec %%r9                                            \n\t" \
                          "jg 3b                                               \n\t" \
                          "addq %[dilateH], %[input]                           \n\t" \
                          "dec %%rbx                                           \n\t" \
                          "jg 2b                                               \n\t" \
                          "addq %[fStep], %[input]                             \n\t" \
                          "subq $0x10, %%rcx                                   \n\t" \
                          "cmpq $0x10, %%rcx                                   \n\t" \
                          "jge 1b                                              \n\t" \
                          "subq %[fStep], %[input]                             \n\t" \
                          "addq %[f8Step], %[input]                            \n\t" \
                          ".align 16                                           \n\t" \
                          "4:                                                  \n\t" \
                          : "+c" (c.ic),               \
                            [input] "+r" (c.input),    \
                            [filter] "+r" (c.filter)   \
                          : [kh] "r" (c.kh),           \
                            [kw] "r" (c.kw),           \
                            [stepC16] "r" (c.stepC16), \
                            [dilateW] "r" (c.dilateW), \
                            [dilateH] "r" (c.dilateH), \
                            [fStep] "r" (c.fStep),     \
                            [off] "r" (c.off),     \
                            [f8Step] "r" (c.f8Step)    \
                          : "%rax", "%rbx", "%r9", "%r10",                              \
                            "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",       \
                            "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",     \
                            "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", \
                            "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                            "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", \
                            "%zmm30", "%zmm31", "memory", "cc");                        \
    __asm__ __volatile__("movq %[output], %%rax                             \n\t" \
                         "movq %[ostepC16], %%rbx                           \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x1, %%rcx                                   \n\t" \
                         "je 0f                                             \n\t" \
                         add_##wsize##x32(vpaddd, %%rax, %%rbx) \
                         ".align 16                                         \n\t" \
                         "0:                                                \n\t" \
                         "cmpq $0x0, %[scale]                               \n\t" \
                         "jne 1f                                            \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%zmm, vpxord, vpmaxsd) \
                         "jmp 4f                                            \n\t" \
                         ".align 16                                         \n\t" \
                         "1:                                                \n\t" \
                         convert##rnum##RegsI32ToF32(%[scale], %%zmm) \
                         ".align 16                                         \n\t" \
                         "2:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x2, %%rcx                                   \n\t" \
                         "je 3f                                             \n\t" \
                         add_##wsize##x32(vaddps, %[eltwise], %%rbx) \
                         ".align 16                                         \n\t" \
                         "3:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(%%zmm, vxorps, vmaxps) \
                         ".align 16                                         \n\t" \
                         "4:                                                \n\t" \
                         store_##wsize##x32(%%rax, %%rbx) \
                         : \
                         : [output] "r" (c.output), \
                           [eltwise] "r" (c.eltwise), \
                           [ostepC16] "r" (c.ostepC16), \
                           [flags] "r" (c.flags), \
                           [scale] "r" (c.scale) \
                         : "%rax", "%rbx", "%rcx", \
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", \
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", \
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", \
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                           "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", \
                           "%zmm30", "%zmm31", "memory", "cc");

#define convKernelForLoopXx16(rnum, wsize, nSize, rtype, idx, cross, off0, off1, off2, off3, off4) \
    __asm__ __volatile__("vmovups (%[filter]), "#rtype"24              \n\t" \
                         "addq $"#off1", %[filter]                     \n\t" \
                         "mov $1, %%eax                                \n\t" \
                         "vmovd %%eax, %%xmm0                          \n\t" \
                         "vpbroadcastw %%xmm0, "#rtype"31              \n\t" \
                         "movq %[flags], %%rax                         \n\t" \
                         "andq $0x1, %%rax                             \n\t" \
                         "jne 0f                                       \n\t" \
                         load1BiasToRegs(rnum, %[bias], rtype)               \
                         "jmp 1f                                       \n\t" \
                         ".align 16                                    \n\t" \
                         "0:                                           \n\t" \
                         clear##rnum##Regs(rtype)                            \
                         ".align 16                                    \n\t" \
                         "1:                                           \n\t" \
                         : [filter] "+r" (c.filter)                          \
                         : [bias] "r" (c.bias),                              \
                           [flags] "r" (c.flags)                             \
                         : "%rax",                                           \
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",       \
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",     \
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", \
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                           "memory", "cc"); \
    __asm__ __volatile__("movq (%[stepC16]), %%r10              \n\t" \
                         ".align 16                             \n\t" \
                         "1:                                    \n\t" \
                         "mov %[kh], %%rbx                      \n\t" \
                         ".align 16                             \n\t" \
                         "2:                                    \n\t" \
                         "mov %[kw], %%r9                       \n\t" \
                         ".align 16                             \n\t" \
                         "3:                                    \n\t" \
                         "movq %[input], %%rax                  \n\t" \
                         convKernel##wsize##x16c4_##cross(            \
                            %%rax, rtype##24, off0, rtype##25, rtype) \
                         "movq %[input], %%rax                  \n\t" \
                         "addq $0x4, %%rax                      \n\t" \
                         convKernel##wsize##x16c4_##cross(            \
                            %%rax, rtype##25, off1, rtype##24, rtype) \
                         "movq %[input], %%rax                  \n\t" \
                         "addq $0x8, %%rax                      \n\t" \
                         convKernel##wsize##x16c4_##cross(            \
                            %%rax, rtype##24, off2, rtype##25, rtype) \
                         "movq %[input], %%rax                  \n\t" \
                         "addq $0xC, %%rax                      \n\t" \
                         convKernel##wsize##x16c4_##cross(            \
                            %%rax, rtype##25, off3, rtype##24, rtype) \
                         "addq $"#off4", %[filter]              \n\t" \
                         "addq %[dilateW], %[input]             \n\t" \
                         "dec %%r9                              \n\t" \
                         "jg 3b                                 \n\t" \
                         "addq %[dilateH], %[input]             \n\t" \
                         "dec %%rbx                             \n\t" \
                         "jg 2b                                 \n\t" \
                         "addq %[fStep], %[input]               \n\t" \
                         "subq $0x10, %%rcx                     \n\t" \
                         "cmpq $0x10, %%rcx                     \n\t" \
                         "jge 1b                                \n\t" \
                         "subq %[fStep], %[input]               \n\t" \
                         "addq %[f8Step], %[input]              \n\t" \
                         ".align 16                             \n\t" \
                         "4:                                    \n\t" \
                         : "+c" (c.ic),                               \
                           [input] "+r" (c.input),                    \
                           [filter] "+r" (c.filter)                   \
                         : [kh] "r" (c.kh),                           \
                           [kw] "r" (c.kw),                           \
                           [stepC16] "r" (c.stepC16),                 \
                           [dilateW] "r" (c.dilateW),                 \
                           [dilateH] "r" (c.dilateH),                 \
                           [fStep] "r" (c.fStep),                     \
                           [f8Step] "r" (c.f8Step)                    \
                         : "%rax", "%rbx", "%r9", "%r10", "%r11",             \
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",       \
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",     \
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", \
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                           "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", \
                           "%zmm30", "%zmm31", "memory", "cc"); \
    __asm__ __volatile__("movq %[output], %%rax                             \n\t" \
                         "movq %[ostepC16], %%rbx                           \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x1, %%rcx                                   \n\t" \
                         "je 0f                                             \n\t" \
                         add_##wsize##x##nSize(vpaddd, %%rax, %%rbx) \
                         ".align 16                                         \n\t" \
                         "0:                                                \n\t" \
                         "cmpq $0x0, %[scale]                               \n\t" \
                         "jne 1f                                            \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(rtype, vpxord, vpmaxsd) \
                         "jmp 4f                                            \n\t" \
                         ".align 16                                         \n\t" \
                         "1:                                                \n\t" \
                         convert##rnum##RegsI32ToF32(%[scale], rtype) \
                         ".align 16                                         \n\t" \
                         "2:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0x2, %%rcx                                   \n\t" \
                         "je 3f                                             \n\t" \
                         add_##wsize##x##nSize(vaddps, %[eltwise], %%rbx) \
                         ".align 16                                         \n\t" \
                         "3:                                                \n\t" \
                         "movq %[flags], %%rcx                              \n\t" \
                         "and $0xC, %%rcx                                   \n\t" \
                         "je 4f                                             \n\t" \
                         relu##rnum##Regs(rtype, vxorps, vmaxps) \
                         ".align 16                                         \n\t" \
                         "4:                                                \n\t" \
                         store_##wsize##x##nSize(%%rax, %%rbx) \
                         : \
                         : [output] "r" (c.output), \
                           [eltwise] "r" (c.eltwise), \
                           [ostepC16] "r" (c.ostepC16), \
                           [flags] "r" (c.flags), \
                           [scale] "r" (c.scale) \
                         : "%rax", "%rbx", "%rcx", \
                           "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", \
                           "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11", \
                           "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17", \
                           "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23", \
                           "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29", \
                           "%zmm30", "%zmm31", "memory", "cc");

void Avx512ConvKernel12x32(ConvController &c) {
    if (c.cross) {
        convKernelForLoopXx32(24, 12, 0, 2)
    } else if (c.stride > 1) {
        convKernelForLoopXx32(24, 12, 0, 1)
    } else {
        if (c.idx == 0) {
            convKernelForLoopXx32(24, 12, 0, 0)
        } else if (c.idx == 1) {
            convKernelForLoopXx32(24, 12, 1, 0)
        } else if (c.idx == 2) {
            convKernelForLoopXx32(24, 12, 2, 0)
        } else if (c.idx == 3) {
            convKernelForLoopXx32(24, 12, 3, 0)
        } else if (c.idx == 4) {
            convKernelForLoopXx32(24, 12, 4, 0)
        } else if (c.idx == 5) {
            convKernelForLoopXx32(24, 12, 5, 0)
        } else if (c.idx == 6) {
            convKernelForLoopXx32(24, 12, 6, 0)
        } else if (c.idx == 7) {
            convKernelForLoopXx32(24, 12, 7, 0)
        } else if (c.idx == 8) {
            convKernelForLoopXx32(24, 12, 8, 0)
        } else if (c.idx == 9) {
            convKernelForLoopXx32(24, 12, 9, 0)
        } else if (c.idx == 10) {
            convKernelForLoopXx32(24, 12, 10, 0)
        } else if (c.idx == 11) {
            convKernelForLoopXx32(24, 12, 11, 0)
        } else if (c.idx == 12) {
            convKernelForLoopXx32(24, 12, 0, 0)
        }
    }
}


#define convKernelx48(nReg, wSize) \
    void Avx512ConvKernel##wSize##x48(ConvController &c) { \
        if (c.cross) { \
            convKernelForLoopXx48(nReg, wSize, 0, 1) \
        } else { \
            convKernelForLoopXx48(nReg, wSize, 0, 0) \
        } \
    }

#define convKernelx32(nReg, wSize) \
    void Avx512ConvKernel##wSize##x32(ConvController &c) { \
        if (c.cross) { \
            convKernelForLoopXx32(nReg, wSize, 0, 2) \
        } else if (c.stride > 1) { \
            convKernelForLoopXx32(nReg, wSize, 0, 1) \
        } else { \
            convKernelForLoopXx32(nReg, wSize, 0, 0) \
        } \
    }

#define convKernelx16(nReg, wSize) \
    void Avx512ConvKernel##wSize##x16(ConvController &c) { \
        if (c.cross) { \
            convKernelForLoopXx16(nReg, wSize, 16, %%zmm, 0, 1, 0x0, 0x40, 0x80, 0xC0, 0x100) \
        } else { \
            convKernelForLoopXx16(nReg, wSize, 16, %%zmm, 0, 0, 0x0, 0x40, 0x80, 0xC0, 0x100) \
        } \
    }

#define convKernelx8(nReg, wSize) \
    void Avx512ConvKernel##wSize##x8(ConvController &c) { \
        if (c.cross) { \
            convKernelForLoopXx16(nReg, wSize, 8, %%ymm, 0, 1, 0x0, 0x20, 0x40, 0x60, 0x80) \
        } else { \
            convKernelForLoopXx16(nReg, wSize, 8, %%ymm, 0, 0, 0x0, 0x20, 0x40, 0x60, 0x80) \
        } \
    }

convKernelx48(24, 8)
convKernelx48(21, 7)
convKernelx48(18, 6)
convKernelx48(15, 5)
convKernelx48(12, 4)
convKernelx48(9, 3)
convKernelx48(6, 2)
convKernelx48(3, 1)
convKernelx32(22, 11)
convKernelx32(20, 10)
convKernelx32(18, 9)
convKernelx32(16, 8)
convKernelx32(14, 7)
convKernelx32(12, 6)
convKernelx32(10, 5)
convKernelx32(8, 4)
convKernelx32(6, 3)
convKernelx32(4, 2)
convKernelx32(2, 1)
convKernelx16(24, 24)
convKernelx16(23, 23)
convKernelx16(22, 22)
convKernelx16(21, 21)
convKernelx16(20, 20)
convKernelx16(19, 19)
convKernelx16(18, 18)
convKernelx16(17, 17)
convKernelx16(16, 16)
convKernelx16(15, 15)
convKernelx16(14, 14)
convKernelx16(13, 13)
convKernelx16(12, 12)
convKernelx16(11, 11)
convKernelx16(10, 10)
convKernelx16(9, 9)
convKernelx16(8, 8)
convKernelx16(7, 7)
convKernelx16(6, 6)
convKernelx16(5, 5)
convKernelx16(4, 4)
convKernelx16(3, 3)
convKernelx16(2, 2)
convKernelx16(1, 1)
convKernelx8(24, 24)
convKernelx8(23, 23)
convKernelx8(22, 22)
convKernelx8(21, 21)
convKernelx8(20, 20)
convKernelx8(19, 19)
convKernelx8(18, 18)
convKernelx8(17, 17)
convKernelx8(16, 16)
convKernelx8(15, 15)
convKernelx8(14, 14)
convKernelx8(13, 13)
convKernelx8(12, 12)
convKernelx8(11, 11)
convKernelx8(10, 10)
convKernelx8(9, 9)
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

    if (fdf != DF_NCHWC2NxC4 || (idf != DF_NCHWC16 && idf != DF_NCHW && idf != DF_NCHWC8)) {
        CHECK_STATUS(NOT_MATCH);
    }

    // get kernels
    U32 ocSizeArray[4] = {8, 16, 32, 48};
    U32 wSizeArray[4] = {24, 24, 12, 8};

    const kernelFunc kernel[24][4] = {
        {Avx512ConvKernel1x8,  Avx512ConvKernel1x16,  Avx512ConvKernel1x32, Avx512ConvKernel1x48}, 
        {Avx512ConvKernel2x8,  Avx512ConvKernel2x16,  Avx512ConvKernel2x32, Avx512ConvKernel2x48}, 
        {Avx512ConvKernel3x8,  Avx512ConvKernel3x16,  Avx512ConvKernel3x32, Avx512ConvKernel3x48}, 
        {Avx512ConvKernel4x8,  Avx512ConvKernel4x16,  Avx512ConvKernel4x32, Avx512ConvKernel4x48}, 
        {Avx512ConvKernel5x8,  Avx512ConvKernel5x16,  Avx512ConvKernel5x32, Avx512ConvKernel5x48}, 
        {Avx512ConvKernel6x8,  Avx512ConvKernel6x16,  Avx512ConvKernel6x32, Avx512ConvKernel6x48}, 
        {Avx512ConvKernel7x8,  Avx512ConvKernel7x16,  Avx512ConvKernel7x32, Avx512ConvKernel7x48}, 
        {Avx512ConvKernel8x8,  Avx512ConvKernel8x16,  Avx512ConvKernel8x32, Avx512ConvKernel8x48}, 
        {Avx512ConvKernel9x8,  Avx512ConvKernel9x16,  Avx512ConvKernel9x32, Avx512ConvKernel8x48}, 
        {Avx512ConvKernel10x8, Avx512ConvKernel10x16, Avx512ConvKernel10x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel11x8, Avx512ConvKernel11x16, Avx512ConvKernel11x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel12x8, Avx512ConvKernel12x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel13x8, Avx512ConvKernel13x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel14x8, Avx512ConvKernel14x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel15x8, Avx512ConvKernel15x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel16x8, Avx512ConvKernel16x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel17x8, Avx512ConvKernel17x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel18x8, Avx512ConvKernel18x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel19x8, Avx512ConvKernel19x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel20x8, Avx512ConvKernel20x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel21x8, Avx512ConvKernel21x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel22x8, Avx512ConvKernel22x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel23x8, Avx512ConvKernel23x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48},
        {Avx512ConvKernel24x8, Avx512ConvKernel24x16, Avx512ConvKernel12x32, Avx512ConvKernel8x48}};

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
    ConvController convCtl;
    convCtl.ostepC16 = oh * ow * 16 * 4;
    convCtl.dilateW = dilateW * SIMDW;
    convCtl.kw = fw;
    convCtl.kh = fh;
    convCtl.dilateH = ((I32)iw_pad - convCtl.kw * dilateW + (dilateH - 1) * (I32)iw_pad) * SIMDW;
    convCtl.fStep = (((I32)ih_pad - convCtl.kh * dilateH) * (I32)iw_pad) * SIMDW;
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
    I64 step[24];
    I64 normalStep = strideW * 16;
    I64 lastStep = (iw_pad - (ow - 1) * strideW + (strideH - 1) * iw_pad) * 16;
    convCtl.off = lastStep - 0x10;
    for (U32 i = 0; i < 24; ++i) {
        step[i] = strideW * 16;
    }
    convCtl.stepC16 = step;
    convCtl.stride = strideW;
    convCtl.idx = 0;

    for (U32 n = 0; n < in; ++n) {
        UINT8 *bInArray = inArray + n * ic * ih * iw;
        if (idf == DF_NCHWC16 && paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0) {
            if (ic % SIMDW != 0) {
                PaddingChannelNCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
            } else {
                tmpInput = bInArray;
            }
        } else if (idf == DF_NCHWC16) {
            PaddingNCHWCx(bInArray, tmpInput, inputDesc, convParamSpec);
        } else if (idf == DF_NCHWC8) {
            PaddingNCHWC8ToNCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
        } else {
            PaddingNCHW2NCHWC16(bInArray, tmpInput, inputDesc, convParamSpec);
        }

        ic = UNI_ALIGN(ic, SIMDW);

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
            U32 simdOc = SIMDW;

            U32 hwSize = 0;
            for (U32 hw = 0; hw < oh * ow; hw += hwSize) {
                hwSize = UNI_MIN(BLOCK_HW_DIM, oh * ow - hw);
                U32 ocSize = 0;
                for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                    ocSize = UNI_MIN(unrollOc, oc - ocb);
                    ocSize = ocSizeArray[ocSize >> 4];
                    simdOc = UNI_MIN(SIMDW, ocSize);
                    convCtl.bias = offsetC + ocb;
                    UINT8 *curI = tmpInput + icbb * ih_pad * iw_pad;
                    U32 wSize = 8;
                    U32 unrollW = wSizeArray[ocSize >> 4];
                    for (U32 ihw = hw; ihw < hw + hwSize; ihw += wSize) {
                        wSize = UNI_MIN(hw + hwSize - ihw, unrollW);
                        U32 in_h = ihw / ow * strideH;
                        U32 in_w = ihw % ow * strideW;
                        convCtl.input = curI + in_h * iw_pad * SIMDW + in_w * SIMDW;
                        convCtl.output = output + ((n * oc + ocb) * ohow + ihw * simdOc) * oBytes;
                        convCtl.eltwise = eltwiseInput + (n * oc + ocb) * ohow + ihw * simdOc;
                        convCtl.filter = filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                        convCtl.cross = false;
                        U32 lane = (ihw % ow + wSize) / ow - ((ihw % ow + wSize) % ow == 0);
                        if ((ocSize != 32) || (lane > 1) || ((lane == 1) && (wSize < unrollW || strideW > 1))) {
                            for (U32 ui = 0; ui < lane; ++ui) {
                                convCtl.stepC16[(ihw / ow + ui + 1) * ow - ihw - 1] = lastStep;
                            }
                            convCtl.cross = true;
                        } else {
                            convCtl.idx = UNI_MIN((ow - ihw % ow), wSize);
                        }
                        convCtl.ic = icSize;
                        kernel[wSize - 1][ocSize >> 4](convCtl);

                        if ((ocSize != 32) || (lane > 1) || ((lane == 1) && wSize < unrollW) || (strideW > 1)) {
                            for (U32 ui = 0; ui < lane; ++ui) {
                                convCtl.stepC16[(ihw / ow + ui + 1) * ow - ihw - 1] = normalStep;
                            }
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
        F32 *xo = (F32 *)outArray;

        *scaleO = scales[0];
    }

    return SUCCESS;
}
