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
#define BLOCK_HW_DIM 128
#define UNROLL_IC_BLOCK_DIM 8
#define align_addr(addr, unit) (((uintptr_t)addr + unit - 1) / unit * unit)

// clang-format off
#define kernel4x3(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                \n\t" \
    "vbroadcastss "#m0"("#r1"), %%ymm13                \n\t" \
    "vbroadcastss "#m0"("#r2"), %%ymm14                \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t" \
    "vmovups "#m2"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t" \
    "vmovups "#m3"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t" \
    "vmovups "#m4"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

#define kernel4x2(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                \n\t" \
    "vbroadcastss "#m0"("#r1"), %%ymm13                \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t" \
    "vmovups "#m2"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t" \
    "vmovups "#m3"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t" \
    "vmovups "#m4"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

#define kernel4x1(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vmovups "#m2"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t" \
    "vmovups "#m3"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t" \
    "vmovups "#m4"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t" \

#define kernel2x3(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                \n\t" \
    "vbroadcastss "#m0"("#r1"), %%ymm13                \n\t" \
    "vbroadcastss "#m0"("#r2"), %%ymm14                \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t" \
    "vmovups "#m2"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t" \

#define kernel2x2(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                \n\t" \
    "vbroadcastss "#m0"("#r1"), %%ymm13                \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t" \
    "vmovups "#m2"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t" \

#define kernel2x1(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vmovups "#m2"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t" \

#define kernel1x3(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                \n\t" \
    "vbroadcastss "#m0"("#r1"), %%ymm13                \n\t" \
    "vbroadcastss "#m0"("#r2"), %%ymm14                \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t" \

#define kernel1x2(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                \n\t" \
    "vbroadcastss "#m0"("#r1"), %%ymm13                \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                     \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t" \

#define kernel1x1(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                        \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                          \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \

#define kernel4c8(r, r0, r1, r2, r3) \
    kernel4x##r(0x0, r0, r1, r2, r3, 0x0, 0x20, 0x40, 0x60) \
    kernel4x##r(0x4, r0, r1, r2, r3, 0x80, 0xA0, 0xC0, 0xE0) \
    kernel4x##r(0x8, r0, r1, r2, r3, 0x100, 0x120, 0x140, 0x160) \
    kernel4x##r(0xC, r0, r1, r2, r3, 0x180, 0x1A0, 0x1C0, 0x1E0) \
    kernel4x##r(0x10, r0, r1, r2, r3, 0x200, 0x220, 0x240, 0x260) \
    kernel4x##r(0x14, r0, r1, r2, r3, 0x280, 0x2A0, 0x2C0, 0x2E0) \
    kernel4x##r(0x18, r0, r1, r2, r3, 0x300, 0x320, 0x340, 0x360) \
    kernel4x##r(0x1C, r0, r1, r2, r3, 0x380, 0x3A0, 0x3C0, 0x3E0)

#define kernel2c8(r, r0, r1, r2, r3) \
    kernel4x##r(0x0, r0, r1, r2, r3, 0x0, 0x20, 0, 0) \
    kernel4x##r(0x4, r0, r1, r2, r3, 0x40, 0x60, 0, 0) \
    kernel4x##r(0x8, r0, r1, r2, r3, 0x80, 0xA0, 0, 0) \
    kernel4x##r(0xC, r0, r1, r2, r3, 0xC0, 0xE0, 0, 0) \
    kernel4x##r(0x10, r0, r1, r2, r3, 0x100, 0x120, 0, 0) \
    kernel4x##r(0x14, r0, r1, r2, r3, 0x140, 0x160, 0, 0) \
    kernel4x##r(0x18, r0, r1, r2, r3, 0x180, 0x1A0, 0, 0) \
    kernel4x##r(0x1C, r0, r1, r2, r3, 0x1C0, 0x1E0, 0, 0)

#define kernel1c8(r, r0, r1, r2, r3) \
    kernel4x##r(0x0, r0, r1, r2, r3, 0x0,  0, 0, 0) \
    kernel4x##r(0x4, r0, r1, r2, r3, 0x20, 0, 0, 0) \
    kernel4x##r(0x8, r0, r1, r2, r3, 0x40, 0, 0, 0) \
    kernel4x##r(0xC, r0, r1, r2, r3, 0x60, 0, 0, 0) \
    kernel4x##r(0x10, r0, r1, r2, r3, 0x80, 0, 0, 0) \
    kernel4x##r(0x14, r0, r1, r2, r3, 0xA0, 0, 0, 0) \
    kernel4x##r(0x18, r0, r1, r2, r3, 0xC0, 0, 0, 0) \
    kernel4x##r(0x1C, r0, r1, r2, r3, 0xE0, 0, 0, 0)

typedef void (*kernel_func)(F32 *curI, const F32 *curW, F32 *curO, U32 fw, U32 fh, U32 oStep, U32 iStep, U32 store, const F32 *curB, U32 dw, F32 *in_1, F32 *in_2);

void avx2_conv_kernel_3x32c8(F32 *curI, const F32 *curW, F32 *curO, U32 fw, U32 fh, U32 oStep, U32 iStep, U32 store, const F32 *curB, U32 dw, F32 *in_1, F32 *in_2) {
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups (%8), %%ymm1                       \n\t"
                         "vmovups (%8), %%ymm2                       \n\t"
                         "vmovups 0x20(%8), %%ymm3                       \n\t"
                         "vmovups 0x20(%8), %%ymm4                   \n\t"
                         "vmovups 0x20(%8), %%ymm5                   \n\t"
                         "vmovups 0x40(%8), %%ymm6                   \n\t"
                         "vmovups 0x40(%8), %%ymm7                   \n\t"
                         "vmovups 0x40(%8), %%ymm8                   \n\t"
                         "vmovups 0x60(%8), %%ymm9                   \n\t"
                         "vmovups 0x60(%8), %%ymm10                 \n\t"
                         "vmovups 0x60(%8), %%ymm11                 \n\t"
                         "jmp 1f                                             \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %1, %%r9                                     \n\t"
                         "vmovups (%%r9), %%ymm0                     \n\t"
                         "vmovups 0x20(%%r9), %%ymm1                     \n\t"
                         "vmovups 0x40(%%r9), %%ymm2                     \n\t"
                         "add %4, %%r9                                     \n\t"
                         "vmovups (%%r9), %%ymm3                     \n\t"
                         "vmovups 0x20(%%r9), %%ymm4                     \n\t"
                         "vmovups 0x40(%%r9), %%ymm5                     \n\t"
                         "add %4, %%r9                                     \n\t"
                         "vmovups (%%r9), %%ymm6                     \n\t"
                         "vmovups 0x20(%%r9), %%ymm7                     \n\t"
                         "vmovups 0x40(%%r9), %%ymm8                     \n\t"
                         "add %4, %%r9                                     \n\t"
                         "vmovups (%%r9), %%ymm9                     \n\t"
                         "vmovups 0x20(%%r9), %%ymm10                  \n\t"
                         "vmovups 0x40(%%r9), %%ymm11                  \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         kernel4c8(3, %0, %10, %11, %2)
                         "add %9, %0                                      \n\t"
                         "add %9, %10                                      \n\t"
                         "add %9, %11                                      \n\t"
                         "add $0x400, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 2b                                             \n\t"
                         "add %6, %0                                     \n\t"
                         "add %6, %10                                     \n\t"
                         "add %6, %11                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 1b                                             \n\t"
                         // relu
                         "and $0x6, %7                                      \n\t"
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
                         "and $0x4, %7                                      \n\t"
                         "je 3f                                             \n\t"
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
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         "vmovups %%ymm2, 0x40(%1)                          \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm3, (%1)                              \n\t"
                         "vmovups %%ymm4, 0x20(%1)                          \n\t"
                         "vmovups %%ymm5, 0x40(%1)                          \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm6, (%1)                              \n\t"
                         "vmovups %%ymm7, 0x20(%1)                          \n\t"
                         "vmovups %%ymm8, 0x40(%1)                          \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm9, (%1)                              \n\t"
                         "vmovups %%ymm10, 0x20(%1)                         \n\t"
                         "vmovups %%ymm11, 0x40(%1)                         \n\t"
                         :
                         : "r" (curI), "r" (curO), "r" (curW), "r" (fw), 
                           "r" (I64(oStep)), "b" (fh), "r" (I64(iStep)), "r" (store), 
                           "r" (curB), "r" (I64(dw)), "r" (in_1), "r" (in_2)
                         : "%ecx", "%r9", 
                           "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3",
                           "%ymm4",  "%ymm5",  "%ymm6",  "%ymm7",
                           "%ymm8",  "%ymm9",  "%ymm10", "%ymm11",
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                           "memory", "cc");
}

void avx2_conv_kernel_1x32c8(F32 *curI, const F32 *curW, F32 *curO, U32 fw, U32 fh, U32 oStep, U32 iStep, U32 store, const F32 *curB, U32 dw, F32 *in_1, F32 *in_2) {
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups 0x20(%8), %%ymm3                       \n\t"
                         "vmovups 0x40(%8), %%ymm6                   \n\t"
                         "vmovups 0x60(%8), %%ymm9                   \n\t"
                         "jmp 1f                                             \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %1, %%r9                                     \n\t"
                         "vmovups (%%r9), %%ymm0                     \n\t"
                         "add %4, %%r9                                     \n\t"
                         "vmovups (%%r9), %%ymm3                     \n\t"
                         "add %4, %%r9                                     \n\t"
                         "vmovups (%%r9), %%ymm6                     \n\t"
                         "add %4, %%r9                                     \n\t"
                         "vmovups (%%r9), %%ymm9                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         kernel4c8(1, %0, 0, 0, %2)
                         "add %9, %0                                     \n\t"
                         "add $0x400, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 2b                                             \n\t"
                         "add %6, %0                                     \n\t"
                         "sub $1, %%ebx                                     \n\t"
                         "jg 1b                                             \n\t"
                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
                         "vmaxps %%ymm15, %%ymm6, %%ymm6                    \n\t"
                         "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"
                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
                         "vminps %%ymm12, %%ymm6, %%ymm6                    \n\t"
                         "vminps %%ymm12, %%ymm9, %%ymm9                    \n\t"
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm3, (%1)                              \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm6, (%1)                              \n\t"
                         "add %4, %1                                     \n\t"
                         "vmovups %%ymm9, (%1)                              \n\t"
                         :
                         : "r" (curI), "r" (curO), "r" (curW), "r" (fw), 
                           "r" (I64(oStep)), "b" (fh), "r" (I64(iStep)), "r" (store), 
                           "r" (curB), "r" (I64(dw))
                         : "%ecx", "%r9",
                           "%ymm0",  "%ymm3", "%ymm6", "%ymm9", "%ymm12", "%ymm15",
                           "memory", "cc");
}

void avx2_conv_kernel_3x16c8(F32 *curI, const F32 *curW, F32 *curO, U32 fw, U32 fh, U32 oStep, U32 iStep, U32 store, const F32 *curB, U32 dw, F32 *in_1, F32 *in_2) {
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups (%8), %%ymm1                       \n\t"
                         "vmovups (%8), %%ymm2                       \n\t"
                         "vmovups 0x20(%8), %%ymm3                       \n\t"
                         "vmovups 0x20(%8), %%ymm4                   \n\t"
                         "vmovups 0x20(%8), %%ymm5                   \n\t"
                         "jmp 1f                                             \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%1), %%ymm0                     \n\t"
                         "vmovups 0x20(%1), %%ymm1                     \n\t"
                         "vmovups 0x40(%1), %%ymm2                     \n\t"
                         "vmovups (%1, %4), %%ymm3                     \n\t"
                         "vmovups 0x20(%1, %4), %%ymm4                     \n\t"
                         "vmovups 0x40(%1, %4), %%ymm5                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         kernel2c8(3, %0, %10, %11, %2)
                         "add %9, %0                                      \n\t"
                         "add %9, %10                                      \n\t"
                         "add %9, %11                                      \n\t"
                         "add $0x200, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 2b                                             \n\t"
                         "add %6, %0                                     \n\t"
                         "add %6, %10                                     \n\t"
                         "add %6, %11                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 1b                                             \n\t"
                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
                         "vmaxps %%ymm15, %%ymm2, %%ymm2                    \n\t"
                         "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
                         "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
                         "vmaxps %%ymm15, %%ymm5, %%ymm5                    \n\t"
                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
                         "vminps %%ymm12, %%ymm2, %%ymm2                    \n\t"
                         "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
                         "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
                         "vminps %%ymm12, %%ymm5, %%ymm5                    \n\t"
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         "vmovups %%ymm2, 0x40(%1)                          \n\t"
                         "vmovups %%ymm3, (%1, %4)                              \n\t"
                         "vmovups %%ymm4, 0x20(%1, %4)                          \n\t"
                         "vmovups %%ymm5, 0x40(%1, %4)                          \n\t"
                         :
                         : "r" (curI), "r" (curO), "r" (curW), "r" (fw), 
                           "r" (I64(oStep)), "b" (fh), "r" (I64(iStep)), "r" (store), 
                           "r" (curB), "r" (I64(dw)), "r" (in_1), "r" (in_2)
                         : "%ecx", 
                           "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3",
                           "%ymm4",  "%ymm5",  
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                           "memory", "cc");
}

void avx2_conv_kernel_1x16c8(F32 *curI, const F32 *curW, F32 *curO, U32 fw, U32 fh, U32 oStep, U32 iStep, U32 store, const F32 *curB, U32 dw, F32 *in_1, F32 *in_2) {
    __asm__ __volatile__("mov %7, %%ecx                                  \n\t"
                         "and $0x1, %%ecx                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%8), %%ymm0                       \n\t"
                         "vmovups 0x20(%8), %%ymm3                       \n\t"
                         "jmp 1f                                             \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%1), %%ymm0                     \n\t"
                         "vmovups (%1, %4), %%ymm3                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         kernel2c8(1, %0, 0, 0, %2)
                         "add %9, %0                                     \n\t"
                         "add $0x200, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 2b                                             \n\t"
                         "add %6, %0                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 1b                                             \n\t"
                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%ecx                            \n\t"
                         "vmovd %%ecx, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm3, (%1, %4)                              \n\t"
                         :
                         : "r" (curI), "r" (curO), "r" (curW), "r" (fw), 
                           "r" (I64(oStep)), "b" (fh), "r" (I64(iStep)), "r" (store), 
                           "r" (curB), "r" (I64(dw))
                         : "%ecx", 
                           "%ymm0", "%ymm3", "%ymm12", "%ymm15",
                           "memory", "cc");
}

void avx2_conv_kernel_3x8c8(F32 *curI, const F32 *curW, F32 *curO, U32 fw, U32 fh, U32 oStep, U32 iStep, U32 store, const F32 *curB, U32 dw, F32 *in_1, F32 *in_2) {
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
                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         kernel1c8(3, %0, %10, %11, %2)
                         "add %9, %0                                      \n\t"
                         "add %9, %10                                      \n\t"
                         "add %9, %11                                      \n\t"
                         "add $0x100, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 2b                                             \n\t"
                         "add %6, %0                                     \n\t"
                         "add %6, %10                                     \n\t"
                         "add %6, %11                                     \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 1b                                             \n\t"
                         // relu
                         "and $0x6, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
                         "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
                         "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
                         "vmaxps %%ymm15, %%ymm2, %%ymm2                    \n\t"
                         // relu6
                         "and $0x4, %7                                      \n\t"
                         "je 3f                                             \n\t"
                         "mov $0x40C00000, %%eax                            \n\t"
                         "vmovd %%eax, %%xmm12                              \n\t"
                         "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
                         "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
                         "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
                         "vminps %%ymm12, %%ymm2, %%ymm2                    \n\t"
                         "3:                                                \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         "vmovups %%ymm2, 0x40(%1)                          \n\t"
                         :
                         : "r" (curI), "r" (curO), "r" (curW), "r" (fw), 
                           "r" (I64(oStep)), "b" (fh), "r" (I64(iStep)), "r" (store), 
                           "r" (curB), "r" (I64(dw)), "r" (in_1), "r" (in_2)
                         : "%ecx", 
                           "%ymm0",  "%ymm1",  "%ymm2", 
                           "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                           "memory", "cc");
}

void avx2_conv_kernel_1x8c8(F32 *curI, const F32 *curW, F32 *curO, U32 fw, U32 fh, U32 oStep, U32 iStep, U32 store, const F32 *curB, U32 dw, F32 *in_1, F32 *in_2) {
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
                         "mov %3, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "2:                                                \n\t"
                         kernel1c8(1, %0, 0, 0, %2)
                         "add %9, %0                                     \n\t"
                         "add $0x100, %2                                    \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 2b                                             \n\t"
                         "add %6, %0                                     \n\t"
                         "dec %%ebx                                         \n\t"
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
                         : "r" (curI), "r" (curO), "r" (curW), "r" (fw), 
                           "r" (I64(oStep)), "b" (fh), "r" (I64(iStep)), "r" (store), 
                           "r" (curB), "r" (I64(dw))
                         : "%ecx", 
                           "%ymm0",  "%ymm12", "%ymm15",
                           "memory", "cc");
}

EE convolution_direct(TensorDesc inputDesc,
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

    if ((2 == fh) && (2 == fw)) {
        return convolution_2x2_direct(inputDesc, inArray, filterDesc, filterArray, convParamSpec,
            biasDesc, biasArray, tmpBytes, tmp, outputDesc, outArray, activationDesc);
    }

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

    F32 *ftmp = (F32 *)align_addr(tmp, 32);
    filterArray = (F32 *)align_addr(filterArray, 32);

    U32 icAlignSize = 8;
    U32 icPadding = (ic + icAlignSize - 1) / icAlignSize * icAlignSize;
    U32 ih_pad = ih + paddingT + paddingB;
    U32 iw_pad = iw + paddingL + paddingR;
    I32 ohow = oh * ow;

    U32 oStep = oh * ow * UNROLL_OC_DIM * 4;
    U32 iStep = (iw_pad - fw * dilateW + (dilateH - 1) * iw_pad) * UNROLL_IC_BLOCK_DIM * 4;
    U32 sw = strideW * UNROLL_IC_BLOCK_DIM * 4;
    U32 dw = dilateW * UNROLL_IC_BLOCK_DIM * 4;
    kernel_func kernel[3][2] = {{avx2_conv_kernel_1x8c8, avx2_conv_kernel_3x8c8},
                                {avx2_conv_kernel_1x16c8, avx2_conv_kernel_3x16c8},
                                {avx2_conv_kernel_1x32c8, avx2_conv_kernel_3x32c8}};
    U32 ocblocks[3] = {8, 16, 32};

#ifdef _USE_OPENMP
    U32 alpha = (ohow + OMP_NUM_THREADS * BLOCK_HW_DIM - 1) / (OMP_NUM_THREADS * BLOCK_HW_DIM);
    U32 block_hw_dim = (ohow + OMP_NUM_THREADS * alpha - 1 ) / (OMP_NUM_THREADS * alpha);
#else
    U32 block_hw_dim = BLOCK_HW_DIM;
#endif

    U32 icSize = 0;
    U32 hwBlockNums = (ohow + block_hw_dim - 1 ) / block_hw_dim;
    U32 ocBlockNums = oc / BLOCK_OC_DIM;
    U32 ocbArray[4] = {0};
    U32 oc_remain = oc % BLOCK_OC_DIM;
    for (U32 i = 0, j = 0; i < oc_remain; i += icSize, ++j) {
        icSize = ocblocks[(oc_remain - i)>>4];
        ocbArray[j + 1] = icSize + ocbArray[j];
        ++ocBlockNums;
    }
    U32 hwocBlockNums = hwBlockNums * ocBlockNums;

    for (U32 n = 0; n < in; ++n) {
        if ((paddingT == 0) && (paddingB == 0) && (paddingL == 0) && (paddingR == 0)) {
            ftmp = inArray;
        } else {
            PaddingNCHWC8(inArray, ftmp, inputDesc, convParamSpec);
        }
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
        {
#endif
            U32 private_icSize = icSize;
            for (U32 icbb = 0; icbb < ic; icbb += private_icSize) {
                private_icSize = UNI_MIN(BLOCK_IC_DIM, ic - icbb);
#ifdef _USE_OPENMP
#pragma omp for
#endif
                for (U32 bIdx = 0; bIdx < hwocBlockNums; ++bIdx) {
                    U32 hw = (bIdx / ocBlockNums) * block_hw_dim;
                    U32 hwSize = UNI_MIN(block_hw_dim, ohow - hw);
                    U32 ocIdx = bIdx % ocBlockNums;
                    U32 ocb = ocIdx * BLOCK_OC_DIM;
                    if (ocIdx > oc / BLOCK_OC_DIM) {
                        ocb += ocbArray[ocIdx - oc / BLOCK_OC_DIM];
                    }
                    U32 ocSize = UNI_MIN(BLOCK_OC_DIM, oc - ocb);
                    ocSize = ocblocks[ocSize >> 4];
                    const F32 *curB = biasArray + ocb;
                    U32 store = 0, icbSize = 0;
                    for (U32 icb = icbb; icb < icbb + private_icSize; icb += icbSize) {
                        icbSize = UNI_MIN(icbb + private_icSize - icb, UNROLL_IC_BLOCK_DIM);
                        const F32 *calW = filterArray + ocb * icPadding * fh * fw + ocSize * icb * fh * fw;
                        F32 *curI = ftmp + icb * ih_pad * iw_pad;

                        store |= (icb > 0);
                        if (icb == ic - icbSize) {
                            store |= U32(activationDesc.mode) << 1;
                        }
                        U32 wSize = 0;
                        for (U32 ihw = hw; ihw < hw + hwSize; ihw += wSize) {
                            wSize = UNI_MIN(hw + hwSize - ihw, UNROLL_W);
                            if (wSize < 3) {
                                wSize = 1;
                            }
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

                            kernel[ocSize>>4][wSize>>1](in_0, calW, out_ptr, fw, fh, oStep, iStep, 
                                store, curB, dw, in_1, in_2);
                        }
                    }
                }
            }
#ifdef _USE_OPENMP
        }
#endif
        inArray += ic * ih * iw;
        outArray += oc * oh * ow;
    }
    return SUCCESS;
}
