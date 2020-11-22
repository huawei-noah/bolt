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

#define UNROLL_W 4
#define UNROLL_OC_DIM 8
#define BLOCK_OC_DIM 24
#define BLOCK_IC_DIM 32
#define BLOCK_HW_DIM 768
#define UNROLL_IC_BLOCK_DIM 8
#define align_addr(addr, unit) (((uintptr_t)addr + unit - 1) / unit * unit)

#define kernel4x3(m0, r0, r1, r2, r3, m1, m2, m3, m4) \
    "vbroadcastss "#m0"("#r0"), %%ymm12                        \n\t" \
    "vbroadcastss "#m0"("#r1"), %%ymm13                       \n\t" \
    "vbroadcastss "#m0"("#r2"), %%ymm14                       \n\t" \
    "vmovups "#m1"("#r3"), %%ymm15                          \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t" \
    "vmovups "#m2"("#r3"), %%ymm15                         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t" \
    "vmovups "#m3"("#r3"), %%ymm15                         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t" \
    "vmovups "#m4"("#r3"), %%ymm15                         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

typedef void (*kernel_func)(F32 *in_0, F32 *in_1, F32 *in_2, F32 *in_3, const F32 *curW, F32 *curO, const F32 *curB, 
    U32 fw, U32 fh, U32 oStep, U32 hStep, U32 store, U32 dw, U32 ic, U32 iStep, U32 fwStep, U32 fhStep);

void avx2_conv_kernel_3x32(F32 *in_0, F32 *in_1, F32 *in_2, F32 *in_3, const F32 *curW, F32 *curO, const F32 *curB, 
    U32 fw, U32 fh, U32 oStep, U32 hStep, U32 store, U32 dw, U32 ic, U32 iStep, U32 fwStep, U32 fhStep) {
    __asm__ __volatile__("mov %3, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
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
                         "jmp 1f                                             \n\t"
                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %0, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm0                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm1                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm2                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm3                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm4                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm5                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm6                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm7                     \n\t"
                         "vmovups 0x40(%%rax), %%ymm8                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm9                     \n\t"
                         "vmovups 0x20(%%rax), %%ymm10                  \n\t"
                         "vmovups 0x40(%%rax), %%ymm11                  \n\t"
                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : "r" (curO), "r" (curB), "r" (I64(oStep)), "r" (store)
                         : "%eax", "%rax",
                           "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3",
                           "%ymm4",  "%ymm5",  "%ymm6",  "%ymm7",
                           "%ymm8",  "%ymm9",  "%ymm10", "%ymm11",
                           "memory", "cc");

    if ((fw == 7) && (fh > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"
                             "mov %4, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x0, 0x20, 0x40, 0x60)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x80, 0xA0, 0xC0, 0xE0)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x100, 0x120, 0x140, 0x160)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x180, 0x1A0, 0x1C0, 0x1E0)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x200, 0x220, 0x240, 0x260)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x280, 0x2A0, 0x2C0, 0x2E0)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x300, 0x320, 0x340, 0x360)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             "add $0x380, %3                                    \n\t"
                             "add %7, %0                                     \n\t"
                             "add %7, %1                                     \n\t"
                             "add %7, %2                                     \n\t"
                             "add %10, %3                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"
                             "add %9, %0                                     \n\t"
                             "add %9, %1                                     \n\t"
                             "add %9, %2                                     \n\t"
                             "add %11, %3                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r" (in_0), "r" (in_1), "r" (in_2), 
                               "r" (curW), "r" (fh), "r" (fw), "a" (ic), 
                               "r" (I64(hStep)), "r" (I64(dw)), "r" (I64(iStep)), "r" (I64(fwStep)), "r" (I64(fhStep))
                             : "%ecx", "%ebx",
                               "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3",
                               "%ymm4",  "%ymm5",  "%ymm6",  "%ymm7",
                               "%ymm8",  "%ymm9",  "%ymm10", "%ymm11",
                               "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                               "memory", "cc");
    } else if ((fw == 5) && (fh > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"
                             "mov %4, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x0, 0x20, 0x40, 0x60)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x80, 0xA0, 0xC0, 0xE0)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x100, 0x120, 0x140, 0x160)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x180, 0x1A0, 0x1C0, 0x1E0)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x200, 0x220, 0x240, 0x260)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             "add $0x280, %3                                    \n\t"
                             "add %7, %0                                     \n\t"
                             "add %7, %1                                     \n\t"
                             "add %7, %2                                     \n\t"
                             "add %10, %3                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"
                             "add %9, %0                                     \n\t"
                             "add %9, %1                                     \n\t"
                             "add %9, %2                                     \n\t"
                             "add %11, %3                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r" (in_0), "r" (in_1), "r" (in_2), 
                               "r" (curW), "r" (fh), "r" (fw), "a" (ic), 
                               "r" (I64(hStep)), "r" (I64(dw)), "r" (I64(iStep)), "r" (I64(fwStep)), "r" (I64(fhStep))
                             : "%ecx", "%ebx",
                               "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3",
                               "%ymm4",  "%ymm5",  "%ymm6",  "%ymm7",
                               "%ymm8",  "%ymm9",  "%ymm10", "%ymm11",
                               "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                               "memory", "cc");
    } else if ((fw == 3) && (fh == 3)) {
        __asm__ __volatile__("add %8, %7                                     \n\t"
                             ".align 16                                         \n\t"
                             "0:                                                \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x0, 0x20, 0x40, 0x60)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x80, 0xA0, 0xC0, 0xE0)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x100, 0x120, 0x140, 0x160)
                             "add %7, %0                                     \n\t"
                             "add %7, %1                                     \n\t"
                             "add %7, %2                                     \n\t"
                             "add %10, %3                                    \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x180, 0x1A0, 0x1C0, 0x1E0)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x200, 0x220, 0x240, 0x260)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x280, 0x2A0, 0x2C0, 0x2E0)
                             "add %7, %0                                     \n\t"
                             "add %7, %1                                     \n\t"
                             "add %7, %2                                     \n\t"
                             "add %10, %3                                    \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x300, 0x320, 0x340, 0x360)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x380, 0x3A0, 0x3C0, 0x3E0)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x400, 0x420, 0x440, 0x460)
                             "add %7, %0                                     \n\t"
                             "add %7, %1                                     \n\t"
                             "add %7, %2                                     \n\t"
                             "add %10, %3                                    \n\t"
                             "add $0x480, %3                                    \n\t"
                             "add %9, %0                                     \n\t"
                             "add %9, %1                                     \n\t"
                             "add %9, %2                                     \n\t"
                             "add %11, %3                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r" (in_0), "r" (in_1), "r" (in_2), 
                               "r" (curW), "r" (fh), "r" (fw), "a" (ic), 
                               "r" (I64(hStep)), "r" (I64(dw)), "r" (I64(iStep)), "r" (I64(fwStep)), "r" (I64(fhStep))
                             : "%ecx",
                               "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3",
                               "%ymm4",  "%ymm5",  "%ymm6",  "%ymm7",
                               "%ymm8",  "%ymm9",  "%ymm10", "%ymm11",
                               "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                               "memory", "cc");
    } else if ((fh > 0) && (fw > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"
                             "mov %4, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"
                             "mov %5, %%ecx                                     \n\t"
                             ".align 16                                         \n\t"
                             "2:                                                \n\t"
                             kernel4x3(0x0, %0, %1, %2, %3, 0x0, 0x20, 0x40, 0x60)
                             "add %8, %0                                      \n\t"
                             "add %8, %1                                      \n\t"
                             "add %8, %2                                      \n\t"
                             "add $0x80, %3                                    \n\t"
                             "dec %%ecx                                         \n\t"
                             "jg 2b                                             \n\t"
                             "add %7, %0                                     \n\t"
                             "add %7, %1                                     \n\t"
                             "add %7, %2                                     \n\t"
                             "add %10, %3                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"
                             "add %9, %0                                     \n\t"
                             "add %9, %1                                     \n\t"
                             "add %9, %2                                     \n\t"
                             "add %11, %3                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r" (in_0), "r" (in_1), "r" (in_2), 
                               "r" (curW), "r" (fh), "r" (fw), "a" (ic), 
                               "r" (I64(hStep)), "r" (I64(dw)), "r" (I64(iStep)), "r" (I64(fwStep)), "r" (I64(fhStep))
                             : "%ecx", "%ebx",
                               "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3",
                               "%ymm4",  "%ymm5",  "%ymm6",  "%ymm7",
                               "%ymm8",  "%ymm9",  "%ymm10", "%ymm11",
                               "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                               "memory", "cc");   
    }

    __asm__ __volatile__(
        // relu
        "and $0x6, %2                                      \n\t"
        "je 0f                                             \n\t"
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
        "and $0x4, %2                                      \n\t"
        "je 0f                                             \n\t"
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
        "vminps %%ymm12, %%ymm10, %%ymm10                  \n\t"
        "vminps %%ymm12, %%ymm11, %%ymm11                  \n\t"

        "0:                                                \n\t"
        "vmovups %%ymm0, (%0)                          \n\t"
        "vmovups %%ymm1, 0x20(%0)                      \n\t"
        "vmovups %%ymm2, 0x40(%0)                      \n\t"
        "add %1, %0                                  \n\t"
        "vmovups %%ymm3, (%0)                      \n\t"
        "vmovups %%ymm4, 0x20(%0)                          \n\t"
        "vmovups %%ymm5, 0x40(%0)                      \n\t"
        "add %1, %0                                  \n\t"
        "vmovups %%ymm6, (%0)                      \n\t"
        "vmovups %%ymm7, 0x20(%0)                      \n\t"
        "vmovups %%ymm8, 0x40(%0)                       \n\t"
        "add %1, %0                                  \n\t"
        "vmovups %%ymm9, (%0)                   \n\t"
        "vmovups %%ymm10, 0x20(%0)                  \n\t"
        "vmovups %%ymm11, 0x40(%0)                  \n\t"
        :
        : "r"(curO), "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_1x32(F32 *in_0,
    F32 *in_1,
    F32 *in_2,
    F32 *in_3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 ic,
    U32 iStep,
    U32 fwStep,
    U32 fhStep)
{
    __asm__ __volatile__("mov %3, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups 0x20(%1), %%ymm3                       \n\t"
                         "vmovups 0x40(%1), %%ymm6                   \n\t"
                         "vmovups 0x60(%1), %%ymm9                   \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "mov %0, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm0                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm3                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm6                     \n\t"
                         "add %2, %%rax                                  \n\t"
                         "vmovups (%%rax), %%ymm9                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store)
                         : "%eax", "%rax", "%ymm0", "%ymm3", "%ymm6", "%ymm9", "memory", "cc");

    if ((fh == 3) && (fw == 3)) {
        __asm__ __volatile__("add %8, %7                                     \n\t"
                             ".align 16                                         \n\t"
                             "0:                                                \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps (%3), %%ymm11                             \n\t"
                             "vmovaps 0x20(%3), %%ymm13                             \n\t"
                             "vmovaps 0x40(%3), %%ymm14                             \n\t"
                             "vmovaps 0x60(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %8, %0                                      \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps 0x80(%3), %%ymm11                             \n\t"
                             "vmovaps 0xA0(%3), %%ymm13                             \n\t"
                             "vmovaps 0xC0(%3), %%ymm14                             \n\t"
                             "vmovaps 0xE0(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %8, %0                                      \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps 0x100(%3), %%ymm11                             \n\t"
                             "vmovaps 0x120(%3), %%ymm13                             \n\t"
                             "vmovaps 0x140(%3), %%ymm14                             \n\t"
                             "vmovaps 0x160(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %7, %0                                      \n\t"
                             "add %10, %3                                    \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps 0x180(%3), %%ymm11                             \n\t"
                             "vmovaps 0x1A0(%3), %%ymm13                             \n\t"
                             "vmovaps 0x1C0(%3), %%ymm14                             \n\t"
                             "vmovaps 0x1E0(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %8, %0                                      \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps 0x200(%3), %%ymm11                             \n\t"
                             "vmovaps 0x220(%3), %%ymm13                             \n\t"
                             "vmovaps 0x240(%3), %%ymm14                             \n\t"
                             "vmovaps 0x260(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %8, %0                                      \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps 0x280(%3), %%ymm11                             \n\t"
                             "vmovaps 0x2A0(%3), %%ymm13                             \n\t"
                             "vmovaps 0x2C0(%3), %%ymm14                             \n\t"
                             "vmovaps 0x2E0(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %7, %0                                      \n\t"
                             "add %10, %3                                    \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps 0x300(%3), %%ymm11                             \n\t"
                             "vmovaps 0x320(%3), %%ymm13                             \n\t"
                             "vmovaps 0x340(%3), %%ymm14                             \n\t"
                             "vmovaps 0x360(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %8, %0                                      \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps 0x380(%3), %%ymm11                             \n\t"
                             "vmovaps 0x3A0(%3), %%ymm13                             \n\t"
                             "vmovaps 0x3C0(%3), %%ymm14                             \n\t"
                             "vmovaps 0x3E0(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %8, %0                                      \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps 0x400(%3), %%ymm11                             \n\t"
                             "vmovaps 0x420(%3), %%ymm13                             \n\t"
                             "vmovaps 0x440(%3), %%ymm14                             \n\t"
                             "vmovaps 0x460(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %7, %0                                      \n\t"
                             "add %10, %3                                    \n\t"
                             "add $0x480, %3                                    \n\t"
                             "add %9, %0                                     \n\t"
                             "add %11, %3                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r" (in_0), "r" (in_1), "r" (in_2), 
                               "r" (curW), "r" (fh), "r" (fw), "a" (ic), 
                               "r" (I64(hStep)), "r" (I64(dw)), "r" (I64(iStep)), "r" (I64(fwStep)), "r" (I64(fhStep))
                             : 
                               "%ymm0", "%ymm3", "%ymm6", "%ymm9", "%ymm11",
                               "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                               "memory", "cc");
    } else if ((fh > 0) && (fw > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"
                             "mov %4, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"
                             "mov %5, %%ecx                                     \n\t"
                             ".align 16                                         \n\t"
                             "2:                                                \n\t"
                             "vbroadcastss (%0), %%ymm12                     \n\t"
                             "vmovaps (%3), %%ymm11                             \n\t"
                             "vmovaps 0x20(%3), %%ymm13                             \n\t"
                             "vmovaps 0x40(%3), %%ymm14                             \n\t"
                             "vmovaps 0x60(%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                             "add %8, %0                                      \n\t"
                             "add $0x80, %3                                    \n\t"
                             "dec %%ecx                                         \n\t"
                             "jg 2b                                             \n\t"
                             "add %7, %0                                     \n\t"
                             "add %10, %3                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"
                             "add %9, %0                                     \n\t"
                             "add %11, %3                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r" (in_0), "r" (in_1), "r" (in_2), 
                               "r" (curW), "r" (fh), "r" (fw), "a" (ic), 
                               "r" (I64(hStep)), "r" (I64(dw)), "r" (I64(iStep)), "r" (I64(fwStep)), "r" (I64(fhStep))
                             : "%ecx", "%ebx",
                               "%ymm0", "%ymm3", "%ymm6", "%ymm9", "%ymm11",
                               "%ymm12", "%ymm13", "%ymm14", "%ymm15",
                               "memory", "cc");
    }

    __asm__ __volatile__(
        // relu
        "and $0x6, %2                                      \n\t"
        "je 0f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"
        "vmaxps %%ymm15, %%ymm6, %%ymm6                    \n\t"
        "vmaxps %%ymm15, %%ymm9, %%ymm9                    \n\t"

        // relu6
        "and $0x4, %2                                      \n\t"
        "je 0f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"
        "vminps %%ymm12, %%ymm6, %%ymm6                    \n\t"
        "vminps %%ymm12, %%ymm9, %%ymm9                    \n\t"

        "0:                                                \n\t"
        "vmovups %%ymm0, (%0)                          \n\t"
        "add %1, %0                                  \n\t"
        "vmovups %%ymm3, (%0)                      \n\t"
        "add %1, %0                                  \n\t"
        "vmovups %%ymm6, (%0)                      \n\t"
        "add %1, %0                                  \n\t"
        "vmovups %%ymm9, (%0)                   \n\t"
        :
        : "r"(curO), "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm3", "%ymm6", "%ymm9", "%ymm12", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_4x24(F32 *in_0,
    F32 *in_1,
    F32 *in_2,
    F32 *in_3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 ic,
    U32 iStep,
    U32 fwStep,
    U32 fhStep)
{
    __asm__ __volatile__("mov %3, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups (%1), %%ymm1                       \n\t"
                         "vmovups (%1), %%ymm2                       \n\t"
                         "vmovups (%1), %%ymm3                       \n\t"
                         "vmovups 0x20(%1), %%ymm4                   \n\t"
                         "vmovups 0x20(%1), %%ymm5                   \n\t"
                         "vmovups 0x20(%1), %%ymm6                   \n\t"
                         "vmovups 0x20(%1), %%ymm7                   \n\t"
                         "vmovups 0x40(%1), %%ymm8                   \n\t"
                         "vmovups 0x40(%1), %%ymm9                   \n\t"
                         "vmovups 0x40(%1), %%ymm10                 \n\t"
                         "vmovups 0x40(%1), %%ymm11                 \n\t"
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
                         "vmovups 0x40(%0, %2, 2), %%ymm10                  \n\t"
                         "vmovups 0x60(%0, %2, 2), %%ymm11                  \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store)
                         : "%eax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                         "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "memory", "cc");

    if ((fw == 3) && (fh == 3)) {
        __asm__ __volatile__("mov %7, %%eax                                     \n\t"
                             ".align 16                                         \n\t"
                             "0:                                                \n\t"

                             "vmovaps (%4), %%ymm12                     \n\t"
                             "vmovaps 0x20(%4), %%ymm13                     \n\t"
                             "vmovaps 0x40(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "vmovaps 0x60(%4), %%ymm12                     \n\t"
                             "vmovaps 0x80(%4), %%ymm13                     \n\t"
                             "vmovaps 0xA0(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "vmovaps 0xC0(%4), %%ymm12                     \n\t"
                             "vmovaps 0xE0(%4), %%ymm13                     \n\t"
                             "vmovaps 0x100(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "add %8, %0                                     \n\t"
                             "add %8, %1                                     \n\t"
                             "add %8, %2                                     \n\t"
                             "add %8, %3                                     \n\t"
                             "add %11, %4                                    \n\t"

                             "vmovaps 0x120(%4), %%ymm12                     \n\t"
                             "vmovaps 0x140(%4), %%ymm13                     \n\t"
                             "vmovaps 0x160(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "vmovaps 0x180(%4), %%ymm12                     \n\t"
                             "vmovaps 0x1A0(%4), %%ymm13                     \n\t"
                             "vmovaps 0x1C0(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "vmovaps 0x1E0(%4), %%ymm12                     \n\t"
                             "vmovaps 0x200(%4), %%ymm13                     \n\t"
                             "vmovaps 0x220(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "add %8, %0                                     \n\t"
                             "add %8, %1                                     \n\t"
                             "add %8, %2                                     \n\t"
                             "add %8, %3                                     \n\t"
                             "add %11, %4                                    \n\t"

                             "vmovaps 0x240(%4), %%ymm12                     \n\t"
                             "vmovaps 0x260(%4), %%ymm13                     \n\t"
                             "vmovaps 0x280(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "vmovaps 0x2A0(%4), %%ymm12                     \n\t"
                             "vmovaps 0x2C0(%4), %%ymm13                     \n\t"
                             "vmovaps 0x2E0(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "vmovaps 0x300(%4), %%ymm12                     \n\t"
                             "vmovaps 0x320(%4), %%ymm13                     \n\t"
                             "vmovaps 0x340(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"

                             "add %8, %0                                     \n\t"
                             "add %8, %1                                     \n\t"
                             "add %8, %2                                     \n\t"
                             "add %8, %3                                     \n\t"

                             "add $0x360, %4                                    \n\t"
                             "add %10, %0                                     \n\t"
                             "add %10, %1                                     \n\t"
                             "add %10, %2                                     \n\t"
                             "add %10, %3                                     \n\t"
                             "add %12, %4                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r"(in_0), "r"(in_1), "r"(in_2), "r"(in_3), "r"(curW), "r"(fh),
                             "r"(fw), "r"(ic), "r"(I64(hStep)), "r"(I64(dw)), "r"(I64(iStep)),
                             "r"(I64(fwStep)), "r"(I64(fhStep))
                             : "%ecx", "%ebx", "%eax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                             "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",
                             "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");

    } else if ((fh > 0) && (fw > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"

                             "mov %5, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"

                             "mov %6, %%ecx                                     \n\t"
                             ".align 16                                         \n\t"
                             "2:                                                \n\t"

                             "vmovaps (%4), %%ymm12                     \n\t"
                             "vmovaps 0x20(%4), %%ymm13                     \n\t"
                             "vmovaps 0x40(%4), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm9              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm10              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"
                             "add $0x60, %4                                    \n\t"
                             "dec %%ecx                                         \n\t"
                             "jg 2b                                             \n\t"

                             "add %8, %0                                     \n\t"
                             "add %8, %1                                     \n\t"
                             "add %8, %2                                     \n\t"
                             "add %8, %3                                     \n\t"
                             "add %11, %4                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"

                             "add %10, %0                                     \n\t"
                             "add %10, %1                                     \n\t"
                             "add %10, %2                                     \n\t"
                             "add %10, %3                                     \n\t"
                             "add %12, %4                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r"(in_0), "r"(in_1), "r"(in_2), "r"(in_3), "r"(curW), "r"(fh),
                             "r"(fw), "a"(ic), "r"(I64(hStep)), "r"(I64(dw)), "r"(I64(iStep)),
                             "r"(I64(fwStep)), "r"(I64(fhStep))
                             : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                             "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
                             "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
    }

    __asm__ __volatile__(
        // relu
        "and $0x6, %2                                      \n\t"
        "je 0f                                             \n\t"
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
        "and $0x4, %2                                      \n\t"
        "je 0f                                             \n\t"
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
        "vminps %%ymm12, %%ymm10, %%ymm10                  \n\t"
        "vminps %%ymm12, %%ymm11, %%ymm11                  \n\t"

        "0:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm1, 0x20(%0)                          \n\t"
        "vmovups %%ymm2, 0x40(%0)                          \n\t"
        "vmovups %%ymm3, 0x60(%0)                          \n\t"
        "vmovups %%ymm4, (%0, %1)                          \n\t"
        "vmovups %%ymm5, 0x20(%0, %1)                          \n\t"
        "vmovups %%ymm6, 0x40(%0, %1)                              \n\t"
        "vmovups %%ymm7, 0x60(%0, %1)                          \n\t"
        "vmovups %%ymm8, (%0, %1, 2)                          \n\t"
        "vmovups %%ymm9, 0x20(%0, %1, 2)                              \n\t"
        "vmovups %%ymm10, 0x40(%0, %1, 2)                         \n\t"
        "vmovups %%ymm11, 0x60(%0, %1, 2)                         \n\t"
        :
        : "r"(curO), "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_1x24(F32 *in_0,
    F32 *in_1,
    F32 *in_2,
    F32 *in_3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 ic,
    U32 iStep,
    U32 fwStep,
    U32 fhStep)
{
    __asm__ __volatile__("mov %3, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups 0x20(%1), %%ymm4                   \n\t"
                         "vmovups 0x40(%1), %%ymm8                   \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups (%0, %2), %%ymm4                     \n\t"
                         "vmovups (%0, %2, 2), %%ymm8                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store)
                         : "%eax", "%ymm0", "%ymm4", "%ymm8", "memory", "cc");

    if ((fh > 0) && (fw > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"

                             "mov %2, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"

                             "mov %3, %%ecx                                     \n\t"
                             ".align 16                                         \n\t"
                             "2:                                                \n\t"

                             "vmovaps (%1), %%ymm12                     \n\t"
                             "vmovaps 0x20(%1), %%ymm13                     \n\t"
                             "vmovaps 0x40(%1), %%ymm14                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

                             "add %6, %0                                      \n\t"
                             "add $0x60, %1                                    \n\t"
                             "dec %%ecx                                         \n\t"
                             "jg 2b                                             \n\t"

                             "add %5, %0                                     \n\t"
                             "add %8, %1                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"

                             "add %7, %0                                     \n\t"
                             "add %9, %1                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r"(in_0), "r"(curW), "r"(fh), "r"(fw), "a"(ic), "r"(I64(hStep)),
                             "r"(I64(dw)), "r"(I64(iStep)), "r"(I64(fwStep)), "r"(I64(fhStep))
                             : "%ecx", "%ebx", "%ymm0", "%ymm4", "%ymm8", "%ymm12", "%ymm13",
                             "%ymm14", "%ymm15", "memory", "cc");
    }

    __asm__ __volatile__(
        // relu
        "and $0x6, %2                                      \n\t"
        "je 3f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"
        "vmaxps %%ymm15, %%ymm8, %%ymm8                    \n\t"

        // relu6
        "and $0x4, %2                                      \n\t"
        "je 3f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"
        "vminps %%ymm12, %%ymm8, %%ymm8                    \n\t"

        "3:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm4, (%0, %1)                          \n\t"
        "vmovups %%ymm8, (%0, %1, 2)                       \n\t"
        :
        : "r"(curO), "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_4x16(F32 *in_0,
    F32 *in_1,
    F32 *in_2,
    F32 *in_3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 ic,
    U32 iStep,
    U32 fwStep,
    U32 fhStep)
{
    __asm__ __volatile__("mov %3, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups (%1), %%ymm1                       \n\t"
                         "vmovups (%1), %%ymm2                       \n\t"
                         "vmovups (%1), %%ymm3                       \n\t"
                         "vmovups 0x20(%1), %%ymm4                   \n\t"
                         "vmovups 0x20(%1), %%ymm5                   \n\t"
                         "vmovups 0x20(%1), %%ymm6                   \n\t"
                         "vmovups 0x20(%1), %%ymm7                   \n\t"
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

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store)
                         : "%eax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                         "%ymm7", "memory", "cc");

    if ((fh > 0) && (fw > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"

                             "mov %5, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"

                             "mov %6, %%ecx                                     \n\t"
                             ".align 16                                         \n\t"
                             "2:                                                \n\t"

                             "vmovaps (%4), %%ymm12                     \n\t"
                             "vmovaps 0x20(%4), %%ymm13                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                             "vbroadcastss (%1), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm5              \n\t"
                             "vbroadcastss (%2), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm6              \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm7             \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"
                             "add $0x40, %4                                    \n\t"
                             "dec %%ecx                                         \n\t"
                             "jg 2b                                             \n\t"

                             "add %8, %0                                     \n\t"
                             "add %8, %1                                     \n\t"
                             "add %8, %2                                     \n\t"
                             "add %8, %3                                     \n\t"
                             "add %11, %4                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"

                             "add %10, %0                                     \n\t"
                             "add %10, %1                                     \n\t"
                             "add %10, %2                                     \n\t"
                             "add %10, %3                                     \n\t"
                             "add %12, %4                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r"(in_0), "r"(in_1), "r"(in_2), "r"(in_3), "r"(curW), "r"(fh),
                             "r"(fw), "a"(ic), "r"(I64(hStep)), "r"(I64(dw)), "r"(I64(iStep)),
                             "r"(I64(fwStep)), "r"(I64(fhStep))
                             : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                             "%ymm6", "%ymm7", "%ymm12", "%ymm13", "%ymm15", "memory", "cc");
    }

    __asm__ __volatile__(
        // relu
        "and $0x6, %2                                      \n\t"
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
        "and $0x4, %2                                      \n\t"
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

        "3:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm1, 0x20(%0)                          \n\t"
        "vmovups %%ymm2, 0x40(%0)                          \n\t"
        "vmovups %%ymm3, 0x60(%0)                          \n\t"
        "vmovups %%ymm4, (%0, %1)                          \n\t"
        "vmovups %%ymm5, 0x20(%0, %1)                          \n\t"
        "vmovups %%ymm6, 0x40(%0, %1)                              \n\t"
        "vmovups %%ymm7, 0x60(%0, %1)                          \n\t"
        :
        : "r"(curO), "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm12",
        "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_1x16(F32 *in_0,
    F32 *in_1,
    F32 *in_2,
    F32 *in_3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 ic,
    U32 iStep,
    U32 fwStep,
    U32 fhStep)
{
    __asm__ __volatile__("mov %3, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups 0x20(%1), %%ymm4                   \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups (%0, %2), %%ymm4                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(I64(oStep)), "r"(store)
                         : "%eax", "%ymm0", "%ymm4", "memory", "cc");

    if ((fh > 0) && (fw > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"

                             "mov %2, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"

                             "mov %3, %%ecx                                     \n\t"
                             ".align 16                                         \n\t"
                             "2:                                                \n\t"

                             "vmovaps (%1), %%ymm12                     \n\t"
                             "vmovaps 0x20(%1), %%ymm13                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

                             "add %6, %0                                      \n\t"
                             "add $0x40, %1                                    \n\t"
                             "dec %%ecx                                         \n\t"
                             "jg 2b                                             \n\t"

                             "add %5, %0                                     \n\t"
                             "add %8, %1                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"

                             "add %7, %0                                     \n\t"
                             "add %9, %1                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r"(in_0), "r"(curW), "r"(fh), "r"(fw), "a"(ic), "r"(I64(hStep)),
                             "r"(I64(dw)), "r"(I64(iStep)), "r"(I64(fwStep)), "r"(I64(fhStep))
                             : "%ecx", "%ebx", "%ymm0", "%ymm4", "%ymm8", "%ymm12", "%ymm13",
                             "%ymm15", "memory", "cc");
    }

    __asm__ __volatile__(
        // relu
        "and $0x6, %2                                      \n\t"
        "je 3f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm4, %%ymm4                    \n\t"

        // relu6
        "and $0x4, %2                                      \n\t"
        "je 3f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm4, %%ymm4                    \n\t"

        "3:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm4, (%0, %1)                          \n\t"
        :
        : "r"(curO), "r"(I64(oStep)), "r"(store)
        : "%ecx", "%ymm0", "%ymm4", "%ymm12", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_4x8(F32 *in_0,
    F32 *in_1,
    F32 *in_2,
    F32 *in_3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 ic,
    U32 iStep,
    U32 fwStep,
    U32 fhStep)
{
    __asm__ __volatile__("mov %2, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "vmovups (%1), %%ymm1                       \n\t"
                         "vmovups (%1), %%ymm2                       \n\t"
                         "vmovups (%1), %%ymm3                       \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups 0x20(%0), %%ymm1                     \n\t"
                         "vmovups 0x40(%0), %%ymm2                     \n\t"
                         "vmovups 0x60(%0), %%ymm3                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(store)
                         : "%eax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "memory", "cc");

    if ((fh > 0) && (fw > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"

                             "mov %5, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"

                             "mov %6, %%ecx                                     \n\t"
                             ".align 16                                         \n\t"
                             "2:                                                \n\t"

                             "vmovaps (%4), %%ymm12                     \n\t"
                             "vbroadcastss (%0), %%ymm11                             \n\t"
                             "vbroadcastss (%1), %%ymm13                             \n\t"
                             "vbroadcastss (%2), %%ymm14                             \n\t"
                             "vbroadcastss (%3), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm11, %%ymm12, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm14, %%ymm12, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

                             "add %9, %0                                      \n\t"
                             "add %9, %1                                      \n\t"
                             "add %9, %2                                      \n\t"
                             "add %9, %3                                      \n\t"
                             "add $0x20, %4                                    \n\t"
                             "dec %%ecx                                         \n\t"
                             "jg 2b                                             \n\t"

                             "add %8, %0                                     \n\t"
                             "add %8, %1                                     \n\t"
                             "add %8, %2                                     \n\t"
                             "add %8, %3                                     \n\t"
                             "add %11, %4                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"

                             "add %10, %0                                     \n\t"
                             "add %10, %1                                     \n\t"
                             "add %10, %2                                     \n\t"
                             "add %10, %3                                     \n\t"
                             "add %12, %4                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r"(in_0), "r"(in_1), "r"(in_2), "r"(in_3), "r"(curW), "r"(fh),
                             "r"(fw), "a"(ic), "r"(I64(hStep)), "r"(I64(dw)), "r"(I64(iStep)),
                             "r"(I64(fwStep)), "r"(I64(fhStep))
                             : "%ecx", "%ebx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm11",
                             "%ymm14", "%ymm12", "%ymm13", "%ymm15", "memory", "cc");
    }

    __asm__ __volatile__(
        // relu
        "and $0x6, %1                                      \n\t"
        "je 3f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"
        "vmaxps %%ymm15, %%ymm1, %%ymm1                    \n\t"
        "vmaxps %%ymm15, %%ymm2, %%ymm2                    \n\t"
        "vmaxps %%ymm15, %%ymm3, %%ymm3                    \n\t"

        // relu6
        "and $0x4, %1                                      \n\t"
        "je 3f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"
        "vminps %%ymm12, %%ymm1, %%ymm1                    \n\t"
        "vminps %%ymm12, %%ymm2, %%ymm2                    \n\t"
        "vminps %%ymm12, %%ymm3, %%ymm3                    \n\t"

        "3:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        "vmovups %%ymm1, 0x20(%0)                          \n\t"
        "vmovups %%ymm2, 0x40(%0)                          \n\t"
        "vmovups %%ymm3, 0x60(%0)                          \n\t"
        :
        : "r"(curO), "r"(store)
        : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm12", "%ymm15", "memory", "cc");
}

void avx2_conv_kernel_1x8(F32 *in_0,
    F32 *in_1,
    F32 *in_2,
    F32 *in_3,
    const F32 *curW,
    F32 *curO,
    const F32 *curB,
    U32 fw,
    U32 fh,
    U32 oStep,
    U32 hStep,
    U32 store,
    U32 dw,
    U32 ic,
    U32 iStep,
    U32 fwStep,
    U32 fhStep)
{
    __asm__ __volatile__("mov %2, %%eax                                  \n\t"
                         "and $0x1, %%eax                                  \n\t"
                         "jne 0f                                             \n\t"
                         "vmovups (%1), %%ymm0                       \n\t"
                         "jmp 1f                                             \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"

                         ".align 16                                         \n\t"
                         "1:                                      \n\t"
                         :
                         : "r"(curO), "r"(curB), "r"(store)
                         : "%eax", "%ymm0", "memory", "cc");

    if ((fh > 0) && (fw > 0)) {
        __asm__ __volatile__(".align 16                                         \n\t"
                             "0:                                                \n\t"

                             "mov %2, %%ebx                                     \n\t"
                             ".align 16                                         \n\t"
                             "1:                                                \n\t"

                             "mov %3, %%ecx                                     \n\t"
                             ".align 16                                         \n\t"
                             "2:                                                \n\t"

                             "vmovaps (%1), %%ymm12                     \n\t"
                             "vbroadcastss (%0), %%ymm15                             \n\t"
                             "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

                             "add %6, %0                                      \n\t"
                             "add $0x20, %1                                    \n\t"
                             "dec %%ecx                                         \n\t"
                             "jg 2b                                             \n\t"

                             "add %5, %0                                     \n\t"
                             "add %8, %1                                    \n\t"
                             "dec %%ebx                                         \n\t"
                             "jg 1b                                             \n\t"

                             "add %7, %0                                     \n\t"
                             "add %9, %1                                    \n\t"
                             "dec %%eax                                         \n\t"
                             "jg 0b                                             \n\t"
                             :
                             : "r"(in_0), "r"(curW), "r"(fh), "r"(fw), "a"(ic), "r"(I64(hStep)),
                             "r"(I64(dw)), "r"(I64(iStep)), "r"(I64(fwStep)), "r"(I64(fhStep))
                             : "%ecx", "%ebx", "%ymm0", "%ymm12", "%ymm15", "memory", "cc");
    }

    __asm__ __volatile__(
        // relu
        "and $0x6, %1                                      \n\t"
        "je 0f                                             \n\t"
        "vxorps %%ymm15, %%ymm15, %%ymm15                  \n\t"
        "vmaxps %%ymm15, %%ymm0, %%ymm0                    \n\t"

        // relu6
        "and $0x4, %1                                      \n\t"
        "je 0f                                             \n\t"
        "mov $0x40C00000, %%ecx                            \n\t"
        "vmovd %%ecx, %%xmm12                              \n\t"
        "vpermps %%ymm12, %%ymm15, %%ymm12                 \n\t"
        "vminps %%ymm12, %%ymm0, %%ymm0                    \n\t"

        "0:                                                \n\t"
        "vmovups %%ymm0, (%0)                              \n\t"
        :
        : "r"(curO), "r"(store)
        : "%ecx", "%ymm0", "%ymm12", "%ymm15", "memory", "cc");
}

EE convolution_direct_nchw(TensorDesc inputDesc,
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

    I32 strideH = convParamSpec.stride_h;
    I32 strideW = convParamSpec.stride_w;
    I32 paddingT = convParamSpec.padding_top;
    I32 paddingB = convParamSpec.padding_bottom;
    I32 paddingL = convParamSpec.padding_left;
    I32 paddingR = convParamSpec.padding_right;
    I32 dilateH = convParamSpec.dilatedRate_h;
    I32 dilateW = convParamSpec.dilatedRate_w;

    if (((fdf != DF_NCHWCxN24) && (fdf != DF_NCHWCxN32)) || (idf != DF_NCHW)) {
        CHECK_STATUS(NOT_MATCH);
    }

    F32 *curI, *curO, *calI, *calO;
    const F32 *curW, *curB, *calW;
    F32 *ftmp = inArray;
    filterArray = (F32 *)align_addr(filterArray, 32);

    U32 oStep = oh * ow * UNROLL_OC_DIM * 4;
    U32 iStep = ((ih - fh) * iw) * 4;
    U32 hStep = (iw - fw * dilateW + (dilateH - 1) * iw) * 4;
    U32 dw = dilateW * 4;
    U32 wSize = 0, store = 0, ocSize = 0, icSize = 0, hwSize = 0, icbSize = 0;
    I32 ih_idx = 0;
    kernel_func kernel[4][2] = {{avx2_conv_kernel_1x8, avx2_conv_kernel_4x8},
        {avx2_conv_kernel_1x16, avx2_conv_kernel_4x16},
        {avx2_conv_kernel_1x24, avx2_conv_kernel_4x24},
        {avx2_conv_kernel_1x32, avx2_conv_kernel_3x32}};
    U32 ocblocks[4] = {8, 16, 24, 32};
    U32 wblocks[4] = {4, 4, 4, 3};
    U32 unroll_w = UNROLL_W, unroll_oc = BLOCK_OC_DIM;
    I32 ohow = oh * ow;

    U32 k24 = (oc + 23) / 24 * (ohow + 3) / 4;
    U32 k32 = (oc + 31) / 32 * (ohow + 2) / 3;
    if (k32 < k24) {
        unroll_oc = 32;
    }

    I32 oh_padding_t = 0;
    I32 oh_padding_b = 0;

    for (U32 n = 0; n < in; ++n) {
        store = 0;
        for (U32 icbb = 0; icbb < ic; icbb += icSize) {
            icSize = UNI_MIN(BLOCK_IC_DIM, ic - icbb);
            store |= (icbb > 0);
            if (icbb == ic - icSize) {
                store |= U32(activationDesc.mode) << 1;
            }
            if ((paddingL == 0) && (paddingR == 0) && (paddingT != 0 || paddingB != 0)) {
                oh_padding_t = UNI_MIN((paddingT - 1) / strideH + 1, oh);
                oh_padding_b = UNI_MIN((paddingB - 1) / strideH + 1, oh - oh_padding_t);
                if (((ih + paddingT - fh) / strideH + 1) >= oh) {
                    oh_padding_b = 0;
                }
                for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                    ocSize = UNI_MIN(unroll_oc, oc - ocb);
                    ocSize = ocblocks[(ocSize >> 3) - 1];
                    unroll_w = wblocks[(ocSize >> 3) - 1];
                    curW = filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                    curB = biasArray + ocb;
                    curI = ftmp + icbb * ih * iw;
                    for (I32 h = 0; h < oh_padding_t; ++h) {
                        I32 in_h_0 = h * strideH - paddingT;
                        U32 tfh = UNI_MIN(fh + in_h_0, ih);
                        iStep = ((ih - tfh) * iw) * 4;
                        for (I32 w = 0; w < ow; w += wSize) {
                            wSize = UNI_MIN(ow - w, unroll_w);
                            if (wSize < unroll_w) {
                                wSize = 1;
                            }
                            U32 in_w_0 = w * strideW;
                            U32 in_w_1 = (w + 1) * strideW;
                            U32 in_w_2 = (w + 2) * strideW;
                            U32 in_w_3 = (w + 3) * strideW;
                            F32 *out_ptr = outArray + (n * oc + ocb) * ohow + (h * ow + w) * 8;
                            F32 *in_0 = curI + in_w_0;
                            F32 *in_1 = curI + in_w_1;
                            F32 *in_2 = curI + in_w_2;
                            F32 *in_3 = curI + in_w_3;
                            kernel[(ocSize >> 3) - 1][wSize > 1](in_0, in_1, in_2, in_3, curW + (fh - tfh) * fw * ocSize,
                                out_ptr, curB, fw, tfh, oStep, hStep, store, dw, icSize, iStep, 0, fw * (fh - tfh) * ocSize * 4);
                        }
                    }
                }
            }
            if ((paddingL == 0) && (paddingR == 0)) {
                iStep = ((ih - fh) * iw) * 4;
                for (I32 hw = oh_padding_t * ow; hw < ohow - oh_padding_b * ow; hw += hwSize) {
                    hwSize = UNI_MIN(BLOCK_HW_DIM, ohow - oh_padding_b * ow - hw);
                    for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                        ocSize = UNI_MIN(unroll_oc, oc - ocb);
                        ocSize = ocblocks[(ocSize >> 3) - 1];
                        unroll_w = wblocks[(ocSize >> 3) - 1];
                        curW = filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                        curB = biasArray + ocb;
                        curI = ftmp + icbb * ih * iw;
                        for (I32 ihw = hw; ihw < hw + hwSize; ihw += wSize) {
                            wSize = UNI_MIN(hw + hwSize - ihw, unroll_w);
                            if (wSize < unroll_w) {
                                wSize = 1;
                            }
                            U32 in_h_0 = ihw / ow * strideH - paddingT;
                            U32 in_w_0 = ihw % ow * strideW;
                            U32 in_h_1 = (ihw + 1) / ow * strideH - paddingT;
                            U32 in_w_1 = (ihw + 1) % ow * strideW;
                            U32 in_h_2 = (ihw + 2) / ow * strideH - paddingT;
                            U32 in_w_2 = (ihw + 2) % ow * strideW;
                            U32 in_h_3 = (ihw + 3) / ow * strideH - paddingT;
                            U32 in_w_3 = (ihw + 3) % ow * strideW;
                            F32 *out_ptr = outArray + (n * oc + ocb) * ohow + ihw * 8;
                            F32 *in_0 = curI + in_h_0 * iw + in_w_0;
                            F32 *in_1 = curI + in_h_1 * iw + in_w_1;
                            F32 *in_2 = curI + in_h_2 * iw + in_w_2;
                            F32 *in_3 = curI + in_h_3 * iw + in_w_3;
                            kernel[(ocSize >> 3) - 1][wSize > 1](in_0, in_1, in_2, in_3, curW,
                                out_ptr, curB, fw, fh, oStep, hStep, store, dw, icSize, iStep, 0, 0);
                        }
                    }
                }
            }
            if ((paddingL == 0) && (paddingR == 0) && (paddingT != 0 || paddingB != 0)) {
                for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                    ocSize = UNI_MIN(unroll_oc, oc - ocb);
                    ocSize = ocblocks[(ocSize >> 3) - 1];
                    unroll_w = wblocks[(ocSize >> 3) - 1];
                    curW = filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw;
                    curB = biasArray + ocb;
                    curI = ftmp + icbb * ih * iw;
                    for (I32 h = oh - oh_padding_b; h < oh; ++h) {
                        I32 in_h_0 = h * strideH - paddingT;
                        U32 tfh = ih - in_h_0;
                        iStep = ((ih - tfh) * iw) * 4;
                        for (I32 w = 0; w < ow; w += wSize) {
                            wSize = UNI_MIN(ow - w, unroll_w);
                            if (wSize < unroll_w) {
                                wSize = 1;
                            }
                            U32 in_w_0 = w * strideW;
                            U32 in_w_1 = (w + 1) * strideW;
                            U32 in_w_2 = (w + 2) * strideW;
                            U32 in_w_3 = (w + 3) * strideW;
                            F32 *out_ptr = outArray + (n * oc + ocb) * ohow + (h * ow + w) * 8;
                            F32 *in_0 = curI + in_h_0 * iw + in_w_0;
                            F32 *in_1 = curI + in_h_0 * iw + in_w_1;
                            F32 *in_2 = curI + in_h_0 * iw + in_w_2;
                            F32 *in_3 = curI + in_h_0 * iw + in_w_3;
                            kernel[(ocSize >> 3) - 1][wSize > 1](in_0, in_1, in_2, in_3, curW,
                                out_ptr, curB, fw, tfh, oStep, hStep, store, dw, icSize, iStep, 0, fw * (fh - tfh) * ocSize * 4);
                        }
                    }
                }
            }
            if ((paddingL != 0) || (paddingR != 0)) {
                I32 tfw = fw, tfh = fh, wh = 0;
                I32 in_h = 0, in_w = 0;
                I32 ow_padding_l = UNI_MIN((paddingL - 1) / strideW + 1, ow);
                I32 ow_padding_r = UNI_MIN((paddingR - 1) / strideW + 1, ow - ow_padding_l);
                if (((iw + paddingL - fw) / strideW + 1) >= ow) {
                    ow_padding_r = 0;
                }
                for (I32 h = 0; h < oh; ++h) {
                    tfh = fh;
                    in_h = h * strideH - paddingT;
                    calW = curW;
                    wh = 0;
                    if (in_h < 0) {
                        tfh = UNI_MIN(fh + in_h, ih);
                        in_h = 0;
                        wh = fh - tfh;
                    } else if (in_h + fh >= ih) {
                        tfh = ih - in_h;
                        curW = filterArray;
                    }
                    iStep = ((ih - tfh) * iw) * 4;
                    for (U32 ocb = 0; ocb < oc; ocb += ocSize) {
                        ocSize = UNI_MIN(unroll_oc, oc - ocb);
                        ocSize = ocblocks[(ocSize >> 3) - 1];
                        unroll_w = wblocks[(ocSize >> 3) - 1];
                        curW = filterArray + ocb * ic * fh * fw + ocSize * icbb * fh * fw +
                            wh * fw * ocSize;
                        curB = biasArray + ocb;
                        curI = ftmp + icbb * ih * iw + in_h * iw;
                        curO = outArray + ocb * ohow + h * ow * 8;
                        I32 w = 0;
                        for (; w < ow_padding_l; ++w) {
                            I32 in_w = w * strideW - paddingL;
                            tfw = UNI_MIN(fw + in_w, iw);
                            const F32 *useW = curW + (fw - tfw) * ocSize;
                            hStep = (iw - tfw * dilateW + (dilateH - 1) * iw) * 4;
                            calO = curO + w * 8;
                            kernel[(ocSize >> 3) - 1][0](curI, nullptr, nullptr, nullptr, useW,
                                calO, curB, tfw, tfh, oStep, hStep, store, dw, icSize, iStep,
                                (fw - tfw) * ocSize * 4, fw * (fh - tfh) * ocSize * 4);
                        }
                        for (; w < ow - ow_padding_r; w += wSize) {
                            hStep = (iw - fw * dilateW + (dilateH - 1) * iw) * 4;
                            wSize = UNI_MIN(ow - ow_padding_r - w, unroll_w);
                            if (wSize < unroll_w) {
                                wSize = 1;
                            }
                            F32 *in_0 = curI + w * strideW - paddingL;
                            F32 *in_1 = curI + (w + 1) * strideW - paddingL;
                            F32 *in_2 = curI + (w + 2) * strideW - paddingL;
                            F32 *in_3 = curI + (w + 3) * strideW - paddingL;
                            calO = curO + w * 8;
                            kernel[(ocSize >> 3) - 1][wSize > 1](in_0, in_1, in_2, in_3, curW, calO,
                                curB, fw, tfh, oStep, hStep, store, dw, icSize, iStep, 0,
                                fw * (fh - tfh) * ocSize * 4);
                        }
                        for (; w < ow; ++w) {
                            I32 in_w = w * strideW - paddingL;
                            tfw = iw - in_w;
                            hStep = (iw - tfw * dilateW + (dilateH - 1) * iw) * 4;
                            F32 *in_0 = curI + in_w;
                            calO = curO + w * 8;
                            kernel[(ocSize >> 3) - 1][0](in_0, nullptr, nullptr, nullptr, curW,
                                calO, curB, tfw, tfh, oStep, hStep, store, dw, icSize, iStep,
                                (fw - tfw) * ocSize * 4, fw * (fh - tfh) * ocSize * 4);
                        }
                    }
                }
            }
        }
        inArray += ic * ih * iw;
        outArray += oc * oh * ow;
    }
    return SUCCESS;
}
