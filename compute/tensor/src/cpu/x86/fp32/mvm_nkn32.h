// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MVM_NKN32
#define _H_MVM_NKN32
#include "tensor_desc.h"
#include "thread_affinity.h"

inline void mvm_nkn32_with_bias(
    U32 fn, U32 fk, const F32 *filterArray, const F32 *input, F32 *output, const F32 *bias)
{
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 n = 0; n < fn; ++n) {
        const F32 *f = filterArray + n * fk * 32;
        F32 *out = output + n * 32;
        const F32 *b = bias + n * 32;
        if (bias == nullptr) {
            __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                                 "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                                 "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                                 "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
                                 :
                                 :
                                 : "%ymm0", "%ymm1", "%ymm2", "%ymm3");
        } else {
            __asm__ __volatile__("vmovups (%0), %%ymm0                     \n\t"
                                 "vmovups 0x20(%0), %%ymm1                     \n\t"
                                 "vmovups 0x40(%0), %%ymm2                     \n\t"
                                 "vmovups 0x60(%0), %%ymm3                     \n\t"
                                 :
                                 : "r"(b)
                                 : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "memory");
        }
        __asm__ __volatile__("mov %1, %%rax                                     \n\t"
                             "mov %3, %%ecx                                     \n\t"
                             "shr $3, %%ecx                                     \n\t"
                             "je 1f                                \n\t"
                             ".align 16                                         \n\t"
                             "0:                                      \n\t"

                             "vmovups (%0), %%ymm4                             \n\t"
                             "vmovups 0x20(%0), %%ymm5                         \n\t"
                             "vmovups 0x40(%0), %%ymm6                         \n\t"
                             "vmovups 0x60(%0), %%ymm7                         \n\t"
                             "vbroadcastss 0x0(%%rax), %%ymm8                     \n\t"
                             "vmovups 0x80(%0), %%ymm9                             \n\t"
                             "vmovups 0xA0(%0), %%ymm10                         \n\t"
                             "vmovups 0xC0(%0), %%ymm11                         \n\t"
                             "vmovups 0xE0(%0), %%ymm12                         \n\t"
                             "vbroadcastss 0x4(%%rax), %%ymm13                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"

                             "vmovups 0x100(%0), %%ymm4                             \n\t"
                             "vmovups 0x120(%0), %%ymm5                         \n\t"
                             "vmovups 0x140(%0), %%ymm6                         \n\t"
                             "vmovups 0x160(%0), %%ymm7                         \n\t"
                             "vbroadcastss 0x8(%%rax), %%ymm8                     \n\t"
                             "vfmadd231ps %%ymm13, %%ymm9, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm10, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm11, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                             "vmovups 0x180(%0), %%ymm9                              \n\t"
                             "vmovups 0x1A0(%0), %%ymm10                         \n\t"
                             "vmovups 0x1C0(%0), %%ymm11                         \n\t"
                             "vmovups 0x1E0(%0), %%ymm12                         \n\t"
                             "vbroadcastss 0xC(%%rax), %%ymm13                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"

                             "vmovups 0x200(%0), %%ymm4                             \n\t"
                             "vmovups 0x220(%0), %%ymm5                         \n\t"
                             "vmovups 0x240(%0), %%ymm6                         \n\t"
                             "vmovups 0x260(%0), %%ymm7                         \n\t"
                             "vbroadcastss 0x10(%%rax), %%ymm8                     \n\t"
                             "vfmadd231ps %%ymm13, %%ymm9 , %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm10, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm11, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                             "vmovups 0x280(%0), %%ymm9                              \n\t"
                             "vmovups 0x2A0(%0), %%ymm10                         \n\t"
                             "vmovups 0x2C0(%0), %%ymm11                         \n\t"
                             "vmovups 0x2E0(%0), %%ymm12                         \n\t"
                             "vbroadcastss 0x14(%%rax), %%ymm13                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"

                             "vmovups 0x300(%0), %%ymm4                             \n\t"
                             "vmovups 0x320(%0), %%ymm5                         \n\t"
                             "vmovups 0x340(%0), %%ymm6                         \n\t"
                             "vmovups 0x360(%0), %%ymm7                         \n\t"
                             "vbroadcastss 0x18(%%rax), %%ymm8                     \n\t"
                             "vfmadd231ps %%ymm13, %%ymm9 , %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm10, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm11, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                             "vmovups 0x380(%0), %%ymm9                              \n\t"
                             "vmovups 0x3A0(%0), %%ymm10                         \n\t"
                             "vmovups 0x3C0(%0), %%ymm11                         \n\t"
                             "vmovups 0x3E0(%0), %%ymm12                         \n\t"
                             "vbroadcastss 0x1C(%%rax), %%ymm13                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"

                             "vfmadd231ps %%ymm13, %%ymm9 , %%ymm0              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm10, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm11, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm13, %%ymm12, %%ymm3              \n\t"

                             "add $0x400, %0                                    \n\t"
                             "add $0x20, %%rax                                     \n\t"

                             "sub $1, %%ecx                                     \n\t"
                             "jg 0b                                    \n\t"
                             ".align 16                                         \n\t"
                             "1:                                  \n\t"

                             "mov %3, %%ecx                                     \n\t"
                             "and $7, %%ecx                                     \n\t"
                             "je 3f                         \n\t"
                             "2:                               \n\t"
                             "vmovups (%0), %%ymm4                             \n\t"
                             "vmovups 0x20(%0), %%ymm5                         \n\t"
                             "vmovups 0x40(%0), %%ymm6                         \n\t"
                             "vmovups 0x60(%0), %%ymm7                         \n\t"
                             "vbroadcastss (%%rax), %%ymm8                     \n\t"
                             "vfmadd231ps %%ymm8, %%ymm4, %%ymm0              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm5, %%ymm1              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm6, %%ymm2              \n\t"
                             "vfmadd231ps %%ymm8, %%ymm7, %%ymm3              \n\t"
                             "add $0x80, %0                                     \n\t"
                             "add $0x4, %%rax                                     \n\t"
                             "sub $1, %%ecx                                     \n\t"
                             "jg 2b                             \n\t"

                             "3:                           \n\t"
                             "vmovups %%ymm0,  (%2)                             \n\t"
                             "vmovups %%ymm1,  0x20(%2)                         \n\t"
                             "vmovups %%ymm2,  0x40(%2)                         \n\t"
                             "vmovups %%ymm3,  0x60(%2)                         \n\t"

                             : "+r"(f)
                             : "r"(input), "r"(out), "r"(fk)
                             : "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",
                             "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
                             "%ymm13", "memory");
    }
}
#endif
