// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_POOLING_KERNEL
#define _H_POOLING_KERNEL

inline void pooling_max_w4(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    __asm__ __volatile__("mov %%eax, %%eax                                  \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "mov %%rax, %%rdi                                  \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "mov %%rax, %%r9                                  \n\t"
                         "add %%r9, %%r9                                  \n\t"
                         "mov %%rax, %%r10                                  \n\t"
                         "add %%r9, %%r10                                  \n\t"
                         "add %0, %%rax                                  \n\t"
                         "add %0, %%r9                                  \n\t"
                         "add %0, %%r10                                  \n\t"

                         "vmovups (%0), %%ymm0                     \n\t"
                         "vmovups (%%rax), %%ymm1                     \n\t"
                         "vmovups (%%r9), %%ymm2                     \n\t"
                         "vmovups (%%r10), %%ymm3                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %2, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovups (%0), %%ymm4                     \n\t"
                         "vmovups (%%rax), %%ymm5                     \n\t"
                         "vmovups (%%r9), %%ymm6                     \n\t"
                         "vmovups (%%r10), %%ymm7                     \n\t"

                         "vmaxps %%ymm0, %%ymm4, %%ymm0                     \n\t"
                         "vmaxps %%ymm1, %%ymm5, %%ymm1                     \n\t"
                         "vmaxps %%ymm2, %%ymm6, %%ymm2                     \n\t"
                         "vmaxps %%ymm3, %%ymm7, %%ymm3                     \n\t"

                         "add $0x20, %0                                      \n\t"
                         "add $0x20, %%rax                                      \n\t"
                         "add $0x20, %%r9                                      \n\t"
                         "add $0x20, %%r10                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %%rdi, %0                                      \n\t"
                         "add %%rdi, %%rax                                      \n\t"
                         "add %%rdi, %%r9                                      \n\t"
                         "add %%rdi, %%r10                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "vmovups %%ymm0, (%1)                              \n\t"
                         "vmovups %%ymm1, 0x20(%1)                          \n\t"
                         "vmovups %%ymm2, 0x40(%1)                          \n\t"
                         "vmovups %%ymm3, 0x60(%1)                          \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
                         : "%eax", "%rax", "%ecx", "%r10", "%r9", "%rdi", "%ymm0", "%ymm1", "%ymm2",
                         "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "memory", "cc");
}

inline void pooling_max_w2(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %%eax, %%eax                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "vmovups (%0), %%ymm0                     \n\t"
        "vmovups (%%rax), %%ymm1                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%ymm4                     \n\t"
        "vmovups (%%rax), %%ymm5                     \n\t"
        "vmaxps %%ymm0, %%ymm4, %%ymm0                     \n\t"
        "vmaxps %%ymm1, %%ymm5, %%ymm1                     \n\t"
        "add $0x20, %0                                      \n\t"
        "add $0x20, %%rax                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "vmovups %%ymm1, 0x20(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
        : "%eax", "%rax", "%ecx", "%rdi", "%ymm0", "%ymm1", "%ymm4", "%ymm5", "memory", "cc");
}

inline void pooling_max_w1(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    __asm__ __volatile__("mov %%eax, %%eax                                  \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "mov %%rax, %%rdi                                  \n\t"
                         "vmovups (%0), %%ymm0                     \n\t"
                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "mov %2, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "vmovups (%0), %%ymm4                     \n\t"
                         "vmaxps %%ymm0, %%ymm4, %%ymm0                     \n\t"
                         "add $0x20, %0                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"
                         "add %%rdi, %0                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"
                         "vmovups %%ymm0, (%1)                              \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
                         : "%eax", "%rax", "%ecx", "%rdi", "%ymm0", "%ymm4", "memory", "cc");
}

inline void pooling_mean_w4(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "mov %%rax, %%r9                                  \n\t"
        "add %%r9, %%r9                                  \n\t"
        "mov %%rax, %%r10                                  \n\t"
        "add %%r9, %%r10                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "add %0, %%r9                                  \n\t"
        "add %0, %%r10                                  \n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%ymm4                     \n\t"
        "vmovups (%%rax), %%ymm5                     \n\t"
        "vmovups (%%r9), %%ymm6                     \n\t"
        "vmovups (%%r10), %%ymm7                     \n\t"
        "vaddps %%ymm0, %%ymm4, %%ymm0                     \n\t"
        "vaddps %%ymm1, %%ymm5, %%ymm1                     \n\t"
        "vaddps %%ymm2, %%ymm6, %%ymm2                     \n\t"
        "vaddps %%ymm3, %%ymm7, %%ymm3                     \n\t"
        "add $0x20, %0                                      \n\t"
        "add $0x20, %%rax                                      \n\t"
        "add $0x20, %%r9                                      \n\t"
        "add $0x20, %%r10                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "add %%rdi, %%r9                                      \n\t"
        "add %%rdi, %%r10                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%ymm4                     \n\t"
        "vdivps %%ymm4, %%ymm0, %%ymm0                     \n\t"
        "vdivps %%ymm4, %%ymm1, %%ymm1                     \n\t"
        "vdivps %%ymm4, %%ymm2, %%ymm2                     \n\t"
        "vdivps %%ymm4, %%ymm3, %%ymm3                     \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "vmovups %%ymm1, 0x20(%1)                          \n\t"
        "vmovups %%ymm2, 0x40(%1)                          \n\t"
        "vmovups %%ymm3, 0x60(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%r10", "%r9", "%rdi", "%ymm0", "%ymm1", "%ymm2", "%ymm3",
        "%ymm4", "%ymm5", "%ymm6", "%ymm7", "memory", "cc");
}

inline void pooling_mean_w2(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%ymm4                     \n\t"
        "vmovups (%%rax), %%ymm5                     \n\t"
        "vaddps %%ymm0, %%ymm4, %%ymm0                     \n\t"
        "vaddps %%ymm1, %%ymm5, %%ymm1                     \n\t"
        "add $0x20, %0                                      \n\t"
        "add $0x20, %%rax                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%ymm4                     \n\t"
        "vdivps %%ymm4, %%ymm0, %%ymm0                     \n\t"
        "vdivps %%ymm4, %%ymm1, %%ymm1                     \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        "vmovups %%ymm1, 0x20(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%rdi", "%ymm0", "%ymm1", "%ymm4", "%ymm5", "memory", "cc");
}

inline void pooling_mean_w1(const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%ymm4                     \n\t"
        "vaddps %%ymm0, %%ymm4, %%ymm0                     \n\t"
        "add $0x20, %0                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%ymm4                     \n\t"
        "vdivps %%ymm4, %%ymm0, %%ymm0                     \n\t"
        "vmovups %%ymm0, (%1)                              \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%rdi", "%ymm0", "%ymm4", "memory", "cc");
}

inline void pooling_c16_max_w4(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    __asm__ __volatile__("mov %%eax, %%eax                                  \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "mov %%rax, %%rdi                                  \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "mov %5, %%eax                                  \n\t"
                         "mov %%rax, %%r9                                  \n\t"
                         "add %%r9, %%r9                                  \n\t"
                         "mov %%rax, %%r10                                  \n\t"
                         "add %%r9, %%r10                                  \n\t"
                         "add %0, %%rax                                  \n\t"
                         "add %0, %%r9                                  \n\t"
                         "add %0, %%r10                                  \n\t"

                         "vmovups (%0), %%zmm0                     \n\t"
                         "vmovups (%%rax), %%zmm1                     \n\t"
                         "vmovups (%%r9), %%zmm2                     \n\t"
                         "vmovups (%%r10), %%zmm3                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                                \n\t"

                         "mov %2, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"

                         "vmovups (%0), %%zmm4                     \n\t"
                         "vmovups (%%rax), %%zmm5                     \n\t"
                         "vmovups (%%r9), %%zmm6                     \n\t"
                         "vmovups (%%r10), %%zmm7                     \n\t"

                         "vmaxps %%zmm0, %%zmm4, %%zmm0                     \n\t"
                         "vmaxps %%zmm1, %%zmm5, %%zmm1                     \n\t"
                         "vmaxps %%zmm2, %%zmm6, %%zmm2                     \n\t"
                         "vmaxps %%zmm3, %%zmm7, %%zmm3                     \n\t"

                         "add $0x40, %0                                      \n\t"
                         "add $0x40, %%rax                                      \n\t"
                         "add $0x40, %%r9                                      \n\t"
                         "add $0x40, %%r10                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"

                         "add %%rdi, %0                                      \n\t"
                         "add %%rdi, %%rax                                      \n\t"
                         "add %%rdi, %%r9                                      \n\t"
                         "add %%rdi, %%r10                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "vmovups %%zmm0, (%1)                              \n\t"
                         "vmovups %%zmm1, 0x40(%1)                          \n\t"
                         "vmovups %%zmm2, 0x80(%1)                          \n\t"
                         "vmovups %%zmm3, 0xC0(%1)                          \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
                         : "%eax", "%rax", "%ecx", "%r10", "%r9", "%rdi", "%zmm0", "%zmm1", "%zmm2",
                         "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7", "memory", "cc");
}

inline void pooling_c16_max_w2(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %%eax, %%eax                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "vmovups (%0), %%zmm0                     \n\t"
        "vmovups (%%rax), %%zmm1                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%zmm4                     \n\t"
        "vmovups (%%rax), %%zmm5                     \n\t"
        "vmaxps %%zmm0, %%zmm4, %%zmm0                     \n\t"
        "vmaxps %%zmm1, %%zmm5, %%zmm1                     \n\t"
        "add $0x40, %0                                      \n\t"
        "add $0x40, %%rax                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vmovups %%zmm0, (%1)                              \n\t"
        "vmovups %%zmm1, 0x40(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
        : "%eax", "%rax", "%ecx", "%rdi", "%zmm0", "%zmm1", "%zmm4", "%zmm5", "memory", "cc");
}

inline void pooling_c16_max_w1(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    __asm__ __volatile__("mov %%eax, %%eax                                  \n\t"
                         "mov %4, %%eax                                  \n\t"
                         "mov %%rax, %%rdi                                  \n\t"
                         "vmovups (%0), %%zmm0                     \n\t"
                         ".align 16                                         \n\t"
                         "0:                                                \n\t"
                         "mov %2, %%ecx                                     \n\t"
                         ".align 16                                         \n\t"
                         "1:                                                \n\t"
                         "vmovups (%0), %%zmm4                     \n\t"
                         "vmaxps %%zmm0, %%zmm4, %%zmm0                     \n\t"
                         "add $0x40, %0                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 1b                                             \n\t"
                         "add %%rdi, %0                                      \n\t"
                         "dec %%ebx                                         \n\t"
                         "jg 0b                                             \n\t"
                         "vmovups %%zmm0, (%1)                              \n\t"
                         :
                         : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride)
                         : "%eax", "%rax", "%ecx", "%rdi", "%zmm0", "%zmm4", "memory", "cc");
}

inline void pooling_c16_mean_w4(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "mov %%rax, %%r9                                  \n\t"
        "add %%r9, %%r9                                  \n\t"
        "mov %%rax, %%r10                                  \n\t"
        "add %%r9, %%r10                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "add %0, %%r9                                  \n\t"
        "add %0, %%r10                                  \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        "vxorps %%zmm2, %%zmm2, %%zmm2                     \n\t"
        "vxorps %%zmm3, %%zmm3, %%zmm3                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%zmm4                     \n\t"
        "vmovups (%%rax), %%zmm5                     \n\t"
        "vmovups (%%r9), %%zmm6                     \n\t"
        "vmovups (%%r10), %%zmm7                     \n\t"
        "vaddps %%zmm0, %%zmm4, %%zmm0                     \n\t"
        "vaddps %%zmm1, %%zmm5, %%zmm1                     \n\t"
        "vaddps %%zmm2, %%zmm6, %%zmm2                     \n\t"
        "vaddps %%zmm3, %%zmm7, %%zmm3                     \n\t"
        "add $0x40, %0                                      \n\t"
        "add $0x40, %%rax                                      \n\t"
        "add $0x40, %%r9                                      \n\t"
        "add $0x40, %%r10                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "add %%rdi, %%r9                                      \n\t"
        "add %%rdi, %%r10                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%zmm4                     \n\t"
        "vdivps %%zmm4, %%zmm0, %%zmm0                     \n\t"
        "vdivps %%zmm4, %%zmm1, %%zmm1                     \n\t"
        "vdivps %%zmm4, %%zmm2, %%zmm2                     \n\t"
        "vdivps %%zmm4, %%zmm3, %%zmm3                     \n\t"
        "vmovups %%zmm0, (%1)                              \n\t"
        "vmovups %%zmm1, 0x40(%1)                          \n\t"
        "vmovups %%zmm2, 0x80(%1)                          \n\t"
        "vmovups %%zmm3, 0xC0(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%r10", "%r9", "%rdi", "%zmm0", "%zmm1", "%zmm2", "%zmm3",
        "%zmm4", "%zmm5", "%zmm6", "%zmm7", "memory", "cc");
}

inline void pooling_c16_mean_w2(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "mov %5, %%eax                                  \n\t"
        "add %0, %%rax                                  \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        "vxorps %%zmm1, %%zmm1, %%zmm1                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%zmm4                     \n\t"
        "vmovups (%%rax), %%zmm5                     \n\t"
        "vaddps %%zmm0, %%zmm4, %%zmm0                     \n\t"
        "vaddps %%zmm1, %%zmm5, %%zmm1                     \n\t"
        "add $0x40, %0                                      \n\t"
        "add $0x40, %%rax                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "add %%rdi, %%rax                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%zmm4                     \n\t"
        "vdivps %%zmm4, %%zmm0, %%zmm0                     \n\t"
        "vdivps %%zmm4, %%zmm1, %%zmm1                     \n\t"
        "vmovups %%zmm0, (%1)                              \n\t"
        "vmovups %%zmm1, 0x40(%1)                          \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%rdi", "%zmm0", "%zmm1", "%zmm4", "%zmm5", "memory", "cc");
}

inline void pooling_c16_mean_w1(
    const F32 *curI, F32 *curO, U32 kw, U32 kh, U32 iStep, U32 stride, F32 poolSize)
{
    __asm__ __volatile__(
        "mov %%eax, %%eax                                  \n\t"
        "mov %4, %%eax                                  \n\t"
        "mov %%rax, %%rdi                                  \n\t"
        "vxorps %%zmm0, %%zmm0, %%zmm0                     \n\t"
        ".align 16                                         \n\t"
        "0:                                                \n\t"
        "mov %2, %%ecx                                     \n\t"
        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vmovups (%0), %%zmm4                     \n\t"
        "vaddps %%zmm0, %%zmm4, %%zmm0                     \n\t"
        "add $0x40, %0                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 1b                                             \n\t"
        "add %%rdi, %0                                      \n\t"
        "dec %%ebx                                         \n\t"
        "jg 0b                                             \n\t"
        "vbroadcastss (%6), %%zmm4                     \n\t"
        "vdivps %%zmm4, %%zmm0, %%zmm0                     \n\t"
        "vmovups %%zmm0, (%1)                              \n\t"
        :
        : "r"(curI), "r"(curO), "r"(kw), "b"(kh), "r"(iStep), "r"(stride), "r"(&poolSize)
        : "%eax", "%rax", "%ecx", "%rdi", "%zmm0", "%zmm4", "memory", "cc");
}

inline void pooling_max_with_idx_w4(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    stride /= 4;
    iStep /= 4;
    __m256 x1, x2, x3, x4;
    __m256 t1, t2, t3, t4;
    __m256 b1, b2;
    __m256 i1, i2, i3, i4;
    x1 = _mm256_loadu_ps(curI);
    x2 = _mm256_loadu_ps(curI + stride);
    x3 = _mm256_loadu_ps(curI + stride * 2);
    x4 = _mm256_loadu_ps(curI + stride * 3);
    __m256 diff = _mm256_set_ps(ihw * 7, ihw * 6, ihw * 5, ihw * 4, ihw * 3, ihw * 2, ihw, 0);
    i1 = _mm256_add_ps(diff, _mm256_set1_ps(w));
    i2 = _mm256_add_ps(diff, _mm256_set1_ps(w + stride / 8));
    i3 = _mm256_add_ps(diff, _mm256_set1_ps(w + stride / 8 * 2));
    i4 = _mm256_add_ps(diff, _mm256_set1_ps(w + stride / 8 * 3));
    for (U32 fh = 0; fh < kh; ++fh) {
        for (U32 fw = 0; fw < kw; ++fw) {
            t1 = _mm256_loadu_ps(curI);
            t2 = _mm256_loadu_ps(curI + stride);
            t3 = _mm256_loadu_ps(curI + stride * 2);
            t4 = _mm256_loadu_ps(curI + stride * 3);
            b1 = _mm256_cmp_ps(x1, t1, 1);
            b2 = _mm256_cmp_ps(x2, t2, 1);
            x1 = _mm256_blendv_ps(x1, t1, b1);
            x2 = _mm256_blendv_ps(x2, t2, b2);
            i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
            i2 = _mm256_blendv_ps(i2, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + stride / 8)), b2);
            b1 = _mm256_cmp_ps(x3, t3, 1);
            b2 = _mm256_cmp_ps(x4, t4, 1);
            x3 = _mm256_blendv_ps(x3, t3, b1);
            x4 = _mm256_blendv_ps(x4, t4, b2);
            i3 = _mm256_blendv_ps(i3, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + stride / 8 * 2)), b1);
            i4 = _mm256_blendv_ps(i4, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + stride / 8 * 3)), b2);
            curI += 8;
        }
        curI += iStep;
    }
    _mm256_storeu_ps(curO, x1);
    _mm256_storeu_ps(curO + 8, x2);
    _mm256_storeu_ps(curO + 16, x3);
    _mm256_storeu_ps(curO + 24, x4);
    _mm256_store_si256((__m256i*)(idx), _mm256_cvtps_epi32(i1));
    _mm256_store_si256((__m256i*)(idx + 8), _mm256_cvtps_epi32(i2));
    _mm256_store_si256((__m256i*)(idx + 16), _mm256_cvtps_epi32(i3));
    _mm256_store_si256((__m256i*)(idx + 24), _mm256_cvtps_epi32(i4));
}

inline void pooling_max_with_idx_w2(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    stride /= 4;
    iStep /= 4;
    __m256 x1, x2;
    __m256 t1, t2;
    __m256 b1, b2;
    __m256 i1, i2;
    x1 = _mm256_loadu_ps(curI);
    x2 = _mm256_loadu_ps(curI + stride);
    __m256 diff = _mm256_set_ps(ihw * 7, ihw * 6, ihw * 5, ihw * 4, ihw * 3, ihw * 2, ihw, 0);
    i1 = _mm256_add_ps(diff, _mm256_set1_ps(w));
    i2 = _mm256_add_ps(diff, _mm256_set1_ps(w + stride / 8));
    for (U32 fh = 0; fh < kh; ++fh) {
        for (U32 fw = 0; fw < kw; ++fw) {
            t1 = _mm256_loadu_ps(curI);
            t2 = _mm256_loadu_ps(curI + stride);
            b1 = _mm256_cmp_ps(x1, t1, 1);
            b2 = _mm256_cmp_ps(x2, t2, 1);
            x1 = _mm256_blendv_ps(x1, t1, b1);
            x2 = _mm256_blendv_ps(x2, t2, b2);
            i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
            i2 = _mm256_blendv_ps(i2, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w + stride / 8)), b2);
            curI += 8;
        }
        curI += iStep;
    }
    _mm256_storeu_ps(curO, x1);
    _mm256_storeu_ps(curO + 8, x2);
    _mm256_store_si256((__m256i*)(idx), _mm256_cvtps_epi32(i1));
    _mm256_store_si256((__m256i*)(idx + 8), _mm256_cvtps_epi32(i2));
}

inline void pooling_max_with_idx_w1(const F32 *curI, F32 *curO, I32 *idx, U32 kw, U32 kh, U32 iw, U32 ihw, U32 w, U32 iStep, U32 stride)
{
    stride /= 4;
    iStep /= 4;
    __m256 x1;
    __m256 t1;
    __m256 b1;
    __m256 i1;
    x1 = _mm256_loadu_ps(curI);
    __m256 diff = _mm256_set_ps(ihw * 7, ihw * 6, ihw * 5, ihw * 4, ihw * 3, ihw * 2, ihw, 0);
    i1 = _mm256_add_ps(diff, _mm256_set1_ps(w));
    for (U32 fh = 0; fh < kh; ++fh) {
        for (U32 fw = 0; fw < kw; ++fw) {
            t1 = _mm256_loadu_ps(curI);
            b1 = _mm256_cmp_ps(x1, t1, 1);
            x1 = _mm256_blendv_ps(x1, t1, b1);
            i1 = _mm256_blendv_ps(i1, _mm256_add_ps(diff, _mm256_set1_ps(fh * iw + fw + w)), b1);
            curI += 8;
        }
        curI += iStep;
    }
    _mm256_storeu_ps(curO, x1);
    _mm256_store_si256((__m256i*)(idx), _mm256_cvtps_epi32(i1));
}

#endif
