// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/int8/blas_int8.h"
#include "thread_affinity.h"

#define UNROLL_N 48
#define UNROLL_M 8
#define BOLCK_M_DIM 384
#define BOLCK_K_DIM 4096

typedef void (*kernel_func)(U32 um,
    U32 un,
    U32 bk,
    UINT8 *matrixA,
    INT8 *matrixB,
    I32 *matrixC,
    UINT8 *u8Result,
    I32 *offsetC,
    U32 N,
    U32 stepK,
    const F32 *scale,
    U32 nmask,
    UINT8 *resK,
    U32 flags);

// clang-format off
#define loadOffset_1_1(rtype) \
    "vmovups (%[offset]), "#rtype"0            \n\t"

#define loadOffset_6_1(rtype) \
    loadOffset_1_1(rtype) \
    "vmovups "#rtype"0, "#rtype"1              \n\t" \
    "vmovups "#rtype"0, "#rtype"2              \n\t" \
    "vmovups "#rtype"0, "#rtype"3              \n\t" \
    "vmovups "#rtype"0, "#rtype"4              \n\t" \
    "vmovups "#rtype"0, "#rtype"5              \n\t"

#define loadOffset_12_1(rtype) \
    loadOffset_6_1(rtype) \
    "vmovups "#rtype"0, "#rtype"6              \n\t" \
    "vmovups "#rtype"0, "#rtype"7              \n\t" \
    "vmovups "#rtype"0, "#rtype"8              \n\t" \
    "vmovups "#rtype"0, "#rtype"9              \n\t" \
    "vmovups "#rtype"0, "#rtype"10             \n\t" \
    "vmovups "#rtype"0, "#rtype"11             \n\t"

#define loadOffset_24_1(rtype) \
    loadOffset_12_1(rtype) \
    "vmovups "#rtype"0, "#rtype"12             \n\t" \
    "vmovups "#rtype"0, "#rtype"13             \n\t" \
    "vmovups "#rtype"0, "#rtype"14             \n\t" \
    "vmovups "#rtype"0, "#rtype"15             \n\t" \
    "vmovups "#rtype"0, "#rtype"16             \n\t" \
    "vmovups "#rtype"0, "#rtype"17             \n\t" \
    "vmovups "#rtype"0, "#rtype"18             \n\t" \
    "vmovups "#rtype"0, "#rtype"19             \n\t" \
    "vmovups "#rtype"0, "#rtype"20             \n\t" \
    "vmovups "#rtype"0, "#rtype"21             \n\t" \
    "vmovups "#rtype"0, "#rtype"22             \n\t" \
    "vmovups "#rtype"0, "#rtype"23             \n\t"

#define loadOffset_1_2 \
    loadOffset_1_1(%%zmm) \
    "vmovups 0x40(%[offset]), %%zmm1           \n\t"

#define loadOffset_3_2 \
    loadOffset_1_2 \
    "vmovups %%zmm0, %%zmm2                    \n\t" \
    "vmovups %%zmm1, %%zmm3                    \n\t" \
    "vmovups %%zmm0, %%zmm4                    \n\t" \
    "vmovups %%zmm1, %%zmm5                    \n\t"

#define loadOffset_6_2 \
    loadOffset_3_2 \
    "vmovups %%zmm0, %%zmm6                    \n\t" \
    "vmovups %%zmm1, %%zmm7                    \n\t" \
    "vmovups %%zmm0, %%zmm8                    \n\t" \
    "vmovups %%zmm1, %%zmm9                    \n\t" \
    "vmovups %%zmm0, %%zmm10                   \n\t" \
    "vmovups %%zmm1, %%zmm11                   \n\t"

#define loadOffset_12_2 \
    loadOffset_6_2 \
    "vmovups %%zmm0, %%zmm12                   \n\t" \
    "vmovups %%zmm1, %%zmm13                   \n\t" \
    "vmovups %%zmm0, %%zmm14                   \n\t" \
    "vmovups %%zmm1, %%zmm15                   \n\t" \
    "vmovups %%zmm0, %%zmm16                   \n\t" \
    "vmovups %%zmm1, %%zmm17                   \n\t" \
    "vmovups %%zmm0, %%zmm18                   \n\t" \
    "vmovups %%zmm1, %%zmm19                   \n\t" \
    "vmovups %%zmm0, %%zmm20                   \n\t" \
    "vmovups %%zmm1, %%zmm21                   \n\t" \
    "vmovups %%zmm0, %%zmm22                   \n\t" \
    "vmovups %%zmm1, %%zmm23                   \n\t"

#define loadOffset_1_3 \
    loadOffset_1_2 \
    "vmovups 0x80(%[offset]), %%zmm2           \n\t"

#define loadOffset_2_3 \
    loadOffset_1_3 \
    "vmovups %%zmm0, %%zmm3                    \n\t" \
    "vmovups %%zmm1, %%zmm4                    \n\t" \
    "vmovups %%zmm2, %%zmm5                    \n\t"

#define loadOffset_4_3 \
    loadOffset_2_3 \
    "vmovups %%zmm0, %%zmm6                    \n\t" \
    "vmovups %%zmm1, %%zmm7                    \n\t" \
    "vmovups %%zmm2, %%zmm8                    \n\t" \
    "vmovups %%zmm0, %%zmm9                    \n\t" \
    "vmovups %%zmm1, %%zmm10                   \n\t" \
    "vmovups %%zmm2, %%zmm11                   \n\t"

#define loadOffset_8_3 \
    loadOffset_4_3 \
    "vmovups %%zmm0, %%zmm12                   \n\t" \
    "vmovups %%zmm1, %%zmm13                   \n\t" \
    "vmovups %%zmm2, %%zmm14                   \n\t" \
    "vmovups %%zmm0, %%zmm15                   \n\t" \
    "vmovups %%zmm1, %%zmm16                   \n\t" \
    "vmovups %%zmm2, %%zmm17                   \n\t" \
    "vmovups %%zmm0, %%zmm18                   \n\t" \
    "vmovups %%zmm1, %%zmm19                   \n\t" \
    "vmovups %%zmm2, %%zmm20                   \n\t" \
    "vmovups %%zmm0, %%zmm21                   \n\t" \
    "vmovups %%zmm1, %%zmm22                   \n\t" \
    "vmovups %%zmm2, %%zmm23                   \n\t"

#define addC_1_1(rtype, C) \
    "movq "#C", %%rax  \n\t" \
    "vpaddd (%%rax), "#rtype"0, "#rtype"0       \n\t"

#define addC_6_1(rtype, C) \
    addC_1_1(rtype, C) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"1, "#rtype"1       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"2, "#rtype"2       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"3, "#rtype"3       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"4, "#rtype"4       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"5, "#rtype"5       \n\t"

#define addC_12_1(rtype, C) \
    addC_6_1(rtype, C) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"6, "#rtype"6       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"7, "#rtype"7       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"8, "#rtype"8       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"9, "#rtype"9       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"10, "#rtype"10     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"11, "#rtype"11     \n\t"

#define addC_24_1(rtype, C) \
    addC_12_1(rtype, C) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"12, "#rtype"12     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"13, "#rtype"13     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"14, "#rtype"14     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"15, "#rtype"15     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"16, "#rtype"16     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"17, "#rtype"17     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"18, "#rtype"18     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"19, "#rtype"19     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"20, "#rtype"20     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"21, "#rtype"21     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"22, "#rtype"22     \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), "#rtype"23, "#rtype"23     \n\t"

#define addC_1_2(C) \
    addC_1_1(%%zmm, C) \
    "vpaddd 0x40(%%rax), %%zmm1, %%zmm1         \n\t"

#define addC_3_2(C) \
    addC_1_2(C) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm2, %%zmm2             \n\t" \
    "vpaddd 0x40(%%rax), %%zmm3, %%zmm3         \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm4, %%zmm4             \n\t" \
    "vpaddd 0x40(%%rax), %%zmm5, %%zmm5         \n\t"

#define addC_6_2(C) \
    addC_3_2(C) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm6, %%zmm6             \n\t" \
    "vpaddd 0x40(%%rax), %%zmm7, %%zmm7         \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm8, %%zmm8             \n\t" \
    "vpaddd 0x40(%%rax), %%zmm9, %%zmm9         \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm10, %%zmm10           \n\t" \
    "vpaddd 0x40(%%rax), %%zmm11, %%zmm11       \n\t"

#define addC_12_2(C) \
    addC_6_2(C) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm12, %%zmm12           \n\t" \
    "vpaddd 0x40(%%rax), %%zmm13, %%zmm13       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm14, %%zmm14           \n\t" \
    "vpaddd 0x40(%%rax), %%zmm15, %%zmm15       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm16, %%zmm16           \n\t" \
    "vpaddd 0x40(%%rax), %%zmm17, %%zmm17       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm18, %%zmm18           \n\t" \
    "vpaddd 0x40(%%rax), %%zmm19, %%zmm19       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm20, %%zmm20           \n\t" \
    "vpaddd 0x40(%%rax), %%zmm21, %%zmm21       \n\t" \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%zmm22, %%zmm22           \n\t" \
    "vpaddd 0x40(%%rax), %%zmm23, %%zmm23       \n\t"

#define addC_1_3(C) \
    "vpaddd ("#C"), %%zmm0, %%zmm0              \n\t" \
    "vpaddd 0x40("#C"), %%zmm1, %%zmm1          \n\t" \
    "vpaddd 0x80("#C"), %%zmm2, %%zmm2          \n\t"

#define addC_2_3(C) \
    "vpaddd ("#C"), %%zmm0, %%zmm0              \n\t" \
    "vpaddd 0x40("#C"), %%zmm1, %%zmm1          \n\t" \
    "vpaddd 0x80("#C"), %%zmm2, %%zmm2          \n\t" \
    "vpaddd ("#C", %[N]), %%zmm3, %%zmm3        \n\t" \
    "vpaddd 0x40("#C", %[N]), %%zmm4, %%zmm4    \n\t" \
    "vpaddd 0x80("#C", %[N]), %%zmm5, %%zmm5    \n\t"

#define addC_4_3(C) \
    addC_2_3(C) \
    "addq %%rcx, "#C"  \n\t" \
    "vpaddd ("#C"), %%zmm6, %%zmm6              \n\t" \
    "vpaddd 0x40("#C"), %%zmm7, %%zmm7          \n\t" \
    "vpaddd 0x80("#C"), %%zmm8, %%zmm8          \n\t" \
    "vpaddd ("#C", %[N]), %%zmm9, %%zmm9        \n\t" \
    "vpaddd 0x40("#C", %[N]), %%zmm10, %%zmm10  \n\t" \
    "vpaddd 0x80("#C", %[N]), %%zmm11, %%zmm11  \n\t"

#define addC_8_3(C) \
    addC_4_3(C) \
    "addq %%rcx, "#C"  \n\t" \
    "vpaddd ("#C"), %%zmm12, %%zmm12                   \n\t" \
    "vpaddd 0x40("#C"), %%zmm13, %%zmm13                   \n\t" \
    "vpaddd 0x80("#C"), %%zmm14, %%zmm14                   \n\t" \
    "vpaddd ("#C", %[N]), %%zmm15, %%zmm15                   \n\t" \
    "vpaddd 0x40("#C", %[N]), %%zmm16, %%zmm16                   \n\t" \
    "vpaddd 0x80("#C", %[N]), %%zmm17, %%zmm17                   \n\t" \
    "addq %%rcx, "#C"  \n\t" \
    "vpaddd ("#C"), %%zmm18, %%zmm18                   \n\t" \
    "vpaddd 0x40("#C"), %%zmm19, %%zmm19                   \n\t" \
    "vpaddd 0x80("#C"), %%zmm20, %%zmm20                   \n\t" \
    "vpaddd ("#C", %[N]), %%zmm21, %%zmm21                   \n\t" \
    "vpaddd 0x40("#C", %[N]), %%zmm22, %%zmm22                   \n\t" \
    "vpaddd 0x80("#C", %[N]), %%zmm23, %%zmm23                   \n\t" \

#define storeC_1_1_0(op, rtype, C, off0, off1) \
    "movq "#C", %%rax  \n\t" \
    #op" "#rtype"0, (%%rax)       \n\t"

#define storeC_2_1_0(op, rtype, C, off0, off1) \
    storeC_1_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"1, (%%rax)       \n\t"

#define storeC_3_1_0(op, rtype, C, off0, off1) \
    storeC_2_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"2, (%%rax)       \n\t"

#define storeC_4_1_0(op, rtype, C, off0, off1) \
    storeC_3_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"3, (%%rax)       \n\t"

#define storeC_5_1_0(op, rtype, C, off0, off1) \
    storeC_4_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"4, (%%rax)       \n\t"

#define storeC_6_1_0(op, rtype, C, off0, off1) \
    storeC_5_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"5, (%%rax)       \n\t"

#define storeC_7_1_0(op, rtype, C, off0, off1) \
    storeC_6_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"6, (%%rax)       \n\t"

#define storeC_8_1_0(op, rtype, C, off0, off1) \
    storeC_7_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"7, (%%rax)       \n\t"

#define storeC_9_1_0(op, rtype, C, off0, off1) \
    storeC_8_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"8, (%%rax)       \n\t"

#define storeC_10_1_0(op, rtype, C, off0, off1) \
    storeC_9_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"9, (%%rax)       \n\t"

#define storeC_11_1_0(op, rtype, C, off0, off1) \
    storeC_10_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"10, (%%rax)       \n\t"

#define storeC_12_1_0(op, rtype, C, off0, off1) \
    storeC_11_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"11, (%%rax)       \n\t"

#define storeC_13_1_0(op, rtype, C, off0, off1) \
    storeC_12_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"12, (%%rax)       \n\t"

#define storeC_14_1_0(op, rtype, C, off0, off1) \
    storeC_13_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"13, (%%rax)       \n\t"

#define storeC_15_1_0(op, rtype, C, off0, off1) \
    storeC_14_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"14, (%%rax)       \n\t"

#define storeC_16_1_0(op, rtype, C, off0, off1) \
    storeC_15_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"15, (%%rax)       \n\t"

#define storeC_17_1_0(op, rtype, C, off0, off1) \
    storeC_16_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"16, (%%rax)       \n\t"

#define storeC_18_1_0(op, rtype, C, off0, off1) \
    storeC_17_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"17, (%%rax)       \n\t"

#define storeC_19_1_0(op, rtype, C, off0, off1) \
    storeC_18_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"18, (%%rax)       \n\t"

#define storeC_20_1_0(op, rtype, C, off0, off1) \
    storeC_19_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"19, (%%rax)       \n\t"

#define storeC_21_1_0(op, rtype, C, off0, off1) \
    storeC_20_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"20, (%%rax)       \n\t"

#define storeC_22_1_0(op, rtype, C, off0, off1) \
    storeC_21_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"21, (%%rax)       \n\t"

#define storeC_23_1_0(op, rtype, C, off0, off1) \
    storeC_22_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"22, (%%rax)       \n\t"

#define storeC_24_1_0(op, rtype, C, off0, off1) \
    storeC_23_1_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"23, (%%rax)       \n\t"

#define storeC_1_2_0(op, rtype, C, off0, off1) \
    storeC_1_1_0(op, rtype, C, off0, off1) \
    #op" "#rtype"1, "#off0"(%%rax)       \n\t"

#define storeC_2_2_0(op, rtype, C, off0, off1) \
    storeC_1_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"2, (%%rax)                        \n\t" \
    #op" "#rtype"3, "#off0"(%%rax)                    \n\t"

#define storeC_3_2_0(op, rtype, C, off0, off1) \
    storeC_2_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"4, (%%rax)                        \n\t" \
    #op" "#rtype"5, "#off0"(%%rax)                    \n\t"

#define storeC_4_2_0(op, rtype, C, off0, off1) \
    storeC_3_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"6, (%%rax)                        \n\t" \
    #op" "#rtype"7, "#off0"(%%rax)                    \n\t"

#define storeC_5_2_0(op, rtype, C, off0, off1) \
    storeC_4_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"8, (%%rax)                        \n\t" \
    #op" "#rtype"9, "#off0"(%%rax)                    \n\t"

#define storeC_6_2_0(op, rtype, C, off0, off1) \
    storeC_5_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"10, (%%rax)                        \n\t" \
    #op" "#rtype"11, "#off0"(%%rax)                    \n\t"

#define storeC_7_2_0(op, rtype, C, off0, off1) \
    storeC_6_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"12, (%%rax)                        \n\t" \
    #op" "#rtype"13, "#off0"(%%rax)                    \n\t"

#define storeC_8_2_0(op, rtype, C, off0, off1) \
    storeC_7_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"14, (%%rax)                        \n\t" \
    #op" "#rtype"15, "#off0"(%%rax)                    \n\t"

#define storeC_9_2_0(op, rtype, C, off0, off1) \
    storeC_8_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"16, (%%rax)                        \n\t" \
    #op" "#rtype"17, "#off0"(%%rax)                    \n\t"

#define storeC_10_2_0(op, rtype, C, off0, off1) \
    storeC_9_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"18, (%%rax)                        \n\t" \
    #op" "#rtype"19, "#off0"(%%rax)                    \n\t"

#define storeC_11_2_0(op, rtype, C, off0, off1) \
    storeC_10_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"20, (%%rax)                        \n\t" \
    #op" "#rtype"21, "#off0"(%%rax)                    \n\t"

#define storeC_12_2_0(op, rtype, C, off0, off1) \
    storeC_11_2_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"22, (%%rax)                        \n\t" \
    #op" "#rtype"23, "#off0"(%%rax)                    \n\t"

#define storeC_1_3_0(op, rtype, C, off0, off1) \
    "movq "#C", %%rax  \n\t" \
    #op" "#rtype"0, (%%rax)       \n\t" \
    #op" "#rtype"1, "#off0"(%%rax)       \n\t" \
    #op" "#rtype"2, "#off1"(%%rax)       \n\t"

#define storeC_2_3_0(op, rtype, C, off0, off1) \
    storeC_1_3_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"3, (%%rax)                        \n\t" \
    #op" "#rtype"4, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"5, "#off1"(%%rax)                    \n\t"

#define storeC_3_3_0(op, rtype, C, off0, off1) \
    storeC_2_3_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"6, (%%rax)                        \n\t" \
    #op" "#rtype"7, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"8, "#off1"(%%rax)                    \n\t"

#define storeC_4_3_0(op, rtype, C, off0, off1) \
    storeC_3_3_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"9, (%%rax)                        \n\t" \
    #op" "#rtype"10, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"11, "#off1"(%%rax)                    \n\t"

#define storeC_5_3_0(op, rtype, C, off0, off1) \
    storeC_4_3_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"12, (%%rax)                        \n\t" \
    #op" "#rtype"13, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"14, "#off1"(%%rax)                    \n\t"

#define storeC_6_3_0(op, rtype, C, off0, off1) \
    storeC_5_3_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"15, (%%rax)                        \n\t" \
    #op" "#rtype"16, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"17, "#off1"(%%rax)                    \n\t"

#define storeC_7_3_0(op, rtype, C, off0, off1) \
    storeC_6_3_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"18, (%%rax)                        \n\t" \
    #op" "#rtype"19, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"20, "#off1"(%%rax)                    \n\t"

#define storeC_8_3_0(op, rtype, C, off0, off1) \
    storeC_7_3_0(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"21, (%%rax)                        \n\t" \
    #op" "#rtype"22, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"23, "#off1"(%%rax)                    \n\t"

#define storeC_1_1_1(op, rtype, C, off0, off1) \
    "movq "#C", %%rax  \n\t" \
    "kmovw %[nmask], %%k1  \n\t" \
    #op" "#rtype"0, (%%rax) %{%%k1%}       \n\t"

#define storeC_2_1_1(op, rtype, C, off0, off1) \
    storeC_1_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"1, (%%rax) %{%%k1%}       \n\t"

#define storeC_3_1_1(op, rtype, C, off0, off1) \
    storeC_2_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"2, (%%rax) %{%%k1%}       \n\t"

#define storeC_4_1_1(op, rtype, C, off0, off1) \
    storeC_3_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"3, (%%rax) %{%%k1%}       \n\t"

#define storeC_5_1_1(op, rtype, C, off0, off1) \
    storeC_4_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"4, (%%rax) %{%%k1%}       \n\t"

#define storeC_6_1_1(op, rtype, C, off0, off1) \
    storeC_5_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"5, (%%rax) %{%%k1%}       \n\t"

#define storeC_7_1_1(op, rtype, C, off0, off1) \
    storeC_6_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"6, (%%rax) %{%%k1%}       \n\t"

#define storeC_8_1_1(op, rtype, C, off0, off1) \
    storeC_7_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"7, (%%rax) %{%%k1%}       \n\t"

#define storeC_9_1_1(op, rtype, C, off0, off1) \
    storeC_8_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"8, (%%rax) %{%%k1%}       \n\t"

#define storeC_10_1_1(op, rtype, C, off0, off1) \
    storeC_9_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"9, (%%rax) %{%%k1%}       \n\t"

#define storeC_11_1_1(op, rtype, C, off0, off1) \
    storeC_10_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"10, (%%rax) %{%%k1%}       \n\t"

#define storeC_12_1_1(op, rtype, C, off0, off1) \
    storeC_11_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"11, (%%rax) %{%%k1%}       \n\t"

#define storeC_13_1_1(op, rtype, C, off0, off1) \
    storeC_12_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"12, (%%rax) %{%%k1%}       \n\t"

#define storeC_14_1_1(op, rtype, C, off0, off1) \
    storeC_13_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"13, (%%rax) %{%%k1%}       \n\t"

#define storeC_15_1_1(op, rtype, C, off0, off1) \
    storeC_14_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"14, (%%rax) %{%%k1%}       \n\t"

#define storeC_16_1_1(op, rtype, C, off0, off1) \
    storeC_15_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"15, (%%rax) %{%%k1%}       \n\t"

#define storeC_17_1_1(op, rtype, C, off0, off1) \
    storeC_16_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"16, (%%rax) %{%%k1%}       \n\t"

#define storeC_18_1_1(op, rtype, C, off0, off1) \
    storeC_17_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"17, (%%rax) %{%%k1%}       \n\t"

#define storeC_19_1_1(op, rtype, C, off0, off1) \
    storeC_18_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"18, (%%rax) %{%%k1%}       \n\t"

#define storeC_20_1_1(op, rtype, C, off0, off1) \
    storeC_19_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"19, (%%rax) %{%%k1%}       \n\t"

#define storeC_21_1_1(op, rtype, C, off0, off1) \
    storeC_20_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"20, (%%rax) %{%%k1%}       \n\t"

#define storeC_22_1_1(op, rtype, C, off0, off1) \
    storeC_21_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"21, (%%rax) %{%%k1%}       \n\t"

#define storeC_23_1_1(op, rtype, C, off0, off1) \
    storeC_22_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"22, (%%rax) %{%%k1%}       \n\t"

#define storeC_24_1_1(op, rtype, C, off0, off1) \
    storeC_23_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"23, (%%rax) %{%%k1%}       \n\t"

#define storeC_1_2_1(op, rtype, C, off0, off1) \
    "kmovw %[nmask], %%k1  \n\t" \
    storeC_1_1_0(op, rtype, C, off0, off1) \
    #op" "#rtype"1, "#off0"(%%rax) %{%%k1%}       \n\t"

#define storeC_2_2_1(op, rtype, C, off0, off1) \
    storeC_1_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"2, (%%rax)                        \n\t" \
    #op" "#rtype"3, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_3_2_1(op, rtype, C, off0, off1) \
    storeC_2_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"4, (%%rax)                        \n\t" \
    #op" "#rtype"5, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_4_2_1(op, rtype, C, off0, off1) \
    storeC_3_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"6, (%%rax)                        \n\t" \
    #op" "#rtype"7, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_5_2_1(op, rtype, C, off0, off1) \
    storeC_4_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"8, (%%rax)                        \n\t" \
    #op" "#rtype"9, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_6_2_1(op, rtype, C, off0, off1) \
    storeC_5_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"10, (%%rax)                        \n\t" \
    #op" "#rtype"11, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_7_2_1(op, rtype, C, off0, off1) \
    storeC_6_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"12, (%%rax)                        \n\t" \
    #op" "#rtype"13, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_8_2_1(op, rtype, C, off0, off1) \
    storeC_7_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"14, (%%rax)                        \n\t" \
    #op" "#rtype"15, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_9_2_1(op, rtype, C, off0, off1) \
    storeC_8_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"16, (%%rax)                        \n\t" \
    #op" "#rtype"17, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_10_2_1(op, rtype, C, off0, off1) \
    storeC_9_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"18, (%%rax)                        \n\t" \
    #op" "#rtype"19, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_11_2_1(op, rtype, C, off0, off1) \
    storeC_10_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"20, (%%rax)                        \n\t" \
    #op" "#rtype"21, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_12_2_1(op, rtype, C, off0, off1) \
    storeC_11_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"22, (%%rax)                        \n\t" \
    #op" "#rtype"23, "#off0"(%%rax) %{%%k1%}                    \n\t"

#define storeC_1_3_1(op, rtype, C, off0, off1) \
    "kmovw %[nmask], %%k1  \n\t" \
    storeC_1_2_0(op, rtype, C, off0, off1) \
    #op" "#rtype"2, "#off1"(%%rax) %{%%k1%}       \n\t"

#define storeC_2_3_1(op, rtype, C, off0, off1) \
    storeC_1_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"3, (%%rax)                        \n\t" \
    #op" "#rtype"4, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"5, "#off1"(%%rax) %{%%k1%}                    \n\t"

#define storeC_3_3_1(op, rtype, C, off0, off1) \
    storeC_2_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"6, (%%rax)                        \n\t" \
    #op" "#rtype"7, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"8, "#off1"(%%rax) %{%%k1%}                    \n\t"

#define storeC_4_3_1(op, rtype, C, off0, off1) \
    storeC_3_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"9, (%%rax)                        \n\t" \
    #op" "#rtype"10, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"11, "#off1"(%%rax) %{%%k1%}                    \n\t"

#define storeC_5_3_1(op, rtype, C, off0, off1) \
    storeC_4_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"12, (%%rax)                        \n\t" \
    #op" "#rtype"13, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"14, "#off1"(%%rax) %{%%k1%}                    \n\t"

#define storeC_6_3_1(op, rtype, C, off0, off1) \
    storeC_5_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"15, (%%rax)                        \n\t" \
    #op" "#rtype"16, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"17, "#off1"(%%rax) %{%%k1%}                    \n\t"

#define storeC_7_3_1(op, rtype, C, off0, off1) \
    storeC_6_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"18, (%%rax)                        \n\t" \
    #op" "#rtype"19, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"20, "#off1"(%%rax) %{%%k1%}                    \n\t"

#define storeC_8_3_1(op, rtype, C, off0, off1) \
    storeC_7_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    #op" "#rtype"21, (%%rax)                        \n\t" \
    #op" "#rtype"22, "#off0"(%%rax)                    \n\t" \
    #op" "#rtype"23, "#off1"(%%rax) %{%%k1%}                    \n\t"

#define clear1Regs(rtype) \
    "vxorps "#rtype"0, "#rtype"0, "#rtype"0    \n\t"

#define clear2Regs(rtype) \
    clear1Regs(rtype) \
    "vxorps "#rtype"1, "#rtype"1, "#rtype"1    \n\t"

#define clear3Regs(rtype) \
    clear2Regs(rtype) \
    "vxorps "#rtype"2, "#rtype"2, "#rtype"2    \n\t"

#define clear4Regs(rtype) \
    clear3Regs(rtype) \
    "vxorps "#rtype"3, "#rtype"3, "#rtype"3    \n\t"

#define clear6Regs(rtype) \
    clear4Regs(rtype) \
    "vxorps "#rtype"4, "#rtype"4, "#rtype"4    \n\t" \
    "vxorps "#rtype"5, "#rtype"5, "#rtype"5    \n\t"

#define clear8Regs(rtype) \
    clear6Regs(rtype) \
    "vxorps "#rtype"6, "#rtype"6, "#rtype"6    \n\t" \
    "vxorps "#rtype"7, "#rtype"7, "#rtype"7    \n\t"

#define clear9Regs(rtype) \
    clear8Regs(rtype) \
    "vxorps "#rtype"8, "#rtype"8, "#rtype"8    \n\t"

#define clear12Regs(rtype) \
    clear9Regs(rtype) \
    "vxorps "#rtype"9, "#rtype"9, "#rtype"9    \n\t" \
    "vxorps "#rtype"10, "#rtype"10, "#rtype"10 \n\t" \
    "vxorps "#rtype"11, "#rtype"11, "#rtype"11 \n\t"

#define clear24Regs(rtype) \
    clear12Regs(rtype) \
    "vxorps "#rtype"12, "#rtype"12, "#rtype"12 \n\t" \
    "vxorps "#rtype"13, "#rtype"13, "#rtype"13 \n\t" \
    "vxorps "#rtype"14, "#rtype"14, "#rtype"14 \n\t" \
    "vxorps "#rtype"15, "#rtype"15, "#rtype"15 \n\t" \
    "vxorps "#rtype"16, "#rtype"16, "#rtype"16 \n\t" \
    "vxorps "#rtype"17, "#rtype"17, "#rtype"17 \n\t" \
    "vxorps "#rtype"18, "#rtype"18, "#rtype"18 \n\t" \
    "vxorps "#rtype"19, "#rtype"19, "#rtype"19 \n\t" \
    "vxorps "#rtype"20, "#rtype"20, "#rtype"20 \n\t" \
    "vxorps "#rtype"21, "#rtype"21, "#rtype"21 \n\t" \
    "vxorps "#rtype"22, "#rtype"22, "#rtype"22 \n\t" \
    "vxorps "#rtype"23, "#rtype"23, "#rtype"23 \n\t"

#define convert1I32Regs2Ps(rtype, sReg) \
    "vbroadcastss ("#sReg"), "#rtype"24                        \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                  \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0            \n\t"

#define convert2I32Regs2Ps(rtype, sReg) \
    "vbroadcastss ("#sReg"), "#rtype"24                        \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                  \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                  \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0            \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1            \n\t"

#define convert3I32Regs2Ps(rtype, sReg) \
    "vbroadcastss ("#sReg"), "#rtype"24                        \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                  \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                  \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                  \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0            \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1            \n\t" \
    "vmulps "#rtype"2, "#rtype"24, "#rtype"2            \n\t"

#define convert4I32Regs2Ps(rtype, sReg) \
    "vbroadcastss ("#sReg"), "#rtype"24                        \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                  \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                  \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                  \n\t" \
    "vcvtdq2ps "#rtype"3, "#rtype"3                  \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0            \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1            \n\t" \
    "vmulps "#rtype"2, "#rtype"24, "#rtype"2            \n\t" \
    "vmulps "#rtype"3, "#rtype"24, "#rtype"3            \n\t"

#define convert6I32Regs2Ps(rtype, sReg) \
    "vbroadcastss ("#sReg"), "#rtype"24                        \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                  \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                  \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                  \n\t" \
    "vcvtdq2ps "#rtype"3, "#rtype"3                  \n\t" \
    "vcvtdq2ps "#rtype"4, "#rtype"4                  \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                  \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0            \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1            \n\t" \
    "vmulps "#rtype"2, "#rtype"24, "#rtype"2            \n\t" \
    "vmulps "#rtype"3, "#rtype"24, "#rtype"3            \n\t" \
    "vmulps "#rtype"4, "#rtype"24, "#rtype"4            \n\t" \
    "vmulps "#rtype"5, "#rtype"24, "#rtype"5            \n\t"

#define convert12I32Regs2Ps(rtype, sReg) \
    convert6I32Regs2Ps(rtype, sReg) \
    "vcvtdq2ps "#rtype"6, "#rtype"6                  \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                  \n\t" \
    "vcvtdq2ps "#rtype"8, "#rtype"8                  \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                  \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                \n\t" \
    "vmulps "#rtype"6, "#rtype"24, "#rtype"6            \n\t" \
    "vmulps "#rtype"7, "#rtype"24, "#rtype"7            \n\t" \
    "vmulps "#rtype"8, "#rtype"24, "#rtype"8            \n\t" \
    "vmulps "#rtype"9, "#rtype"24, "#rtype"9            \n\t" \
    "vmulps "#rtype"10, "#rtype"24, "#rtype"10          \n\t" \
    "vmulps "#rtype"11, "#rtype"24, "#rtype"11          \n\t"

#define convert24I32Regs2Ps(rtype, sReg) \
    convert12I32Regs2Ps(rtype, sReg) \
    "vcvtdq2ps "#rtype"12, "#rtype"12                \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                \n\t" \
    "vcvtdq2ps "#rtype"15, "#rtype"15                \n\t" \
    "vcvtdq2ps "#rtype"16, "#rtype"16                \n\t" \
    "vcvtdq2ps "#rtype"17, "#rtype"17                \n\t" \
    "vcvtdq2ps "#rtype"18, "#rtype"18                \n\t" \
    "vcvtdq2ps "#rtype"19, "#rtype"19                \n\t" \
    "vcvtdq2ps "#rtype"20, "#rtype"20                \n\t" \
    "vcvtdq2ps "#rtype"21, "#rtype"21                \n\t" \
    "vcvtdq2ps "#rtype"22, "#rtype"22                \n\t" \
    "vcvtdq2ps "#rtype"23, "#rtype"23                \n\t" \
    "vmulps "#rtype"12, "#rtype"24, "#rtype"12          \n\t" \
    "vmulps "#rtype"13, "#rtype"24, "#rtype"13          \n\t" \
    "vmulps "#rtype"14, "#rtype"24, "#rtype"14          \n\t" \
    "vmulps "#rtype"15, "#rtype"24, "#rtype"15          \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16          \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17          \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18          \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19          \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20          \n\t" \
    "vmulps "#rtype"21, "#rtype"24, "#rtype"21          \n\t" \
    "vmulps "#rtype"22, "#rtype"24, "#rtype"22          \n\t" \
    "vmulps "#rtype"23, "#rtype"24, "#rtype"23          \n\t"

#define convert1PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm25                            \n\t" \
    "vbroadcastss %%xmm25, "#rtype"24                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                  \n\t" \
    "vpaddd "#rtype"0, "#rtype"24, "#rtype"0         \n\t"

#define convert2PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm25                            \n\t" \
    "vbroadcastss %%xmm25, "#rtype"24                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                  \n\t" \
    "vcvtps2dq "#rtype"1, "#rtype"1                  \n\t" \
    "vpaddd "#rtype"0, "#rtype"24, "#rtype"0         \n\t" \
    "vpaddd "#rtype"1, "#rtype"24, "#rtype"1         \n\t"

#define convert3PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm25                            \n\t" \
    "vbroadcastss %%xmm25, "#rtype"24                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                  \n\t" \
    "vcvtps2dq "#rtype"1, "#rtype"1                  \n\t" \
    "vcvtps2dq "#rtype"2, "#rtype"2                  \n\t" \
    "vpaddd "#rtype"0, "#rtype"24, "#rtype"0         \n\t" \
    "vpaddd "#rtype"1, "#rtype"24, "#rtype"1         \n\t" \
    "vpaddd "#rtype"2, "#rtype"24, "#rtype"2         \n\t"

#define convert4PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm25                            \n\t" \
    "vbroadcastss %%xmm25, "#rtype"24                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                  \n\t" \
    "vcvtps2dq "#rtype"1, "#rtype"1                  \n\t" \
    "vcvtps2dq "#rtype"2, "#rtype"2                  \n\t" \
    "vcvtps2dq "#rtype"3, "#rtype"3                  \n\t" \
    "vpaddd "#rtype"0, "#rtype"24, "#rtype"0         \n\t" \
    "vpaddd "#rtype"1, "#rtype"24, "#rtype"1         \n\t" \
    "vpaddd "#rtype"2, "#rtype"24, "#rtype"2         \n\t" \
    "vpaddd "#rtype"3, "#rtype"24, "#rtype"3         \n\t"

#define convert6PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm25                            \n\t" \
    "vbroadcastss %%xmm25, "#rtype"24                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                  \n\t" \
    "vcvtps2dq "#rtype"1, "#rtype"1                  \n\t" \
    "vcvtps2dq "#rtype"2, "#rtype"2                  \n\t" \
    "vcvtps2dq "#rtype"3, "#rtype"3                  \n\t" \
    "vcvtps2dq "#rtype"4, "#rtype"4                  \n\t" \
    "vcvtps2dq "#rtype"5, "#rtype"5                  \n\t" \
    "vpaddd "#rtype"0, "#rtype"24, "#rtype"0         \n\t" \
    "vpaddd "#rtype"1, "#rtype"24, "#rtype"1         \n\t" \
    "vpaddd "#rtype"2, "#rtype"24, "#rtype"2         \n\t" \
    "vpaddd "#rtype"3, "#rtype"24, "#rtype"3         \n\t" \
    "vpaddd "#rtype"4, "#rtype"24, "#rtype"4         \n\t" \
    "vpaddd "#rtype"5, "#rtype"24, "#rtype"5         \n\t"

#define convert12PsRegs2U8(rtype) \
    convert6PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"6, "#rtype"6                  \n\t" \
    "vcvtps2dq "#rtype"7, "#rtype"7                  \n\t" \
    "vcvtps2dq "#rtype"8, "#rtype"8                  \n\t" \
    "vcvtps2dq "#rtype"9, "#rtype"9                  \n\t" \
    "vcvtps2dq "#rtype"10, "#rtype"10                \n\t" \
    "vcvtps2dq "#rtype"11, "#rtype"11                \n\t" \
    "vpaddd "#rtype"6, "#rtype"24, "#rtype"6         \n\t" \
    "vpaddd "#rtype"7, "#rtype"24, "#rtype"7         \n\t" \
    "vpaddd "#rtype"8, "#rtype"24, "#rtype"8         \n\t" \
    "vpaddd "#rtype"9, "#rtype"24, "#rtype"9         \n\t" \
    "vpaddd "#rtype"10, "#rtype"24, "#rtype"10       \n\t" \
    "vpaddd "#rtype"11, "#rtype"24, "#rtype"11       \n\t"

#define convert24PsRegs2U8(rtype) \
    convert12PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"12, "#rtype"12                \n\t" \
    "vcvtps2dq "#rtype"13, "#rtype"13                \n\t" \
    "vcvtps2dq "#rtype"14, "#rtype"14                \n\t" \
    "vcvtps2dq "#rtype"15, "#rtype"15                \n\t" \
    "vcvtps2dq "#rtype"16, "#rtype"16                \n\t" \
    "vcvtps2dq "#rtype"17, "#rtype"17                \n\t" \
    "vcvtps2dq "#rtype"18, "#rtype"18                \n\t" \
    "vcvtps2dq "#rtype"19, "#rtype"19                \n\t" \
    "vcvtps2dq "#rtype"20, "#rtype"20                \n\t" \
    "vcvtps2dq "#rtype"21, "#rtype"21                \n\t" \
    "vcvtps2dq "#rtype"22, "#rtype"22                \n\t" \
    "vcvtps2dq "#rtype"23, "#rtype"23                \n\t" \
    "vpaddd "#rtype"12, "#rtype"24, "#rtype"12       \n\t" \
    "vpaddd "#rtype"13, "#rtype"24, "#rtype"13       \n\t" \
    "vpaddd "#rtype"14, "#rtype"24, "#rtype"14       \n\t" \
    "vpaddd "#rtype"15, "#rtype"24, "#rtype"15       \n\t" \
    "vpaddd "#rtype"16, "#rtype"24, "#rtype"16       \n\t" \
    "vpaddd "#rtype"17, "#rtype"24, "#rtype"17       \n\t" \
    "vpaddd "#rtype"18, "#rtype"24, "#rtype"18       \n\t" \
    "vpaddd "#rtype"19, "#rtype"24, "#rtype"19       \n\t" \
    "vpaddd "#rtype"20, "#rtype"24, "#rtype"20       \n\t" \
    "vpaddd "#rtype"21, "#rtype"24, "#rtype"21       \n\t" \
    "vpaddd "#rtype"22, "#rtype"24, "#rtype"22       \n\t" \
    "vpaddd "#rtype"23, "#rtype"24, "#rtype"23       \n\t"

#define mmm_1_48(A, K) \
    "movq "#A", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd 0x4(%%rax), %%zmm31       \n\t" \
    "vmovups (%[B]), %%zmm27               \n\t" \
    "vmovups 0x40(%[B]), %%zmm28           \n\t" \
    "vmovups 0x80(%[B]), %%zmm29           \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm0     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm1     \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm2     \n\t" \
    "vmovups 0xC0(%[B]), %%zmm24           \n\t" \
    "vmovups 0x100(%[B]), %%zmm25          \n\t" \
    "vmovups 0x140(%[B]), %%zmm26          \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm0     \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm1     \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm2     \n\t"

#define mmm_2_48(A, K) \
    "movq "#A", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "prefetcht0 0xC0(%1)                   \n\t" \
    "prefetcht0 0x100(%1)                  \n\t" \
    "prefetcht0 0x140(%1)                  \n\t" \
    "vmovups (%[B]), %%zmm27               \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm0     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm1     \n\t" \
    "vmovups 0x40(%[B]), %%zmm28           \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm2     \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm3     \n\t" \
    "vmovups 0x80(%[B]), %%zmm29           \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm4     \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm5     \n\t" \
    "vpbroadcastd 0x4(%%rax), %%zmm30       \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%zmm31 \n\t" \
    "prefetcht0 0x180(%[B])                \n\t" \
    "prefetcht0 0x1C0(%[B])                \n\t" \
    "prefetcht0 0x200(%[B])                \n\t" \
    "vmovups 0xC0(%[B]), %%zmm24           \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm0     \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm1     \n\t" \
    "vmovups 0x100(%[B]), %%zmm25          \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm2     \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm3     \n\t" \
    "vmovups 0x140(%[B]), %%zmm26          \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm4     \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm5     \n\t"

#define mmm_4_48(A, K) \
    "movq "#A", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "prefetcht0 0xC0(%1)                   \n\t" \
    "prefetcht0 0x100(%1)                  \n\t" \
    "prefetcht0 0x140(%1)                  \n\t" \
    "vmovups (%[B]), %%zmm27               \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm0     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm1     \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm2     \n\t" \
    "vmovups 0x40(%[B]), %%zmm28           \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm3     \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm4     \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm5     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "vmovups 0x80(%[B]), %%zmm29           \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm6     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm7     \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm8     \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm9     \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm10    \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm11    \n\t" \
    "movq "#A", %%rax                         \n\t" \
    "addq $0x4, %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "prefetcht0 0x180(%[B])                \n\t" \
    "prefetcht0 0x1C0(%[B])                \n\t" \
    "prefetcht0 0x200(%[B])                \n\t" \
    "vmovups 0xC0(%[B]), %%zmm24           \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm0     \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm1     \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm2     \n\t" \
    "vmovups 0x100(%[B]), %%zmm25          \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm3     \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm4     \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm5     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "vmovups 0x140(%[B]), %%zmm26          \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm6     \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm7     \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm8     \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm9     \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm10    \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm11    \n\t"

#define mmm_8_48(A, K) \
    "movq "#A", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "prefetcht0 0xC0(%[B])                 \n\t" \
    "prefetcht0 0x100(%[B])                \n\t" \
    "prefetcht0 0x140(%[B])                \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm0     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm1     \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm2     \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm3     \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm4     \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm5     \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "vmovups (%[B]), %%zmm27               \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm6     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm7     \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm8     \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm9     \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm10    \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm11    \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "vmovups 0x40(%[B]), %%zmm28           \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm12    \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm13    \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm14    \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm15    \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm16    \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm17    \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "vmovups 0x80(%[B]), %%zmm29           \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm18    \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm19    \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm20    \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm21    \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm22    \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm23    \n\t" \
    "movq "#A", %%rax                      \n\t" \
    "addq $0x4, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "prefetcht0 0x180(%[B])                \n\t" \
    "prefetcht0 0x1C0(%[B])                \n\t" \
    "prefetcht0 0x200(%[B])                \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm0     \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm1     \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm2     \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm3     \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm4     \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm5     \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "vmovups 0xC0(%[B]), %%zmm24           \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm6     \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm7     \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm8     \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm9     \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm10    \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm11    \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "vmovups 0x100(%[B]), %%zmm25          \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm12    \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm13    \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm14    \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm15    \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm16    \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm17    \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "vmovups 0x140(%[B]), %%zmm26          \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm18    \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm19    \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm20    \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm21    \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm22    \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm23    \n\t"

#define mmm_1_32(A, K) \
    "movq "#A", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28         \n\t" \
    "vpbroadcastd 0x4(%%rax), %%zmm29      \n\t" \
    "prefetcht0 0x80(%[B])                 \n\t" \
    "prefetcht0 0xC0(%[B])                 \n\t" \
    "vmovups (%[B]), %%zmm26               \n\t" \
    "vmovups 0x40(%[B]), %%zmm27           \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm0     \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm1     \n\t" \
    "prefetcht0 0x100(%[B])                \n\t" \
    "prefetcht0 0x140(%[B])                \n\t" \
    "vmovups 0x80(%[B]), %%zmm24           \n\t" \
    "vmovups 0xC0(%[B]), %%zmm25           \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm0     \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm1     \n\t"

#define mmm_3_32(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "prefetcht0 0x80(%[B])                  \n\t" \
    "prefetcht0 0xC0(%[B])                  \n\t" \
    "vmovups (%[B]), %%zmm26                \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm0      \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm1      \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm2      \n\t" \
    "vmovups 0x40(%[B]), %%zmm27            \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm3      \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm5      \n\t" \
    "movq "#A", %%rax                       \n\t" \
    "addq $0x4, %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "prefetcht0 0x100(%[B])                 \n\t" \
    "prefetcht0 0x140(%[B])                 \n\t" \
    "vmovups 0x80(%[B]), %%zmm24            \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm0      \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm1      \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm2      \n\t" \
    "vmovups 0xC0(%[B]), %%zmm25            \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm3      \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm5      \n\t"

#define mmm_6_32(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "movq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "prefetcht0 0x80(%[B])                  \n\t" \
    "prefetcht0 0xC0(%[B])                  \n\t" \
    "vmovups (%[B]), %%zmm26                \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm0      \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm1      \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm2      \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm3      \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm5      \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm6      \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm7      \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vmovups 0x40(%[B]), %%zmm27            \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm8      \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm9      \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm10     \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm11     \n\t" \
    "movq "#A", %%rax                       \n\t" \
    "addq $0x4, %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "prefetcht0 0x100(%[B])                 \n\t" \
    "prefetcht0 0x140(%[B])                 \n\t" \
    "vmovups 0x80(%[B]), %%zmm24            \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm0      \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm1      \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm2      \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm3      \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm5      \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm6      \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm7      \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vmovups 0xC0(%[B]), %%zmm25            \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm8      \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm9      \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm10     \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm11     \n\t"

#define mmm_12_32(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "movq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "prefetcht0 0x80(%[B])                  \n\t" \
    "prefetcht0 0xC0(%[B])                  \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm0      \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm1      \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm2      \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm3      \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm5      \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm6      \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm7      \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "vmovups (%[B]), %%zmm26                \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm8      \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm9      \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm10     \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm11     \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm12     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm13     \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm14     \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm15     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "vmovups 0x40(%[B]), %%zmm27            \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm16     \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm17     \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm18     \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm19     \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm20     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm21     \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm22     \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm23     \n\t" \
    "movq "#A", %%rax                       \n\t" \
    "addq $0x4, %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "prefetcht0 0x100(%[B])                 \n\t" \
    "prefetcht0 0x140(%[B])                 \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm0      \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm1      \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm2      \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm3      \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm5      \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm6      \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm7      \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "vmovups 0x80(%[B]), %%zmm24            \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm8      \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm9      \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm10     \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm11     \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm12     \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm13     \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm14     \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm15     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "vmovups 0xC0(%[B]), %%zmm25            \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm16     \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm17     \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm18     \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm19     \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm20     \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm21     \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm22     \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm23     \n\t"

#define mmm_1_16(A, K, rtype, off) \
    "vpbroadcastd ("#A"), "#rtype"25                     \n\t"        \
    "vpbroadcastd 0x4("#A"), "#rtype"26                     \n\t"     \
    "vmovups (%[B]), "#rtype"31                             \n\t"     \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"0              \n\t"        \
    "vmovups "#off"(%[B]), "#rtype"24                             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"0              \n\t"

#define mmm_6_16(A, K, rtype, off) \
    "movq "#A", %%rax  \n\t"                                          \
    "movq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"0              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"1              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"2              \n\t"           \
    "vmovups (%[B]), "#rtype"31                             \n\t"        \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"3              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"4              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"5              \n\t"           \
    "movq "#A", %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"0              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"1              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"2              \n\t"           \
    "vmovups "#off"(%[B]), "#rtype"24                             \n\t"    \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"3              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"4              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"5              \n\t"           \
    
#define mmm_12_16(A, K, rtype, off) \
    "movq "#A", %%rax  \n\t"                                          \
    "movq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"0              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"1              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"2              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"3              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"4              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"5              \n\t"           \
    "vmovups (%[B]), "#rtype"31                             \n\t"        \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"6              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"7              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"8              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"9              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"10              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"11              \n\t"          \
    "movq "#A", %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"0              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"1              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"2              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"3              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"4              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"5              \n\t"           \
    "vmovups "#off"(%[B]), "#rtype"24                             \n\t"    \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"6              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"7              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"8              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"9              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"10              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"11              \n\t"

#define mmm_24_16(A, K, rtype, off) \
    "movq "#A", %%rax  \n\t"                                          \
    "movq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "prefetcht0 0x80(%[B])                              \n\t"         \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"0              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"1              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"2              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"3              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"4              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"5              \n\t"           \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"6              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"7              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"8              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"9              \n\t"           \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"10              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"11              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vmovups (%[B]), "#rtype"31                             \n\t"        \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"12              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"13              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"14              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"15              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"16              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"17              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"18              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"19              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"20              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"21              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"22              \n\t"          \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"23              \n\t"          \
    "movq "#A", %%rax  \n\t"                                          \
    "addq $0x4, %%rax  \n\t"                                        \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "prefetcht0 0xC0(%[B])                              \n\t"         \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"0              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"1              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"2              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"3              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"4              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"5              \n\t"           \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"6              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"7              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"8              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"9              \n\t"           \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"10              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"11              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vmovups "#off"(%[B]), "#rtype"24                             \n\t"    \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"12              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"13              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"14              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"15              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"16              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"17              \n\t"          \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"25                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27                     \n\t" \
    "addq %%rbx, %%rax  \n\t"                                       \
    "vpbroadcastd (%%rax), "#rtype"28                     \n\t"        \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29                     \n\t"    \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30                     \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"18              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"19              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"20              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"21              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"22              \n\t"          \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"23              \n\t"

#define mmm_m_48_asm(m, n, nRegs, mRegs, edge) \
    __asm__ __volatile__(                                                  \
        "prefetcht0 0xC0(%[B])                              \n\t" \
        "prefetcht0 0x100(%[B])                              \n\t" \
        "prefetcht0 0x140(%[B])                              \n\t" \
        "vmovups (%[B]), %%zmm24                                     \n\t" \
        "vmovups 0x40(%[B]), %%zmm25                                 \n\t" \
        "vmovups 0x80(%[B]), %%zmm26                                 \n\t" \
        "add $0xC0, %[B]                                             \n\t" \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x1, %%rax                                            \n\t" \
        "jne 0f                                                      \n\t" \
        loadOffset_##m##_##n                                                     \
        "jmp 1f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "0:                                                          \n\t" \
        clear##nRegs##Regs(%%zmm)                                                 \
        ".align 16                                                   \n\t" \
        "1:                                                          \n\t" \
        "movq %[C], %%rax  \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 0x40(%%rax)                              \n\t"     \
        "prefetcht0 0x80(%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x40(%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x80(%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 0x40(%%rax)                              \n\t"     \
        "prefetcht0 0x80(%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x40(%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x80(%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 0x40(%%rax)                              \n\t"     \
        "prefetcht0 0x80(%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x40(%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x80(%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 0x40(%%rax)                              \n\t"     \
        "prefetcht0 0x80(%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x40(%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x80(%%rax, %[N])                              \n\t"     \
        "movq %[bk], %%rcx                                           \n\t" \
        "shr $3, %%rcx                                \n\t" \
        "je 3f                                        \n\t" \
        ".align 16                                                   \n\t" \
        "2:                                                          \n\t" \
        mmm_##m##_48(%[A], %[K])                                                     \
        "add $0x180, %[B]                                            \n\t" \
        "add $0x8, %[A]                                              \n\t" \
        "dec %%rcx                                                   \n\t" \
        "jg 2b                                                       \n\t" \
        ".align 16                                                   \n\t" \
        "3:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "and $7, %%rcx                                \n\t" \
        "je 4f                                        \n\t" \
        "movq $8, %%rcx                                           \n\t" \
        mmm_##m##_48(%[resK], %%rcx)                                                     \
        ".align 16                                                   \n\t" \
        "4:                                                          \n\t" \
        "movq %[C], %%rax  \n\t" \
        "movq %[N], %%rcx  \n\t" \
        "addq %[N], %%rcx                                     \n\t" \
        addC_##m##_##n(%%rax)                                                     \
        "cmpq $0x0, %[s]                                             \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##I32Regs2Ps(%%zmm, %[s])                                   \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x2, %%rax                                            \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##PsRegs2U8(%%zmm)                                          \
        storeC_##mRegs##_##n##_##edge(vpmovusdb, %%zmm, %[u8C], 0x10, 0x20)                   \
        "jmp 6f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "5:                                                          \n\t" \
        storeC_##mRegs##_##n##_##edge(vmovups, %%zmm, %[C], 0x40, 0x80)                       \
        ".align 16                                                   \n\t" \
        "6:                                                          \n\t" \
        : [B] "+r" (matrixB)                                               \
        : [A] "r" (matrixA),                                               \
          [C] "r" (matrixC),                                               \
          [bk] "r" ((int64_t)bk),                                          \
          [N]"r" ((int64_t)(N * 4)),                                       \
          [s] "r" (scale),                                                 \
          [K] "r" ((int64_t)stepK),                                        \
          [offset] "r" (offsetC),                                          \
          [flags] "b" ((int64_t)flags),                                    \
          [u8C] "r" (u8Result),                                             \
          [nmask] "r" (nmask),                                             \
          [resK] "r" (resK)                                             \
        : "%rax", "%rcx",                                                  \
          "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",            \
          "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",          \
          "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",      \
          "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",      \
          "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",      \
          "%zmm30", "%zmm31", "memory", "cc");

#define mmm_m_32_asm(m, n, nRegs, mRegs, edge) \
    __asm__ __volatile__(                                                  \
        "prefetcht0 0xC0(%[B])                              \n\t" \
        "prefetcht0 0x100(%[B])                              \n\t" \
        "vmovups (%[B]), %%zmm24                                     \n\t" \
        "vmovups 0x40(%[B]), %%zmm25                                 \n\t" \
        "add $0x80, %[B]                                             \n\t" \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x1, %%rax                                            \n\t" \
        "jne 0f                                                      \n\t" \
        loadOffset_##m##_##n                                                     \
        "jmp 1f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "0:                                                          \n\t" \
        clear##nRegs##Regs(%%zmm)                                                 \
        ".align 16                                                   \n\t" \
        "1:                                                          \n\t" \
        "movq %[C], %%rax  \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 0x40(%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x40(%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 0x40(%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x40(%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 0x40(%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x40(%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 0x40(%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "prefetcht0 0x40(%%rax, %[N])                              \n\t"     \
        "movq %[bk], %%rcx                                           \n\t" \
        "shr $3, %%rcx                                \n\t" \
        "je 3f                                        \n\t" \
        ".align 16                                                   \n\t" \
        "2:                                                          \n\t" \
        mmm_##m##_32(%[A], %[K])                                                     \
        "add $0x100, %[B]                                            \n\t" \
        "add $0x8, %[A]                                              \n\t" \
        "dec %%rcx                                                   \n\t" \
        "jg 2b                                                       \n\t" \
        ".align 16                                                   \n\t" \
        "3:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "and $7, %%rcx                                \n\t" \
        "je 4f                                        \n\t" \
        "movq $8, %%rcx                                           \n\t" \
        mmm_##m##_32(%[resK], %%rcx)                                                     \
        ".align 16                                                   \n\t" \
        "4:                                                          \n\t" \
        addC_##m##_##n(%[C])                                                     \
        "cmpq $0x0, %[s]                                             \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##I32Regs2Ps(%%zmm, %[s])                                   \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x2, %%rax                                            \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##PsRegs2U8(%%zmm)                                          \
        storeC_##mRegs##_##n##_##edge(vpmovusdb, %%zmm, %[u8C], 0x10, 0x20)                   \
        "jmp 6f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "5:                                                          \n\t" \
        storeC_##mRegs##_##n##_##edge(vmovups, %%zmm, %[C], 0x40, 0x80)                       \
        ".align 16                                                   \n\t" \
        "6:                                                          \n\t" \
        : [B] "+r" (matrixB)                                               \
        : [A] "r" (matrixA),                                               \
          [C] "r" (matrixC),                                               \
          [bk] "r" ((int64_t)bk),                                          \
          [N]"r" ((int64_t)(N * 4)),                                       \
          [s] "r" (scale),                                                 \
          [K] "r" ((int64_t)stepK),                                        \
          [offset] "r" (offsetC),                                          \
          [flags] "b" ((int64_t)flags),                                    \
          [u8C] "r" (u8Result),                                             \
          [nmask] "r" (nmask),                                             \
          [resK] "r" (resK)                                             \
        : "%rax", "%rcx",                                                  \
          "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",            \
          "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",          \
          "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",      \
          "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",      \
          "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",      \
          "%zmm30", "%zmm31", "memory", "cc");

#define mmm_m_16_8_asm(m, n, nRegs, mRegs, rtype, off0, off1, edge) \
    __asm__ __volatile__(                                                  \
        "vmovups (%[B]), "#rtype"24                                     \n\t" \
        "add $"#off0", %[B]                                             \n\t" \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x1, %%rax                                            \n\t" \
        "jne 0f                                                      \n\t" \
        loadOffset_##m##_##n(rtype)                                                     \
        "jmp 1f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "0:                                                          \n\t" \
        clear##nRegs##Regs(rtype)                                                 \
        ".align 16                                                   \n\t" \
        "1:                                                          \n\t" \
        "movq %[C], %%rax  \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "add %[N], %%rax                                     \n\t"     \
        "prefetcht0 (%%rax)                              \n\t"     \
        "prefetcht0 (%%rax, %[N])                              \n\t"     \
        "movq %[bk], %%rcx                                           \n\t" \
        "shr $3, %%rcx                                \n\t" \
        "je 3f                                        \n\t" \
        ".align 16                                                   \n\t" \
        "2:                                                          \n\t" \
        mmm_##m##_16(%[A], %[K], rtype, off0)                                                     \
        "add $"#off1", %[B]                                            \n\t" \
        "add $0x8, %[A]                                              \n\t" \
        "dec %%rcx                                                   \n\t" \
        "jg 2b                                                       \n\t" \
        ".align 16                                                   \n\t" \
        "3:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "and $7, %%rcx                                \n\t" \
        "je 4f                                        \n\t" \
        "movq $8, %%rcx                                           \n\t" \
        mmm_##m##_16(%[resK], %%rcx, rtype, off0)                                                     \
        ".align 16                                                   \n\t" \
        "4:                                                          \n\t" \
        addC_##m##_##n(rtype, %[C])                                                     \
        "cmpq $0x0, %[s]                                             \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##I32Regs2Ps(rtype, %[s])                                   \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x2, %%rax                                            \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##PsRegs2U8(rtype)                                          \
        storeC_##mRegs##_##n##_##edge(vpmovusdb, rtype, %[u8C], 0x0, 0x0)                   \
        "jmp 6f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "5:                                                          \n\t" \
        storeC_##mRegs##_##n##_##edge(vmovups, rtype, %[C], 0x0, 0x0)                       \
        ".align 16                                                   \n\t" \
        "6:                                                          \n\t" \
        : [B] "+r" (matrixB)                                               \
        : [A] "r" (matrixA),                                               \
          [C] "r" (matrixC),                                               \
          [bk] "r" ((int64_t)bk),                                          \
          [N]"r" ((int64_t)(N * 4)),                                       \
          [s] "r" (scale),                                                 \
          [K] "r" ((int64_t)stepK),                                        \
          [offset] "r" (offsetC),                                          \
          [flags] "r" ((int64_t)flags),                                    \
          [u8C] "r" (u8Result),                                             \
          [nmask] "r" (nmask),                                             \
          [resK] "r" (resK)                                             \
        : "%rax", "%rbx", "%rcx",                                                  \
          "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",            \
          "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",          \
          "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",      \
          "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",      \
          "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",      \
          "%zmm30", "%zmm31", "memory", "cc");

#define mmm_m_16_asm(m, n, nRegs, mRegs, edge) \
    mmm_m_16_8_asm(m, n, nRegs, mRegs, %%zmm, 0x40, 0x80, edge)

#define mmm_m_8_asm(m, n, nRegs, mRegs, edge) \
    mmm_m_16_8_asm(m, n, nRegs, mRegs, %%ymm, 0x20, 0x40, edge)

#define mmm_m_n_asm(m, n, nRegs, mRegs, regs) \
    void mmm_avx512_##mRegs##x##n##_asm(U32 um, \
        U32 un, \
        U32 bk, \
        UINT8 *matrixA, \
        INT8 *matrixB, \
        I32 *matrixC, \
        UINT8 *u8Result, \
        I32 *offsetC, \
        U32 N, \
        U32 stepK, \
        const F32 *scale, \
        U32 nmask, \
        UINT8 *resK, \
        U32 flags) \
    { \
        if (nmask == 0) { \
            mmm_m_##n##_asm(m, nRegs, regs, mRegs, 0) \
        } else { \
            mmm_m_##n##_asm(m, nRegs, regs, mRegs, 1) \
        } \
    }

mmm_m_n_asm(8, 48, 3, 8, 24)
mmm_m_n_asm(8, 48, 3, 7, 24)
mmm_m_n_asm(8, 48, 3, 6, 24)
mmm_m_n_asm(8, 48, 3, 5, 24)
mmm_m_n_asm(4, 48, 3, 4, 12)
mmm_m_n_asm(4, 48, 3, 3, 12)
mmm_m_n_asm(2, 48, 3, 2, 6)
mmm_m_n_asm(1, 48, 3, 1, 1)

mmm_m_n_asm(12, 32, 2, 12, 24)
mmm_m_n_asm(12, 32, 2, 11, 24)
mmm_m_n_asm(12, 32, 2, 10, 24)
mmm_m_n_asm(12, 32, 2, 9, 24)
mmm_m_n_asm(12, 32, 2, 8, 24)
mmm_m_n_asm(12, 32, 2, 7, 24)
mmm_m_n_asm(6, 32, 2, 6, 12)
mmm_m_n_asm(6, 32, 2, 5, 12)
mmm_m_n_asm(6, 32, 2, 4, 12)
mmm_m_n_asm(3, 32, 2, 3, 6)
mmm_m_n_asm(3, 32, 2, 2, 6)
mmm_m_n_asm(1, 32, 2, 1, 1)

mmm_m_n_asm(24, 16, 1, 24, 24)
mmm_m_n_asm(24, 16, 1, 23, 24)
mmm_m_n_asm(24, 16, 1, 22, 24)
mmm_m_n_asm(24, 16, 1, 21, 24)
mmm_m_n_asm(24, 16, 1, 20, 24)
mmm_m_n_asm(24, 16, 1, 19, 24)
mmm_m_n_asm(24, 16, 1, 18, 24)
mmm_m_n_asm(24, 16, 1, 17, 24)
mmm_m_n_asm(24, 16, 1, 16, 24)
mmm_m_n_asm(24, 16, 1, 15, 24)
mmm_m_n_asm(24, 16, 1, 14, 24)
mmm_m_n_asm(24, 16, 1, 13, 24)
mmm_m_n_asm(12, 16, 1, 12, 12)
mmm_m_n_asm(12, 16, 1, 11, 12)
mmm_m_n_asm(12, 16, 1, 10, 12)
mmm_m_n_asm(12, 16, 1, 9, 12)
mmm_m_n_asm(12, 16, 1, 8, 12)
mmm_m_n_asm(12, 16, 1, 7, 12)
mmm_m_n_asm(6, 16, 1, 6, 6)
mmm_m_n_asm(6, 16, 1, 5, 6)
mmm_m_n_asm(6, 16, 1, 4, 6)
mmm_m_n_asm(6, 16, 1, 3, 6)
mmm_m_n_asm(6, 16, 1, 2, 6)
mmm_m_n_asm(1, 16, 1, 1, 1)

mmm_m_n_asm(24, 8, 1, 24, 24)
mmm_m_n_asm(24, 8, 1, 23, 24)
mmm_m_n_asm(24, 8, 1, 22, 24)
mmm_m_n_asm(24, 8, 1, 21, 24)
mmm_m_n_asm(24, 8, 1, 20, 24)
mmm_m_n_asm(24, 8, 1, 19, 24)
mmm_m_n_asm(24, 8, 1, 18, 24)
mmm_m_n_asm(24, 8, 1, 17, 24)
mmm_m_n_asm(24, 8, 1, 16, 24)
mmm_m_n_asm(24, 8, 1, 15, 24)
mmm_m_n_asm(24, 8, 1, 14, 24)
mmm_m_n_asm(24, 8, 1, 13, 24)
mmm_m_n_asm(12, 8, 1, 12, 12)
mmm_m_n_asm(12, 8, 1, 11, 12)
mmm_m_n_asm(12, 8, 1, 10, 12)
mmm_m_n_asm(12, 8, 1, 9, 12)
mmm_m_n_asm(12, 8, 1, 8, 12)
mmm_m_n_asm(12, 8, 1, 7, 12)
mmm_m_n_asm(6, 8, 1, 6, 6)
mmm_m_n_asm(6, 8, 1, 5, 6)
mmm_m_n_asm(6, 8, 1, 4, 6)
mmm_m_n_asm(6, 8, 1, 3, 6)
mmm_m_n_asm(6, 8, 1, 2, 6)
mmm_m_n_asm(1, 8, 1, 1, 1)


void matrix_matrix_multiply_tmp_bytes_int8(
    U32 row1, U32 col1, U32 row2, U32 col2, DataFormat df, DataType dt, U32 *bytes)
{
    U32 alignedN = UNI_ALIGN(col2, 16);
    U32 alignedK = UNI_ALIGN(row2, 8);
    *bytes = 2 * alignedN * bytesOf(DT_I32) + alignedN * alignedK;
    if (df == DF_NORMAL) {
        *bytes += 32 * col1;
        if (col1 % 8 != 0) {
            *bytes += UNI_ALIGN(row1, 24) * 8;
        }
    } else if (df == DF_TRANSPOSE) {
        *bytes += UNI_ALIGN(col1, 24) * UNI_MIN(BOLCK_K_DIM, alignedK);
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    *bytes += 64;
}

// clang-format on

EE matrix_matrix_multiply_transform_rhsN_int8(TensorDesc desc, INT8 *src, INT8 *packB)
{
    DataType dt;
    DataFormat df;
    U32 N, K, blockSizeK, unrollSizeN;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    U32 unrollSize[4] = {8, 16, 32, 48};
    INT8 *tmpS = src;
    I32 *offsetCBias = (I32 *)(packB + UNI_ALIGN(K, SIMDW) * UNI_ALIGN(N, 16));

    for (U32 bk = 0; bk < K; bk += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        for (U32 un = 0; un < N; un += unrollSizeN) {
            unrollSizeN = UNI_MIN(UNROLL_N, N - un);
            U32 alignedN = (unrollSizeN > 8) ? UNI_ALIGN(unrollSizeN, 16) : 8;
            matrix2_trans_l(unrollSizeN, alignedN, blockSizeK, N, SIMDW, tmpS + un, packB);
            packB += alignedN * UNI_ALIGN(blockSizeK, SIMDW);
        }
        tmpS += blockSizeK * N;
    }

    for (U32 n = 0; n < N; ++n) {
        I32 tmp = 0;
        for (U32 k = 0; k < K; ++k) {
            tmp += (I32)(src[k * N + n]);
        }
        offsetCBias[n] = tmp * (-128);
    }
    return SUCCESS;
}

EE matrix_matrix_multiply_transform_rhsT_int8(TensorDesc desc, INT8 *src, INT8 *packB)
{
    DataType dt;
    DataFormat df;
    U32 N, K, blockSizeK, unrollSizeN;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    U32 unrollSize[4] = {8, 16, 32, 48};
    INT8 *tmpS = src;
    I32 *offsetCBias = (I32 *)(packB + UNI_ALIGN(K, SIMDW) * UNI_ALIGN(N, 16));

    for (U32 bk = 0; bk < K; bk += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        for (U32 un = 0; un < N; un += unrollSizeN) {
            unrollSizeN = UNI_MIN(UNROLL_N, N - un);
            U32 alignedN = (unrollSizeN > 8) ? UNI_ALIGN(unrollSizeN, 16) : 8;
            matrix1_trans_l(unrollSizeN, alignedN, blockSizeK, K, SIMDW, tmpS + un * K, packB);
            packB += alignedN * UNI_ALIGN(blockSizeK, SIMDW);
        }
        tmpS += blockSizeK;
    }

    for (U32 n = 0; n < N; ++n) {
        I32 tmp = 0;
        for (U32 k = 0; k < K; ++k) {
            tmp += (I32)(src[n * K + k]);
        }
        offsetCBias[n] = tmp * (-128);
    }

    return SUCCESS;
}

//TODO: matrixC alloc
EE mmm_avx512_vnni_int8(U32 N,
    U32 M,
    U32 K,
    DataFormat matrix1Df,
    UINT8 *matrix1,
    INT8 *packB,
    UINT8 *tmp,
    UINT8 *result,
    const F32 *scale)
{
    UINT8 *packA = matrix1;
    kernel_func kernel[24][4] = {
        {mmm_avx512_1x8_asm, mmm_avx512_1x16_asm, mmm_avx512_1x32_asm, mmm_avx512_1x48_asm},
        {mmm_avx512_2x8_asm, mmm_avx512_2x16_asm, mmm_avx512_2x32_asm, mmm_avx512_2x48_asm},
        {mmm_avx512_3x8_asm, mmm_avx512_3x16_asm, mmm_avx512_3x32_asm, mmm_avx512_3x48_asm},
        {mmm_avx512_4x8_asm, mmm_avx512_4x16_asm, mmm_avx512_4x32_asm, mmm_avx512_4x48_asm},
        {mmm_avx512_5x8_asm, mmm_avx512_5x16_asm, mmm_avx512_5x32_asm, mmm_avx512_5x48_asm},
        {mmm_avx512_6x8_asm, mmm_avx512_6x16_asm, mmm_avx512_6x32_asm, mmm_avx512_6x48_asm},
        {mmm_avx512_7x8_asm, mmm_avx512_7x16_asm, mmm_avx512_7x32_asm, mmm_avx512_7x48_asm},
        {mmm_avx512_8x8_asm, mmm_avx512_8x16_asm, mmm_avx512_8x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_9x8_asm, mmm_avx512_9x16_asm, mmm_avx512_9x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_10x8_asm, mmm_avx512_10x16_asm, mmm_avx512_10x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_11x8_asm, mmm_avx512_11x16_asm, mmm_avx512_11x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_12x8_asm, mmm_avx512_12x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_13x8_asm, mmm_avx512_13x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_14x8_asm, mmm_avx512_14x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_15x8_asm, mmm_avx512_15x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_16x8_asm, mmm_avx512_16x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_17x8_asm, mmm_avx512_17x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_18x8_asm, mmm_avx512_18x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_19x8_asm, mmm_avx512_19x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_20x8_asm, mmm_avx512_20x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_21x8_asm, mmm_avx512_21x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_22x8_asm, mmm_avx512_22x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_23x8_asm, mmm_avx512_23x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm},
        {mmm_avx512_24x8_asm, mmm_avx512_24x16_asm, mmm_avx512_12x32_asm, mmm_avx512_8x48_asm}};
    U32 unrollNSizes[4] = {8, 16, 32, 48};
    U32 unrollMSizes[5] = {24, 24, 12, 8};
    U32 alignedK = (K + 7) / 8 * 8;

    I32 *offsetC = (I32 *)(tmp);
    tmp += N * bytesOf(DT_I32);

    U32 flags = 0;
    F32 *factorPtr = nullptr;
    F32 factor = 0;
    I32 *i32Result = (I32 *)result;
    UINT8 *u8Result = result;
    if (scale != nullptr) {
        if (scale[0] < 0) {
            // when use offline scale, the output datatype is U8_Q, you need more tmp buffer
            flags |= 1 << 1;
            factor = scale[1];
            i32Result = (I32 *)tmp;
            UNI_MEMSET(i32Result, 0, M * N * bytesOf(DT_I32));
            tmp += M * N * bytesOf(DT_I32);
        } else {
            factor = 1 / (scale[0]);
        }
        factorPtr = &factor;
    }

    auto getEdgeMSize = [](U32 resM, U32 unrollM) {
        U32 unit = unrollM / 2;
        U32 low = unrollM / 4;
        return (resM > 1) ? ((resM > low) ? ((resM + unit - 1) / unit * unit) : low) : resM;
    };
    auto getMNum = [](U32 mDim, U32 unrollM) { return mDim / unrollM + ((mDim % unrollM) > 0); };

    U32 resN = N % UNROLL_N;
    U32 edgeNSize = (resN > 8) ? UNI_ALIGN(resN, 16) : 8;
    U32 resM = M % UNROLL_M;
    U32 mainEdgeMSize = getEdgeMSize(resM, UNROLL_M);
    UINT8 *lastMainBlockA = packA + M / UNROLL_M * UNROLL_M * K;
    if (resM < mainEdgeMSize && matrix1Df == DF_NORMAL) {  // padding last block
        UNI_MEMCPY(tmp, lastMainBlockA, resM * K);
        UNI_MEMSET(tmp + resM * K, 128, (mainEdgeMSize - resM) * K);
        lastMainBlockA = tmp;
        tmp += mainEdgeMSize * K;
    }
    U32 mloopNum = getMNum(BOLCK_M_DIM, UNROLL_M) * (M / BOLCK_M_DIM) +
        getMNum(M % BOLCK_M_DIM, UNROLL_M) * (M % BOLCK_M_DIM > 0);

    U32 newUnrollM = unrollMSizes[edgeNSize >> 4];
    resM = M % newUnrollM;
    U32 resEdgeMSize = getEdgeMSize(resM, newUnrollM);
    UINT8 *lastResBlockA = packA + M / newUnrollM * newUnrollM * K;
    if (resM < resEdgeMSize && matrix1Df == DF_NORMAL) {  // padding last block
        UNI_MEMCPY(tmp, lastResBlockA, resM * K);
        UNI_MEMSET(tmp + resM * K, 128, (resEdgeMSize - resM) * K);
        lastResBlockA = tmp;
        tmp += resEdgeMSize * K;
    }
    U32 resMloopNum = getMNum(BOLCK_M_DIM, newUnrollM) * (M / BOLCK_M_DIM) +
        getMNum(M % BOLCK_M_DIM, newUnrollM) * (M % BOLCK_M_DIM > 0);

    U32 padM = UNI_MAX(UNI_ALIGN(M, UNROLL_M), UNI_ALIGN(M, newUnrollM));
    UINT8 *tmpK = tmp;
    U32 resK = K % SIMDW;
    if (resK > 0 && matrix1Df == DF_NORMAL) {
        for (U32 i = 0; i < M; ++i) {
            UNI_MEMCPY(tmpK + i * SIMDW, packA + (i + 1) * K - resK, resK);
            UNI_MEMSET(tmpK + i * SIMDW + resK, 128, SIMDW - resK);
        }
        UNI_MEMSET(tmpK + M * SIMDW, 128, (padM - M) * SIMDW);
        tmp += padM * SIMDW;
    }
    U32 mNNum = N / UNROLL_N;
    U32 alginedN = mNNum * UNROLL_N + (resN > 0) * edgeNSize;
    U32 nmask = pow(2, N % 16) - 1;
    U32 loopNum = mNNum * mloopNum + (resN > 0) * resMloopNum;
    U32 bmLoopNum =
        mNNum * getMNum(BOLCK_M_DIM, UNROLL_M) + (resN > 0) * getMNum(BOLCK_M_DIM, newUnrollM);

#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        U32 blockSizeK = 0;
        for (U32 k = 0; k < K; k += blockSizeK) {
            blockSizeK = UNI_MIN(BOLCK_K_DIM, K - k);
            F32 *useFactor = nullptr;
            flags |= (k > 0);
            if (k == K - blockSizeK) {
                useFactor = factorPtr;
            }

            U32 realK = blockSizeK;
            U32 stepK = K;
            if (matrix1Df == DF_TRANSPOSE) {
                matrix2_trans_r(M, blockSizeK, M, SIMDW, packA, tmp);
                realK = UNI_ALIGN(realK, SIMDW);
                packA = tmp;
                stepK = realK;
            }

#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
            for (U32 l = 0; l < loopNum; ++l) {
                U32 bm = l / bmLoopNum * BOLCK_M_DIM;
                U32 blockSizeM = UNI_MIN(BOLCK_M_DIM, M - bm);
                U32 mMNum = getMNum(blockSizeM, UNROLL_M);
                U32 bn = l % bmLoopNum;
                U32 nLoop = bn / mMNum;
                U32 n = nLoop * UNROLL_N;
                U32 mLoop = bn % mMNum;
                U32 m = mLoop * UNROLL_M;
                U32 edgeMSize = mainEdgeMSize;
                U32 unrollM = UNROLL_M;
                U32 mNum = mMNum;
                U32 nSize = UNROLL_N;
                UINT8 *lastBlockA = lastMainBlockA;
                if (bn >= mNNum * mMNum) {
                    nLoop = mNNum;
                    n = mNNum * UNROLL_N;
                    mLoop = bn - mNNum * mMNum;
                    m = mLoop * newUnrollM;
                    edgeMSize = resEdgeMSize;
                    lastBlockA = lastResBlockA;
                    unrollM = newUnrollM;
                    mNum = getMNum(blockSizeM, newUnrollM);
                    nSize = edgeNSize;
                }

                U32 um = (unrollM + m > blockSizeM) ? edgeMSize : unrollM;
                U32 rm = UNI_MIN(unrollM, blockSizeM - m);
                INT8 *curB = packB + k * alginedN + n * UNI_ALIGN(realK, SIMDW);
                UINT8 *curA = packA + (m + bm) * stepK + k;
                if ((mLoop == (mNum - 1)) && (M - bm <= BOLCK_M_DIM) && (resM < edgeMSize) &&
                    (matrix1Df == DF_NORMAL)) {
                    curA = lastBlockA + k;
                }
                UINT8 *kpad = tmpK + (m + bm) * SIMDW;
                U32 tnmask = (nLoop == mNNum - 1 + (resN > 0)) ? nmask : 0;
                kernel[rm - 1][nSize >> 4](um, nSize, realK, curA, curB,
                    i32Result + (m + bm) * N + n, u8Result + (m + bm) * N + n, offsetC + n, N,
                    stepK, useFactor, tnmask, kpad, flags);
            }
        }
    }
    return SUCCESS;
}
