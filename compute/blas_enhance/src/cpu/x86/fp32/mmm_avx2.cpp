// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/x86/fp32/blas_fp32.h"
#include "blas_enhance.h"

#define UNROLL_K 4
#define UNROLL_N 24
#define UNROLL_M 4
#define BOLCK_M_DIM 1024
#define BOLCK_K_DIM 1024
#define align_addr(addr, unit) (((uintptr_t)addr + unit - 1) / unit * unit)

void matrix_matrix_multiply_transform_rhs_bytes_fp32(
    U32 N, U32 K, DataFormat bdf, U32 *bytes, U32 *rhsBytes)
{
    U32 matrix = 0;
    U32 pad = 0;
    if (bdf != matrix_matrix_multiply_rhs_format(DT_F32)) {
        matrix = UNI_ALIGN(N, 8) * K * bytesOf(DT_F32);
        pad = matrix + 32;
    }
    if (rhsBytes != nullptr) {
        *rhsBytes = matrix;
    }
    if (bytes != nullptr) {
        *bytes = pad;
    }
}

void matrix_matrix_multiply_tmp_bytes_fp32(
    U32 N, U32 M, U32 K, DataFormat adf, DataFormat bdf, U32 *bytes)
{
    matrix_matrix_multiply_transform_rhs_bytes_fp32(N, K, bdf, bytes, nullptr);
    *bytes += M * K * bytesOf(DT_F32);
    *bytes += 32;
}

EE matrix_matrix_multiply_transform_rhsN_fp32(TensorDesc desc, F32 *src, F32 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    F32 unrollSize[4] = {4, 8, 16, 24};
    U32 resN = N % UNROLL_N;
    U32 edgeBlockNSizeIdx = (resN > 4) ? ((resN + 7) / 8) : 0;
    U32 edgeBlockNSize = (resN > 0) ? unrollSize[edgeBlockNSizeIdx] : 0;
    I32 blockNNum = N / UNROLL_N + (resN > 0);
    U32 blockKNum = (K + BOLCK_K_DIM - 1) / BOLCK_K_DIM;
    I32 alginedN = (blockNNum - 1) * UNROLL_N + edgeBlockNSize;
    if (edgeBlockNSize == 0) {
        alginedN += UNROLL_N;
    }
    U32 loopNum = blockKNum * blockNNum;

    // buffer addr algined to 32
    F32 *packB = (F32 *)align_addr(dst, 32);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 l = 0; l < loopNum; ++l) {
        U32 bk = l / blockNNum * BOLCK_K_DIM;
        U32 blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        U32 un = (l % blockNNum) * UNROLL_N;
        U32 unrollSizeN = UNI_MAX(UNI_MIN(UNROLL_N, N - un), edgeBlockNSize);
        F32 *curB = packB + bk * alginedN + un * blockSizeK;
        matrix2_trans_w(
            unrollSizeN, UNI_MIN(N - un, unrollSizeN), blockSizeK, N, src + bk * N + un, curB);
    }
    return SUCCESS;
}

EE matrix_matrix_multiply_transform_rhsT_fp32(TensorDesc desc, F32 *src, F32 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    F32 unrollSize[4] = {4, 8, 16, 24};
    U32 resN = N % UNROLL_N;
    U32 edgeBlockNSizeIdx = (resN > 4) ? ((resN + 7) / 8) : 0;
    U32 edgeBlockNSize = (resN > 0) ? unrollSize[edgeBlockNSizeIdx] : 0;
    I32 blockNNum = N / UNROLL_N + (resN > 0);
    U32 blockKNum = (K + BOLCK_K_DIM - 1) / BOLCK_K_DIM;
    I32 alginedN = (blockNNum - 1) * UNROLL_N + edgeBlockNSize;
    if (edgeBlockNSize == 0) {
        alginedN += UNROLL_N;
    }
    U32 loopNum = blockKNum * blockNNum;

    // buffer addr aligned to 32
    F32 *packB = (F32 *)align_addr(dst, 32);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
    for (U32 l = 0; l < loopNum; ++l) {
        U32 bk = l / blockNNum * BOLCK_K_DIM;
        U32 blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        U32 un = (l % blockNNum) * UNROLL_N;
        U32 unrollSizeN = UNI_MAX(UNI_MIN(UNROLL_N, N - un), edgeBlockNSize);
        F32 *curB = packB + bk * alginedN + un * blockSizeK;
        matrix1_trans_w(
            unrollSizeN, UNI_MIN(N - un, unrollSizeN), blockSizeK, K, src + bk + un * K, curB);
    }

    return SUCCESS;
}

typedef void (*kernel_func)(U32 um,
    U32 un,
    U32 bk,
    F32 *matrixA,
    F32 *matrixB,
    F32 *matrixC,
    U32 ldc,
    I32 *mask,
    F32 *A1,
    F32 *A2,
    F32 *A3);

// clang-format off
#define clear1Regs(rtype) \
    "vxorps "#rtype"0, "#rtype"0, "#rtype"0                     \n\t"

#define clear2Regs(rtype) \
    clear1Regs(rtype) \
    "vxorps "#rtype"1, "#rtype"1, "#rtype"1                     \n\t"

#define clear3Regs(rtype) \
    clear2Regs(rtype) \
    "vxorps "#rtype"2, "#rtype"2, "#rtype"2                     \n\t"

#define clear4Regs(rtype) \
    clear3Regs(rtype) \
    "vxorps "#rtype"3, "#rtype"3, "#rtype"3                     \n\t"

#define clear6Regs(rtype) \
    clear4Regs(rtype) \
    "vxorps "#rtype"4, "#rtype"4, "#rtype"4                     \n\t" \
    "vxorps "#rtype"5, "#rtype"5, "#rtype"5                     \n\t"

#define clear8Regs(rtype) \
    clear6Regs(rtype) \
    "vxorps "#rtype"6, "#rtype"6, "#rtype"6                     \n\t" \
    "vxorps "#rtype"7, "#rtype"7, "#rtype"7                     \n\t"

#define clear9Regs(rtype) \
    clear8Regs(rtype) \
    "vxorps "#rtype"8, "#rtype"8, "#rtype"8                     \n\t"

#define clear12Regs(rtype) \
    clear9Regs(rtype) \
    "vxorps "#rtype"9, "#rtype"9, "#rtype"9                     \n\t" \
    "vxorps "#rtype"10, "#rtype"10, "#rtype"10                  \n\t" \
    "vxorps "#rtype"11, "#rtype"11, "#rtype"11                  \n\t"

#define asm_1x24_kernel(i0, f0, f1, f2) \
    "vbroadcastss "#i0"(%[A0]), %%ymm15                   \n\t" \
    "vmovaps "#f0"(%[B]), %%ymm12                        \n\t" \
    "vmovaps "#f1"(%[B]), %%ymm13                    \n\t" \
    "vmovaps "#f2"(%[B]), %%ymm14                    \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm0         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm1         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm2         \n\t"

#define asm_2x24_kernel(i0, f0, f1, f2) \
    asm_1x24_kernel(i0, f0, f1, f2) \
    "vbroadcastss "#i0"(%[A1]), %%ymm15                   \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm3         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm4         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm5         \n\t"

#define asm_3x24_kernel(i0, f0, f1, f2) \
    asm_2x24_kernel(i0, f0, f1, f2) \
    "vbroadcastss "#i0"(%[A2]), %%ymm15                   \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm6         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm7         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm8         \n\t"

#define asm_4x24_kernel(i0, f0, f1, f2) \
    asm_3x24_kernel(i0, f0, f1, f2) \
    "vbroadcastss "#i0"(%[A3]), %%ymm15                   \n\t" \
    "vfmadd231ps %%ymm15, %%ymm12, %%ymm9         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm13, %%ymm10         \n\t" \
    "vfmadd231ps %%ymm15, %%ymm14, %%ymm11         \n\t"

#define store_1x24_0(N) \
    "vaddps (%[C]), %%ymm0, %%ymm0                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm1, %%ymm1                   \n\t" \
    "vaddps 0x40(%[C]), %%ymm2, %%ymm2                   \n\t" \
    "vmovups %%ymm0,  (%[C])                             \n\t" \
    "vmovups %%ymm1,  0x20(%[C])                         \n\t" \
    "vmovups %%ymm2,  0x40(%[C])                         \n\t"

#define store_2x24_0(N) \
    store_1x24_0(N) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), %%ymm3, %%ymm3                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm4, %%ymm4                   \n\t" \
    "vaddps 0x40(%[C]), %%ymm5, %%ymm5                   \n\t" \
    "vmovups %%ymm3,  (%[C])                             \n\t" \
    "vmovups %%ymm4,  0x20(%[C])                         \n\t" \
    "vmovups %%ymm5,  0x40(%[C])                         \n\t"

#define store_3x24_0(N) \
    store_2x24_0(N) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), %%ymm6, %%ymm6                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm7, %%ymm7                   \n\t" \
    "vaddps 0x40(%[C]), %%ymm8, %%ymm8                   \n\t" \
    "vmovups %%ymm6,  (%[C])                             \n\t" \
    "vmovups %%ymm7,  0x20(%[C])                         \n\t" \
    "vmovups %%ymm8,  0x40(%[C])                         \n\t"

#define store_4x24_0(N) \
    store_3x24_0(N) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), %%ymm9, %%ymm9                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm10, %%ymm10                 \n\t" \
    "vaddps 0x40(%[C]), %%ymm11, %%ymm11                 \n\t" \
    "vmovups %%ymm9,  (%[C])                             \n\t" \
    "vmovups %%ymm10, 0x20(%[C])                         \n\t" \
    "vmovups %%ymm11, 0x40(%[C])                         \n\t"

#define store_1x24_1(N) \
    "vmovups (%[mask]), %%ymm15                             \n\t" \
    "vmaskmovps 0x40(%[C]), %%ymm15, %%ymm14                       \n\t" \
    "vaddps (%[C]), %%ymm0, %%ymm0                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm1, %%ymm1                   \n\t" \
    "vaddps %%ymm14, %%ymm2, %%ymm2                   \n\t" \
    "vmovups %%ymm0,  (%[C])                             \n\t" \
    "vmovups %%ymm1,  0x20(%[C])                         \n\t" \
    "vmaskmovps %%ymm2, %%ymm15,  0x40(%[C])                         \n\t"

#define store_2x24_1(N) \
    store_1x24_1(N) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps 0x40(%[C]), %%ymm15, %%ymm14                       \n\t" \
    "vaddps (%[C]), %%ymm3, %%ymm3                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm4, %%ymm4                   \n\t" \
    "vaddps %%ymm14, %%ymm5, %%ymm5                   \n\t" \
    "vmovups %%ymm3,  (%[C])                             \n\t" \
    "vmovups %%ymm4,  0x20(%[C])                         \n\t" \
    "vmaskmovps %%ymm5, %%ymm15,  0x40(%[C])                         \n\t"

#define store_3x24_1(N) \
    store_2x24_1(N) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps 0x40(%[C]), %%ymm15, %%ymm14                       \n\t" \
    "vaddps (%[C]), %%ymm6, %%ymm6                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm7, %%ymm7                   \n\t" \
    "vaddps %%ymm14, %%ymm8, %%ymm8                   \n\t" \
    "vmovups %%ymm6,  (%[C])                             \n\t" \
    "vmovups %%ymm7,  0x20(%[C])                         \n\t" \
    "vmaskmovps %%ymm8, %%ymm15,  0x40(%[C])                         \n\t"

#define store_4x24_1(N) \
    store_3x24_1(N) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps 0x40(%[C]), %%ymm15, %%ymm14                       \n\t" \
    "vaddps (%[C]), %%ymm9, %%ymm9                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm10, %%ymm10                 \n\t" \
    "vaddps %%ymm14, %%ymm11, %%ymm11                 \n\t" \
    "vmovups %%ymm9,  (%[C])                             \n\t" \
    "vmovups %%ymm10, 0x20(%[C])                         \n\t" \
    "vmaskmovps %%ymm11, %%ymm15, 0x40(%[C])                         \n\t"


#define asm_1x16_kernel(i0, f0, f1) \
    "vbroadcastss "#i0"(%[A0]), %%ymm10                   \n\t" \
    "vmovaps "#f0"(%[B]), %%ymm8                        \n\t" \
    "vmovaps "#f1"(%[B]), %%ymm9                    \n\t" \
    "vfmadd231ps %%ymm10, %%ymm8, %%ymm0         \n\t" \
    "vfmadd231ps %%ymm10, %%ymm9, %%ymm1         \n\t" \

#define asm_2x16_kernel(i0, f0, f1) \
    asm_1x16_kernel(i0, f0, f1) \
    "vbroadcastss "#i0"(%[A1]), %%ymm10                   \n\t" \
    "vfmadd231ps %%ymm10, %%ymm8, %%ymm2         \n\t" \
    "vfmadd231ps %%ymm10, %%ymm9, %%ymm3         \n\t" \

#define asm_3x16_kernel(i0, f0, f1) \
    asm_2x16_kernel(i0, f0, f1) \
    "vbroadcastss "#i0"(%[A2]), %%ymm10                   \n\t" \
    "vfmadd231ps %%ymm10, %%ymm8, %%ymm4         \n\t" \
    "vfmadd231ps %%ymm10, %%ymm9, %%ymm5         \n\t" \

#define asm_4x16_kernel(i0, f0, f1) \
    asm_3x16_kernel(i0, f0, f1) \
    "vbroadcastss "#i0"(%[A3]), %%ymm10                   \n\t" \
    "vfmadd231ps %%ymm10, %%ymm8, %%ymm6         \n\t" \
    "vfmadd231ps %%ymm10, %%ymm9, %%ymm7         \n\t" \

#define store_1x16_0(N) \
    "vaddps (%[C]), %%ymm0, %%ymm0                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm1, %%ymm1                   \n\t" \
    "vmovups %%ymm0,  (%[C])                             \n\t" \
    "vmovups %%ymm1,  0x20(%[C])                         \n\t" \

#define store_2x16_0(N) \
    store_1x16_0(N) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), %%ymm2, %%ymm2                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm3, %%ymm3                   \n\t" \
    "vmovups %%ymm2,  (%[C])                             \n\t" \
    "vmovups %%ymm3,  0x20(%[C])                         \n\t" \

#define store_3x16_0(N) \
    store_2x16_0(N) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), %%ymm4, %%ymm4                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm5, %%ymm5                   \n\t" \
    "vmovups %%ymm4,  (%[C])                             \n\t" \
    "vmovups %%ymm5,  0x20(%[C])                         \n\t" \

#define store_4x16_0(N) \
    store_3x16_0(N) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), %%ymm6, %%ymm6                       \n\t" \
    "vaddps 0x20(%[C]), %%ymm7, %%ymm7                   \n\t" \
    "vmovups %%ymm6,  (%[C])                             \n\t" \
    "vmovups %%ymm7,  0x20(%[C])                         \n\t" \

#define store_1x16_1(N) \
    "vmovups (%[mask]), %%ymm10                             \n\t" \
    "vmaskmovps 0x20(%[C]), %%ymm10, %%ymm9                       \n\t" \
    "vaddps (%[C]), %%ymm0, %%ymm0                       \n\t" \
    "vaddps %%ymm9, %%ymm1, %%ymm1                   \n\t" \
    "vmovups %%ymm0,  (%[C])                             \n\t" \
    "vmaskmovps %%ymm1, %%ymm10,  0x20(%[C])                         \n\t" \

#define store_2x16_1(N) \
    store_1x16_1(N) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps 0x20(%[C]), %%ymm10, %%ymm9                       \n\t" \
    "vaddps (%[C]), %%ymm2, %%ymm2                       \n\t" \
    "vaddps %%ymm9, %%ymm3, %%ymm3                   \n\t" \
    "vmovups %%ymm2,  (%[C])                             \n\t" \
    "vmaskmovps %%ymm3, %%ymm10,  0x20(%[C])                         \n\t" \

#define store_3x16_1(N) \
    store_2x16_1(N) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps 0x20(%[C]), %%ymm10, %%ymm9                       \n\t" \
    "vaddps (%[C]), %%ymm4, %%ymm4                       \n\t" \
    "vaddps %%ymm9, %%ymm5, %%ymm5                   \n\t" \
    "vmovups %%ymm4,  (%[C])                             \n\t" \
    "vmaskmovps %%ymm5, %%ymm10,  0x20(%[C])                         \n\t" \

#define store_4x16_1(N) \
    store_3x16_1(N) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps 0x20(%[C]), %%ymm10, %%ymm9                       \n\t" \
    "vaddps (%[C]), %%ymm6, %%ymm6                       \n\t" \
    "vaddps %%ymm9, %%ymm7, %%ymm7                   \n\t" \
    "vmovups %%ymm6,  (%[C])                             \n\t" \
    "vmaskmovps %%ymm7, %%ymm10,  0x20(%[C])                         \n\t" \


#define asm_1x8_kernel(i0, f0, rtype) \
    "vmovaps "#f0"(%[B]), "#rtype"4                              \n\t" \
    "vbroadcastss "#i0"(%[A0]), "#rtype"5                      \n\t" \
    "vfmadd231ps "#rtype"5, "#rtype"4, "#rtype"0                \n\t"

#define asm_2x8_kernel(i0, f0, rtype) \
    asm_1x8_kernel(i0, f0, rtype) \
    "vbroadcastss "#i0"(%[A1]), "#rtype"5                   \n\t" \
    "vfmadd231ps "#rtype"5, "#rtype"4, "#rtype"1                \n\t"

#define asm_3x8_kernel(i0, f0, rtype) \
    asm_2x8_kernel(i0, f0, rtype) \
    "vbroadcastss "#i0"(%[A2]), "#rtype"5                   \n\t" \
    "vfmadd231ps "#rtype"5, "#rtype"4, "#rtype"2                \n\t"

#define asm_4x8_kernel(i0, f0, rtype) \
    asm_3x8_kernel(i0, f0, rtype) \
    "vbroadcastss "#i0"(%[A3]), "#rtype"5                      \n\t" \
    "vfmadd231ps "#rtype"5, "#rtype"4, "#rtype"3                \n\t"

#define store_1x8_0(N, rtype) \
    "vaddps (%[C]), "#rtype"0, "#rtype"0                       \n\t" \
    "vmovups "#rtype"0,  (%[C])                             \n\t"

#define store_2x8_0(N, rtype) \
    store_1x8_0(N, rtype) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), "#rtype"1, "#rtype"1                       \n\t" \
    "vmovups "#rtype"1,  (%[C])                             \n\t"

#define store_3x8_0(N, rtype) \
    store_2x8_0(N, rtype) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), "#rtype"2, "#rtype"2                       \n\t" \
    "vmovups "#rtype"2,  (%[C])                             \n\t"

#define store_4x8_0(N, rtype) \
    store_3x8_0(N, rtype) \
    "add "#N", %[C]                                     \n\t" \
    "vaddps (%[C]), "#rtype"3, "#rtype"3                       \n\t" \
    "vmovups "#rtype"3,  (%[C])                             \n\t"

#define store_1x8_1(N, rtype) \
    "vmovups (%[mask]), "#rtype"5                             \n\t" \
    "vmaskmovps (%[C]), "#rtype"5, "#rtype"4                       \n\t" \
    "vaddps "#rtype"4, "#rtype"0, "#rtype"0                       \n\t" \
    "vmaskmovps "#rtype"0, "#rtype"5, (%[C])                             \n\t"

#define store_2x8_1(N, rtype) \
    store_1x8_1(N, rtype) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps (%[C]), "#rtype"5, "#rtype"4                       \n\t" \
    "vaddps "#rtype"4, "#rtype"1, "#rtype"1                       \n\t" \
    "vmaskmovps "#rtype"1, "#rtype"5, (%[C])                             \n\t"

#define store_3x8_1(N, rtype) \
    store_2x8_1(N, rtype) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps (%[C]), "#rtype"5, "#rtype"4                       \n\t" \
    "vaddps "#rtype"4, "#rtype"2, "#rtype"2                       \n\t" \
    "vmaskmovps "#rtype"2, "#rtype"5, (%[C])                             \n\t"

#define store_4x8_1(N, rtype) \
    store_3x8_1(N, rtype) \
    "add "#N", %[C]                                     \n\t" \
    "vmaskmovps (%[C]), "#rtype"5, "#rtype"4                       \n\t" \
    "vaddps "#rtype"4, "#rtype"3, "#rtype"3                       \n\t" \
    "vmaskmovps "#rtype"3, "#rtype"5, (%[C])                             \n\t"

#define kernel_24_4_loop(m) \
    "prefetcht0 0x140(%[B])                              \n\t" \
    "prefetcht0 0x180(%[B])                              \n\t" \
    asm_##m##x24_kernel(0x0, 0x0, 0x20, 0x40) \
    "prefetcht0 0x1C0(%[B])                              \n\t" \
    asm_##m##x24_kernel(0x4, 0x60, 0x80, 0xA0) \
    "prefetcht0 0x200(%[B])                              \n\t" \
    "prefetcht0 0x240(%[B])                              \n\t" \
    asm_##m##x24_kernel(0x8, 0xC0, 0xE0, 0x100) \
    "prefetcht0 0x280(%[B])                              \n\t" \
    asm_##m##x24_kernel(0xC, 0x120, 0x140, 0x160) \
    "add $0x180, %[B]                             \n\t"

#define kernel_16_4_loop(m) \
    "prefetcht0 0x140(%1)                              \n\t" \
    asm_##m##x16_kernel(0x0, 0x0, 0x20) \
    "prefetcht0 0x180(%1)                              \n\t" \
    asm_##m##x16_kernel(0x4, 0x40, 0x60) \
    "prefetcht0 0x1C0(%1)                              \n\t" \
    asm_##m##x16_kernel(0x8, 0x80, 0xA0) \
    "prefetcht0 0x200(%1)                              \n\t" \
    asm_##m##x16_kernel(0xC, 0xC0, 0xE0) \
    "add $0x100, %[B]                             \n\t"

#define kernel_8_4_loop(m) \
    asm_##m##x8_kernel(0x0, 0x0, %%ymm) \
    asm_##m##x8_kernel(0x4, 0x20, %%ymm) \
    asm_##m##x8_kernel(0x8, 0x40, %%ymm) \
    asm_##m##x8_kernel(0xC, 0x60, %%ymm) \
    "add $0x80, %[B]                             \n\t"

#define kernel_4_4_loop(m) \
    asm_##m##x8_kernel(0x0, 0x0, %%xmm) \
    asm_##m##x8_kernel(0x4, 0x10, %%xmm) \
    asm_##m##x8_kernel(0x8, 0x20, %%xmm) \
    asm_##m##x8_kernel(0xC, 0x30, %%xmm) \
    "add $0x40, %[B]                             \n\t"

#define m_24_kernel(m, x, edge) \
    __asm__ __volatile__(clear##x##Regs(%%ymm)                               \
                         "mov %[bk], %%ecx                             \n\t" \
                         "shr $2, %%ecx                                \n\t" \
                         "je 1f                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "0:                                           \n\t" \
                         kernel_24_4_loop(m)                                 \
                         "add $0x10, %[A0]                             \n\t" \
                         "add $0x10, %[A1]                             \n\t" \
                         "add $0x10, %[A2]                             \n\t" \
                         "add $0x10, %[A3]                             \n\t" \
                         "dec %%ecx                                    \n\t" \
                         "jg 0b                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "1:                                           \n\t" \
                         "mov %[bk], %%ecx                             \n\t" \
                         "and $3, %%ecx                                \n\t" \
                         "je 3f                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "2:                                           \n\t" \
                         asm_##m##x24_kernel(0x0, 0x0, 0x20, 0x40)           \
                         "add $0x60, %[B]                              \n\t" \
                         "add $0x4, %[A0]                              \n\t" \
                         "add $0x4, %[A1]                              \n\t" \
                         "add $0x4, %[A2]                              \n\t" \
                         "add $0x4, %[A3]                              \n\t" \
                         "dec %%ecx                                    \n\t" \
                         "jg 2b                                        \n\t" \
                         "3:                                           \n\t" \
                         "shl $2, %%rax                                \n\t" \
                         store_##m##x24_##edge(%%rax)                        \
                         : [B] "+r" (matrixB),                               \
                           [A0] "+r" (matrixA),                              \
                           [A1] "+r" (A1),                                   \
                           [A2] "+r" (A2),                                   \
                           [A3] "+r" (A3),                                   \
                           [C] "+r" (matrixC)                                \
                         : "a"((I64)N),                                      \
                           [bk] "r" (bk),                                    \
                           [mask] "r" (mask)                                 \
                         : "%ecx",                                           \
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",      \
                           "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",      \
                           "%ymm10", "%ymm11", "%ymm12", "%ymm13",           \
                           "%ymm14", "%ymm15", "memory");

#define m_16_kernel(m, x, edge) \
    __asm__ __volatile__(clear##x##Regs(%%ymm)                               \
                         "mov %[bk], %%ecx                             \n\t" \
                         "shr $2, %%ecx                                \n\t" \
                         "je 1f                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "0:                                           \n\t" \
                         kernel_16_4_loop(m)                                 \
                         "add $0x10, %[A0]                             \n\t" \
                         "add $0x10, %[A1]                             \n\t" \
                         "add $0x10, %[A2]                             \n\t" \
                         "add $0x10, %[A3]                             \n\t" \
                         "dec %%ecx                                    \n\t" \
                         "jg 0b                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "1:                                           \n\t" \
                         "mov %[bk], %%ecx                             \n\t" \
                         "and $3, %%ecx                                \n\t" \
                         "je 3f                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "2:                                           \n\t" \
                         asm_##m##x16_kernel(0x0, 0x0, 0x20)                 \
                         "add $0x40, %[B]                              \n\t" \
                         "add $0x4, %[A0]                              \n\t" \
                         "add $0x4, %[A1]                              \n\t" \
                         "add $0x4, %[A2]                              \n\t" \
                         "add $0x4, %[A3]                              \n\t" \
                         "dec %%ecx                                    \n\t" \
                         "jg 2b                                        \n\t" \
                         "3:                                           \n\t" \
                         "shl $2, %%rax                                \n\t" \
                         store_##m##x16_##edge(%%rax)                               \
                         : [B] "+r" (matrixB),                               \
                           [A0] "+r" (matrixA),                              \
                           [A1] "+r" (A1),                                   \
                           [A2] "+r" (A2),                                   \
                           [A3] "+r" (A3),                                   \
                           [C] "+r" (matrixC)                                \
                         : "a"((I64)N),                                      \
                           [bk] "r" (bk),                                     \
                           [mask] "r" (mask)                                     \
                         : "%ecx",                                           \
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",      \
                           "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",      \
                           "%ymm10", "memory");

#define asm_4_kernel(m) \
    asm_##m##x8_kernel(0x0, 0x0, %%xmm)                 \
    "add $0x10, %[B]                              \n\t"

#define asm_8_kernel(m) \
    asm_##m##x8_kernel(0x0, 0x0, %%ymm)                 \
    "add $0x20, %[B]                              \n\t"

#define m_8_kernel_wrap(m, n, x, rtype, edge) \
    __asm__ __volatile__(clear##x##Regs(rtype)                               \
                         "mov %[bk], %%ecx                             \n\t" \
                         "shr $2, %%ecx                                \n\t" \
                         "je 1f                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "0:                                           \n\t" \
                         kernel_##n##_4_loop(m)                                 \
                         "add $0x10, %[A0]                             \n\t" \
                         "add $0x10, %[A1]                             \n\t" \
                         "add $0x10, %[A2]                             \n\t" \
                         "add $0x10, %[A3]                             \n\t" \
                         "dec %%ecx                                    \n\t" \
                         "jg 0b                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "1:                                           \n\t" \
                         "mov %[bk], %%ecx                             \n\t" \
                         "and $3, %%ecx                                \n\t" \
                         "je 3f                                        \n\t" \
                         ".align 16                                    \n\t" \
                         "2:                                           \n\t" \
                         asm_##n##_kernel(m)                                     \
                         "add $0x4, %[A0]                              \n\t" \
                         "add $0x4, %[A1]                              \n\t" \
                         "add $0x4, %[A2]                              \n\t" \
                         "add $0x4, %[A3]                              \n\t" \
                         "dec %%ecx                                    \n\t" \
                         "jg 2b                                        \n\t" \
                         "3:                                           \n\t" \
                         "shl $2, %%rax                                \n\t" \
                         store_##m##x8_##edge(%%rax, rtype)                         \
                         : [B] "+r" (matrixB),                               \
                           [A0] "+r" (matrixA),                              \
                           [A1] "+r" (A1),                                   \
                           [A2] "+r" (A2),                                   \
                           [A3] "+r" (A3),                                   \
                           [C] "+r" (matrixC)                                \
                         : "a"((I64)N),                                      \
                           [bk] "r" (bk),                                     \
                           [mask] "r" (mask)                                     \
                         : "%ecx",                                           \
                           "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",      \
                           "%ymm5", "memory");

#define m_8_kernel(m, x, edge) \
    m_8_kernel_wrap(m, 8, x, %%ymm, edge)

#define m_4_kernel(m, x, edge) \
    m_8_kernel_wrap(m, 4, x, %%xmm, edge)

#define mmm_mxn_asm(m, n, regNum) \
    void mmm_avx2_##m##x##n##_asm( \
        U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, \
        F32 *matrixC, U32 N, I32 *mask, F32 *A1, F32 *A2, F32 *A3) \
{ \
    if (mask == nullptr) { \
        m_##n##_kernel(m, regNum, 0) \
    } else { \
        m_##n##_kernel(m, regNum, 1) \
    } \
}

mmm_mxn_asm(4, 24, 12)
mmm_mxn_asm(3, 24, 9)
mmm_mxn_asm(2, 24, 6)
mmm_mxn_asm(1, 24, 3)
mmm_mxn_asm(4, 16, 8)
mmm_mxn_asm(3, 16, 6)
mmm_mxn_asm(2, 16, 4)
mmm_mxn_asm(1, 16, 2)
mmm_mxn_asm(4, 8, 4)
mmm_mxn_asm(3, 8, 3)
mmm_mxn_asm(2, 8, 2)
mmm_mxn_asm(1, 8, 1)
mmm_mxn_asm(4, 4, 4)
mmm_mxn_asm(3, 4, 3)
mmm_mxn_asm(2, 4, 2)
mmm_mxn_asm(1, 4, 1)

    // clang-format on

    void mmm_avx2_n_mtail(U32 um,
        U32 un,
        U32 bk,
        F32 *matrixA,
        F32 *matrixB,
        F32 *matrixC,
        U32 N,
        I32 *mask,
        F32 *A1,
        F32 *A2,
        F32 *A3)
{
    F32 *ar[4] = {matrixA, A1, A2, A3};
    for (U32 i = 0; i < um; ++i) {
        for (U32 j = 0; j < un; ++j) {
            for (U32 k = 0; k < bk; ++k) {
                matrixC[i * N + j] += ar[i][k] * matrixB[k * un + j];
            }
        }
    }
}

EE mmm_avx2_fp32(
    int N, int M, int K, DataFormat matrix1Df, F32 *matrix1, F32 *matrix2, F32 *tmp, F32 *result)
{
    // buffer addr algined to 32
    F32 *packA = (F32 *)align_addr(tmp, 32);
    F32 *packB = (F32 *)align_addr(matrix2, 32);
    kernel_func kernel[4][5] = {
        {mmm_avx2_n_mtail, mmm_avx2_1x4_asm, mmm_avx2_1x8_asm, mmm_avx2_1x16_asm, mmm_avx2_1x24_asm},
        {mmm_avx2_n_mtail, mmm_avx2_2x4_asm, mmm_avx2_2x8_asm, mmm_avx2_2x16_asm, mmm_avx2_2x24_asm},
        {mmm_avx2_n_mtail, mmm_avx2_3x4_asm, mmm_avx2_3x8_asm, mmm_avx2_3x16_asm, mmm_avx2_3x24_asm},
        {mmm_avx2_n_mtail, mmm_avx2_4x4_asm, mmm_avx2_4x8_asm, mmm_avx2_4x16_asm, mmm_avx2_4x24_asm}};
    F32 unrollNSize[4] = {4, 8, 16, 24};
    F32 unrollMSize[4] = {1, 2, 3, 4};
    I32 resN = N % UNROLL_N;
    I32 blockNNum = N / UNROLL_N + (resN > 0);
    I32 edgeBlockNSizeIdx = (resN > 4) ? ((resN + 7) / 8) : 0;
    I32 edgeBlockNSize = (resN > 0) ? unrollNSize[edgeBlockNSizeIdx] : 0;
    I32 mask[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if (resN != edgeBlockNSize) {
        UNI_MEMSET(mask + resN % 8, 0, (edgeBlockNSize - resN) * 4);
    }
    I32 *maskPtr = (N % 4 != 0) ? mask : nullptr;
    I32 alginedN = (blockNNum - 1) * UNROLL_N + edgeBlockNSize;
    if (edgeBlockNSize == 0) {
        alginedN += UNROLL_N;
    }
    I32 blockNum = (M + 3) / 4 * blockNNum;
    I32 mainBlockNum = (BOLCK_M_DIM + 3) / 4 * blockNNum;

#ifdef _USE_OPENMP
    int in_parallel = omp_in_parallel();
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (in_parallel == 0)
#endif
    {
        I32 blockSizeK = 0;
        for (int k = 0; k < K; k += blockSizeK) {
            blockSizeK = UNI_MIN(BOLCK_K_DIM, K - k);
            if (matrix1Df == DF_TRANSPOSE) {
                matrix1_trans(blockSizeK, M, M, matrix1 + k * M, packA);
            }

#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
            for (int mnIdx = 0; mnIdx < blockNum; ++mnIdx) {
                I32 j = mnIdx / mainBlockNum * BOLCK_M_DIM;
                I32 blockSizeM = UNI_MIN(BOLCK_M_DIM, M - j);
                I32 blockMNum = (blockSizeM + 3) / 4;

                I32 n = (mnIdx % mainBlockNum) / blockMNum * UNROLL_N;
                I32 blockSizeN = UNI_MAX(UNI_MIN(UNROLL_N, N - n), edgeBlockNSize);
                F32 *curB = packB + k * alginedN + n * blockSizeK;
                maskPtr = ((blockSizeN + n) > N) ? mask : nullptr;

                I32 m = ((mnIdx % mainBlockNum) % blockMNum) * UNROLL_M;
                I32 unrollSizeM = UNI_MIN(UNROLL_M, blockSizeM - m);

                F32 *curA, *A1, *A2, *A3;
                if (matrix1Df == DF_TRANSPOSE) {
                    curA = packA + m * blockSizeK;
                    A1 = curA + blockSizeK;
                    A2 = curA + 2 * blockSizeK;
                    A3 = curA + 3 * blockSizeK;
                } else {
                    curA = matrix1 + k + (j + m) * K;
                    A1 = curA + K;
                    A2 = curA + 2 * K;
                    A3 = curA + 3 * K;
                }

                kernel[unrollSizeM - 1][(blockSizeN >> 3) + (blockSizeN > 3)](unrollSizeM,
                    blockSizeN, blockSizeK, curA, curB, result + (m + j) * N + n, N, maskPtr, A1,
                    A2, A3);
            }
        }
    }
    return SUCCESS;
}
