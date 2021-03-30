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
#include "error.h"

#ifdef _USE_OPENMP
#include <omp.h>
#endif

#define UNROLL_K 4
#define UNROLL_N 24
#define UNROLL_M 4
#define BOLCK_M_DIM 768
#define BOLCK_K_DIM 768
#define align_addr(addr, unit) (((uintptr_t)addr + unit - 1) / unit * unit)

typedef void (*kernel_func)(
    U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 ldc);

void matrix_matrix_multiply_tmp_bytes_fp32(
    U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes)
{
    *bytes = row1 * col1;
    *bytes *= sizeof(dt);
    *bytes += 32;
}

EE matrix_matrix_multiply_transform_rhsN_fp32(TensorDesc desc, F32 *src, F32 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K, blockSizeK, unrollSizeN;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    F32 unrollSize[4] = {4, 8, 16, 24};

    // buffer addr algined to 32
    F32 *packB = (F32 *)align_addr(dst, 32);
    for (U32 bk = 0; bk < K; bk += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        for (U32 un = 0; un < N; un += unrollSizeN) {
            unrollSizeN = UNI_MIN(UNROLL_N, N - un);
            unrollSizeN = UNI_MIN(unrollSize[unrollSizeN / 8], unrollSizeN);
            matrix2_trans(unrollSizeN, blockSizeK, N, src + un, packB);
            packB += unrollSizeN * blockSizeK;
        }
        src += blockSizeK * N;
    }
    return SUCCESS;
}

EE matrix_matrix_multiply_transform_rhsT_fp32(TensorDesc desc, F32 *src, F32 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K, blockSizeK, unrollSizeN;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    F32 unrollSize[4] = {4, 8, 16, 24};

    // buffer addr aligned to 32
    F32 *packB = (F32 *)align_addr(dst, 32);
    for (U32 bk = 0; bk < K; bk += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        for (U32 un = 0; un < N; un += unrollSizeN) {
            unrollSizeN = UNI_MIN(UNROLL_N, N - un);
            unrollSizeN = UNI_MIN(unrollSize[unrollSizeN >> 3], unrollSizeN);
            matrix1_trans(unrollSizeN, blockSizeK, K, src + un * K, packB);
            packB += unrollSizeN * blockSizeK;
        }
        src += blockSizeK;
    }
    return SUCCESS;
}

void mmm_avx2_4x24_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                         "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
                         "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
                         "vxorps %%ymm5, %%ymm5, %%ymm5                     \n\t"
                         "vxorps %%ymm6, %%ymm6, %%ymm6                     \n\t"
                         "vxorps %%ymm7, %%ymm7, %%ymm7                     \n\t"
                         "vxorps %%ymm8, %%ymm8, %%ymm8                     \n\t"
                         "vxorps %%ymm9, %%ymm9, %%ymm9                     \n\t"
                         "vxorps %%ymm10, %%ymm10, %%ymm10                  \n\t"
                         "vxorps %%ymm11, %%ymm11, %%ymm11                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_4x24_end                                \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_4x24:                                      \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"
                         "prefetcht0 0x180(%1)                              \n\t"
                         "prefetcht0 0x140(%2)                              \n\t"

                         "vmovaps (%1), %%ymm12                             \n\t"
                         "vmovaps 0x20(%1), %%ymm13                         \n\t"
                         "vmovaps 0x40(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vbroadcastss 0x8(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0xC(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "prefetcht0 0x1C0(%1)                              \n\t"

                         "vmovaps 0x60(%1), %%ymm12                         \n\t"
                         "vmovaps 0x80(%1), %%ymm13                         \n\t"
                         "vmovaps 0xA0(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x10(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vbroadcastss 0x14(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vbroadcastss 0x18(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "prefetcht0 0x200(%1)                              \n\t"
                         "prefetcht0 0x240(%1)                              \n\t"

                         "vmovaps 0xC0(%1), %%ymm12                         \n\t"
                         "vmovaps 0xE0(%1), %%ymm13                         \n\t"
                         "vmovaps 0x100(%1), %%ymm14                        \n\t"
                         "vbroadcastss 0x20(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vbroadcastss 0x24(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vbroadcastss 0x28(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x2C(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "prefetcht0 0x280(%1)                              \n\t"

                         "vmovaps 0x120(%1), %%ymm12                        \n\t"
                         "vmovaps 0x140(%1), %%ymm13                        \n\t"
                         "vmovaps 0x160(%1), %%ymm14                        \n\t"
                         "vbroadcastss 0x30(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vbroadcastss 0x34(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vbroadcastss 0x38(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0x3C(%2), %%ymm15                    \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

                         "add $0x180, %1                                    \n\t"
                         "add $0x40, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_4x24                                    \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_4x24_end:                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_4x24_remain_end                         \n\t"
                         ".k_loop_4x24_remain:                               \n\t"
                         "vmovaps (%1), %%ymm12                             \n\t"
                         "vmovaps 0x20(%1), %%ymm13                         \n\t"
                         "vmovaps 0x40(%1), %%ymm14                         \n\t"
                         "vbroadcastss 0x0(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
                         "vbroadcastss 0x4(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
                         "vbroadcastss 0x8(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
                         "vbroadcastss 0xC(%2), %%ymm15                     \n\t"
                         "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
                         "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
                         "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"
                         "add $0x60, %1                                     \n\t"
                         "add $0x10, %2                                     \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_4x24_remain                             \n\t"

                         ".k_loop_4x24_remain_end:                           \n\t"
                         "mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "prefetcht0 0x40(%3)                               \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "prefetcht0 0x40(%3, %%rax)                        \n\t"
                         "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
                         "vaddps 0x20(%3), %%ymm1, %%ymm1                   \n\t"
                         "vaddps 0x40(%3), %%ymm2, %%ymm2                   \n\t"
                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1,  0x20(%3)                         \n\t"
                         "vmovups %%ymm2,  0x40(%3)                         \n\t"
                         "add %%rax, %3                                     \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "prefetcht0 0x40(%3, %%rax)                        \n\t"
                         "vaddps (%3), %%ymm3, %%ymm3                       \n\t"
                         "vaddps 0x20(%3), %%ymm4, %%ymm4                   \n\t"
                         "vaddps 0x40(%3), %%ymm5, %%ymm5                   \n\t"
                         "vmovups %%ymm3,  (%3)                             \n\t"
                         "vmovups %%ymm4,  0x20(%3)                         \n\t"
                         "vmovups %%ymm5,  0x40(%3)                         \n\t"
                         "add %%rax, %3                                     \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "prefetcht0 0x40(%3, %%rax)                        \n\t"
                         "vaddps (%3), %%ymm6, %%ymm6                       \n\t"
                         "vaddps 0x20(%3), %%ymm7, %%ymm7                   \n\t"
                         "vaddps 0x40(%3), %%ymm8, %%ymm8                   \n\t"
                         "vmovups %%ymm6,  (%3)                             \n\t"
                         "vmovups %%ymm7,  0x20(%3)                         \n\t"
                         "vmovups %%ymm8,  0x40(%3)                         \n\t"
                         "add %%rax, %3                                     \n\t"
                         "prefetcht0 0x40(%3)                               \n\t"
                         "vaddps (%3), %%ymm9, %%ymm9                       \n\t"
                         "vaddps 0x20(%3), %%ymm10, %%ymm10                 \n\t"
                         "vaddps 0x40(%3), %%ymm11, %%ymm11                 \n\t"
                         "vmovups %%ymm9,  (%3)                             \n\t"
                         "vmovups %%ymm10, 0x20(%3)                         \n\t"
                         "vmovups %%ymm11, 0x40(%3)                         \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                         "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12",
                         "%ymm13", "%ymm14", "%ymm15", "memory");
}

void mmm_avx2_4x16_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                         "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
                         "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
                         "vxorps %%ymm5, %%ymm5, %%ymm5                     \n\t"
                         "vxorps %%ymm6, %%ymm6, %%ymm6                     \n\t"
                         "vxorps %%ymm7, %%ymm7, %%ymm7                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_4x16_end                                \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_4x16:                                      \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"
                         "prefetcht0 0x140(%2)                              \n\t"

                         "vmovaps (%1), %%ymm8                              \n\t"
                         "vmovaps 0x20(%1), %%ymm9                          \n\t"
                         "vbroadcastss 0x0(%2), %%ymm10                     \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm1               \n\t"
                         "vbroadcastss 0x4(%2), %%ymm10                     \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm3               \n\t"
                         "vbroadcastss 0x8(%2), %%ymm10                     \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm4               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm5               \n\t"
                         "vbroadcastss 0xC(%2), %%ymm10                     \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm6               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm7               \n\t"

                         "prefetcht0 0x180(%1)                              \n\t"

                         "vmovaps 0x40(%1), %%ymm8                          \n\t"
                         "vmovaps 0x60(%1), %%ymm9                          \n\t"
                         "vbroadcastss 0x10(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm1               \n\t"
                         "vbroadcastss 0x14(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm3               \n\t"
                         "vbroadcastss 0x18(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm4               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm5               \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm6               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm7               \n\t"

                         "prefetcht0 0x1C0(%1)                              \n\t"

                         "vmovaps 0x80(%1), %%ymm8                          \n\t"
                         "vmovaps 0xA0(%1), %%ymm9                          \n\t"
                         "vbroadcastss 0x20(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm1               \n\t"
                         "vbroadcastss 0x24(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm3               \n\t"
                         "vbroadcastss 0x28(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm4               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm5               \n\t"
                         "vbroadcastss 0x2C(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm6               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm7               \n\t"

                         "prefetcht0 0x200(%1)                              \n\t"

                         "vmovaps 0xC0(%1), %%ymm8                          \n\t"
                         "vmovaps 0xE0(%1), %%ymm9                          \n\t"
                         "vbroadcastss 0x30(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm1               \n\t"
                         "vbroadcastss 0x34(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm3               \n\t"
                         "vbroadcastss 0x38(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm4               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm5               \n\t"
                         "vbroadcastss 0x3C(%2), %%ymm10                    \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm6               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm7               \n\t"

                         "add $0x100, %1                                    \n\t"
                         "add $0x40, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_4x16                                    \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_4x16_end:                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_4x16_remain_end                         \n\t"
                         ".k_loop_4x16_remain:                               \n\t"
                         "vmovaps (%1), %%ymm8                              \n\t"
                         "vmovaps 0x20(%1), %%ymm9                          \n\t"
                         "vbroadcastss 0x0(%2), %%ymm10                     \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm0               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm1               \n\t"
                         "vbroadcastss 0x4(%2), %%ymm10                     \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm2               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm3               \n\t"
                         "vbroadcastss 0x8(%2), %%ymm10                     \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm4               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm5               \n\t"
                         "vbroadcastss 0xC(%2), %%ymm10                     \n\t"
                         "vfmadd231ps %%ymm10, %%ymm8, %%ymm6               \n\t"
                         "vfmadd231ps %%ymm10, %%ymm9, %%ymm7               \n\t"
                         "add $0x40, %1                                     \n\t"
                         "add $0x10, %2                                     \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_4x16_remain                             \n\t"

                         ".k_loop_4x16_remain_end:                           \n\t"
                         "mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
                         "vaddps 0x20(%3), %%ymm1, %%ymm1                   \n\t"
                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1,  0x20(%3)                         \n\t"
                         "add %%rax, %3                                     \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "vaddps (%3), %%ymm2, %%ymm2                       \n\t"
                         "vaddps 0x20(%3), %%ymm3, %%ymm3                   \n\t"
                         "vmovups %%ymm2,  (%3)                             \n\t"
                         "vmovups %%ymm3,  0x20(%3)                         \n\t"
                         "add %%rax, %3                                     \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "vaddps (%3), %%ymm4, %%ymm4                       \n\t"
                         "vaddps 0x20(%3), %%ymm5, %%ymm5                   \n\t"
                         "vmovups %%ymm4,  (%3)                             \n\t"
                         "vmovups %%ymm5,  0x20(%3)                         \n\t"
                         "add %%rax, %3                                     \n\t"
                         "vaddps (%3), %%ymm6, %%ymm6                       \n\t"
                         "vaddps 0x20(%3), %%ymm7, %%ymm7                   \n\t"
                         "vmovups %%ymm6,  (%3)                             \n\t"
                         "vmovups %%ymm7, 0x20(%3)                          \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                         "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "memory");
}

void mmm_avx2_4x8_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"

        "mov %0, %%ecx                                     \n\t"
        "shr $2, %%ecx                                     \n\t"
        "je .k_loop_4x8_end                                 \n\t"
        ".align 16                                         \n\t"
        ".k_loop_4x8:                                       \n\t"

        "prefetcht0 0x140(%1)                              \n\t"
        "prefetcht0 0x140(%2)                              \n\t"

        "vmovaps (%1), %%ymm4                              \n\t"
        "vbroadcastss 0x0(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm0                \n\t"
        "vbroadcastss 0x4(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm1                \n\t"
        "vbroadcastss 0x8(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm2                \n\t"
        "vbroadcastss 0xC(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm3                \n\t"

        "vmovaps 0x20(%1), %%ymm4                          \n\t"
        "vbroadcastss 0x10(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm0                \n\t"
        "vbroadcastss 0x14(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm1                \n\t"
        "vbroadcastss 0x18(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm2                \n\t"
        "vbroadcastss 0x1C(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm3                \n\t"

        "prefetcht0 0x180(%1)                              \n\t"

        "vmovaps 0x40(%1), %%ymm4                          \n\t"
        "vbroadcastss 0x20(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm0                \n\t"
        "vbroadcastss 0x24(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm1                \n\t"
        "vbroadcastss 0x28(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm2                \n\t"
        "vbroadcastss 0x2C(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm3                \n\t"

        "vmovaps 0x60(%1), %%ymm4                          \n\t"
        "vbroadcastss 0x30(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm0                \n\t"
        "vbroadcastss 0x34(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm1                \n\t"
        "vbroadcastss 0x38(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm2                \n\t"
        "vbroadcastss 0x3C(%2), %%ymm5                     \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm3                \n\t"

        "add $0x80, %1                                     \n\t"
        "add $0x40, %2                                     \n\t"

        "sub $1, %%ecx                                     \n\t"
        "jg .k_loop_4x8                                     \n\t"
        ".align 16                                         \n\t"
        ".k_loop_4x8_end:                                   \n\t"

        "mov %0, %%ecx                                     \n\t"
        "and $3, %%ecx                                     \n\t"
        "je .k_loop_4x8_remain_end                          \n\t"
        ".k_loop_4x8_remain:                                \n\t"
        "vmovaps (%1), %%ymm4                              \n\t"
        "vbroadcastss 0x0(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm0                \n\t"
        "vbroadcastss 0x4(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm1                \n\t"
        "vbroadcastss 0x8(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm2                \n\t"
        "vbroadcastss 0xC(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm4, %%ymm3                \n\t"
        "add $0x20, %1                                     \n\t"
        "add $0x10, %2                                     \n\t"
        "sub $1, %%ecx                                     \n\t"
        "jg .k_loop_4x8_remain                              \n\t"

        ".k_loop_4x8_remain_end:                            \n\t"
        "mov %4, %%eax                                     \n\t"
        "shl $2, %%eax                                     \n\t"
        "mov %%eax, %%eax                                  \n\t"
        "prefetcht0 (%3, %%rax)                            \n\t"
        "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
        "vmovups %%ymm0,  (%3)                             \n\t"
        "add %%rax, %3                                     \n\t"
        "prefetcht0 (%3, %%rax)                            \n\t"
        "vaddps (%3), %%ymm1, %%ymm1                       \n\t"
        "vmovups %%ymm1,  (%3)                             \n\t"
        "add %%rax, %3                                     \n\t"
        "prefetcht0 (%3, %%rax)                            \n\t"
        "vaddps (%3), %%ymm2, %%ymm2                       \n\t"
        "vmovups %%ymm2,  (%3)                             \n\t"
        "add %%rax, %3                                     \n\t"
        "vaddps (%3), %%ymm3, %%ymm3                       \n\t"
        "vmovups %%ymm3,  (%3)                             \n\t"
        :
        : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
        : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "memory");
}

void mmm_avx2_4x4_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__(
        "vxorps %%xmm0, %%xmm0, %%xmm0                     \n\t"
        "vxorps %%xmm1, %%xmm1, %%xmm1                     \n\t"
        "vxorps %%xmm2, %%xmm2, %%xmm2                     \n\t"
        "vxorps %%xmm3, %%xmm3, %%xmm3                     \n\t"

        "mov %0, %%ecx                                     \n\t"
        "shr $2, %%ecx                                     \n\t"
        "je .k_loop_4x4_end                                 \n\t"
        ".align 16                                         \n\t"
        ".k_loop_4x4:                                       \n\t"

        "prefetcht0 0x140(%1)                              \n\t"
        "prefetcht0 0x140(%2)                              \n\t"

        "vmovaps (%1), %%xmm4                              \n\t"
        "vbroadcastss 0x0(%2), %%xmm5                      \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm0                \n\t"
        "vbroadcastss 0x4(%2), %%xmm5                      \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm1                \n\t"
        "vbroadcastss 0x8(%2), %%xmm5                      \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm2                \n\t"
        "vbroadcastss 0xC(%2), %%xmm5                      \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm3                \n\t"

        "vmovaps 0x10(%1), %%xmm4                          \n\t"
        "vbroadcastss 0x10(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm0                \n\t"
        "vbroadcastss 0x14(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm1                \n\t"
        "vbroadcastss 0x18(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm2                \n\t"
        "vbroadcastss 0x1C(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm3                \n\t"

        "vmovaps 0x20(%1), %%xmm4                          \n\t"
        "vbroadcastss 0x20(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm0                \n\t"
        "vbroadcastss 0x24(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm1                \n\t"
        "vbroadcastss 0x28(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm2                \n\t"
        "vbroadcastss 0x2C(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm3                \n\t"

        "vmovaps 0x30(%1), %%xmm4                          \n\t"
        "vbroadcastss 0x30(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm0                \n\t"
        "vbroadcastss 0x34(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm1                \n\t"
        "vbroadcastss 0x38(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm2                \n\t"
        "vbroadcastss 0x3C(%2), %%xmm5                     \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm3                \n\t"

        "add $0x40, %1                                     \n\t"
        "add $0x40, %2                                     \n\t"

        "sub $1, %%ecx                                     \n\t"
        "jg .k_loop_4x4                                     \n\t"
        ".align 16                                         \n\t"
        ".k_loop_4x4_end:                                   \n\t"

        "mov %0, %%ecx                                     \n\t"
        "and $3, %%ecx                                     \n\t"
        "je .k_loop_4x4_remain_end                          \n\t"

        ".k_loop_4x4_remain:                                \n\t"
        "vmovaps (%1), %%xmm4                              \n\t"
        "vbroadcastss 0x0(%2), %%xmm5                      \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm0                \n\t"
        "vbroadcastss 0x4(%2), %%xmm5                      \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm1                \n\t"
        "vbroadcastss 0x8(%2), %%xmm5                      \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm2                \n\t"
        "vbroadcastss 0xC(%2), %%xmm5                      \n\t"
        "vfmadd231ps %%xmm5, %%xmm4, %%xmm3                \n\t"
        "add $0x10, %1                                     \n\t"
        "add $0x10, %2                                     \n\t"
        "sub $1, %%ecx                                     \n\t"
        "jg .k_loop_4x4_remain                              \n\t"

        ".k_loop_4x4_remain_end:                            \n\t"
        "mov %4, %%eax                                     \n\t"
        "shl $2, %%eax                                     \n\t"
        "mov %%eax, %%eax                                  \n\t"
        "prefetcht0 (%3, %%rax)                            \n\t"
        "vaddps (%3), %%xmm0, %%xmm0                       \n\t"
        "vmovups %%xmm0,  (%3)                             \n\t"
        "add %%rax, %3                                     \n\t"
        "prefetcht0 (%3, %%rax)                            \n\t"
        "vaddps (%3), %%xmm1, %%xmm1                       \n\t"
        "vmovups %%xmm1,  (%3)                             \n\t"
        "add %%rax, %3                                     \n\t"
        "prefetcht0 (%3, %%rax)                            \n\t"
        "vaddps (%3), %%xmm2, %%xmm2                       \n\t"
        "vmovups %%xmm2,  (%3)                             \n\t"
        "add %%rax, %3                                     \n\t"
        "vaddps (%3), %%xmm3, %%xmm3                       \n\t"
        "vmovups %%xmm3,  (%3)                             \n\t"
        :
        : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
        : "%eax", "%rax", "%ecx", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "memory");
}

void mmm_avx2_2x24_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"

                         "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                         "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
                         "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
                         "vxorps %%ymm5, %%ymm5, %%ymm5                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_2x24_end                                \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x24:                                      \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"
                         "prefetcht0 0x180(%1)                              \n\t"

                         "vmovaps (%1), %%ymm6                              \n\t"
                         "vmovaps 0x20(%1), %%ymm7                          \n\t"
                         "vmovaps 0x40(%1), %%ymm8                          \n\t"
                         "vbroadcastss 0x0(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm2                \n\t"
                         "vbroadcastss 0x4(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm5                \n\t"

                         "prefetcht0 0x1C0(%1)                              \n\t"

                         "vmovaps 0x60(%1), %%ymm6                          \n\t"
                         "vmovaps 0x80(%1), %%ymm7                          \n\t"
                         "vmovaps 0xA0(%1), %%ymm8                          \n\t"
                         "vbroadcastss 0x8(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm2                \n\t"
                         "vbroadcastss 0xC(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm5                \n\t"

                         "prefetcht0 0x200(%1)                              \n\t"
                         "prefetcht0 0x240(%1)                              \n\t"

                         "vmovaps 0xC0(%1), %%ymm6                          \n\t"
                         "vmovaps 0xE0(%1), %%ymm7                          \n\t"
                         "vmovaps 0x100(%1), %%ymm8                         \n\t"
                         "vbroadcastss 0x10(%2), %%ymm9                     \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm2                \n\t"
                         "vbroadcastss 0x14(%2), %%ymm9                     \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm5                \n\t"

                         "prefetcht0 0x280(%1)                              \n\t"

                         "vmovaps 0x120(%1), %%ymm6                         \n\t"
                         "vmovaps 0x140(%1), %%ymm7                         \n\t"
                         "vmovaps 0x160(%1), %%ymm8                         \n\t"
                         "vbroadcastss 0x18(%2), %%ymm9                     \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm2                \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm9                     \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm5                \n\t"

                         "add $0x180, %1                                    \n\t"
                         "add $0x20, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_2x24                                    \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_2x24_end:                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_2x24_remain_end                         \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x24_remain:                               \n\t"
                         "vmovaps (%1), %%ymm6                              \n\t"
                         "vmovaps 0x20(%1), %%ymm7                          \n\t"
                         "vmovaps 0x40(%1), %%ymm8                          \n\t"
                         "vbroadcastss 0x0(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm2                \n\t"
                         "vbroadcastss 0x4(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm8, %%ymm5                \n\t"
                         "add $0x60, %1                                     \n\t"
                         "add $0x8, %2                                      \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_2x24_remain                             \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x24_remain_end:                           \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "prefetcht0 0x40(%3, %%rax)                        \n\t"
                         "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
                         "vaddps 0x20(%3), %%ymm1, %%ymm1                   \n\t"
                         "vaddps 0x40(%3), %%ymm2, %%ymm2                   \n\t"
                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1,  0x20(%3)                         \n\t"
                         "vmovups %%ymm2,  0x40(%3)                         \n\t"
                         "add %%rax, %3                                     \n\t"
                         "vaddps (%3), %%ymm3, %%ymm3                       \n\t"
                         "vaddps 0x20(%3), %%ymm4, %%ymm4                   \n\t"
                         "vaddps 0x40(%3), %%ymm5, %%ymm5                   \n\t"
                         "vmovups %%ymm3,  (%3)                             \n\t"
                         "vmovups %%ymm4,  0x20(%3)                         \n\t"
                         "vmovups %%ymm5,  0x40(%3)                         \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                         "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "memory");
}

void mmm_avx2_2x16_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"

                         "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
                         "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_2x16_end                                \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x16:                                      \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"

                         "vmovaps (%1), %%ymm6                              \n\t"
                         "vmovaps 0x20(%1), %%ymm7                          \n\t"
                         "vbroadcastss 0x0(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vbroadcastss 0x4(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"

                         "prefetcht0 0x180(%1)                              \n\t"

                         "vmovaps 0x40(%1), %%ymm6                          \n\t"
                         "vmovaps 0x60(%1), %%ymm7                          \n\t"
                         "vbroadcastss 0x8(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vbroadcastss 0xC(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"

                         "prefetcht0 0x1C0(%1)                              \n\t"

                         "vmovaps 0x80(%1), %%ymm6                          \n\t"
                         "vmovaps 0xA0(%1), %%ymm7                          \n\t"
                         "vbroadcastss 0x10(%2), %%ymm9                     \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vbroadcastss 0x14(%2), %%ymm9                     \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"

                         "prefetcht0 0x200(%1)                              \n\t"

                         "vmovaps 0xC0(%1), %%ymm6                         \n\t"
                         "vmovaps 0xE0(%1), %%ymm7                         \n\t"
                         "vbroadcastss 0x18(%2), %%ymm9                     \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm9                     \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"

                         "add $0x100, %1                                    \n\t"
                         "add $0x20, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_2x16                                    \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_2x16_end:                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_2x16_remain_end                         \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x16_remain:                               \n\t"
                         "vmovaps (%1), %%ymm6                              \n\t"
                         "vmovaps 0x20(%1), %%ymm7                          \n\t"
                         "vbroadcastss 0x0(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm1                \n\t"
                         "vbroadcastss 0x4(%2), %%ymm9                      \n\t"
                         "vfmadd231ps %%ymm9, %%ymm6, %%ymm3                \n\t"
                         "vfmadd231ps %%ymm9, %%ymm7, %%ymm4                \n\t"
                         "add $0x40, %1                                     \n\t"
                         "add $0x8, %2                                      \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_2x16_remain                             \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x16_remain_end:                           \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
                         "vaddps 0x20(%3), %%ymm1, %%ymm1                   \n\t"
                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1,  0x20(%3)                         \n\t"
                         "add %%rax, %3                                     \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "vaddps (%3), %%ymm3, %%ymm3                       \n\t"
                         "vaddps 0x20(%3), %%ymm4, %%ymm4                   \n\t"
                         "vmovups %%ymm3,  (%3)                             \n\t"
                         "vmovups %%ymm4,  0x20(%3)                         \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm3", "%ymm4", "%ymm6",
                         "%ymm7", "%ymm9", "memory");
}

void mmm_avx2_2x8_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"

                         "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_2x8_end                                 \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x8:                                       \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"
                         "vmovaps (%1), %%ymm2                              \n\t"
                         "vbroadcastss 0x0(%2), %%ymm3                      \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm0                \n\t"
                         "vbroadcastss 0x4(%2), %%ymm3                      \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm1                \n\t"

                         "vmovaps 0x20(%1), %%ymm2                          \n\t"
                         "vbroadcastss 0x8(%2), %%ymm3                      \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm0                \n\t"
                         "vbroadcastss 0xC(%2), %%ymm3                      \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm1                \n\t"

                         "prefetcht0 0x180(%1)                              \n\t"
                         "vmovaps 0x40(%1), %%ymm2                          \n\t"
                         "vbroadcastss 0x10(%2), %%ymm3                     \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm0                \n\t"
                         "vbroadcastss 0x14(%2), %%ymm3                     \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm1                \n\t"

                         "vmovaps 0x60(%1), %%ymm2                          \n\t"
                         "vbroadcastss 0x18(%2), %%ymm3                     \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm0                \n\t"
                         "vbroadcastss 0x1C(%2), %%ymm3                     \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm1                \n\t"

                         "add $0x80, %1                                     \n\t"
                         "add $0x20, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_2x8                                     \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_2x8_end:                                   \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_2x8_remain_end                          \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x8_remain:                                \n\t"
                         "vmovaps (%1), %%ymm2                              \n\t"
                         "vbroadcastss 0x0(%2), %%ymm3                      \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm0                \n\t"
                         "vbroadcastss 0x4(%2), %%ymm3                      \n\t"
                         "vfmadd231ps %%ymm3, %%ymm2, %%ymm1                \n\t"
                         "add $0x20, %1                                     \n\t"
                         "add $0x8, %2                                      \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_2x8_remain                              \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x8_remain_end:                            \n\t"

                         "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "add %%rax, %3                                     \n\t"
                         "vaddps (%3), %%ymm1, %%ymm1                       \n\t"
                         "vmovups %%ymm1,  (%3)                             \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "memory");
}

void mmm_avx2_2x4_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"

                         "vxorps %%xmm0, %%xmm0, %%xmm0                     \n\t"
                         "vxorps %%xmm1, %%xmm1, %%xmm1                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_2x4_end                                 \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x4:                                       \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"
                         "vmovaps (%1), %%xmm2                              \n\t"
                         "vbroadcastss 0x0(%2), %%xmm3                      \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm0                \n\t"
                         "vbroadcastss 0x4(%2), %%xmm3                      \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm1                \n\t"

                         "vmovaps 0x10(%1), %%xmm2                          \n\t"
                         "vbroadcastss 0x8(%2), %%xmm3                      \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm0                \n\t"
                         "vbroadcastss 0xC(%2), %%xmm3                      \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm1                \n\t"

                         "vmovaps 0x20(%1), %%xmm2                          \n\t"
                         "vbroadcastss 0x10(%2), %%xmm3                     \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm0                \n\t"
                         "vbroadcastss 0x14(%2), %%xmm3                     \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm1                \n\t"

                         "vmovaps 0x30(%1), %%xmm2                          \n\t"
                         "vbroadcastss 0x18(%2), %%xmm3                     \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm0                \n\t"
                         "vbroadcastss 0x1C(%2), %%xmm3                     \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm1                \n\t"

                         "add $0x40, %1                                     \n\t"
                         "add $0x20, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_2x4                                     \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_2x4_end:                                   \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_2x4_remain_end                          \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x4_remain:                                \n\t"
                         "vmovaps (%1), %%xmm2                              \n\t"
                         "vbroadcastss 0x0(%2), %%xmm3                      \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm0                \n\t"
                         "vbroadcastss 0x4(%2), %%xmm3                      \n\t"
                         "vfmadd231ps %%xmm3, %%xmm2, %%xmm1                \n\t"
                         "add $0x10, %1                                     \n\t"
                         "add $0x8, %2                                      \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_2x4_remain                              \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_2x4_remain_end:                            \n\t"

                         "vaddps (%3), %%xmm0, %%xmm0                       \n\t"
                         "vmovups %%xmm0, (%3)                              \n\t"
                         "add %%rax, %3                                     \n\t"
                         "vaddps (%3), %%xmm1, %%xmm1                       \n\t"
                         "vmovups %%xmm1, (%3)                              \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory");
}

void mmm_avx2_1x24_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"

                         "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_1x24_end                                \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_1x24:                                      \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"
                         "prefetcht0 0x180(%1)                              \n\t"

                         "vmovaps (%1), %%ymm3                              \n\t"
                         "vmovaps 0x20(%1), %%ymm4                          \n\t"
                         "vmovaps 0x40(%1), %%ymm5                          \n\t"
                         "vbroadcastss 0x0(%2), %%ymm6                      \n\t"
                         "vfmadd231ps %%ymm6, %%ymm3, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm4, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm5, %%ymm2                \n\t"

                         "prefetcht0 0x1C0(%1)                              \n\t"

                         "vmovaps 0x60(%1), %%ymm3                          \n\t"
                         "vmovaps 0x80(%1), %%ymm4                          \n\t"
                         "vmovaps 0xA0(%1), %%ymm5                          \n\t"
                         "vbroadcastss 0x4(%2), %%ymm6                      \n\t"
                         "vfmadd231ps %%ymm6, %%ymm3, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm4, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm5, %%ymm2                \n\t"

                         "prefetcht0 0x200(%1)                              \n\t"
                         "prefetcht0 0x240(%1)                              \n\t"

                         "vmovaps 0xC0(%1), %%ymm3                          \n\t"
                         "vmovaps 0xE0(%1), %%ymm4                          \n\t"
                         "vmovaps 0x100(%1), %%ymm5                         \n\t"
                         "vbroadcastss 0x8(%2), %%ymm6                      \n\t"
                         "vfmadd231ps %%ymm6, %%ymm3, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm4, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm5, %%ymm2                \n\t"

                         "prefetcht0 0x280(%1)                              \n\t"

                         "vmovaps 0x120(%1), %%ymm3                         \n\t"
                         "vmovaps 0x140(%1), %%ymm4                         \n\t"
                         "vmovaps 0x160(%1), %%ymm5                         \n\t"
                         "vbroadcastss 0xC(%2), %%ymm6                      \n\t"
                         "vfmadd231ps %%ymm6, %%ymm3, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm4, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm5, %%ymm2                \n\t"

                         "add $0x180, %1                                    \n\t"
                         "add $0x10, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_1x24                                    \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_1x24_end:                                  \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_1x24_remain_end                         \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_1x24_remain:                               \n\t"
                         "vmovaps (%1), %%ymm3                              \n\t"
                         "vmovaps 0x20(%1), %%ymm4                          \n\t"
                         "vmovaps 0x40(%1), %%ymm5                          \n\t"
                         "vbroadcastss (%2), %%ymm6                         \n\t"
                         "vfmadd231ps %%ymm6, %%ymm3, %%ymm0                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm4, %%ymm1                \n\t"
                         "vfmadd231ps %%ymm6, %%ymm5, %%ymm2                \n\t"
                         "add $0x60, %1                                     \n\t"
                         "add $0x4, %2                                      \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_1x24_remain                             \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_1x24_remain_end:                           \n\t"
                         "prefetcht0 (%3, %%rax)                            \n\t"
                         "prefetcht0 0x40(%3, %%rax)                        \n\t"
                         "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
                         "vaddps 0x20(%3), %%ymm1, %%ymm1                   \n\t"
                         "vaddps 0x40(%3), %%ymm2, %%ymm2                   \n\t"
                         "vmovups %%ymm0,  (%3)                             \n\t"
                         "vmovups %%ymm1,  0x20(%3)                         \n\t"
                         "vmovups %%ymm2,  0x40(%3)                         \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                         "%ymm5", "%ymm6", "memory");
}

void mmm_avx2_1x16_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__(
        "mov %4, %%eax                                     \n\t"
        "shl $2, %%eax                                     \n\t"
        "mov %%eax, %%eax                                  \n\t"

        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"

        "mov %0, %%ecx                                     \n\t"
        "shr $2, %%ecx                                     \n\t"
        "je .k_loop_1x16_end                                \n\t"

        ".align 16                                         \n\t"
        ".k_loop_1x16:                                      \n\t"

        "prefetcht0 0x140(%1)                              \n\t"

        "vmovaps (%1), %%ymm2                              \n\t"
        "vmovaps 0x20(%1), %%ymm3                          \n\t"
        "vbroadcastss (%2), %%ymm5                         \n\t"
        "vfmadd231ps %%ymm5, %%ymm2, %%ymm0                \n\t"
        "vfmadd231ps %%ymm5, %%ymm3, %%ymm1                \n\t"

        "prefetcht0 0x180(%1)                              \n\t"

        "vmovaps 0x40(%1), %%ymm2                          \n\t"
        "vmovaps 0x60(%1), %%ymm3                          \n\t"
        "vbroadcastss 0x4(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm2, %%ymm0                \n\t"
        "vfmadd231ps %%ymm5, %%ymm3, %%ymm1                \n\t"

        "prefetcht0 0x1C0(%1)                              \n\t"

        "vmovaps 0x80(%1), %%ymm2                          \n\t"
        "vmovaps 0xA0(%1), %%ymm3                          \n\t"
        "vbroadcastss 0x8(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm2, %%ymm0                \n\t"
        "vfmadd231ps %%ymm5, %%ymm3, %%ymm1                \n\t"

        "prefetcht0 0x200(%1)                              \n\t"

        "vmovaps 0xC0(%1), %%ymm2                          \n\t"
        "vmovaps 0xE0(%1), %%ymm3                          \n\t"
        "vbroadcastss 0xC(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm2, %%ymm0                \n\t"
        "vfmadd231ps %%ymm5, %%ymm3, %%ymm1                \n\t"

        "add $0x100, %1                                    \n\t"
        "add $0x10, %2                                     \n\t"

        "sub $1, %%ecx                                     \n\t"
        "jg .k_loop_1x16                                    \n\t"
        ".align 16                                         \n\t"
        ".k_loop_1x16_end:                                  \n\t"

        "mov %0, %%ecx                                     \n\t"
        "and $3, %%ecx                                     \n\t"
        "je .k_loop_1x16_remain_end                         \n\t"

        ".align 16                                         \n\t"
        ".k_loop_1x16_remain:                               \n\t"
        "vmovaps (%1), %%ymm2                              \n\t"
        "vmovaps 0x20(%1), %%ymm3                          \n\t"
        "vbroadcastss 0x0(%2), %%ymm5                      \n\t"
        "vfmadd231ps %%ymm5, %%ymm2, %%ymm0                \n\t"
        "vfmadd231ps %%ymm5, %%ymm3, %%ymm1                \n\t"
        "add $0x40, %1                                     \n\t"
        "add $0x4, %2                                      \n\t"
        "sub $1, %%ecx                                     \n\t"
        "jg .k_loop_1x16_remain                             \n\t"

        ".align 16                                         \n\t"
        ".k_loop_1x16_remain_end:                           \n\t"
        "prefetcht0 (%3, %%rax)                            \n\t"
        "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
        "vaddps 0x20(%3), %%ymm1, %%ymm1                   \n\t"
        "vmovups %%ymm0,  (%3)                             \n\t"
        "vmovups %%ymm1,  0x20(%3)                         \n\t"
        :
        : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
        : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm5", "memory");
}

void mmm_avx2_1x8_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"

                         "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_1x8_end                                 \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_1x8:                                       \n\t"

                         "prefetcht0 0x140(%1)                              \n\t"
                         "vmovaps (%1), %%ymm1                              \n\t"
                         "vbroadcastss (%2), %%ymm2                         \n\t"
                         "vfmadd231ps %%ymm2, %%ymm1, %%ymm0                \n\t"

                         "vmovaps 0x20(%1), %%ymm1                          \n\t"
                         "vbroadcastss 0x4(%2), %%ymm2                      \n\t"
                         "vfmadd231ps %%ymm2, %%ymm1, %%ymm0                \n\t"

                         "prefetcht0 0x180(%1)                              \n\t"
                         "vmovaps 0x40(%1), %%ymm1                          \n\t"
                         "vbroadcastss 0x8(%2), %%ymm2                      \n\t"
                         "vfmadd231ps %%ymm2, %%ymm1, %%ymm0                \n\t"

                         "vmovaps 0x60(%1), %%ymm1                          \n\t"
                         "vbroadcastss 0xC(%2), %%ymm2                      \n\t"
                         "vfmadd231ps %%ymm2, %%ymm1, %%ymm0                \n\t"

                         "add $0x80, %1                                     \n\t"
                         "add $0x10, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_1x8                                     \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_1x8_end:                                   \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_1x8_remain_end                          \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_1x8_remain:                                \n\t"
                         "vmovaps (%1), %%ymm1                              \n\t"
                         "vbroadcastss (%2), %%ymm2                         \n\t"
                         "vfmadd231ps %%ymm2, %%ymm1, %%ymm0                \n\t"
                         "add $0x20, %1                                     \n\t"
                         "add $0x4, %2                                      \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_1x8_remain                              \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_1x8_remain_end:                            \n\t"

                         "vaddps (%3), %%ymm0, %%ymm0                       \n\t"
                         "vmovups %%ymm0,  (%3)                             \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%ymm0", "%ymm1", "%ymm2", "memory");
}

void mmm_avx2_1x4_asm(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    __asm__ __volatile__("mov %4, %%eax                                     \n\t"
                         "shl $2, %%eax                                     \n\t"
                         "mov %%eax, %%eax                                  \n\t"

                         "vxorps %%xmm0, %%xmm0, %%xmm0                     \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "shr $2, %%ecx                                     \n\t"
                         "je .k_loop_1x4_end                                 \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_1x4:                                       \n\t"

                         "prefetcht0 0x40(%1)                               \n\t"

                         "vmovaps (%1), %%xmm1                              \n\t"
                         "vbroadcastss 0x0(%2), %%xmm2                      \n\t"
                         "vfmadd231ps %%xmm2, %%xmm1, %%xmm0                \n\t"

                         "vmovaps 0x10(%1), %%xmm1                          \n\t"
                         "vbroadcastss 0x4(%2), %%xmm2                      \n\t"
                         "vfmadd231ps %%xmm2, %%xmm1, %%xmm0                \n\t"

                         "vmovaps 0x20(%1), %%xmm1                          \n\t"
                         "vbroadcastss 0x8(%2), %%xmm2                      \n\t"
                         "vfmadd231ps %%xmm2, %%xmm1, %%xmm0                \n\t"

                         "vmovaps 0x30(%1), %%xmm1                          \n\t"
                         "vbroadcastss 0xC(%2), %%xmm2                      \n\t"
                         "vfmadd231ps %%xmm2, %%xmm1, %%xmm0                \n\t"

                         "add $0x40, %1                                     \n\t"
                         "add $0x10, %2                                     \n\t"

                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_1x4                                     \n\t"
                         ".align 16                                         \n\t"
                         ".k_loop_1x4_end:                                   \n\t"

                         "mov %0, %%ecx                                     \n\t"
                         "and $3, %%ecx                                     \n\t"
                         "je .k_loop_1x4_remain_end                          \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_1x4_remain:                                \n\t"
                         "vmovaps (%1), %%xmm1                              \n\t"
                         "vbroadcastss 0x0(%2), %%xmm2                      \n\t"
                         "vfmadd231ps %%xmm2, %%xmm1, %%xmm0                \n\t"
                         "add $0x10, %1                                     \n\t"
                         "add $0x4, %2                                      \n\t"
                         "sub $1, %%ecx                                     \n\t"
                         "jg .k_loop_1x4_remain                              \n\t"

                         ".align 16                                         \n\t"
                         ".k_loop_1x4_remain_end:                            \n\t"

                         "vaddps (%3), %%xmm0, %%xmm0                       \n\t"
                         "vmovups %%xmm0,  (%3)                             \n\t"
                         "add %%rax, %3                                     \n\t"
                         :
                         : "r"(bk), "r"(matrixB), "r"(matrixA), "r"(matrixC), "r"(N)
                         : "%eax", "%rax", "%ecx", "%xmm0", "%xmm1", "%xmm2", "memory");
}

void mmm_avx2_n_mtail(U32 um, U32 un, U32 bk, F32 *matrixA, F32 *matrixB, F32 *matrixC, U32 N)
{
    for (U32 i = 0; i < um; ++i) {
        for (U32 j = 0; j < un; ++j) {
            for (U32 k = 0; k < bk; ++k) {
                matrixC[i * N + j] += matrixA[k * um + i] * matrixB[k * un + j];
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
    kernel_func kernel[3][5] = {
        {mmm_avx2_n_mtail, mmm_avx2_1x4_asm, mmm_avx2_1x8_asm, mmm_avx2_1x16_asm, mmm_avx2_1x24_asm},
        {mmm_avx2_n_mtail, mmm_avx2_2x4_asm, mmm_avx2_2x8_asm, mmm_avx2_2x16_asm, mmm_avx2_2x24_asm},
        {mmm_avx2_n_mtail, mmm_avx2_4x4_asm, mmm_avx2_4x8_asm, mmm_avx2_4x16_asm, mmm_avx2_4x24_asm}};
    F32 unrollNSize[4] = {4, 8, 16, 24};
    F32 unrollMSize[3] = {1, 2, 4};
    I32 resN = N % 24;
    I32 blockNNum = N / 24;
    I32 edgeblockNSizeArray[5] = {0};
    for (U32 i = 0; resN > 0; ++i) {
        U32 value = UNI_MIN(unrollNSize[resN >> 3], resN);
        edgeblockNSizeArray[i] += value;
        edgeblockNSizeArray[i + 1] = edgeblockNSizeArray[i];
        resN -= value;
        blockNNum += 1;
    }

#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
    {
#endif
        I32 blockSizeM = 0, blockSizeK = 0;
        for (int k = 0; k < K; k += blockSizeK) {
            blockSizeK = UNI_MIN(BOLCK_K_DIM, K - k);
            for (int j = 0; j < M; j += blockSizeM) {
                blockSizeM = UNI_MIN(BOLCK_M_DIM, M - j);
                I32 blockMNum = blockSizeM / 4 + (blockSizeM % 4 + 1) / 2;
#ifdef _USE_OPENMP
#pragma omp for
#endif
                for (I32 mIdx = 0; mIdx < blockMNum; ++mIdx) {
                    I32 m = mIdx * 4 - ((mIdx * 4) > blockSizeM) * 2;
                    I32 unrollSizeM = UNI_MIN(UNROLL_M, blockSizeM - m);
                    unrollSizeM = unrollMSize[unrollSizeM >> 1];

                    I32 blockSizeN = UNI_MIN(UNROLL_N, N);
                    blockSizeN = UNI_MIN(unrollNSize[blockSizeN >> 3], blockSizeN);

                    F32 *curB = packB + k * N;
                    F32 *curA = packA + m * blockSizeK;
                    if (matrix1Df == DF_TRANSPOSE) {
                        matrix2_trans(unrollSizeM, blockSizeK, M, matrix1 + (j + m) + k * M, curA);
                    } else if (matrix1Df == DF_NORMAL) {
                        matrix1_trans(unrollSizeM, blockSizeK, K, matrix1 + k + (j + m) * K, curA);
                    } else if (matrix1Df == DF_NKN8) {
                        matrix2_trans_c8(
                            unrollSizeM, blockSizeK, M, matrix1 + (j + m) * 8 + k * M, curA);
                    }
                    kernel[unrollSizeM >> 1][(blockSizeN >> 3) + (blockSizeN > 3)](
                        unrollSizeM, blockSizeN, blockSizeK, curA, curB, result + (m + j) * N, N);
                }
#ifdef _USE_OPENMP
#pragma omp for
#endif
                for (int mnIdx = blockMNum; mnIdx < blockNNum * blockMNum; ++mnIdx) {
                    I32 nIdx = mnIdx / blockMNum;
                    I32 n = nIdx * UNROLL_N;
                    if (n >= N) {
                        U32 idx = (n - N) / UNROLL_N;
                        CHECK_REQUIREMENT(idx <= 4);
                        n = N / UNROLL_N * UNROLL_N + edgeblockNSizeArray[idx];
                    }
                    I32 blockSizeN = UNI_MIN(UNROLL_N, N - n);
                    blockSizeN = UNI_MIN(unrollNSize[blockSizeN >> 3], blockSizeN);
                    F32 *curB = packB + k * N + n * blockSizeK;

                    I32 mIdx = mnIdx % blockMNum;
                    I32 m = mIdx * 4 - ((mIdx * 4) > blockSizeM) * 2;
                    I32 unrollSizeM = UNI_MIN(UNROLL_M, blockSizeM - m);
                    unrollSizeM = unrollMSize[unrollSizeM >> 1];
                    kernel[unrollSizeM >> 1][(blockSizeN >> 3) + (blockSizeN > 3)](unrollSizeM,
                        blockSizeN, blockSizeK, packA + m * blockSizeK, curB,
                        result + (m + j) * N + n, N);
                }
            }
        }
#ifdef _USE_OPENMP
    }
#endif
    return SUCCESS;
}
