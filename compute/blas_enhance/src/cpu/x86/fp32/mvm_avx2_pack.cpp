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

#define UNROLL_N 64
#define BOLCK_K_DIM 1024

typedef void (*kernel_func)(U32 bk, F32 *matrix, F32 *vector, F32 *result);

EE matrix_vector_multiply_transform_weight_fp32(TensorDesc desc, F32 *src, F32 *packB)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    F32 unrollSize[4] = {8, 16, 32, 64};
    U32 unrollSizeN = 0;
    EE ret = SUCCESS;
    switch (desc.df) {
        case DF_NORMAL: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
            U32 blockKSize = 0;
            for (U32 bk = 0; bk < K; bk += blockKSize) {
                blockKSize = UNI_MIN(K - bk, BOLCK_K_DIM);
                for (U32 un = 0; un < N; un += unrollSizeN) {
                    unrollSizeN = UNI_MIN(UNROLL_N, N - un);
                    unrollSizeN = unrollSize[unrollSizeN / 16 - (unrollSizeN >= 48)];
                    if (N - un < unrollSizeN) {
                        for (U32 k = 0; k < blockKSize; ++k) {
                            for (U32 i = 0; i < N - un; ++i) {
                                packB[k * (N - un) + i] = src[(un + i) * K + k + bk];
                            }
                        }
                        packB += (N - un) * blockKSize;
                    } else {
                        matrix1_trans(unrollSizeN, blockKSize, K, src + un * K + bk, packB);
                        packB += unrollSizeN * blockKSize;
                    }
                }
            }
            break;
        }
        case DF_TRANSPOSE: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
            U32 blockKSize = 0;
            for (U32 bk = 0; bk < K; bk += blockKSize) {
                blockKSize = UNI_MIN(K - bk, BOLCK_K_DIM);
                for (U32 un = 0; un < N; un += unrollSizeN) {
                    unrollSizeN = UNI_MIN(UNROLL_N, N - un);
                    unrollSizeN = unrollSize[unrollSizeN / 16 - (unrollSizeN >= 48)];
                    if (N - un < unrollSizeN) {
                        for (U32 k = 0; k < blockKSize; ++k) {
                            memcpy(packB + k * (N - un), src + (k + bk) * N + un,
                                (N - un) * sizeof(F32));
                        }
                        packB += (N - un) * blockKSize;
                    } else {
                        matrix2_trans(unrollSizeN, blockKSize, N, src + un + bk * N, packB);
                        packB += unrollSizeN * blockKSize;
                    }
                }
            }
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

void mvm_row_avx_64(U32 bk, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
                         "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
                         "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"
                         "vxorps %%ymm4, %%ymm4, %%ymm4                     \n\t"
                         "vxorps %%ymm5, %%ymm5, %%ymm5                     \n\t"
                         "vxorps %%ymm6, %%ymm6, %%ymm6                     \n\t"
                         "vxorps %%ymm7, %%ymm7, %%ymm7                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vbroadcastss (%2), %%ymm12                     \n\t"
                         "vmovups (%1), %%ymm8                             \n\t"
                         "vmovups 0x20(%1), %%ymm9                         \n\t"
                         "vmovups 0x40(%1), %%ymm10                         \n\t"
                         "vmovups 0x60(%1), %%ymm11                         \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm12, %%ymm9, %%ymm1              \n\t"
                         "vfmadd231ps %%ymm12, %%ymm10, %%ymm2              \n\t"
                         "vfmadd231ps %%ymm12, %%ymm11, %%ymm3              \n\t"
                         "vmovups 0x80(%1), %%ymm8                             \n\t"
                         "vmovups 0xA0(%1), %%ymm9                         \n\t"
                         "vmovups 0xC0(%1), %%ymm10                         \n\t"
                         "vmovups 0xE0(%1), %%ymm11                         \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm4              \n\t"
                         "vfmadd231ps %%ymm12, %%ymm9, %%ymm5              \n\t"
                         "vfmadd231ps %%ymm12, %%ymm10, %%ymm6              \n\t"
                         "vfmadd231ps %%ymm12, %%ymm11, %%ymm7              \n\t"

                         "add $0x4, %2                                      \n\t"
                         "add $0x100, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "vaddps (%3), %%ymm0, %%ymm0                            \n\t"
                         "vmovups %%ymm0, (%3)                            \n\t"
                         "vaddps 0x20(%3), %%ymm1, %%ymm1                            \n\t"
                         "vmovups %%ymm1, 0x20(%3)                            \n\t"
                         "vaddps 0x40(%3), %%ymm2, %%ymm2                            \n\t"
                         "vmovups %%ymm2, 0x40(%3)                            \n\t"
                         "vaddps 0x60(%3), %%ymm3, %%ymm3                            \n\t"
                         "vmovups %%ymm3, 0x60(%3)                            \n\t"
                         "vaddps 0x80(%3), %%ymm4, %%ymm4                            \n\t"
                         "vmovups %%ymm4, 0x80(%3)                            \n\t"
                         "vaddps 0xA0(%3), %%ymm5, %%ymm5                            \n\t"
                         "vmovups %%ymm5, 0xA0(%3)                            \n\t"
                         "vaddps 0xC0(%3), %%ymm6, %%ymm6                            \n\t"
                         "vmovups %%ymm6, 0xC0(%3)                            \n\t"
                         "vaddps 0xE0(%3), %%ymm7, %%ymm7                            \n\t"
                         "vmovups %%ymm7, 0xE0(%3)                            \n\t"
                         :
                         : "c"(bk), "r"(matrix), "r"(vector), "r"(result)
                         : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7",
                         "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "memory");
}

void mvm_row_avx_32(U32 bk, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                     \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                     \n\t"

        ".align 16                                         \n\t"
        "0:                                      \n\t"
        "vbroadcastss (%2), %%ymm12                     \n\t"
        "vmovups (%1), %%ymm8                             \n\t"
        "vmovups 0x20(%1), %%ymm9                         \n\t"
        "vmovups 0x40(%1), %%ymm10                         \n\t"
        "vmovups 0x60(%1), %%ymm11                         \n\t"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm0              \n\t"
        "vfmadd231ps %%ymm12, %%ymm9, %%ymm1              \n\t"
        "vfmadd231ps %%ymm12, %%ymm10, %%ymm2              \n\t"
        "vfmadd231ps %%ymm12, %%ymm11, %%ymm3              \n\t"

        "add $0x4, %2                                      \n\t"
        "add $0x80, %1                                      \n\t"
        "dec %%ecx                                         \n\t"
        "jg 0b                                             \n\t"

        "vaddps (%3), %%ymm0, %%ymm0                            \n\t"
        "vmovups %%ymm0, (%3)                            \n\t"
        "vaddps 0x20(%3), %%ymm1, %%ymm1                            \n\t"
        "vmovups %%ymm1, 0x20(%3)                            \n\t"
        "vaddps 0x40(%3), %%ymm2, %%ymm2                            \n\t"
        "vmovups %%ymm2, 0x40(%3)                            \n\t"
        "vaddps 0x60(%3), %%ymm3, %%ymm3                            \n\t"
        "vmovups %%ymm3, 0x60(%3)                            \n\t"
        :
        : "c"(bk), "r"(matrix), "r"(vector), "r"(result)
        : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm8", "%ymm9", "%ymm10", "%ymm12", "memory");
}

void mvm_row_avx_16(U32 bk, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vxorps %%ymm1, %%ymm1, %%ymm1                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vbroadcastss (%2), %%ymm12                     \n\t"
                         "vmovups (%1), %%ymm8                             \n\t"
                         "vmovups 0x20(%1), %%ymm9                         \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm0              \n\t"
                         "vfmadd231ps %%ymm12, %%ymm9, %%ymm1              \n\t"

                         "add $0x4, %2                                      \n\t"
                         "add $0x40, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "vaddps (%3), %%ymm0, %%ymm0                            \n\t"
                         "vmovups %%ymm0, (%3)                            \n\t"
                         "vaddps 0x20(%3), %%ymm1, %%ymm1                            \n\t"
                         "vmovups %%ymm1, 0x20(%3)                            \n\t"
                         :
                         : "c"(bk), "r"(matrix), "r"(vector), "r"(result)
                         : "%ymm0", "%ymm1", "%ymm8", "%ymm9", "%ymm12", "memory");
}

void mvm_row_avx_8(U32 bk, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vbroadcastss (%2), %%ymm12                     \n\t"
                         "vmovups (%1), %%ymm8                             \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm0              \n\t"

                         "add $0x4, %2                                      \n\t"
                         "add $0x20, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "vaddps (%3), %%ymm0, %%ymm0                            \n\t"
                         "vmovups %%ymm0, (%3)                            \n\t"
                         :
                         : "c"(bk), "r"(matrix), "r"(vector), "r"(result)
                         : "%ymm0", "%ymm8", "%ymm12", "memory");
}

void mvm_row_avx_8_mask(U32 bk, U32 step, I32 *mask, F32 *matrix, F32 *vector, F32 *result)
{
    __asm__ __volatile__("vxorps %%ymm0, %%ymm0, %%ymm0                     \n\t"
                         "vmovdqu (%4), %%ymm1                     \n\t"

                         ".align 16                                         \n\t"
                         "0:                                      \n\t"
                         "vbroadcastss (%2), %%ymm12                     \n\t"
                         "vmaskmovps (%1), %%ymm1, %%ymm8                             \n\t"
                         "vfmadd231ps %%ymm12, %%ymm8, %%ymm0              \n\t"

                         "add $0x4, %2                                      \n\t"
                         "add %5, %1                                      \n\t"
                         "dec %%ecx                                         \n\t"
                         "jg 0b                                             \n\t"

                         "vmaskmovps (%3), %%ymm1, %%ymm12                            \n\t"
                         "vaddps %%ymm12, %%ymm0, %%ymm0                            \n\t"
                         "vmaskmovps %%ymm0, %%ymm1, (%3)                            \n\t"
                         :
                         : "c"(bk), "r"(matrix), "r"(vector), "r"(result), "r"(mask), "r"(I64(step))
                         : "%ymm0", "%ymm1", "%ymm8", "%ymm12", "memory");
}

void mvm_pack_fp32(U32 numRows, U32 numColumns, F32 *packB, F32 *vector, F32 *result)
{
    // Actual layout is NKN64, and vector is K
    kernel_func kernel[4] = {mvm_row_avx_8, mvm_row_avx_16, mvm_row_avx_32, mvm_row_avx_64};
    U32 unrollSize[4] = {8, 16, 32, 64};
    I32 resN = numRows % 64;
    I32 blockNum = numRows / 64;
    I32 edgeblockNSizeArray[6] = {0};
    for (U32 i = 0; resN > 0; ++i) {
        U32 value = UNI_MIN(unrollSize[UNI_MIN(resN >> 4, 2)], resN);
        edgeblockNSizeArray[i] += value;
        edgeblockNSizeArray[i + 1] = edgeblockNSizeArray[i];
        resN -= value;
        blockNum += 1;
    }
#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
    {
#endif
        U32 private_blockKSize = 0;
        for (U32 bk = 0; bk < numColumns; bk += private_blockKSize) {
            private_blockKSize = UNI_MIN(numColumns - bk, BOLCK_K_DIM);
#ifdef _USE_OPENMP
#pragma omp for
#endif
            for (U32 bIdx = 0; bIdx < (U32)(blockNum); ++bIdx) {
                U32 bn = bIdx * UNROLL_N;
                if (bn >= numRows) {
                    U32 idx = (bn - numRows) / UNROLL_N;
                    CHECK_REQUIREMENT(idx <= 5);
                    bn = numRows / UNROLL_N * UNROLL_N + edgeblockNSizeArray[idx];
                }

                int blockNSize = UNI_MIN(numRows - bn, UNROLL_N);
                if ((blockNSize < 8) && (blockNSize % 8 != 0)) {
                    I32 mask[8] = {0};
                    for (int mi = 0; mi < blockNSize; ++mi) {
                        mask[mi] = -1;
                    }
                    mvm_row_avx_8_mask(private_blockKSize, blockNSize * 4, mask,
                        packB + bk * numRows + bn * private_blockKSize, vector + bk, result + bn);
                } else {
                    blockNSize = unrollSize[blockNSize / 16 - (blockNSize >= 48)];
                    kernel[blockNSize / 16 - (blockNSize == 64)](private_blockKSize,
                        packB + bk * numRows + bn * private_blockKSize, vector + bk, result + bn);
                }
            }
        }
#ifdef _USE_OPENMP
    }
#endif
}
