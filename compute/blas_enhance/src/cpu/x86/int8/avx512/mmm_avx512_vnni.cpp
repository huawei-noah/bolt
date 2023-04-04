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
#include "blas_enhance.h"

#define UNROLL_N 48
#define UNROLL_M 8
#define BOLCK_M_DIM 384
#define BOLCK_K_DIM 4096

void matrix_matrix_multiply_transform_rhs_bytes_int8(
    U32 N, U32 K, DataFormat bdf, U32 *bytes, U32 *rhsBytes)
{
    U32 matrix = 0;
    U32 pad = 0;
    if (bdf != matrix_matrix_multiply_rhs_format(DT_I8)) {
        pad += N * bytesOf(DT_I32);
        N = UNI_ALIGN(N, 16);
        K = UNI_ALIGN(K, 8);
        matrix = N * K * bytesOf(DT_I8);
        pad += matrix + 64;
    }
    if (rhsBytes != nullptr) {
        *rhsBytes = matrix;
    }
    if (bytes != nullptr) {
        *bytes = pad;
    }
}

void matrix_matrix_multiply_tmp_bytes_int8(
    U32 N, U32 M, U32 K, DataFormat adf, DataFormat bdf, U32 *bytes)
{
    matrix_matrix_multiply_transform_rhs_bytes_int8(N, K, bdf, bytes, nullptr);
    if (adf == DF_NORMAL) {
        *bytes += 32 * K;
        if (K % 8 != 0) {
            *bytes += UNI_ALIGN(M, 24) * 8;
        }
    } else if (adf == DF_TRANSPOSE) {
        *bytes += UNI_ALIGN(M, 24) * UNI_MIN(BOLCK_K_DIM, UNI_ALIGN(K, 8));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    *bytes += N * bytesOf(DT_I32);
    *bytes += 64;
}

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

#define mmm_2_48_0(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "vmovups (%[B]), %%zmm27                \n\t" \
    "vmovups 0x40(%[B]), %%zmm28            \n\t" \
    "vmovups 0x80(%[B]), %%zmm29            \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm0      \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm1      \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm2      \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm3      \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm4      \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm5      \n\t" \

#define mmm_2_48_1(A, K) \
    "addq $0x4, %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "vmovups 0xC0(%[B]), %%zmm24            \n\t" \
    "vmovups 0x100(%[B]), %%zmm25           \n\t" \
    "vmovups 0x140(%[B]), %%zmm26           \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm0      \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm1      \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm2      \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm3      \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm4      \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm5      \n\t" \

#define mmm_2_48(A, K) \
    mmm_2_48_0(A, K) \
    mmm_2_48_1(A, K)

#define mmm_3_48(A, K) \
    mmm_2_48_0(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm6      \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm7      \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm8      \n\t" \
    "movq "#A", %%rax                       \n\t" \
    mmm_2_48_1(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm6      \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm7      \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm8      \n\t" \

#define mmm_4_48_0(A, K) \
    mmm_2_48_0(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm6      \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm7      \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm8      \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm9      \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm10     \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm11     \n\t" \

#define mmm_4_48_1(A, K) \
    "movq "#A", %%rax                       \n\t" \
    mmm_2_48_1(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31    \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm6      \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm7      \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm8      \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm9      \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm10     \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm11     \n\t" \

#define mmm_4_48(A, K) \
    mmm_4_48_0(A, K) \
    mmm_4_48_1(A, K)

#define mmm_5_48(A, K) \
    mmm_4_48_0(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm12     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm13     \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm14     \n\t" \
    "movq "#A", %%rax                       \n\t" \
    mmm_4_48_1(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm30          \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm12     \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm13     \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm14     \n\t" \

#define mmm_6_48_0(A, K) \
    mmm_4_48_0(A, K) \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm12    \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm13    \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm14    \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm15    \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm16    \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm17    \n\t" \

#define mmm_6_48_1(A, K) \
    mmm_4_48_1(A, K) \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm31   \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm12    \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm13    \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm14    \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm15    \n\t" \
    "vpdpbusd %%zmm28, %%zmm31, %%zmm16    \n\t" \
    "vpdpbusd %%zmm29, %%zmm31, %%zmm17    \n\t" \

#define mmm_6_48(A, K) \
    mmm_6_48_0(A, K) \
    mmm_6_48_1(A, K)

#define mmm_7_48(A, K) \
    mmm_6_48_0(A, K) \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm18    \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm19    \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm20    \n\t" \
    mmm_6_48_1(A, K) \
    "addq "#K", %%rax                      \n\t" \
    "addq "#K", %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm30         \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm18    \n\t" \
    "vpdpbusd %%zmm28, %%zmm30, %%zmm19    \n\t" \
    "vpdpbusd %%zmm29, %%zmm30, %%zmm20    \n\t" \

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

#define mmm_2_32_0(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "prefetcht0 0x80(%[B])                  \n\t" \
    "prefetcht0 0xC0(%[B])                  \n\t" \
    "vmovups (%[B]), %%zmm26                \n\t" \
    "vmovups 0x40(%[B]), %%zmm27            \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm0      \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm1      \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm2      \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm3      \n\t" \

#define mmm_2_32_1(A, K) \
    "addq $0x4, %%rax                       \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "prefetcht0 0x100(%[B])                 \n\t" \
    "prefetcht0 0x140(%[B])                 \n\t" \
    "vmovups 0x80(%[B]), %%zmm24            \n\t" \
    "vmovups 0xC0(%[B]), %%zmm25            \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm0      \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm1      \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm2      \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm3      \n\t" \

#define mmm_2_32(A, K) \
    mmm_2_32_0(A, K) \
    mmm_2_32_1(A, K)

#define mmm_3_32(A, K) \
    mmm_2_32_0(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm5      \n\t" \
    mmm_2_32_1(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm5      \n\t"

#define mmm_4_32_0(A, K) \
    "movq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    "addq "#K", %%rbx                       \n\t" \
    mmm_2_32_0(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm5      \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm6      \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm7      \n\t" \

#define mmm_4_32_1(A, K) \
    "movq "#A", %%rax                       \n\t" \
    mmm_2_32_1(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm4      \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm5      \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm6      \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm7      \n\t" \

#define mmm_4_32(A, K) \
    mmm_4_32_0(A, K) \
    mmm_4_32_1(A, K)

#define mmm_5_32(A, K) \
    mmm_4_32_0(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm8      \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm9      \n\t" \
    mmm_4_32_1(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm8      \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm9      \n\t" \

#define mmm_6_32_0(A, K) \
    mmm_4_32_0(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm8      \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm9      \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm10     \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm11     \n\t" \

#define mmm_6_32_1(A, K) \
    mmm_4_32_1(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm8      \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm9      \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm10     \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm11     \n\t"

#define mmm_6_32(A, K) \
    mmm_6_32_0(A, K) \
    mmm_6_32_1(A, K)

#define mmm_7_32(A, K) \
    mmm_6_32_0(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm12     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm13     \n\t" \
    mmm_6_32_1(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm12     \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm13     \n\t" \

#define mmm_8_32_0(A, K) \
    mmm_6_32_0(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm12     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm13     \n\t" \
    "vpdpbusd %%zmm24, %%zmm31, %%zmm14     \n\t" \
    "vpdpbusd %%zmm25, %%zmm31, %%zmm15     \n\t" \

#define mmm_8_32_1(A, K) \
    mmm_6_32_1(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpbroadcastd (%%rax, %%rbx), %%zmm31   \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm12     \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm13     \n\t" \
    "vpdpbusd %%zmm26, %%zmm31, %%zmm14     \n\t" \
    "vpdpbusd %%zmm27, %%zmm31, %%zmm15     \n\t" \

#define mmm_8_32(A, K) \
    mmm_8_32_0(A, K) \
    mmm_8_32_1(A, K)

#define mmm_9_32(A, K) \
    mmm_8_32_0(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm16     \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm17     \n\t" \
    mmm_8_32_1(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm16     \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm17     \n\t" \

#define mmm_10_32_0(A, K) \
    mmm_8_32_0(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpdpbusd %%zmm24, %%zmm28, %%zmm16     \n\t" \
    "vpdpbusd %%zmm25, %%zmm28, %%zmm17     \n\t" \
    "vpdpbusd %%zmm24, %%zmm29, %%zmm18     \n\t" \
    "vpdpbusd %%zmm25, %%zmm29, %%zmm19     \n\t" \

#define mmm_10_32_1(A, K) \
    mmm_8_32_1(A, K) \
    "addq "#K", %%rax                       \n\t" \
    "addq %%rbx, %%rax                      \n\t" \
    "vpbroadcastd (%%rax), %%zmm28          \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%zmm29    \n\t" \
    "vpdpbusd %%zmm26, %%zmm28, %%zmm16     \n\t" \
    "vpdpbusd %%zmm27, %%zmm28, %%zmm17     \n\t" \
    "vpdpbusd %%zmm26, %%zmm29, %%zmm18     \n\t" \
    "vpdpbusd %%zmm27, %%zmm29, %%zmm19     \n\t" \

#define mmm_10_32(A, K) \
    mmm_10_32_0(A, K) \
    mmm_10_32_1(A, K)

#define mmm_11_32(A, K) \
    mmm_10_32_0(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpdpbusd %%zmm24, %%zmm30, %%zmm20     \n\t" \
    "vpdpbusd %%zmm25, %%zmm30, %%zmm21     \n\t" \
    mmm_10_32_1(A, K) \
    "vpbroadcastd (%%rax, "#K", 2), %%zmm30 \n\t" \
    "vpdpbusd %%zmm26, %%zmm30, %%zmm20     \n\t" \
    "vpdpbusd %%zmm27, %%zmm30, %%zmm21     \n\t" \

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
    "vpbroadcastd ("#A"), "#rtype"25              \n\t" \
    "vpbroadcastd 0x4("#A"), "#rtype"26           \n\t" \
    "vmovups (%[B]), "#rtype"31                   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"0   \n\t" \
    "vmovups "#off"(%[B]), "#rtype"24             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"0   \n\t" \

#define mmm_2_16_0(A, K, rtype, off) \
    "movq "#A", %%rax                             \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vmovups (%[B]), "#rtype"31                   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"0   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"1   \n\t" \

#define mmm_2_16_1(A, K, rtype, off) \
    "addq $0x4, %%rax                             \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vmovups "#off"(%[B]), "#rtype"24             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"0   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"1   \n\t" \

#define mmm_2_16(A, K, rtype, off) \
    mmm_2_16_0(A, K, rtype, off) \
    mmm_2_16_1(A, K, rtype, off)

#define mmm_3_16(A, K, rtype, off) \
    mmm_2_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"2   \n\t" \
    mmm_2_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"2   \n\t" \

#define mmm_4_16_0(A, K, rtype, off) \
    "movq "#K", %%rbx                             \n\t" \
    "addq "#K", %%rbx                             \n\t" \
    "addq "#K", %%rbx                             \n\t" \
    mmm_2_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"2   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"3   \n\t" \

#define mmm_4_16_1(A, K, rtype, off) \
    "movq "#A", %%rax                             \n\t" \
    mmm_2_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"2   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"3   \n\t" \

#define mmm_4_16(A, K, rtype, off) \
    mmm_4_16_0(A, K, rtype, off) \
    mmm_4_16_1(A, K, rtype, off)

#define mmm_5_16(A, K, rtype, off) \
    mmm_4_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"4   \n\t" \
    mmm_4_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"4   \n\t" \

#define mmm_6_16_0(A, K, rtype, off) \
    mmm_4_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"4   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"5   \n\t" \

#define mmm_6_16_1(A, K, rtype, off) \
    mmm_4_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"4   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"5   \n\t" \

#define mmm_6_16(A, K, rtype, off) \
    mmm_6_16_0(A, K, rtype, off) \
    mmm_6_16_1(A, K, rtype, off)

#define mmm_7_16(A, K, rtype, off) \
    mmm_6_16_0(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"6   \n\t" \
    mmm_6_16_1(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"6   \n\t" \

#define mmm_8_16_0(A, K, rtype, off) \
    mmm_6_16_0(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"6   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"7   \n\t" \

#define mmm_8_16_1(A, K, rtype, off) \
    mmm_6_16_1(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"6   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"7   \n\t" \

#define mmm_8_16(A, K, rtype, off) \
    mmm_8_16_0(A, K, rtype, off) \
    mmm_8_16_1(A, K, rtype, off)

#define mmm_9_16(A, K, rtype, off) \
    mmm_8_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"8   \n\t" \
    mmm_8_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"8   \n\t" \

#define mmm_10_16_0(A, K, rtype, off) \
    mmm_8_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"8   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"9   \n\t" \

#define mmm_10_16_1(A, K, rtype, off) \
    mmm_8_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"8   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"9   \n\t" \

#define mmm_10_16(A, K, rtype, off) \
    mmm_10_16_0(A, K, rtype, off) \
    mmm_10_16_1(A, K, rtype, off)

#define mmm_11_16(A, K, rtype, off) \
    mmm_10_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"10  \n\t" \
    mmm_10_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"10  \n\t" \

#define mmm_12_16_0(A, K, rtype, off) \
    mmm_10_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"10  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"11  \n\t" \

#define mmm_12_16_1(A, K, rtype, off) \
    mmm_10_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"10  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"11  \n\t" \

#define mmm_12_16(A, K, rtype, off) \
    mmm_12_16_0(A, K, rtype, off) \
    mmm_12_16_1(A, K, rtype, off)

#define mmm_13_16(A, K, rtype, off) \
    mmm_12_16_0(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"12  \n\t" \
    mmm_12_16_1(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"12  \n\t" \

#define mmm_14_16_0(A, K, rtype, off) \
    mmm_12_16_0(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"12  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"13  \n\t" \

#define mmm_14_16_1(A, K, rtype, off) \
    mmm_12_16_1(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"12  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"13  \n\t" \

#define mmm_14_16(A, K, rtype, off) \
    mmm_14_16_0(A, K, rtype, off) \
    mmm_14_16_1(A, K, rtype, off)

#define mmm_15_16(A, K, rtype, off) \
    mmm_14_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"14  \n\t" \
    mmm_14_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"14  \n\t" \

#define mmm_16_16_0(A, K, rtype, off) \
    mmm_14_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"14  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"15  \n\t" \

#define mmm_16_16_1(A, K, rtype, off) \
    mmm_14_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"14  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"15  \n\t" \

#define mmm_16_16(A, K, rtype, off) \
    mmm_16_16_0(A, K, rtype, off) \
    mmm_16_16_1(A, K, rtype, off)

#define mmm_17_16(A, K, rtype, off) \
    mmm_16_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"16  \n\t" \
    mmm_16_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"16  \n\t" \

#define mmm_18_16_0(A, K, rtype, off) \
    mmm_16_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"16  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"17  \n\t" \

#define mmm_18_16_1(A, K, rtype, off) \
    mmm_16_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"16  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"17  \n\t" \

#define mmm_18_16(A, K, rtype, off) \
    mmm_18_16_0(A, K, rtype, off) \
    mmm_18_16_1(A, K, rtype, off)

#define mmm_19_16(A, K, rtype, off) \
    mmm_18_16_0(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"18  \n\t" \
    mmm_18_16_1(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"18  \n\t" \

#define mmm_20_16_0(A, K, rtype, off) \
    mmm_18_16_0(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"18  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"19  \n\t" \

#define mmm_20_16_1(A, K, rtype, off) \
    mmm_18_16_1(A, K, rtype, off) \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"18  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"19  \n\t" \

#define mmm_20_16(A, K, rtype, off) \
    mmm_20_16_0(A, K, rtype, off) \
    mmm_20_16_1(A, K, rtype, off)

#define mmm_21_16(A, K, rtype, off) \
    mmm_20_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"20  \n\t" \
    mmm_20_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"20  \n\t" \

#define mmm_22_16_0(A, K, rtype, off) \
    mmm_20_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"20  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"21  \n\t" \

#define mmm_22_16_1(A, K, rtype, off) \
    mmm_20_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"20  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"21  \n\t" \

#define mmm_22_16(A, K, rtype, off) \
    mmm_22_16_0(A, K, rtype, off) \
    mmm_22_16_1(A, K, rtype, off)

#define mmm_23_16(A, K, rtype, off) \
    mmm_22_16_0(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"22  \n\t" \
    mmm_22_16_1(A, K, rtype, off) \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"22  \n\t" \

#define mmm_24_16(A, K, rtype, off) \
    "movq "#A", %%rax                             \n\t" \
    "movq "#K", %%rbx                             \n\t" \
    "addq "#K", %%rbx                             \n\t" \
    "addq "#K", %%rbx                             \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "prefetcht0 0x80(%[B])                        \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"0   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"1   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"2   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"3   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"4   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"5   \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"6   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"7   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"8   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"9   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"10  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"11  \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vmovups (%[B]), "#rtype"31                   \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"12  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"13  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"14  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"15  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"16  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"17  \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"25, "#rtype"18  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"26, "#rtype"19  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"27, "#rtype"20  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"28, "#rtype"21  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"29, "#rtype"22  \n\t" \
    "vpdpbusd "#rtype"24, "#rtype"30, "#rtype"23  \n\t" \
    "movq "#A", %%rax                             \n\t" \
    "addq $0x4, %%rax                             \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "prefetcht0 0xC0(%[B])                        \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"0   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"1   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"2   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"3   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"4   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"5   \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"6   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"7   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"8   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"9   \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"10  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"11  \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vmovups "#off"(%[B]), "#rtype"24             \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"12  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"13  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"14  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"15  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"16  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"17  \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"25             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"26       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"27    \n\t" \
    "addq %%rbx, %%rax                            \n\t" \
    "vpbroadcastd (%%rax), "#rtype"28             \n\t" \
    "vpbroadcastd (%%rax, "#K"), "#rtype"29       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), "#rtype"30    \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"25, "#rtype"18  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"26, "#rtype"19  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"27, "#rtype"20  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"28, "#rtype"21  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"29, "#rtype"22  \n\t" \
    "vpdpbusd "#rtype"31, "#rtype"30, "#rtype"23  \n\t" \

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
        loadOffset_##m##_##n(zmm)                                                     \
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
        addC_##m##_##n(zmm, %%rax, 0x40, 0x80)                                                     \
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
        loadOffset_##m##_##n(zmm)                                                     \
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
        addC_##m##_##n(zmm, %[C], 0x40, 0x80)                                                     \
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
          [flags] "r" ((int64_t)flags),                                    \
          [u8C] "r" (u8Result),                                             \
          [nmask] "r" (nmask),                                             \
          [resK] "r" (resK)                                             \
        : "%rax", "%rcx", "%rbx",                                                  \
          "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5",            \
          "%zmm6", "%zmm7", "%zmm8", "%zmm9", "%zmm10", "%zmm11",          \
          "%zmm12", "%zmm13", "%zmm14", "%zmm15", "%zmm16", "%zmm17",      \
          "%zmm18", "%zmm19", "%zmm20", "%zmm21", "%zmm22", "%zmm23",      \
          "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28", "%zmm29",      \
          "%zmm30", "%zmm31", "memory", "cc");

#define mmm_m_16_8_asm(m, n, nRegs, mRegs, rtype, token, off0, off1, edge) \
    __asm__ __volatile__(                                                  \
        "vmovups (%[B]), "#rtype"24                                     \n\t" \
        "add $"#off0", %[B]                                             \n\t" \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x1, %%rax                                            \n\t" \
        "jne 0f                                                      \n\t" \
        loadOffset_##m##_##n(token)                                                     \
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
        addC_##m##_##n(token, %[C], 0x40, 0x80)                                                     \
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
    mmm_m_16_8_asm(m, n, nRegs, mRegs, %%zmm, zmm, 0x40, 0x80, edge)

#define mmm_m_8_asm(m, n, nRegs, mRegs, edge) \
    mmm_m_16_8_asm(m, n, nRegs, mRegs, %%ymm, ymm, 0x20, 0x40, edge)

#define mmm_m_n_asm(m, n, nRegs, mRegs, regs) \
    void asm_##mRegs##x##n(U32 um, \
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
mmm_m_n_asm(7, 48, 3, 7, 21)
mmm_m_n_asm(6, 48, 3, 6, 18)
mmm_m_n_asm(5, 48, 3, 5, 15)
mmm_m_n_asm(4, 48, 3, 4, 12)
mmm_m_n_asm(3, 48, 3, 3, 9)
mmm_m_n_asm(2, 48, 3, 2, 6)
mmm_m_n_asm(1, 48, 3, 1, 3)

mmm_m_n_asm(12, 32, 2, 12, 24)
mmm_m_n_asm(11, 32, 2, 11, 22)
mmm_m_n_asm(10, 32, 2, 10, 20)
mmm_m_n_asm(9, 32, 2, 9, 18)
mmm_m_n_asm(8, 32, 2, 8, 16)
mmm_m_n_asm(7, 32, 2, 7, 14)
mmm_m_n_asm(6, 32, 2, 6, 12)
mmm_m_n_asm(5, 32, 2, 5, 10)
mmm_m_n_asm(4, 32, 2, 4, 8)
mmm_m_n_asm(3, 32, 2, 3, 6)
mmm_m_n_asm(2, 32, 2, 2, 4)
mmm_m_n_asm(1, 32, 2, 1, 2)

mmm_m_n_asm(24, 16, 1, 24, 24)
mmm_m_n_asm(23, 16, 1, 23, 23)
mmm_m_n_asm(22, 16, 1, 22, 22)
mmm_m_n_asm(21, 16, 1, 21, 21)
mmm_m_n_asm(20, 16, 1, 20, 20)
mmm_m_n_asm(19, 16, 1, 19, 19)
mmm_m_n_asm(18, 16, 1, 18, 18)
mmm_m_n_asm(17, 16, 1, 17, 17)
mmm_m_n_asm(16, 16, 1, 16, 16)
mmm_m_n_asm(15, 16, 1, 15, 15)
mmm_m_n_asm(14, 16, 1, 14, 14)
mmm_m_n_asm(13, 16, 1, 13, 13)
mmm_m_n_asm(12, 16, 1, 12, 12)
mmm_m_n_asm(11, 16, 1, 11, 11)
mmm_m_n_asm(10, 16, 1, 10, 10)
mmm_m_n_asm(9, 16, 1, 9, 9)
mmm_m_n_asm(8, 16, 1, 8, 8)
mmm_m_n_asm(7, 16, 1, 7, 7)
mmm_m_n_asm(6, 16, 1, 6, 6)
mmm_m_n_asm(5, 16, 1, 5, 5)
mmm_m_n_asm(4, 16, 1, 4, 4)
mmm_m_n_asm(3, 16, 1, 3, 3)
mmm_m_n_asm(2, 16, 1, 2, 2)
mmm_m_n_asm(1, 16, 1, 1, 1)

mmm_m_n_asm(24, 8, 1, 24, 24)
mmm_m_n_asm(23, 8, 1, 23, 23)
mmm_m_n_asm(22, 8, 1, 22, 22)
mmm_m_n_asm(21, 8, 1, 21, 21)
mmm_m_n_asm(20, 8, 1, 20, 20)
mmm_m_n_asm(19, 8, 1, 19, 19)
mmm_m_n_asm(18, 8, 1, 18, 18)
mmm_m_n_asm(17, 8, 1, 17, 17)
mmm_m_n_asm(16, 8, 1, 16, 16)
mmm_m_n_asm(15, 8, 1, 15, 15)
mmm_m_n_asm(14, 8, 1, 14, 14)
mmm_m_n_asm(13, 8, 1, 13, 13)
mmm_m_n_asm(12, 8, 1, 12, 12)
mmm_m_n_asm(11, 8, 1, 11, 11)
mmm_m_n_asm(10, 8, 1, 10, 10)
mmm_m_n_asm(9, 8, 1, 9, 9)
mmm_m_n_asm(8, 8, 1, 8, 8)
mmm_m_n_asm(7, 8, 1, 7, 7)
mmm_m_n_asm(6, 8, 1, 6, 6)
mmm_m_n_asm(5, 8, 1, 5, 5)
mmm_m_n_asm(4, 8, 1, 4, 4)
mmm_m_n_asm(3, 8, 1, 3, 3)
mmm_m_n_asm(2, 8, 1, 2, 2)
mmm_m_n_asm(1, 8, 1, 1, 1)

// clang-format on
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
        {asm_1x8, asm_1x16, asm_1x32, asm_1x48}, {asm_2x8, asm_2x16, asm_2x32, asm_2x48},
        {asm_3x8, asm_3x16, asm_3x32, asm_3x48}, {asm_4x8, asm_4x16, asm_4x32, asm_4x48},
        {asm_5x8, asm_5x16, asm_5x32, asm_5x48}, {asm_6x8, asm_6x16, asm_6x32, asm_6x48},
        {asm_7x8, asm_7x16, asm_7x32, asm_7x48}, {asm_8x8, asm_8x16, asm_8x32, asm_8x48},
        {asm_9x8, asm_9x16, asm_9x32, asm_8x48}, {asm_10x8, asm_10x16, asm_10x32, asm_8x48},
        {asm_11x8, asm_11x16, asm_11x32, asm_8x48}, {asm_12x8, asm_12x16, asm_12x32, asm_8x48},
        {asm_13x8, asm_13x16, asm_12x32, asm_8x48}, {asm_14x8, asm_14x16, asm_12x32, asm_8x48},
        {asm_15x8, asm_15x16, asm_12x32, asm_8x48}, {asm_16x8, asm_16x16, asm_12x32, asm_8x48},
        {asm_17x8, asm_17x16, asm_12x32, asm_8x48}, {asm_18x8, asm_18x16, asm_12x32, asm_8x48},
        {asm_19x8, asm_19x16, asm_12x32, asm_8x48}, {asm_20x8, asm_20x16, asm_12x32, asm_8x48},
        {asm_21x8, asm_21x16, asm_12x32, asm_8x48}, {asm_22x8, asm_22x16, asm_12x32, asm_8x48},
        {asm_23x8, asm_23x16, asm_12x32, asm_8x48}, {asm_24x8, asm_24x16, asm_12x32, asm_8x48}};
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
    auto getMNum = [](U32 mDim, U32 unrollM) { return (mDim + unrollM - 1) / unrollM; };

    U32 resN = N % UNROLL_N;
    U32 edgeNSize = (resN > 8) ? UNI_ALIGN(resN, 16) : 8;
    U32 BlockMDim = UNI_ALIGN(BOLCK_M_DIM, UNROLL_M);
    U32 mloopNum = getMNum(M, UNROLL_M);

    U32 newUnrollM = unrollMSizes[edgeNSize >> 4];
    U32 resMloopNum = getMNum(M, newUnrollM);
    UINT8 *tmpK = tmp;
    U32 resK = K % SIMDW;
    if (resK > 0 && matrix1Df == DF_NORMAL) {
        for (U32 i = 0; i < M; ++i) {
            UNI_MEMCPY(tmpK + i * SIMDW, packA + (i + 1) * K - resK, resK);
            UNI_MEMSET(tmpK + i * SIMDW + resK, 128, SIMDW - resK);
        }
        tmp += M * SIMDW;
    }
    U32 mNNum = N / UNROLL_N;
    U32 alginedN = mNNum * UNROLL_N + (resN > 0) * edgeNSize;
    U32 nmask = pow(2, N % 16) - 1;
    U32 loopNum = mNNum * mloopNum + (resN > 0) * resMloopNum;
    U32 bmLoopNum =
        mNNum * getMNum(BlockMDim, UNROLL_M) + (resN > 0) * getMNum(BOLCK_M_DIM, newUnrollM);

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
#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
        for (U32 l = 0; l < loopNum; ++l) {
            U32 bm = l / bmLoopNum * BOLCK_M_DIM;
            U32 blockSizeM = UNI_MIN(BOLCK_M_DIM, M - bm);
            U32 mMNum = getMNum(blockSizeM, UNROLL_M);
            U32 bn = l % bmLoopNum;
            U32 nLoop = bn / mMNum;
            U32 n = nLoop * UNROLL_N;
            U32 m = (bn % mMNum) * UNROLL_M;
            U32 unrollM = UNROLL_M;
            U32 nSize = UNROLL_N;
            if (bn >= mNNum * mMNum) {
                nLoop = mNNum;
                n = mNNum * UNROLL_N;
                m = (bn - mNNum * mMNum) * newUnrollM;
                unrollM = newUnrollM;
                nSize = edgeNSize;
            }

            U32 rm = UNI_MIN(unrollM, blockSizeM - m);
            INT8 *curB = packB + k * alginedN + n * UNI_ALIGN(realK, SIMDW);
            UINT8 *curA = packA + (m + bm) * stepK + k;
            UINT8 *kpad = tmpK + (m + bm) * SIMDW;
            U32 tnmask = (nLoop == mNNum - 1 + (resN > 0)) ? nmask : 0;
            kernel[rm - 1][nSize >> 4](rm, nSize, realK, curA, curB,
                i32Result + (m + bm) * N + n, u8Result + (m + bm) * N + n, offsetC + n, N,
                stepK, useFactor, tnmask, kpad, flags);
        }
    }

    return SUCCESS;
}
