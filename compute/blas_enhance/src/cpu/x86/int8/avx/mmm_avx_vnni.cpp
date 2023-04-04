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

#define UNROLL_N 24
#define UNROLL_M 4
#define BOLCK_M_DIM 768
#define BOLCK_K_DIM 1024
#define SIMDW 8

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
    U32 unrollSize[3] = {8, 16, 24};
    INT8 *tmpS = src;
    I32 *offsetCBias = (I32 *)(packB + UNI_ALIGN(K, SIMDW) * UNI_ALIGN(N, 16));

    for (U32 bk = 0; bk < K; bk += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        for (U32 un = 0; un < N; un += unrollSizeN) {
            unrollSizeN = UNI_MIN(UNROLL_N, N - un);
            U32 alignedN = UNI_ALIGN(unrollSizeN, 8);
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
    U32 unrollSize[3] = {8, 16, 24};
    INT8 *tmpS = src;
    I32 *offsetCBias = (I32 *)(packB + UNI_ALIGN(K, SIMDW) * UNI_ALIGN(N, 16));

    for (U32 bk = 0; bk < K; bk += blockSizeK) {
        blockSizeK = UNI_MIN(BOLCK_K_DIM, K - bk);
        for (U32 un = 0; un < N; un += unrollSizeN) {
            unrollSizeN = UNI_MIN(UNROLL_N, N - un);
            U32 alignedN = UNI_ALIGN(unrollSizeN, 8);
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
    I32 *maskPtr,
    UINT8 *resK,
    U32 flags);

#define storeC_1_1_1(op, rtype, C, off0, off1) \
    "movq "#C", %%rax  \n\t" \
    "vmovups (%[maskPtr]), "#rtype"15  \n\t" \
    "vmaskmovps "#rtype"0, "#rtype"15, (%%rax)       \n\t"

#define storeC_2_1_1(op, rtype, C, off0, off1) \
    storeC_1_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"1, "#rtype"15, (%%rax)       \n\t"

#define storeC_3_1_1(op, rtype, C, off0, off1) \
    storeC_2_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"2, "#rtype"15, (%%rax)       \n\t"

#define storeC_4_1_1(op, rtype, C, off0, off1) \
    storeC_3_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"3, "#rtype"15, (%%rax)       \n\t"

#define storeC_5_1_1(op, rtype, C, off0, off1) \
    storeC_4_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"4, "#rtype"15, (%%rax)       \n\t"

#define storeC_6_1_1(op, rtype, C, off0, off1) \
    storeC_5_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"5, "#rtype"15, (%%rax)       \n\t"

#define storeC_7_1_1(op, rtype, C, off0, off1) \
    storeC_6_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"6, "#rtype"15, (%%rax)       \n\t"

#define storeC_8_1_1(op, rtype, C, off0, off1) \
    storeC_7_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"7, "#rtype"15, (%%rax)       \n\t"

#define storeC_9_1_1(op, rtype, C, off0, off1) \
    storeC_8_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"8, "#rtype"15, (%%rax)       \n\t"

#define storeC_10_1_1(op, rtype, C, off0, off1) \
    storeC_9_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"9, "#rtype"15, (%%rax)       \n\t"

#define storeC_11_1_1(op, rtype, C, off0, off1) \
    storeC_10_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"10, "#rtype"15, (%%rax)       \n\t"

#define storeC_12_1_1(op, rtype, C, off0, off1) \
    storeC_11_1_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmaskmovps "#rtype"11, "#rtype"15, (%%rax)       \n\t"

#define storeC_1_2_1(op, rtype, C, off0, off1) \
    "movq "#C", %%rax  \n\t" \
    "vmovups (%[maskPtr]), "#rtype"15  \n\t" \
    "vmovups "#rtype"0, (%%rax)       \n\t" \
    "vmaskmovps "#rtype"1, "#rtype"15, "#off0"(%%rax)       \n\t"

#define storeC_2_2_1(op, rtype, C, off0, off1) \
    storeC_1_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmovups "#rtype"2, (%%rax)                        \n\t" \
    "vmaskmovps "#rtype"3, "#rtype"15, "#off0"(%%rax)                    \n\t"

#define storeC_3_2_1(op, rtype, C, off0, off1) \
    storeC_2_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmovups "#rtype"4, (%%rax)                        \n\t" \
    "vmaskmovps "#rtype"5, "#rtype"15, "#off0"(%%rax)                    \n\t"

#define storeC_4_2_1(op, rtype, C, off0, off1) \
    storeC_3_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmovups "#rtype"6, (%%rax)                        \n\t" \
    "vmaskmovps "#rtype"7, "#rtype"15, "#off0"(%%rax)                    \n\t"

#define storeC_5_2_1(op, rtype, C, off0, off1) \
    storeC_4_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmovups "#rtype"8, (%%rax)                        \n\t" \
    "vmaskmovps "#rtype"9, "#rtype"15, "#off0"(%%rax)                    \n\t"

#define storeC_6_2_1(op, rtype, C, off0, off1) \
    storeC_5_2_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmovups "#rtype"10, (%%rax)                        \n\t" \
    "vmaskmovps "#rtype"11, "#rtype"15, "#off0"(%%rax)                    \n\t"

#define storeC_1_3_1(op, rtype, C, off0, off1) \
    "movq "#C", %%rax  \n\t" \
    "vmovups (%[maskPtr]), "#rtype"15  \n\t" \
    "vmovups "#rtype"0, (%%rax)       \n\t" \
    "vmovups "#rtype"1, "#off0"(%%rax)       \n\t" \
    "vmaskmovps "#rtype"2, "#rtype"15, "#off1"(%%rax)       \n\t"

#define storeC_2_3_1(op, rtype, C, off0, off1) \
    storeC_1_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmovups "#rtype"3, (%%rax)                        \n\t" \
    "vmovups "#rtype"4, "#off0"(%%rax)                    \n\t" \
    "vmaskmovps "#rtype"5, "#rtype"15, "#off1"(%%rax)                    \n\t"

#define storeC_3_3_1(op, rtype, C, off0, off1) \
    storeC_2_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmovups "#rtype"6, (%%rax)                        \n\t" \
    "vmovups "#rtype"7, "#off0"(%%rax)                    \n\t" \
    "vmaskmovps "#rtype"8, "#rtype"15, "#off1"(%%rax)                    \n\t"

#define storeC_4_3_1(op, rtype, C, off0, off1) \
    storeC_3_3_1(op, rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vmovups "#rtype"9, (%%rax)                        \n\t" \
    "vmovups "#rtype"10, "#off0"(%%rax)                    \n\t" \
    "vmaskmovps "#rtype"11, "#rtype"15, "#off1"(%%rax)                    \n\t"

#define mmm_1_24(A, K) \
    "vpbroadcastd ("#A"), %%ymm15          \n\t" \
    "vmovups (%[B]), %%ymm3               \n\t" \
    "vmovups 0x20(%[B]), %%ymm4           \n\t" \
    "vmovups 0x40(%[B]), %%ymm5           \n\t" \
    "vmovups 0x60(%[B]), %%ymm6           \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm14       \n\t" \
    "%{vex%} vpdpbusd %%ymm3, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm4, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm5, %%ymm15, %%ymm2     \n\t" \
    "vmovups 0x80(%[B]), %%ymm7          \n\t" \
    "vmovups 0xA0(%[B]), %%ymm8          \n\t" \
    "%{vex%} vpdpbusd %%ymm6, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm7, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm8, %%ymm14, %%ymm2     \n\t"

#define mmm_2_24(A, K) \
    "vpbroadcastd ("#A"), %%ymm15          \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm14       \n\t" \
    "vmovups (%[B]), %%ymm6               \n\t" \
    "vmovups 0x20(%[B]), %%ymm7           \n\t" \
    "vmovups 0x40(%[B]), %%ymm8           \n\t" \
    "%{vex%} vpdpbusd %%ymm6, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm7, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm8, %%ymm15, %%ymm2     \n\t" \
    "vmovups 0x60(%[B]), %%ymm9           \n\t" \
    "vmovups 0x80(%[B]), %%ymm10          \n\t" \
    "vmovups 0xA0(%[B]), %%ymm11          \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm13          \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm12       \n\t" \
    "%{vex%} vpdpbusd %%ymm9, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm6, %%ymm13, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm7, %%ymm13, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm8, %%ymm13, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm9, %%ymm12, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm12, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm5     \n\t"

#define mmm_3_24(A, K) \
    "vpbroadcastd ("#A"), %%ymm15          \n\t" \
    "vmovups (%[B]), %%ymm9               \n\t" \
    "vmovups 0x20(%[B]), %%ymm10           \n\t" \
    "vmovups 0x40(%[B]), %%ymm11           \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm14       \n\t" \
    "vpbroadcastd ("#A", "#K", 2), %%ymm13       \n\t" \
    "%{vex%} vpdpbusd %%ymm9, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm9, %%ymm14, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm9, %%ymm13, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm13, %%ymm7     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm8     \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm15          \n\t" \
    "vmovups 0x60(%[B]), %%ymm9           \n\t" \
    "vmovups 0x80(%[B]), %%ymm10          \n\t" \
    "vmovups 0xA0(%[B]), %%ymm11          \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm14       \n\t" \
    "vpbroadcastd 0x4("#A", "#K", 2), %%ymm13       \n\t" \
    "%{vex%} vpdpbusd %%ymm9, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm9, %%ymm14, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm9, %%ymm13, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm13, %%ymm7     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm8     \n\t"

#define mmm_4_24(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm15          \n\t" \
    "vmovups (%[B]), %%ymm12               \n\t" \
    "vmovups 0x20(%[B]), %%ymm13           \n\t" \
    "vmovups 0x40(%[B]), %%ymm14           \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm15, %%ymm2     \n\t" \
    "vpbroadcastd (%%rax), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm15, %%ymm5     \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm7     \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm15, %%ymm8     \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm9     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm10     \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm15, %%ymm11     \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm15          \n\t" \
    "vmovups 0x60(%[B]), %%ymm12               \n\t" \
    "vmovups 0x80(%[B]), %%ymm13           \n\t" \
    "vmovups 0xA0(%[B]), %%ymm14           \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm15, %%ymm2     \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm15, %%ymm5     \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm7     \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm15, %%ymm8     \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm15          \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm9     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm10     \n\t" \
    "%{vex%} vpdpbusd %%ymm14, %%ymm15, %%ymm11     \n\t" \

#define mmm_1_16(A, K) \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm15      \n\t" \
    "vmovups (%[B]), %%ymm10               \n\t" \
    "vmovups 0x20(%[B]), %%ymm11           \n\t" \
    "vmovups 0x40(%[B]), %%ymm12           \n\t" \
    "vmovups 0x60(%[B]), %%ymm13           \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm1     \n\t"

#define mmm_2_16(A, K) \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm8         \n\t" \
    "vmovups (%[B]), %%ymm10               \n\t" \
    "vmovups 0x20(%[B]), %%ymm11           \n\t" \
    "vmovups 0x40(%[B]), %%ymm12           \n\t" \
    "vmovups 0x60(%[B]), %%ymm13           \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm15      \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm9      \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm8, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm8, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm9, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm9, %%ymm3     \n\t" \

#define mmm_3_16(A, K) \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm8         \n\t" \
    "vmovups (%[B]), %%ymm10               \n\t" \
    "vmovups 0x20(%[B]), %%ymm11           \n\t" \
    "vmovups 0x40(%[B]), %%ymm12           \n\t" \
    "vmovups 0x60(%[B]), %%ymm13           \n\t" \
    "vpbroadcastd ("#A", "#K", 2), %%ymm6         \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm15      \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm9      \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm8, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm8, %%ymm3     \n\t" \
    "vpbroadcastd 0x4("#A", "#K", 2), %%ymm7      \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm6, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm6, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm9, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm9, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm7, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm7, %%ymm5     \n\t" \

#define mmm_4_16(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd (%%rax), %%ymm15         \n\t" \
    "vmovups (%[B]), %%ymm10               \n\t" \
    "vmovups 0x20(%[B]), %%ymm11           \n\t" \
    "vmovups 0x40(%[B]), %%ymm12           \n\t" \
    "vmovups 0x60(%[B]), %%ymm13           \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm3     \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm8         \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm9         \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm14      \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm15      \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm8, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm8, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm9, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm9, %%ymm7     \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm8      \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm9      \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm8, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm8, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm9, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm9, %%ymm7     \n\t" \

#define mmm_5_16(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm15         \n\t" \
    "vpbroadcastd (%%rax), %%ymm12         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm13         \n\t" \
    "vmovups (%[B]), %%ymm10               \n\t" \
    "vmovups 0x20(%[B]), %%ymm11           \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm3     \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm14         \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm12, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm13, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm7     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm8     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm9     \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm12         \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm13         \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm14         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm15         \n\t" \
    "vmovups 0x40(%[B]), %%ymm10               \n\t" \
    "vmovups 0x60(%[B]), %%ymm11           \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm12, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm13, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm3     \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm12         \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm7     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm12, %%ymm8     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm9     \n\t" \

#define mmm_6_16(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm15         \n\t" \
    "vmovups (%[B]), %%ymm12               \n\t" \
    "vmovups 0x20(%[B]), %%ymm13           \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm0     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm14, %%ymm1     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm3     \n\t" \
    "vpbroadcastd ("#A", "#K", 2), %%ymm14         \n\t" \
    "vpbroadcastd (%%rax), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm14, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm7     \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm14         \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm8     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm14, %%ymm9     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm10     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm11     \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm14         \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm15         \n\t" \
    "vmovups 0x40(%[B]), %%ymm12               \n\t" \
    "vmovups 0x60(%[B]), %%ymm13           \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm14, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm3     \n\t" \
    "vpbroadcastd 0x4("#A", "#K", 2), %%ymm14         \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm14, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm7     \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm14         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm8     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm14, %%ymm9     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm10     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm11     \n\t" \

#define mmm_1_8(A, K) \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm15         \n\t" \
    "vmovups (%[B]), %%ymm12               \n\t" \
    "vmovups 0x20(%[B]), %%ymm13               \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm0     \n\t" \

#define mmm_2_8(A, K) \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm15         \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm10         \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm11         \n\t" \
    "vmovups (%[B]), %%ymm12               \n\t" \
    "vmovups 0x20(%[B]), %%ymm13               \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm10, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm11, %%ymm1     \n\t" \

#define mmm_3_8(A, K) \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm15         \n\t" \
    "vpbroadcastd ("#A", "#K", 2), %%ymm8         \n\t" \
    "vmovups (%[B]), %%ymm12               \n\t" \
    "vmovups 0x20(%[B]), %%ymm13               \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm8, %%ymm2     \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm10         \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm11         \n\t" \
    "vpbroadcastd 0x4("#A", "#K", 2), %%ymm9         \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm10, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm11, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm9, %%ymm2     \n\t" \

#define mmm_4_8(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "vpbroadcastd (%%rax), %%ymm15         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm8         \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm7         \n\t" \
    "vmovups (%[B]), %%ymm12               \n\t" \
    "vmovups 0x20(%[B]), %%ymm13               \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm8, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm7, %%ymm3     \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm10         \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm11         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm9         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm6         \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm10, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm11, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm9, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm6, %%ymm3     \n\t" \

#define mmm_5_8(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm14         \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm15         \n\t" \
    "vpbroadcastd (%%rax), %%ymm11         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm10         \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm9         \n\t" \
    "vmovups (%[B]), %%ymm12               \n\t" \
    "vmovups 0x20(%[B]), %%ymm13               \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm11, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm10, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm12, %%ymm9, %%ymm4     \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm14         \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm15         \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm11         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm10         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm9         \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm15, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm11, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm10, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm13, %%ymm9, %%ymm4     \n\t" \

#define mmm_6_8(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm12         \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A", "#K"), %%ymm13         \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A", "#K", 2), %%ymm14         \n\t" \
    "vpbroadcastd (%%rax), %%ymm15         \n\t" \
    "vmovups (%[B]), %%ymm11               \n\t" \
    "vmovups 0x20(%[B]), %%ymm10               \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm3     \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm12         \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm13        \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm14         \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm1     \n\t" \
    "vpbroadcastd 0x4("#A", "#K", 2), %%ymm12         \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm13         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm14         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm12, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm13, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm5     \n\t" \

#define mmm_7_8(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm12         \n\t" \
    "vpbroadcastd (%%rax), %%ymm13         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm14         \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm15         \n\t" \
    "vmovups (%[B]), %%ymm11               \n\t" \
    "vmovups 0x20(%[B]), %%ymm10               \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm0     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm1     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm2     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm3     \n\t" \
    "vpbroadcastd (%%rax), %%ymm12         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm13        \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm14        \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm15         \n\t" \
    "vpbroadcastd 0x4("#A", "#K"), %%ymm9         \n\t" \
    "vpbroadcastd 0x4("#A", "#K", 2), %%ymm8         \n\t" \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm5     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm0     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm9, %%ymm1     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm8, %%ymm2     \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm12         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm13         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm14         \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm12, %%ymm3     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm13, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm6     \n\t" \

#define mmm_8_8(A, K) \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd ("#A"), %%ymm12         \n\t" \
    "vpbroadcastd (%%rax), %%ymm13         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm14         \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm15         \n\t" \
    "vmovups (%[B]), %%ymm11               \n\t" \
    "vmovups 0x20(%[B]), %%ymm10               \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm0     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm1     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm2     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm3     \n\t" \
    "vpbroadcastd (%%rax), %%ymm12         \n\t" \
    "vpbroadcastd (%%rax, "#K"), %%ymm13        \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm14        \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd (%%rax, "#K", 2), %%ymm15        \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm12, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm13, %%ymm5     \n\t" \
    "movq "#A", %%rax                       \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm14, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm11, %%ymm15, %%ymm7     \n\t" \
    "vpbroadcastd 0x4("#A"), %%ymm12         \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm13         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm14         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm12, %%ymm0     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm13, %%ymm1     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm2     \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm3     \n\t" \
    "vpbroadcastd 0x4(%%rax), %%ymm12         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K"), %%ymm13         \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm14         \n\t" \
    "addq "#K", %%rax                       \n\t" \
    "vpbroadcastd 0x4(%%rax, "#K", 2), %%ymm15         \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm12, %%ymm4     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm13, %%ymm5     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm14, %%ymm6     \n\t" \
    "%{vex%} vpdpbusd %%ymm10, %%ymm15, %%ymm7     \n\t" \

#define mmm_m_24_asm(m, n, nRegs, mRegs, edge) \
    __asm__ __volatile__(                                                  \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x1, %%rax                                            \n\t" \
        "jne 0f                                                      \n\t" \
        loadOffset_##m##_##n(ymm)                                                     \
        "jmp 1f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "0:                                                          \n\t" \
        clear##nRegs##Regs(%%ymm)                                                 \
        ".align 16                                                   \n\t" \
        "1:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "shr $3, %%rcx                                \n\t" \
        "je 3f                                        \n\t" \
        ".align 16                                                   \n\t" \
        "2:                                                          \n\t" \
        mmm_##m##_24(%[A], %[K])                                                     \
        "add $0xC0, %[B]                                            \n\t" \
        "add $0x8, %[A]                                              \n\t" \
        "dec %%rcx                                                   \n\t" \
        "jg 2b                                                       \n\t" \
        ".align 16                                                   \n\t" \
        "3:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "and $7, %%rcx                                \n\t" \
        "je 4f                                        \n\t" \
        "movq $8, %%rcx                                           \n\t" \
        mmm_##m##_24(%[resK], %%rcx)                                                     \
        ".align 16                                                   \n\t" \
        "4:                                                          \n\t" \
        "movq %[C], %%rax  \n\t" \
        "movq %[N], %%rcx  \n\t" \
        "addq %[N], %%rcx                                     \n\t" \
        addC_##m##_##n(ymm, %%rax, 0x20, 0x40)                                                     \
        "cmpq $0x0, %[s]                                             \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##I32Regs2Ps(%%ymm, %[s])                                   \
        "5:                                                          \n\t" \
        storeC_##mRegs##_##n##_##edge(vmovups, %%ymm, %[C], 0x20, 0x40)                       \
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
          [maskPtr] "r" (maskPtr),                                             \
          [resK] "r" (resK)                                             \
        : "%rax", "%rcx",                                                  \
          "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",            \
          "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",          \
          "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");

#define mmm_m_16_asm(m, n, nRegs, mRegs, edge) \
    __asm__ __volatile__(                                                  \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x1, %%rax                                            \n\t" \
        "jne 0f                                                      \n\t" \
        loadOffset_##m##_##n(ymm)                                                     \
        "jmp 1f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "0:                                                          \n\t" \
        clear##nRegs##Regs(%%ymm)                                                 \
        ".align 16                                                   \n\t" \
        "1:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "shr $3, %%rcx                                \n\t" \
        "je 3f                                        \n\t" \
        ".align 16                                                   \n\t" \
        "2:                                                          \n\t" \
        mmm_##m##_16(%[A], %[K])                                                     \
        "add $0x80, %[B]                                            \n\t" \
        "add $0x8, %[A]                                              \n\t" \
        "dec %%rcx                                                   \n\t" \
        "jg 2b                                                       \n\t" \
        ".align 16                                                   \n\t" \
        "3:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "and $7, %%rcx                                \n\t" \
        "je 4f                                        \n\t" \
        "movq $8, %%rcx                                           \n\t" \
        mmm_##m##_16(%[resK], %%rcx)                                                     \
        ".align 16                                                   \n\t" \
        "4:                                                          \n\t" \
        addC_##m##_##n(ymm, %[C], 0x20, 0x40)                                                     \
        "cmpq $0x0, %[s]                                             \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##I32Regs2Ps(%%ymm, %[s])                                   \
        ".align 16                                                   \n\t" \
        "5:                                                          \n\t" \
        storeC_##mRegs##_##n##_##edge(vmovups, %%ymm, %[C], 0x20, 0x40)                       \
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
          [maskPtr] "r" (maskPtr),                                             \
          [resK] "r" (resK)                                             \
        : "%rax", "%rcx", "%rbx",                                                  \
          "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",            \
          "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",          \
          "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");

#define mmm_m_8_asm(m, n, nRegs, mRegs, edge) \
    __asm__ __volatile__(                                                  \
        "movq %[flags], %%rax                                        \n\t" \
        "andq $0x1, %%rax                                            \n\t" \
        "jne 0f                                                      \n\t" \
        loadOffset_##m##_##n(ymm)                                                     \
        "jmp 1f                                                      \n\t" \
        ".align 16                                                   \n\t" \
        "0:                                                          \n\t" \
        clear##nRegs##Regs(%%ymm)                                                 \
        ".align 16                                                   \n\t" \
        "1:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "shr $3, %%rcx                                \n\t" \
        "je 3f                                        \n\t" \
        ".align 16                                                   \n\t" \
        "2:                                                          \n\t" \
        mmm_##m##_8(%[A], %[K])                                                     \
        "add $0x40, %[B]                                            \n\t" \
        "add $0x8, %[A]                                              \n\t" \
        "dec %%rcx                                                   \n\t" \
        "jg 2b                                                       \n\t" \
        ".align 16                                                   \n\t" \
        "3:                                                          \n\t" \
        "movq %[bk], %%rcx                                           \n\t" \
        "and $7, %%rcx                                \n\t" \
        "je 4f                                        \n\t" \
        "movq $8, %%rcx                                           \n\t" \
        mmm_##m##_8(%[resK], %%rcx)                                                     \
        ".align 16                                                   \n\t" \
        "4:                                                          \n\t" \
        addC_##m##_##n(ymm, %[C], 0, 0)                                                     \
        "cmpq $0x0, %[s]                                             \n\t" \
        "je 5f                                                       \n\t" \
        convert##nRegs##I32Regs2Ps(%%ymm, %[s])                                   \
        ".align 16                                                   \n\t" \
        "5:                                                          \n\t" \
        storeC_##mRegs##_##n##_##edge(vmovups, %%ymm, %[C], 0x0, 0x0)                       \
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
          [maskPtr] "r" (maskPtr),                                             \
          [resK] "r" (resK)                                             \
        : "%rax", "%rbx", "%rcx",                                                  \
          "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5",            \
          "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",          \
          "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");

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
        I32* maskPtr, \
        UINT8 *resK, \
        U32 flags) \
    { \
        if (maskPtr == nullptr) { \
            mmm_m_##n##_asm(m, nRegs, regs, mRegs, 0) \
        } else { \
            mmm_m_##n##_asm(m, nRegs, regs, mRegs, 1) \
        } \
    }

mmm_m_n_asm(4, 24, 3, 4, 12)
mmm_m_n_asm(3, 24, 3, 3, 9)
mmm_m_n_asm(2, 24, 3, 2, 6)
mmm_m_n_asm(1, 24, 3, 1, 3)

mmm_m_n_asm(6, 16, 2, 6, 12)
mmm_m_n_asm(5, 16, 2, 5, 10)
mmm_m_n_asm(4, 16, 2, 4, 8)
mmm_m_n_asm(3, 16, 2, 3, 6)
mmm_m_n_asm(2, 16, 2, 2, 4)
mmm_m_n_asm(1, 16, 2, 1, 2)

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
    kernel_func kernel[8][3] = {
        {asm_1x8, asm_1x16, asm_1x24},
        {asm_2x8, asm_2x16, asm_2x24},
        {asm_3x8, asm_3x16, asm_3x24},
        {asm_4x8, asm_4x16, asm_4x24},
        {asm_5x8, asm_5x16, asm_4x24},
        {asm_6x8, asm_6x16, asm_4x24},
        {asm_7x8, asm_6x16, asm_4x24},
        {asm_8x8, asm_6x16, asm_4x24}};
    U32 unrollNSizes[4] = {8, 8, 16, 24};
    U32 unrollMSizes[4] = {8, 8, 6, 4};
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

    auto getMNum = [](U32 mDim, U32 unrollM) { return (mDim + unrollM - 1) / unrollM; };

    U32 resN = N % UNROLL_N;
    U32 edgeNSize = UNI_ALIGN(resN, 8);
    // U32 BlockMDim = UNI_ALIGN(BOLCK_M_DIM, UNROLL_M);
    U32 mloopNum = getMNum(M, UNROLL_M);

    U32 newUnrollM = unrollMSizes[edgeNSize >> 3];
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
    U32 maskPtr = pow(2, N % 8) - 1;
    I32 mask[8] = {0};
    bool maskFlag = false;
    if (N % 8 != 0) {
        for (int i = 0; i < (N % 8); ++i) {
            mask[i] = -1;
        }
        maskFlag = true;
    }
    U32 loopNum = mNNum * mloopNum + (resN > 0) * resMloopNum;
    U32 bmLoopNum =
        mNNum * getMNum(BOLCK_M_DIM, UNROLL_M) + (resN > 0) * getMNum(BOLCK_M_DIM, newUnrollM);

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
            I32 *maskPtr = ((nLoop == mNNum) && maskFlag)? mask: nullptr;
            kernel[rm - 1][(nSize >> 3) - 1](rm, nSize, realK, curA, curB,
                i32Result + (m + bm) * N + n, u8Result + (m + bm) * N + n, offsetC + n, N,
                stepK, useFactor, maskPtr, kpad, flags);
        }
    }

    return SUCCESS;
}
