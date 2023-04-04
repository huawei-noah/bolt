// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_COMMON_INT8
#define _H_BLAS_COMMON_INT8

#include "tensor_desc.h"
#include "uni.h"

#define SIMDW 8

// transform no-transposed B to K4, offline
inline void matrix1_trans_l(
    int size, int alignedN, int blockK, int K, int alignSize, INT8 *src, INT8 *dst)
{
    int alignedBlockK = UNI_ALIGN(blockK, alignSize);
    int blockKF32 = blockK / 4;
    __m256i vindex = _mm256_set_epi32(K * 7, K * 6, K * 5, K * 4, K * 3, K * 2, K, 0);
    int i;
    for (i = 0; i < blockKF32; ++i) {
        int j = 0;
        for (; j < size / 8; ++j) {
            if (i % 16 == 0) {
                _mm_prefetch(dst + i * 4 + j * 8 * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 1) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 2) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 3) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 4) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 5) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 6) * K + 16, _MM_HINT_NTA);
                _mm_prefetch(dst + i * 4 + (j * 8 + 7) * K + 16, _MM_HINT_NTA);
            }
            _mm256_storeu_ps(
                (float *)dst, _mm256_i32gather_ps((float *)(src + i * 4 + j * 8 * K), vindex, 1));
            dst += 32;
        }
        j *= 8;
        for (; j < size; ++j) {
            UNI_MEMCPY(dst, src + i * 4 + j * K, 4);
            dst += 4;
        }
        if (j < alignedN) {
            UNI_MEMSET(dst, 0, 4 * (alignedN - size));
            dst += 4 * (alignedN - size);
        }
    }
    i *= 4;
    for (; i < alignedBlockK; i += 4) {
        int j = 0;
        for (; j < size; ++j) {
            for (int ii = i; ii < i + 4; ++ii) {
                if (ii < blockK) {
                    *(dst++) = src[ii + j * K];
                } else {
                    *(dst++) = 0;
                }
            }
        }
        if (j < alignedN) {
            UNI_MEMSET(dst, 0, 4 * (alignedN - size));
            dst += 4 * (alignedN - size);
        }
    }
}

// transform transposed B to K4, offline
inline void matrix2_trans_l(
    int size, int alignedN, int blockK, int N, int alignSize, INT8 *src, INT8 *dst)
{
    int alignedBlockK = UNI_ALIGN(blockK, alignSize);
    for (int i = 0; i < alignedBlockK; i += 4) {
        int j = 0;
        for (; j < size; ++j) {
            for (int ii = i; ii < (i + 4); ++ii) {
                if (ii < blockK) {
                    *(dst++) = src[ii * N + j];
                } else {
                    *(dst++) = 0;
                }
            }
        }
        if (j < alignedN) {
            UNI_MEMSET(dst, 0, 4 * (alignedN - size));
            dst += 4 * (alignedN - size);
        }
    }
}

// transpose A, online
inline void matrix2_trans_r(int size, int blockK, int M, int alignSize, UINT8 *src, UINT8 *dst)
{
    // TODO: optimize
    int alignedBlockK = UNI_ALIGN(blockK, alignSize);
    for (int j = 0; j < size; ++j) {
        int i = 0;
        for (i = 0; i < blockK; ++i) {
            if (j % 64 == 0) {
                _mm_prefetch(src + i * M + j + 64, _MM_HINT_NTA);
            }
            *(dst++) = *(src + i * M + j);
        }
        for (; i < alignedBlockK; ++i) {
            *(dst++) = 128;
        }
    }
}

// transpose A, online
inline void matrix1_trans_r(int size, int blockK, int K, int alignSize, UINT8 *src, UINT8 *dst)
{
    int alignedBlockK = UNI_ALIGN(blockK, alignSize);
    if (alignedBlockK != blockK) {
        UNI_MEMSET(dst, 0, alignedBlockK * size);
    }
    for (int j = 0; j < size; ++j) {
        UNI_MEMCPY(dst + j * alignedBlockK, src + j * K, blockK);
    }
}

#define loadOffset_1_1(rtype) \
    "vmovups (%[offset]), %%"#rtype"0            \n\t"

#define loadOffset_2_1(rtype) \
    loadOffset_1_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"1              \n\t"

#define loadOffset_3_1(rtype) \
    loadOffset_2_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"2              \n\t"

#define loadOffset_4_1(rtype) \
    loadOffset_3_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"3              \n\t"

#define loadOffset_5_1(rtype) \
    loadOffset_4_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"4              \n\t"

#define loadOffset_6_1(rtype) \
    loadOffset_5_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"5              \n\t"

#define loadOffset_7_1(rtype) \
    loadOffset_6_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"6              \n\t"

#define loadOffset_8_1(rtype) \
    loadOffset_7_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"7              \n\t"

#define loadOffset_9_1(rtype) \
    loadOffset_8_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"8              \n\t"

#define loadOffset_10_1(rtype) \
    loadOffset_9_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"9              \n\t"

#define loadOffset_11_1(rtype) \
    loadOffset_10_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"10              \n\t"

#define loadOffset_12_1(rtype) \
    loadOffset_11_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"11              \n\t"

#define loadOffset_13_1(rtype) \
    loadOffset_12_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"12              \n\t"

#define loadOffset_14_1(rtype) \
    loadOffset_13_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"13              \n\t"

#define loadOffset_15_1(rtype) \
    loadOffset_14_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"14              \n\t"

#define loadOffset_16_1(rtype) \
    loadOffset_15_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"15              \n\t"

#define loadOffset_17_1(rtype) \
    loadOffset_16_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"16              \n\t"

#define loadOffset_18_1(rtype) \
    loadOffset_17_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"17              \n\t"

#define loadOffset_19_1(rtype) \
    loadOffset_18_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"18              \n\t"

#define loadOffset_20_1(rtype) \
    loadOffset_19_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"19              \n\t"

#define loadOffset_21_1(rtype) \
    loadOffset_20_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"20              \n\t"

#define loadOffset_22_1(rtype) \
    loadOffset_21_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"21              \n\t"

#define loadOffset_23_1(rtype) \
    loadOffset_22_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"22              \n\t"

#define loadOffset_24_1(rtype) \
    loadOffset_23_1(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"23              \n\t"

#define loadOffset_1_2_ymm(rtype) \
    loadOffset_1_1(rtype) \
    "vmovups 0x20(%[offset]), %%"#rtype"1           \n\t"

#define loadOffset_1_2_zmm(rtype) \
    loadOffset_1_1(rtype) \
    "vmovups 0x40(%[offset]), %%"#rtype"1           \n\t"

#define loadOffset_1_2(rtype) \
    loadOffset_1_2_##rtype(rtype)

#define loadOffset_2_2(rtype) \
    loadOffset_1_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"2                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"3                    \n\t"

#define loadOffset_3_2(rtype) \
    loadOffset_2_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"4                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"5                    \n\t"

#define loadOffset_4_2(rtype) \
    loadOffset_3_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"6                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"7                    \n\t"

#define loadOffset_5_2(rtype) \
    loadOffset_4_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"8                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"9                    \n\t"

#define loadOffset_6_2(rtype) \
    loadOffset_5_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"10                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"11                   \n\t"

#define loadOffset_7_2(rtype) \
    loadOffset_6_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"12                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"13                    \n\t"

#define loadOffset_8_2(rtype) \
    loadOffset_7_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"14                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"15                    \n\t"

#define loadOffset_9_2(rtype) \
    loadOffset_8_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"16                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"17                    \n\t"

#define loadOffset_10_2(rtype) \
    loadOffset_9_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"18                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"19                    \n\t"

#define loadOffset_11_2(rtype) \
    loadOffset_10_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"20                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"21                   \n\t"

#define loadOffset_12_2(rtype) \
    loadOffset_11_2(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"22                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"23                   \n\t"

#define loadOffset_1_3_ymm(rtype) \
    loadOffset_1_2_ymm(rtype) \
    "vmovups 0x40(%[offset]), %%"#rtype"2           \n\t"

#define loadOffset_1_3_zmm(rtype) \
    loadOffset_1_2_zmm(rtype) \
    "vmovups 0x80(%[offset]), %%"#rtype"2           \n\t"

#define loadOffset_1_3(rtype) \
    loadOffset_1_3_##rtype(rtype) \

#define loadOffset_2_3(rtype) \
    loadOffset_1_3(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"3                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"4                    \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"5                    \n\t"

#define loadOffset_3_3(rtype) \
    loadOffset_2_3(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"6                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"7                    \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"8                    \n\t"

#define loadOffset_4_3(rtype) \
    loadOffset_3_3(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"9                    \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"10                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"11                   \n\t"

#define loadOffset_5_3(rtype) \
    loadOffset_4_3(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"12                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"13                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"14                   \n\t"

#define loadOffset_6_3(rtype) \
    loadOffset_5_3(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"15                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"16                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"17                   \n\t"

#define loadOffset_7_3(rtype) \
    loadOffset_6_3(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"18                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"19                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"20                   \n\t"

#define loadOffset_8_3(rtype) \
    loadOffset_7_3(rtype) \
    "vmovups %%"#rtype"0, %%"#rtype"21                   \n\t" \
    "vmovups %%"#rtype"1, %%"#rtype"22                   \n\t" \
    "vmovups %%"#rtype"2, %%"#rtype"23                   \n\t"

#define addC_1_1(rtype, C, off0, off1) \
    "movq "#C", %%rax  \n\t" \
    "vpaddd (%%rax), %%"#rtype"0, %%"#rtype"0       \n\t"

#define addC_2_1(rtype, C, off0, off1) \
    addC_1_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"1, %%"#rtype"1       \n\t" \

#define addC_3_1(rtype, C, off0, off1) \
    addC_2_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"2, %%"#rtype"2       \n\t" \

#define addC_4_1(rtype, C, off0, off1) \
    addC_3_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"3, %%"#rtype"3       \n\t" \

#define addC_5_1(rtype, C, off0, off1) \
    addC_4_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"4, %%"#rtype"4       \n\t" \

#define addC_6_1(rtype, C, off0, off1) \
    addC_5_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"5, %%"#rtype"5       \n\t"

#define addC_7_1(rtype, C, off0, off1) \
    addC_6_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"6, %%"#rtype"6       \n\t" \

#define addC_8_1(rtype, C, off0, off1) \
    addC_7_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"7, %%"#rtype"7       \n\t" \

#define addC_9_1(rtype, C, off0, off1) \
    addC_8_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"8, %%"#rtype"8       \n\t" \

#define addC_10_1(rtype, C, off0, off1) \
    addC_9_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"9, %%"#rtype"9       \n\t" \

#define addC_11_1(rtype, C, off0, off1) \
    addC_10_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"10, %%"#rtype"10     \n\t" \

#define addC_12_1(rtype, C, off0, off1) \
    addC_11_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"11, %%"#rtype"11     \n\t"

#define addC_13_1(rtype, C, off0, off1) \
    addC_12_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"12, %%"#rtype"12     \n\t" \

#define addC_14_1(rtype, C, off0, off1) \
    addC_13_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"13, %%"#rtype"13     \n\t" \

#define addC_15_1(rtype, C, off0, off1) \
    addC_14_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"14, %%"#rtype"14     \n\t" \

#define addC_16_1(rtype, C, off0, off1) \
    addC_15_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"15, %%"#rtype"15     \n\t" \

#define addC_17_1(rtype, C, off0, off1) \
    addC_16_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"16, %%"#rtype"16     \n\t" \

#define addC_18_1(rtype, C, off0, off1) \
    addC_17_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"17, %%"#rtype"17     \n\t" \

#define addC_19_1(rtype, C, off0, off1) \
    addC_18_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"18, %%"#rtype"18     \n\t" \

#define addC_20_1(rtype, C, off0, off1) \
    addC_19_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"19, %%"#rtype"19     \n\t" \

#define addC_21_1(rtype, C, off0, off1) \
    addC_20_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"20, %%"#rtype"20     \n\t" \

#define addC_22_1(rtype, C, off0, off1) \
    addC_21_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"21, %%"#rtype"21     \n\t" \

#define addC_23_1(rtype, C, off0, off1) \
    addC_22_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"22, %%"#rtype"22     \n\t" \

#define addC_24_1(rtype, C, off0, off1) \
    addC_23_1(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"23, %%"#rtype"23     \n\t"

#define addC_1_2(rtype, C, off0, off1) \
    addC_1_1(rtype, C, off0, off1) \
    "vpaddd "#off0"(%%rax), %%"#rtype"1, %%"#rtype"1         \n\t"

#define addC_2_2(rtype, C, off0, off1) \
    addC_1_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"2, %%"#rtype"2             \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"3, %%"#rtype"3         \n\t" \

#define addC_3_2(rtype, C, off0, off1) \
    addC_2_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"4, %%"#rtype"4             \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"5, %%"#rtype"5         \n\t"

#define addC_4_2(rtype, C, off0, off1) \
    addC_3_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"6, %%"#rtype"6             \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"7, %%"#rtype"7         \n\t" \

#define addC_5_2(rtype, C, off0, off1) \
    addC_4_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"8, %%"#rtype"8             \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"9, %%"#rtype"9         \n\t" \

#define addC_6_2(rtype, C, off0, off1) \
    addC_5_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"10, %%"#rtype"10           \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"11, %%"#rtype"11       \n\t"

#define addC_7_2(rtype, C, off0, off1) \
    addC_6_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"12, %%"#rtype"12           \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"13, %%"#rtype"13       \n\t" \

#define addC_8_2(rtype, C, off0, off1) \
    addC_7_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"14, %%"#rtype"14           \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"15, %%"#rtype"15       \n\t" \

#define addC_9_2(rtype, C, off0, off1) \
    addC_8_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"16, %%"#rtype"16           \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"17, %%"#rtype"17       \n\t" \

#define addC_10_2(rtype, C, off0, off1) \
    addC_9_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"18, %%"#rtype"18           \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"19, %%"#rtype"19       \n\t" \

#define addC_11_2(rtype, C, off0, off1) \
    addC_10_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"20, %%"#rtype"20           \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"21, %%"#rtype"21       \n\t" \

#define addC_12_2(rtype, C, off0, off1) \
    addC_11_2(rtype, C, off0, off1) \
    "addq %[N], %%rax                           \n\t" \
    "vpaddd (%%rax), %%"#rtype"22, %%"#rtype"22           \n\t" \
    "vpaddd "#off0"(%%rax), %%"#rtype"23, %%"#rtype"23       \n\t"

#define addC_1_3(rtype, C, off0, off1) \
    "vpaddd ("#C"), %%"#rtype"0, %%"#rtype"0              \n\t" \
    "vpaddd "#off0"("#C"), %%"#rtype"1, %%"#rtype"1          \n\t" \
    "vpaddd "#off1"("#C"), %%"#rtype"2, %%"#rtype"2          \n\t"

#define addC_2_3(rtype, C, off0, off1) \
    addC_1_3(rtype, C, off0, off1) \
    "vpaddd ("#C", %[N]), %%"#rtype"3, %%"#rtype"3        \n\t" \
    "vpaddd "#off0"("#C", %[N]), %%"#rtype"4, %%"#rtype"4    \n\t" \
    "vpaddd "#off1"("#C", %[N]), %%"#rtype"5, %%"#rtype"5    \n\t"

#define addC_3_3(rtype, C, off0, off1) \
    addC_2_3(rtype, C, off0, off1) \
    "addq %%rcx, "#C"  \n\t" \
    "vpaddd ("#C"), %%"#rtype"6, %%"#rtype"6              \n\t" \
    "vpaddd "#off0"("#C"), %%"#rtype"7, %%"#rtype"7          \n\t" \
    "vpaddd "#off1"("#C"), %%"#rtype"8, %%"#rtype"8          \n\t" \

#define addC_4_3(rtype, C, off0, off1) \
    addC_3_3(rtype, C, off0, off1) \
    "vpaddd ("#C", %[N]), %%"#rtype"9, %%"#rtype"9        \n\t" \
    "vpaddd "#off0"("#C", %[N]), %%"#rtype"10, %%"#rtype"10  \n\t" \
    "vpaddd "#off1"("#C", %[N]), %%"#rtype"11, %%"#rtype"11  \n\t"

#define addC_5_3(rtype, C, off0, off1) \
    addC_4_3(rtype, C, off0, off1) \
    "addq %%rcx, "#C"  \n\t" \
    "vpaddd ("#C"), %%"#rtype"12, %%"#rtype"12                   \n\t" \
    "vpaddd "#off0"("#C"), %%"#rtype"13, %%"#rtype"13                   \n\t" \
    "vpaddd "#off1"("#C"), %%"#rtype"14, %%"#rtype"14                   \n\t" \

#define addC_6_3(rtype, C, off0, off1) \
    addC_5_3(rtype, C, off0, off1) \
    "vpaddd ("#C", %[N]), %%"#rtype"15, %%"#rtype"15                   \n\t" \
    "vpaddd "#off0"("#C", %[N]), %%"#rtype"16, %%"#rtype"16                   \n\t" \
    "vpaddd "#off1"("#C", %[N]), %%"#rtype"17, %%"#rtype"17                   \n\t" \

#define addC_7_3(rtype, C, off0, off1) \
    addC_6_3(rtype, C, off0, off1) \
    "addq %%rcx, "#C"  \n\t" \
    "vpaddd ("#C"), %%"#rtype"18, %%"#rtype"18                   \n\t" \
    "vpaddd "#off0"("#C"), %%"#rtype"19, %%"#rtype"19                   \n\t" \
    "vpaddd "#off1"("#C"), %%"#rtype"20, %%"#rtype"20                   \n\t" \

#define addC_8_3(rtype, C, off0, off1) \
    addC_7_3(rtype, C, off0, off1) \
    "vpaddd ("#C", %[N]), %%"#rtype"21, %%"#rtype"21                   \n\t" \
    "vpaddd "#off0"("#C", %[N]), %%"#rtype"22, %%"#rtype"22                   \n\t" \
    "vpaddd "#off1"("#C", %[N]), %%"#rtype"23, %%"#rtype"23                   \n\t" \

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

#define clear5Regs(rtype) \
    clear4Regs(rtype) \
    "vxorps "#rtype"4, "#rtype"4, "#rtype"4    \n\t"

#define clear6Regs(rtype) \
    clear5Regs(rtype) \
    "vxorps "#rtype"5, "#rtype"5, "#rtype"5    \n\t"

#define clear7Regs(rtype) \
    clear6Regs(rtype) \
    "vxorps "#rtype"6, "#rtype"6, "#rtype"6    \n\t"

#define clear8Regs(rtype) \
    clear7Regs(rtype) \
    "vxorps "#rtype"7, "#rtype"7, "#rtype"7    \n\t"

#define clear9Regs(rtype) \
    clear8Regs(rtype) \
    "vxorps "#rtype"8, "#rtype"8, "#rtype"8    \n\t"

#define clear10Regs(rtype) \
    clear9Regs(rtype) \
    "vxorps "#rtype"9, "#rtype"9, "#rtype"9    \n\t"

#define clear11Regs(rtype) \
    clear10Regs(rtype) \
    "vxorps "#rtype"10, "#rtype"10, "#rtype"10 \n\t"

#define clear12Regs(rtype) \
    clear11Regs(rtype) \
    "vxorps "#rtype"11, "#rtype"11, "#rtype"11 \n\t"

#define clear13Regs(rtype) \
    clear12Regs(rtype) \
    "vxorps "#rtype"12, "#rtype"12, "#rtype"12 \n\t"

#define clear14Regs(rtype) \
    clear13Regs(rtype) \
    "vxorps "#rtype"13, "#rtype"13, "#rtype"13 \n\t"

#define clear15Regs(rtype) \
    clear14Regs(rtype) \
    "vxorps "#rtype"14, "#rtype"14, "#rtype"14 \n\t"

#define clear16Regs(rtype) \
    clear15Regs(rtype) \
    "vxorps "#rtype"15, "#rtype"15, "#rtype"15 \n\t"

#define clear17Regs(rtype) \
    clear16Regs(rtype) \
    "vxorps "#rtype"16, "#rtype"16, "#rtype"16 \n\t"

#define clear18Regs(rtype) \
    clear17Regs(rtype) \
    "vxorps "#rtype"17, "#rtype"17, "#rtype"17 \n\t"

#define clear19Regs(rtype) \
    clear18Regs(rtype) \
    "vxorps "#rtype"18, "#rtype"18, "#rtype"18 \n\t"

#define clear20Regs(rtype) \
    clear19Regs(rtype) \
    "vxorps "#rtype"19, "#rtype"19, "#rtype"19 \n\t"

#define clear21Regs(rtype) \
    clear20Regs(rtype) \
    "vxorps "#rtype"20, "#rtype"20, "#rtype"20 \n\t"

#define clear22Regs(rtype) \
    clear21Regs(rtype) \
    "vxorps "#rtype"21, "#rtype"21, "#rtype"21 \n\t"

#define clear23Regs(rtype) \
    clear22Regs(rtype) \
    "vxorps "#rtype"22, "#rtype"22, "#rtype"22 \n\t"

#define clear24Regs(rtype) \
    clear23Regs(rtype) \
    "vxorps "#rtype"23, "#rtype"23, "#rtype"23 \n\t"

#define convert1I32Regs2Ps(rtype, scalePtr) \
    "vbroadcastss ("#scalePtr"), "#rtype"15               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vmulps "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \

#define convert2I32Regs2Ps(rtype, scalePtr) \
    "vbroadcastss ("#scalePtr"), "#rtype"15               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vmulps "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \

#define convert3I32Regs2Ps(rtype, scalePtr) \
    "vbroadcastss ("#scalePtr"), "#rtype"15               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                       \n\t" \
    "vmulps "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \
    "vmulps "#rtype"2, "#rtype"15, "#rtype"2              \n\t" \

#define convert4I32Regs2Ps(rtype, scalePtr) \
    "vbroadcastss ("#scalePtr"), "#rtype"15               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                       \n\t" \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vmulps "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \
    "vmulps "#rtype"2, "#rtype"15, "#rtype"2              \n\t" \
    "vmulps "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \

#define convert5I32Regs2Ps(rtype, scalePtr) \
    convert3I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vmulps "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \
    "vmulps "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \

#define convert6I32Regs2Ps(rtype, scalePtr) \
    convert3I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vmulps "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \
    "vmulps "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \

#define convert7I32Regs2Ps(rtype, scalePtr) \
    convert4I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vmulps "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \

#define convert8I32Regs2Ps(rtype, scalePtr) \
    convert4I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vmulps "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \

#define convert9I32Regs2Ps(rtype, scalePtr) \
    convert5I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vmulps "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \
    "vmulps "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \

#define convert10I32Regs2Ps(rtype, scalePtr) \
    convert6I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vmulps "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \
    "vmulps "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \
    "vmulps "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \

#define convert11I32Regs2Ps(rtype, scalePtr) \
    convert8I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vmulps "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \
    "vmulps "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \

#define convert12I32Regs2Ps(rtype, scalePtr) \
    convert8I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vmulps "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \
    "vmulps "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \

#define convert13I32Regs2Ps(rtype, scalePtr) \
    convert9I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vmulps "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \
    "vmulps "#rtype"12, "#rtype"15, "#rtype"12            \n\t" \

#define convert14I32Regs2Ps(rtype, scalePtr) \
    convert10I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vmulps "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \
    "vmulps "#rtype"12, "#rtype"15, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"15, "#rtype"13            \n\t" \

#define convert15I32Regs2Ps(rtype, scalePtr) \
    convert11I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vmulps "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \
    "vmulps "#rtype"12, "#rtype"15, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"15, "#rtype"13            \n\t" \
    "vmulps "#rtype"14, "#rtype"15, "#rtype"14            \n\t" \

#define convert12I32Regs2PsZmm(rtype, scalePtr) \
    "vbroadcastss ("#scalePtr"), "#rtype"24               \n\t" \
    "vcvtdq2ps "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtdq2ps "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtdq2ps "#rtype"2, "#rtype"2                       \n\t" \
    "vcvtdq2ps "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtdq2ps "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtdq2ps "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtdq2ps "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtdq2ps "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtdq2ps "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtdq2ps "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtdq2ps "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtdq2ps "#rtype"11, "#rtype"11                     \n\t" \
    "vmulps "#rtype"0, "#rtype"24, "#rtype"0              \n\t" \
    "vmulps "#rtype"1, "#rtype"24, "#rtype"1              \n\t" \
    "vmulps "#rtype"2, "#rtype"24, "#rtype"2              \n\t" \
    "vmulps "#rtype"3, "#rtype"24, "#rtype"3              \n\t" \
    "vmulps "#rtype"4, "#rtype"24, "#rtype"4              \n\t" \
    "vmulps "#rtype"5, "#rtype"24, "#rtype"5              \n\t" \
    "vmulps "#rtype"6, "#rtype"24, "#rtype"6              \n\t" \
    "vmulps "#rtype"7, "#rtype"24, "#rtype"7              \n\t" \
    "vmulps "#rtype"8, "#rtype"24, "#rtype"8              \n\t" \
    "vmulps "#rtype"9, "#rtype"24, "#rtype"9              \n\t" \
    "vmulps "#rtype"10, "#rtype"24, "#rtype"10            \n\t" \
    "vmulps "#rtype"11, "#rtype"24, "#rtype"11            \n\t" \

#define convert16I32Regs2Ps(rtype, scalePtr) \
    convert12I32Regs2PsZmm(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtdq2ps "#rtype"15, "#rtype"15                     \n\t" \
    "vmulps "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vmulps "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \
    "vmulps "#rtype"15, "#rtype"24, "#rtype"15            \n\t" \

#define convert17I32Regs2Ps(rtype, scalePtr) \
    convert12I32Regs2PsZmm(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtdq2ps "#rtype"15, "#rtype"15                     \n\t" \
    "vcvtdq2ps "#rtype"16, "#rtype"16                     \n\t" \
    "vmulps "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vmulps "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \
    "vmulps "#rtype"15, "#rtype"24, "#rtype"15            \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \

#define convert18I32Regs2Ps(rtype, scalePtr) \
    convert12I32Regs2PsZmm(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtdq2ps "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtdq2ps "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtdq2ps "#rtype"15, "#rtype"15                     \n\t" \
    "vcvtdq2ps "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtdq2ps "#rtype"17, "#rtype"17                     \n\t" \
    "vmulps "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vmulps "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vmulps "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \
    "vmulps "#rtype"15, "#rtype"24, "#rtype"15            \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \

#define convert19I32Regs2Ps(rtype, scalePtr) \
    convert16I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtdq2ps "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtdq2ps "#rtype"18, "#rtype"18                     \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \

#define convert20I32Regs2Ps(rtype, scalePtr) \
    convert16I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtdq2ps "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtdq2ps "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtdq2ps "#rtype"19, "#rtype"19                     \n\t" \
    "vmulps "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \

#define convert21I32Regs2Ps(rtype, scalePtr) \
    convert17I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtdq2ps "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtdq2ps "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtdq2ps "#rtype"20, "#rtype"20                     \n\t" \
    "vmulps "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \

#define convert22I32Regs2Ps(rtype, scalePtr) \
    convert18I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtdq2ps "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtdq2ps "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtdq2ps "#rtype"21, "#rtype"21                     \n\t" \
    "vmulps "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vmulps "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \

#define convert23I32Regs2Ps(rtype, scalePtr) \
    convert19I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtdq2ps "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtdq2ps "#rtype"21, "#rtype"21                     \n\t" \
    "vcvtdq2ps "#rtype"22, "#rtype"22                     \n\t" \
    "vmulps "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vmulps "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \
    "vmulps "#rtype"22, "#rtype"24, "#rtype"22            \n\t" \

#define convert24I32Regs2Ps(rtype, scalePtr) \
    convert20I32Regs2Ps(rtype, scalePtr) \
    "vcvtdq2ps "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtdq2ps "#rtype"21, "#rtype"21                     \n\t" \
    "vcvtdq2ps "#rtype"22, "#rtype"22                     \n\t" \
    "vcvtdq2ps "#rtype"23, "#rtype"23                     \n\t" \
    "vmulps "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vmulps "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \
    "vmulps "#rtype"22, "#rtype"24, "#rtype"22            \n\t" \
    "vmulps "#rtype"23, "#rtype"24, "#rtype"23            \n\t" \

#define convert1PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm14                            \n\t" \
    "vbroadcastss %%xmm14, "#rtype"15                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                       \n\t" \
    "vpaddd "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \

#define convert2PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm14                            \n\t" \
    "vbroadcastss %%xmm14, "#rtype"15                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtps2dq "#rtype"1, "#rtype"1                       \n\t" \
    "vpaddd "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vpaddd "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \

#define convert3PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm14                            \n\t" \
    "vbroadcastss %%xmm14, "#rtype"15                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtps2dq "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtps2dq "#rtype"2, "#rtype"2                       \n\t" \
    "vpaddd "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vpaddd "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \
    "vpaddd "#rtype"2, "#rtype"15, "#rtype"2              \n\t" \

#define convert4PsRegs2U8(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm14                            \n\t" \
    "vbroadcastss %%xmm14, "#rtype"15                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtps2dq "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtps2dq "#rtype"2, "#rtype"2                       \n\t" \
    "vcvtps2dq "#rtype"3, "#rtype"3                       \n\t" \
    "vpaddd "#rtype"0, "#rtype"15, "#rtype"0              \n\t" \
    "vpaddd "#rtype"1, "#rtype"15, "#rtype"1              \n\t" \
    "vpaddd "#rtype"2, "#rtype"15, "#rtype"2              \n\t" \
    "vpaddd "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \

#define convert5PsRegs2U8(rtype) \
    convert3PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtps2dq "#rtype"4, "#rtype"4                       \n\t" \
    "vpaddd "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \
    "vpaddd "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \

#define convert6PsRegs2U8(rtype) \
    convert3PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtps2dq "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtps2dq "#rtype"5, "#rtype"5                       \n\t" \
    "vpaddd "#rtype"3, "#rtype"15, "#rtype"3              \n\t" \
    "vpaddd "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vpaddd "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \

#define convert7PsRegs2U8(rtype) \
    convert4PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtps2dq "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtps2dq "#rtype"6, "#rtype"6                       \n\t" \
    "vpaddd "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vpaddd "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vpaddd "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \

#define convert8PsRegs2U8(rtype) \
    convert4PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtps2dq "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtps2dq "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtps2dq "#rtype"7, "#rtype"7                       \n\t" \
    "vpaddd "#rtype"4, "#rtype"15, "#rtype"4              \n\t" \
    "vpaddd "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vpaddd "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vpaddd "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \

#define convert9PsRegs2U8(rtype) \
    convert5PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtps2dq "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtps2dq "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtps2dq "#rtype"8, "#rtype"8                       \n\t" \
    "vpaddd "#rtype"5, "#rtype"15, "#rtype"5              \n\t" \
    "vpaddd "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vpaddd "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \
    "vpaddd "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \

#define convert10PsRegs2U8(rtype) \
    convert6PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtps2dq "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtps2dq "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtps2dq "#rtype"9, "#rtype"9                       \n\t" \
    "vpaddd "#rtype"6, "#rtype"15, "#rtype"6              \n\t" \
    "vpaddd "#rtype"7, "#rtype"15, "#rtype"7              \n\t" \
    "vpaddd "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \
    "vpaddd "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \

#define convert11PsRegs2U8(rtype) \
    convert8PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtps2dq "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtps2dq "#rtype"10, "#rtype"10                     \n\t" \
    "vpaddd "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \
    "vpaddd "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \
    "vpaddd "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \

#define convert12PsRegs2U8(rtype) \
    convert8PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtps2dq "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtps2dq "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtps2dq "#rtype"11, "#rtype"11                     \n\t" \
    "vpaddd "#rtype"8, "#rtype"15, "#rtype"8              \n\t" \
    "vpaddd "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \
    "vpaddd "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vpaddd "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \

#define convert13PsRegs2U8(rtype) \
    convert9PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtps2dq "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtps2dq "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtps2dq "#rtype"12, "#rtype"12                     \n\t" \
    "vpaddd "#rtype"9, "#rtype"15, "#rtype"9              \n\t" \
    "vpaddd "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vpaddd "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \
    "vpaddd "#rtype"12, "#rtype"15, "#rtype"12            \n\t" \

#define convert14PsRegs2U8(rtype) \
    convert10PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtps2dq "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtps2dq "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtps2dq "#rtype"13, "#rtype"13                     \n\t" \
    "vpaddd "#rtype"10, "#rtype"15, "#rtype"10            \n\t" \
    "vpaddd "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \
    "vpaddd "#rtype"12, "#rtype"15, "#rtype"12            \n\t" \
    "vpaddd "#rtype"13, "#rtype"15, "#rtype"13            \n\t" \

#define convert15PsRegs2U8(rtype) \
    convert11PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"11, "#rtype"11                     \n\t" \
    "vcvtps2dq "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtps2dq "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtps2dq "#rtype"13, "#rtype"13                     \n\t" \
    "vpaddd "#rtype"11, "#rtype"15, "#rtype"11            \n\t" \
    "vpaddd "#rtype"12, "#rtype"15, "#rtype"12            \n\t" \
    "vpaddd "#rtype"13, "#rtype"15, "#rtype"13            \n\t" \
    "vpaddd "#rtype"14, "#rtype"15, "#rtype"14            \n\t" \

#define convert12PsRegs2U8Zmm(rtype) \
    "mov $128, %%eax \n\t" \
    "vmovd %%eax, %%xmm25                            \n\t" \
    "vbroadcastss %%xmm25, "#rtype"24                 \n\t" \
    "vcvtps2dq "#rtype"0, "#rtype"0                       \n\t" \
    "vcvtps2dq "#rtype"1, "#rtype"1                       \n\t" \
    "vcvtps2dq "#rtype"2, "#rtype"2                       \n\t" \
    "vcvtps2dq "#rtype"3, "#rtype"3                       \n\t" \
    "vcvtps2dq "#rtype"4, "#rtype"4                       \n\t" \
    "vcvtps2dq "#rtype"5, "#rtype"5                       \n\t" \
    "vcvtps2dq "#rtype"6, "#rtype"6                       \n\t" \
    "vcvtps2dq "#rtype"7, "#rtype"7                       \n\t" \
    "vcvtps2dq "#rtype"8, "#rtype"8                       \n\t" \
    "vcvtps2dq "#rtype"9, "#rtype"9                       \n\t" \
    "vcvtps2dq "#rtype"10, "#rtype"10                     \n\t" \
    "vcvtps2dq "#rtype"11, "#rtype"11                     \n\t" \
    "vpaddd "#rtype"0, "#rtype"24, "#rtype"0              \n\t" \
    "vpaddd "#rtype"1, "#rtype"24, "#rtype"1              \n\t" \
    "vpaddd "#rtype"2, "#rtype"24, "#rtype"2              \n\t" \
    "vpaddd "#rtype"3, "#rtype"24, "#rtype"3              \n\t" \
    "vpaddd "#rtype"4, "#rtype"24, "#rtype"4              \n\t" \
    "vpaddd "#rtype"5, "#rtype"24, "#rtype"5              \n\t" \
    "vpaddd "#rtype"6, "#rtype"24, "#rtype"6              \n\t" \
    "vpaddd "#rtype"7, "#rtype"24, "#rtype"7              \n\t" \
    "vpaddd "#rtype"8, "#rtype"24, "#rtype"8              \n\t" \
    "vpaddd "#rtype"9, "#rtype"24, "#rtype"9              \n\t" \
    "vpaddd "#rtype"10, "#rtype"24, "#rtype"10            \n\t" \
    "vpaddd "#rtype"11, "#rtype"24, "#rtype"11            \n\t" \

#define convert16PsRegs2U8(rtype) \
    convert12PsRegs2U8Zmm(rtype) \
    "vcvtps2dq "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtps2dq "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtps2dq "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtps2dq "#rtype"15, "#rtype"15                     \n\t" \
    "vpaddd "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vpaddd "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vpaddd "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \
    "vpaddd "#rtype"15, "#rtype"24, "#rtype"15            \n\t" \

#define convert17PsRegs2U8(rtype) \
    convert12PsRegs2U8Zmm(rtype) \
    "vcvtps2dq "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtps2dq "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtps2dq "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtps2dq "#rtype"15, "#rtype"15                     \n\t" \
    "vcvtps2dq "#rtype"16, "#rtype"16                     \n\t" \
    "vpaddd "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vpaddd "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vpaddd "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \
    "vpaddd "#rtype"15, "#rtype"24, "#rtype"15            \n\t" \
    "vpaddd "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \

#define convert18PsRegs2U8(rtype) \
    convert12PsRegs2U8Zmm(rtype) \
    "vcvtps2dq "#rtype"12, "#rtype"12                     \n\t" \
    "vcvtps2dq "#rtype"13, "#rtype"13                     \n\t" \
    "vcvtps2dq "#rtype"14, "#rtype"14                     \n\t" \
    "vcvtps2dq "#rtype"15, "#rtype"15                     \n\t" \
    "vcvtps2dq "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtps2dq "#rtype"17, "#rtype"17                     \n\t" \
    "vpaddd "#rtype"12, "#rtype"24, "#rtype"12            \n\t" \
    "vpaddd "#rtype"13, "#rtype"24, "#rtype"13            \n\t" \
    "vpaddd "#rtype"14, "#rtype"24, "#rtype"14            \n\t" \
    "vpaddd "#rtype"15, "#rtype"24, "#rtype"15            \n\t" \
    "vpaddd "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vpaddd "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \

#define convert19PsRegs2U8(rtype) \
    convert16PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtps2dq "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtps2dq "#rtype"18, "#rtype"18                     \n\t" \
    "vpaddd "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vpaddd "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vpaddd "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \

#define convert20PsRegs2U8(rtype) \
    convert16PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"16, "#rtype"16                     \n\t" \
    "vcvtps2dq "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtps2dq "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtps2dq "#rtype"19, "#rtype"19                     \n\t" \
    "vpaddd "#rtype"16, "#rtype"24, "#rtype"16            \n\t" \
    "vpaddd "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vpaddd "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vpaddd "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \

#define convert21PsRegs2U8(rtype) \
    convert17PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"17, "#rtype"17                     \n\t" \
    "vcvtps2dq "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtps2dq "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtps2dq "#rtype"20, "#rtype"20                     \n\t" \
    "vpaddd "#rtype"17, "#rtype"24, "#rtype"17            \n\t" \
    "vpaddd "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vpaddd "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vpaddd "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \

#define convert22PsRegs2U8(rtype) \
    convert18PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"18, "#rtype"18                     \n\t" \
    "vcvtps2dq "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtps2dq "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtps2dq "#rtype"21, "#rtype"21                     \n\t" \
    "vpaddd "#rtype"18, "#rtype"24, "#rtype"18            \n\t" \
    "vpaddd "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vpaddd "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vpaddd "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \

#define convert23PsRegs2U8(rtype) \
    convert19PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"19, "#rtype"19                     \n\t" \
    "vcvtps2dq "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtps2dq "#rtype"21, "#rtype"21                     \n\t" \
    "vcvtps2dq "#rtype"22, "#rtype"22                     \n\t" \
    "vpaddd "#rtype"19, "#rtype"24, "#rtype"19            \n\t" \
    "vpaddd "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vpaddd "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \
    "vpaddd "#rtype"22, "#rtype"24, "#rtype"22            \n\t" \

#define convert24PsRegs2U8(rtype) \
    convert20PsRegs2U8(rtype) \
    "vcvtps2dq "#rtype"20, "#rtype"20                     \n\t" \
    "vcvtps2dq "#rtype"21, "#rtype"21                     \n\t" \
    "vcvtps2dq "#rtype"22, "#rtype"22                     \n\t" \
    "vcvtps2dq "#rtype"23, "#rtype"23                     \n\t" \
    "vpaddd "#rtype"20, "#rtype"24, "#rtype"20            \n\t" \
    "vpaddd "#rtype"21, "#rtype"24, "#rtype"21            \n\t" \
    "vpaddd "#rtype"22, "#rtype"24, "#rtype"22            \n\t" \
    "vpaddd "#rtype"23, "#rtype"24, "#rtype"23            \n\t" \

#endif
