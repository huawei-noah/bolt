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

#include "cpu/x86/fp32/tensor_computing_fp32.h"
#include "cpu/x86/fp32/transform_functions_fp32.h"
#include "cpu/x86/fp32/convolution_functions.h"

#define BLOCK_IC_DIM 32
#define BLOCK_OC_DIM 32

void transformInput4x4_3x3(
    F32 *input, F32 *output, F32 *tmp, U32 iw, U32 ih, U32 ic, U32 wSize, U32 blockIc)
{
    __m256 four = _mm256_set1_ps(4.0f);
    __m256 minusFour = _mm256_set1_ps(-4.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 minusFive = _mm256_set1_ps(-5.0f);
    U32 icb = ic / blockIc;
    U32 cb = blockIc / 8;
    for (U32 w = 0; w < wSize; ++w) {
        for (U32 c = 0; c < icb; ++c) {
            for (U32 cc = 0; cc < cb; ++cc) {
                F32 *curI = input + (c * blockIc + cc * 8) * ih * iw + w * 4 * 8;
                F32 *curO = output + (w * ic + c * blockIc) * 36 + cc * 8;
                for (U32 i = 0; i < 6; ++i) {
                    __m256 xi0 = _mm256_loadu_ps(curI + (i)*8);
                    __m256 xi1 = _mm256_loadu_ps(curI + (iw + i) * 8);
                    __m256 xi2 = _mm256_loadu_ps(curI + (iw * 2 + i) * 8);
                    __m256 xi3 = _mm256_loadu_ps(curI + (iw * 3 + i) * 8);
                    __m256 xi4 = _mm256_loadu_ps(curI + (iw * 4 + i) * 8);
                    __m256 xi5 = _mm256_loadu_ps(curI + (iw * 5 + i) * 8);

                    __m256 t0 = _mm256_fmadd_ps(minusFour, xi2, xi4);
                    __m256 t1 = _mm256_fmadd_ps(minusFour, xi1, xi3);
                    __m256 t2 = _mm256_sub_ps(xi4, xi2);
                    __m256 t3 = _mm256_mul_ps(two, _mm256_sub_ps(xi3, xi1));
                    __m256 t4 = _mm256_fmadd_ps(four, xi0, xi4);
                    __m256 t5 = _mm256_fmadd_ps(four, xi1, xi5);

                    xi0 = _mm256_fmadd_ps(minusFive, xi2, t4);
                    xi5 = _mm256_fmadd_ps(minusFive, xi3, t5);
                    xi1 = _mm256_add_ps(t1, t0);
                    xi2 = _mm256_sub_ps(t0, t1);
                    xi3 = _mm256_add_ps(t3, t2);
                    xi4 = _mm256_sub_ps(t2, t3);

                    _mm256_storeu_ps(tmp + (i)*8, xi0);
                    _mm256_storeu_ps(tmp + (6 + i) * 8, xi1);
                    _mm256_storeu_ps(tmp + (6 * 2 + i) * 8, xi2);
                    _mm256_storeu_ps(tmp + (6 * 3 + i) * 8, xi3);
                    _mm256_storeu_ps(tmp + (6 * 4 + i) * 8, xi4);
                    _mm256_storeu_ps(tmp + (6 * 5 + i) * 8, xi5);
                }

                for (U32 i = 0; i < 6; ++i) {
                    __m256 xi0 = _mm256_loadu_ps(tmp + (i * 6) * 8);
                    __m256 xi1 = _mm256_loadu_ps(tmp + (i * 6 + 1) * 8);
                    __m256 xi2 = _mm256_loadu_ps(tmp + (i * 6 + 2) * 8);
                    __m256 xi3 = _mm256_loadu_ps(tmp + (i * 6 + 3) * 8);
                    __m256 xi4 = _mm256_loadu_ps(tmp + (i * 6 + 4) * 8);
                    __m256 xi5 = _mm256_loadu_ps(tmp + (i * 6 + 5) * 8);

                    if (cc % 2 == 0) {
                        _mm_prefetch(curO + (6 * i) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * i + 1) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * i + 2) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * i + 3) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * i + 4) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * i + 5) * blockIc, _MM_HINT_NTA);
                    }
                    __m256 t0 = _mm256_fmadd_ps(minusFour, xi2, xi4);
                    __m256 t1 = _mm256_fmadd_ps(minusFour, xi1, xi3);
                    __m256 t2 = _mm256_sub_ps(xi4, xi2);
                    __m256 t3 = _mm256_mul_ps(two, _mm256_sub_ps(xi3, xi1));
                    __m256 t4 = _mm256_fmadd_ps(four, xi0, xi4);
                    __m256 t5 = _mm256_fmadd_ps(four, xi1, xi5);

                    xi0 = _mm256_fmadd_ps(minusFive, xi2, t4);
                    xi5 = _mm256_fmadd_ps(minusFive, xi3, t5);
                    xi1 = _mm256_add_ps(t1, t0);
                    xi2 = _mm256_sub_ps(t0, t1);
                    xi3 = _mm256_add_ps(t3, t2);
                    xi4 = _mm256_sub_ps(t2, t3);

                    _mm256_storeu_ps(curO + (6 * i) * blockIc, xi0);
                    _mm256_storeu_ps(curO + (6 * i + 1) * blockIc, xi1);
                    _mm256_storeu_ps(curO + (6 * i + 2) * blockIc, xi2);
                    _mm256_storeu_ps(curO + (6 * i + 3) * blockIc, xi3);
                    _mm256_storeu_ps(curO + (6 * i + 4) * blockIc, xi4);
                    _mm256_storeu_ps(curO + (6 * i + 5) * blockIc, xi5);
                }
            }
        }
    }
}

void transformInputWithPad4x4_3x3(F32 *input,
    F32 *output,
    F32 *tmp,
    U32 iw,
    U32 ih,
    U32 ic,
    U32 wSize,
    U32 blockIc,
    U32 pl,
    U32 pr,
    U32 pt,
    U32 pb,
    U32 h,
    U32 w,
    U32 oh,
    U32 ow)
{
    __m256 four = _mm256_set1_ps(4.0f);
    __m256 minusFour = _mm256_set1_ps(-4.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 minusFive = _mm256_set1_ps(-5.0f);
    U32 icb = ic / blockIc;
    U32 cb = blockIc / 8;

    pt = (h > pt) ? 0 : (pt - h);
    pl = (w > pl) ? 0 : (pl - w);
    for (U32 uw = 0; uw < wSize; ++uw) {
        for (U32 c = 0; c < icb; ++c) {
            for (U32 cc = 0; cc < cb; ++cc) {
                F32 *curI = input + (c * blockIc + cc * 8) * ih * iw;
                F32 *curO = output + (uw * ic + c * blockIc) * 36 + cc * 8;
                U32 i = 0;
                for (; ((i + w) < pl) && (i < 6); ++i) {
                    UNI_MEMSET(tmp + (i)*8, 0, 32);
                    UNI_MEMSET(tmp + (6 + i) * 8, 0, 32);
                    UNI_MEMSET(tmp + (6 * 2 + i) * 8, 0, 32);
                    UNI_MEMSET(tmp + (6 * 3 + i) * 8, 0, 32);
                    UNI_MEMSET(tmp + (6 * 4 + i) * 8, 0, 32);
                    UNI_MEMSET(tmp + (6 * 5 + i) * 8, 0, 32);
                }
                for (; ((i + w + pr) < (ow + 2)) && (i < 6); ++i) {
                    __m256 xi[6];
                    U32 b = 0;
                    for (; ((b + h) < pt) && (b < 6); ++b) {
                        xi[b] = _mm256_setzero_ps();
                    }
                    for (; ((b + h + pb) < (oh + 2)) && (b < 6); ++b) {
                        xi[b] = _mm256_loadu_ps(curI + (iw * (b - pt) + i + uw * 4 - pl) * 8);
                    }
                    for (; ((b + h) < (oh + 2)) && (b < 6); ++b) {
                        xi[b] = _mm256_setzero_ps();
                    }

                    __m256 t0 = _mm256_fmadd_ps(minusFour, xi[2], xi[4]);
                    __m256 t1 = _mm256_fmadd_ps(minusFour, xi[1], xi[3]);
                    __m256 t2 = _mm256_sub_ps(xi[4], xi[2]);
                    __m256 t3 = _mm256_mul_ps(two, _mm256_sub_ps(xi[3], xi[1]));
                    __m256 t4 = _mm256_fmadd_ps(four, xi[0], xi[4]);
                    __m256 t5 = _mm256_fmadd_ps(four, xi[1], xi[5]);

                    xi[0] = _mm256_fmadd_ps(minusFive, xi[2], t4);
                    xi[5] = _mm256_fmadd_ps(minusFive, xi[3], t5);
                    xi[1] = _mm256_add_ps(t1, t0);
                    xi[2] = _mm256_sub_ps(t0, t1);
                    xi[3] = _mm256_add_ps(t3, t2);
                    xi[4] = _mm256_sub_ps(t2, t3);

                    _mm256_storeu_ps(tmp + (i)*8, xi[0]);
                    _mm256_storeu_ps(tmp + (6 + i) * 8, xi[1]);
                    _mm256_storeu_ps(tmp + (6 * 2 + i) * 8, xi[2]);
                    _mm256_storeu_ps(tmp + (6 * 3 + i) * 8, xi[3]);
                    _mm256_storeu_ps(tmp + (6 * 4 + i) * 8, xi[4]);
                    _mm256_storeu_ps(tmp + (6 * 5 + i) * 8, xi[5]);
                }
                for (; ((i + w) < (ow + 2)) && (i < 6); ++i) {
                    UNI_MEMSET(tmp + (i)*8, 0, 32);
                    UNI_MEMSET(tmp + (6 + i) * 8, 0, 32);
                    UNI_MEMSET(tmp + (6 * 2 + i) * 8, 0, 32);
                    UNI_MEMSET(tmp + (6 * 3 + i) * 8, 0, 32);
                    UNI_MEMSET(tmp + (6 * 4 + i) * 8, 0, 32);
                    UNI_MEMSET(tmp + (6 * 5 + i) * 8, 0, 32);
                }

                for (U32 j = 0; j < 6; ++j) {
                    __m256 xi0 = _mm256_loadu_ps(tmp + (j * 6) * 8);
                    __m256 xi1 = _mm256_loadu_ps(tmp + (j * 6 + 1) * 8);
                    __m256 xi2 = _mm256_loadu_ps(tmp + (j * 6 + 2) * 8);
                    __m256 xi3 = _mm256_loadu_ps(tmp + (j * 6 + 3) * 8);
                    __m256 xi4 = _mm256_loadu_ps(tmp + (j * 6 + 4) * 8);
                    __m256 xi5 = _mm256_loadu_ps(tmp + (j * 6 + 5) * 8);

                    if (cc % 2 == 0) {
                        _mm_prefetch(curO + (6 * j) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * j + 1) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * j + 2) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * j + 3) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * j + 4) * blockIc, _MM_HINT_NTA);
                        _mm_prefetch(curO + (6 * j + 5) * blockIc, _MM_HINT_NTA);
                    }
                    __m256 t0 = _mm256_fmadd_ps(minusFour, xi2, xi4);
                    __m256 t1 = _mm256_fmadd_ps(minusFour, xi1, xi3);
                    __m256 t2 = _mm256_sub_ps(xi4, xi2);
                    __m256 t3 = _mm256_mul_ps(two, _mm256_sub_ps(xi3, xi1));
                    __m256 t4 = _mm256_fmadd_ps(four, xi0, xi4);
                    __m256 t5 = _mm256_fmadd_ps(four, xi1, xi5);

                    xi0 = _mm256_fmadd_ps(minusFive, xi2, t4);
                    xi5 = _mm256_fmadd_ps(minusFive, xi3, t5);
                    xi1 = _mm256_add_ps(t1, t0);
                    xi2 = _mm256_sub_ps(t0, t1);
                    xi3 = _mm256_add_ps(t3, t2);
                    xi4 = _mm256_sub_ps(t2, t3);

                    _mm256_storeu_ps(curO + (6 * j) * blockIc, xi0);
                    _mm256_storeu_ps(curO + (6 * j + 1) * blockIc, xi1);
                    _mm256_storeu_ps(curO + (6 * j + 2) * blockIc, xi2);
                    _mm256_storeu_ps(curO + (6 * j + 3) * blockIc, xi3);
                    _mm256_storeu_ps(curO + (6 * j + 4) * blockIc, xi4);
                    _mm256_storeu_ps(curO + (6 * j + 5) * blockIc, xi5);
                }
            }
        }
        w += 4;
    }
}

void transformOutput4x4_3x3(F32 *input,
    F32 *output,
    F32 *tmp,
    const F32 *bias,
    U32 ow,
    U32 oh,
    U32 oc,
    U32 wSize,
    bool addF,
    ActivationMode mode)
{
    I64 flag = (I64)addF | (I64(mode) << 1);
    __m256 four = _mm256_set1_ps(4.0f);
    __m256 eight = _mm256_set1_ps(8.0f);
    U32 ocb = oc / 8;
    for (U32 c = 0; c < ocb; ++c) {
        for (U32 w = 0; w < wSize; ++w) {
            F32 *curI = input + w * oc + c * 8;
            F32 *curO = output + w * 32 + c * 8 * oh * ow;
            I64 stepI = 24 * oc * wSize;
            I64 stepT = 192;
            for (U32 i = 0; i < 6; ++i) {
                F32 *useI0 = curI + i * oc * wSize;
                F32 *useI1 = useI0 + 18 * oc * wSize;
                F32 *useO0 = tmp + i * 8;
                F32 *useO1 = useO0 + 96;
                __asm__ __volatile__(
                    "vmovups (%[input0]), %%ymm0                             \n\t"
                    "vmovups (%[input0], %[stepI]), %%ymm1                             \n\t"
                    "vmovups (%[input0], %[stepI], 2), %%ymm2                             \n\t"
                    "vmovups (%[input1]), %%ymm3                             \n\t"
                    "vmovups (%[input1], %[stepI]), %%ymm4                             \n\t"
                    "vmovups (%[input1], %[stepI], 2), %%ymm5                             \n\t"
                    "vaddps %%ymm2, %%ymm1, %%ymm6                             \n\t"
                    "vaddps %%ymm3, %%ymm4, %%ymm7                             \n\t"
                    "vsubps %%ymm2, %%ymm1, %%ymm8                             \n\t"
                    "vsubps %%ymm4, %%ymm3, %%ymm9                             \n\t"
                    "vaddps %%ymm6, %%ymm7, %%ymm1                             \n\t"
                    "vaddps %%ymm9, %%ymm9, %%ymm3                             \n\t"
                    "vaddps %%ymm0, %%ymm1, %%ymm11                             \n\t"  // xi0
                    "vaddps %%ymm8, %%ymm3, %%ymm12                             \n\t"  // xi1
                    "vmovups %%ymm11, (%[output0])                             \n\t"
                    "vmovups %%ymm12, (%[output0], %[stepT])                    \n\t"
                    "vfmadd231ps %[eight], %%ymm9, %%ymm8              \n\t"
                    "vfmadd231ps %[four], %%ymm7, %%ymm6              \n\t"
                    "vaddps %%ymm5, %%ymm8, %%ymm10                             \n\t"  // xi3
                    "vmovups %%ymm6, (%[output1])                             \n\t"
                    "vmovups %%ymm10, (%[output1], %[stepT])                     \n\t"
                    :
                    : [input0] "r"(useI0), [input1] "r"(useI1), [stepI] "r"(stepI), [output0] "r"(useO0),
                    [output1] "r"(useO1), [stepT] "r"(stepT), [four] "x"(four), [eight] "x"(eight)
                    : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                    "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "memory", "cc");
            }

            stepT = 32;
            stepI = 32;
            for (U32 i = 0; i < 4; ++i) {
                F32 *useI0 = tmp + 48 * i;
                F32 *useI1 = useI0 + 24;
                F32 *useO0 = curO + ow * i * 8;
                F32 *useO1 = useO0 + 16;
                __asm__ __volatile__(
                    "vmovups (%[input0]), %%ymm0                             \n\t"
                    "vmovups (%[input0], %[stepI]), %%ymm1                             \n\t"
                    "vmovups (%[input0], %[stepI], 2), %%ymm2                             \n\t"
                    "vmovups (%[input1]), %%ymm3                             \n\t"
                    "vmovups (%[input1], %[stepI]), %%ymm4                             \n\t"
                    "vmovups (%[input1], %[stepI], 2), %%ymm5                             \n\t"
                    "prefetcht0 (%[output0])                              \n\t"
                    "prefetcht0 (%[output1])                              \n\t"
                    "vaddps %%ymm2, %%ymm1, %%ymm6                             \n\t"
                    "vaddps %%ymm3, %%ymm4, %%ymm7                             \n\t"
                    "vsubps %%ymm2, %%ymm1, %%ymm8                             \n\t"
                    "vsubps %%ymm4, %%ymm3, %%ymm9                             \n\t"
                    "vaddps %%ymm6, %%ymm7, %%ymm1                             \n\t"
                    "vaddps %%ymm9, %%ymm9, %%ymm3                             \n\t"
                    "vaddps %%ymm0, %%ymm1, %%ymm11                             \n\t"  // xi0
                    "vaddps %%ymm8, %%ymm3, %%ymm12                             \n\t"  // xi1
                    "vfmadd231ps %[eight], %%ymm9, %%ymm8              \n\t"
                    "vfmadd231ps %[four], %%ymm7, %%ymm6              \n\t"
                    "vaddps %%ymm5, %%ymm8, %%ymm10                             \n\t"  // xi3
                    "mov %[flag], %%rax                                      \n\t"
                    "and $0x1, %%rax                                      \n\t"
                    "je 0f                                             \n\t"
                    "vaddps (%[output0]), %%ymm11, %%ymm11                             \n\t"
                    "vaddps (%[output0], %[stepT]), %%ymm12, %%ymm12                    \n\t"
                    "vaddps (%[output1]), %%ymm6, %%ymm6                             \n\t"
                    "vaddps (%[output1], %[stepT]), %%ymm10, %%ymm10                     \n\t"
                    "jmp 1f                                             \n\t"
                    ".align 16                                         \n\t"
                    "0:                                                \n\t"
                    "vmovups (%[bias]), %%ymm0                             \n\t"
                    "vaddps %%ymm0, %%ymm11, %%ymm11                             \n\t"
                    "vaddps %%ymm0, %%ymm12, %%ymm12                    \n\t"
                    "vaddps %%ymm0, %%ymm6, %%ymm6                             \n\t"
                    "vaddps %%ymm0, %%ymm10, %%ymm10                     \n\t"
                    ".align 16                                         \n\t"
                    "1:                                                \n\t"
                    "mov %[flag], %%rax                                      \n\t"
                    "or $0x1, %%rax                                      \n\t"
                    "cmp $0x3, %%rax                                      \n\t"
                    "jne 2f                                             \n\t"
                    "vxorps %%ymm0, %%ymm0, %%ymm0                  \n\t"
                    "vmaxps %%ymm0, %%ymm6, %%ymm6                    \n\t"
                    "vmaxps %%ymm0, %%ymm12, %%ymm12                    \n\t"
                    "vmaxps %%ymm0, %%ymm10, %%ymm10                  \n\t"
                    "vmaxps %%ymm0, %%ymm11, %%ymm11                  \n\t"
                    ".align 16                                         \n\t"
                    "2:                                                \n\t"
                    "vmovups %%ymm11, (%[output0])                             \n\t"
                    "vmovups %%ymm12, (%[output0], %[stepT])                    \n\t"
                    "vmovups %%ymm6, (%[output1])                             \n\t"
                    "vmovups %%ymm10, (%[output1], %[stepT])                     \n\t"
                    :
                    : [input0] "r"(useI0), [input1] "r"(useI1), [stepI] "r"(stepI),
                    [output0] "r"(useO0), [output1] "r"(useO1), [stepT] "r"(stepT),
                    [four] "x"(four), [eight] "x"(eight), [flag] "r"(flag), [bias] "r"(bias)
                    : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                    "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "memory", "cc");
            }
        }
        bias += 8;
    }
}

void transformOutputWithPad4x4_3x3(F32 *input,
    F32 *output,
    F32 *tmp,
    const F32 *bias,
    U32 ow,
    U32 oh,
    U32 oc,
    U32 wSize,
    bool addF,
    U32 pr,
    U32 pb,
    U32 h,
    U32 w,
    ActivationMode mode)
{
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 four = _mm256_set1_ps(4.0f);
    __m256 eight = _mm256_set1_ps(8.0f);
    U32 ocb = oc / 8;
    for (U32 c = 0; c < ocb; ++c) {
        for (U32 uw = 0; uw < wSize; ++uw) {
            F32 *curI = input + uw * oc + c * 8;
            F32 *curO = output + uw * 32 + c * 8 * oh * ow;
            I64 stepI = 24 * oc * wSize;
            I64 stepT = 192;
            for (U32 i = 0; i < 6; ++i) {
                F32 *useI0 = curI + i * oc * wSize;
                F32 *useI1 = useI0 + 18 * oc * wSize;
                F32 *useO0 = tmp + i * 8;
                F32 *useO1 = useO0 + 96;
                __asm__ __volatile__(
                    "vmovups (%[input0]), %%ymm0                             \n\t"
                    "vmovups (%[input0], %[stepI]), %%ymm1                             \n\t"
                    "vmovups (%[input0], %[stepI], 2), %%ymm2                             \n\t"
                    "vmovups (%[input1]), %%ymm3                             \n\t"
                    "vmovups (%[input1], %[stepI]), %%ymm4                             \n\t"
                    "vmovups (%[input1], %[stepI], 2), %%ymm5                             \n\t"
                    "vaddps %%ymm2, %%ymm1, %%ymm6                             \n\t"
                    "vaddps %%ymm3, %%ymm4, %%ymm7                             \n\t"
                    "vsubps %%ymm2, %%ymm1, %%ymm8                             \n\t"
                    "vsubps %%ymm4, %%ymm3, %%ymm9                             \n\t"
                    "vaddps %%ymm6, %%ymm7, %%ymm1                             \n\t"
                    "vaddps %%ymm9, %%ymm9, %%ymm3                             \n\t"
                    "vaddps %%ymm0, %%ymm1, %%ymm11                             \n\t"  // xi0
                    "vaddps %%ymm8, %%ymm3, %%ymm12                             \n\t"  // xi1
                    "vmovups %%ymm11, (%[output0])                             \n\t"
                    "vmovups %%ymm12, (%[output0], %[stepT])                    \n\t"
                    "vfmadd231ps %[eight], %%ymm9, %%ymm8              \n\t"
                    "vfmadd231ps %[four], %%ymm7, %%ymm6              \n\t"
                    "vaddps %%ymm5, %%ymm8, %%ymm10                             \n\t"  // xi3
                    "vmovups %%ymm6, (%[output1])                             \n\t"
                    "vmovups %%ymm10, (%[output1], %[stepT])                     \n\t"
                    :
                    : [input0] "r"(useI0), [input1] "r"(useI1), [stepI] "r"(stepI), [output0] "r"(useO0),
                    [output1] "r"(useO1), [stepT] "r"(stepT), [four] "x"(four), [eight] "x"(eight)
                    : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6",
                    "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "memory", "cc");
            }
            for (U32 i = 0; (i < 4) && (i + h < oh); ++i) {
                __m256 xi[6];
                for (U32 j = 0; j < 6; ++j) {
                    xi[j] = _mm256_loadu_ps(tmp + (6 * i + j) * 8);
                }

                __m256 t0 = _mm256_add_ps(xi[1], xi[2]);
                __m256 t1 = _mm256_add_ps(xi[4], xi[3]);
                __m256 t2 = _mm256_sub_ps(xi[1], xi[2]);
                __m256 t3 = _mm256_sub_ps(xi[3], xi[4]);

                xi[0] = _mm256_add_ps(_mm256_add_ps(t0, t1), xi[0]);
                xi[1] = _mm256_fmadd_ps(two, t3, t2);
                xi[2] = _mm256_fmadd_ps(four, t1, t0);
                xi[3] = _mm256_add_ps(_mm256_fmadd_ps(eight, t3, t2), xi[5]);

                if (addF) {
                    for (U32 j = 0; (j < 4) && (j + w + uw * 4 < ow); ++j) {
                        xi[j] = _mm256_add_ps(xi[j],
                            _mm256_loadu_ps(output + (ow * i + uw * 4 + j) * 8 + c * 8 * oh * ow));
                    }
                } else {
                    __m256 b = _mm256_loadu_ps(bias + c * 8);
                    for (U32 j = 0; (j < 4) && (j + w + uw * 4 < ow); ++j) {
                        xi[j] = _mm256_add_ps(xi[j], b);
                    }
                }

                if (mode) {
                    __m256 zero = _mm256_setzero_ps();
                    for (U32 j = 0; (j < 4) && (j + w + uw * 4 < ow); ++j) {
                        xi[j] = _mm256_max_ps(xi[j], zero);
                    }
                }

                for (U32 j = 0; (j < 4) && (j + w + uw * 4 < ow); ++j) {
                    _mm256_storeu_ps(output + (ow * i + uw * 4 + j) * 8 + c * 8 * oh * ow, xi[j]);
                }
            }
        }
    }
}

struct ConvController {
    F32 **input;
    const F32 *filter;
    void *output;
    F32 *eltwise;
    I64 ic;
    I64 fStep;
    I64 flags;
};

typedef void (*kernelFunc)(ConvController &c);

void winoKernel3x32(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                             \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4                             \n\t"
        "vxorps %%ymm5, %%ymm5, %%ymm5                             \n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6                             \n\t"
        "vxorps %%ymm7, %%ymm7, %%ymm7                             \n\t"
        "vxorps %%ymm8, %%ymm8, %%ymm8                             \n\t"
        "vxorps %%ymm9, %%ymm9, %%ymm9                             \n\t"
        "vxorps %%ymm10, %%ymm10, %%ymm10                             \n\t"
        "vxorps %%ymm11, %%ymm11, %%ymm11                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vbroadcastss (%[input1]), %%ymm13                 \n\t"
        "vbroadcastss (%[input2]), %%ymm14              \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x100(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x140(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x4(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x4(%[input2]), %%ymm14              \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x180(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x1C0(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x8(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x8(%[input2]), %%ymm14              \n\t"
        "vmovups 0x100(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x200(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x120(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x140(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x240(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
        "vmovups 0x160(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0xC(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0xC(%[input2]), %%ymm14              \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x280(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x2C0(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x10(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x10(%[input2]), %%ymm14              \n\t"
        "vmovups 0x200(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x300(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x220(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x240(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x340(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
        "vmovups 0x260(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x14(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x14(%[input2]), %%ymm14              \n\t"
        "vmovups 0x280(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x380(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x2A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x2C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x3C0(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
        "vmovups 0x2E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x18(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x18(%[input2]), %%ymm14              \n\t"
        "vmovups 0x300(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x400(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x320(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x340(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x440(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
        "vmovups 0x360(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x1C(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x1C(%[input2]), %%ymm14              \n\t"
        "vmovups 0x380(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x480(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x3A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x3C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x4C0(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"
        "vmovups 0x3E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm11             \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x20, %[input1]                                         \n\t"
        "add $0x20, %[input2]                                         \n\t"
        "add $0x400, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        "vmovups %%ymm6, 0x40(%[output])                             \n\t"
        "vmovups %%ymm9, 0x60(%[output])                            \n\t"
        "vmovups %%ymm1, 0x80(%[output])                             \n\t"
        "vmovups %%ymm4, 0xA0(%[output])                             \n\t"
        "vmovups %%ymm7, 0xC0(%[output])                             \n\t"
        "vmovups %%ymm10, 0xE0(%[output])                             \n\t"
        "vmovups %%ymm2, 0x100(%[output])                             \n\t"
        "vmovups %%ymm5, 0x120(%[output])                             \n\t"
        "vmovups %%ymm8, 0x140(%[output])                             \n\t"
        "vmovups %%ymm11, 0x160(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel2x32(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4                             \n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6                             \n\t"
        "vxorps %%ymm7, %%ymm7, %%ymm7                             \n\t"
        "vxorps %%ymm9, %%ymm9, %%ymm9                             \n\t"
        "vxorps %%ymm10, %%ymm10, %%ymm10                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vbroadcastss (%[input1]), %%ymm13                 \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "prefetcht0 0x100(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "prefetcht0 0x140(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x4(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x180(%[filter])                              \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x1C0(%[filter])                              \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x8(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x100(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x200(%[filter])                              \n\t"
        "vmovups 0x120(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x140(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x240(%[filter])                              \n\t"
        "vmovups 0x160(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0xC(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x280(%[filter])                              \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x2C0(%[filter])                              \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x10(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x200(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x300(%[filter])                              \n\t"
        "vmovups 0x220(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x240(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x340(%[filter])                              \n\t"
        "vmovups 0x260(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x14(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x280(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x380(%[filter])                              \n\t"
        "vmovups 0x2A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x2C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x3C0(%[filter])                              \n\t"
        "vmovups 0x2E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x18(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x300(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x400(%[filter])                              \n\t"
        "vmovups 0x320(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x340(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x440(%[filter])                              \n\t"
        "vmovups 0x360(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x1C(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x380(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x480(%[filter])                              \n\t"
        "vmovups 0x3A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x3C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "prefetcht0 0x4C0(%[filter])                              \n\t"
        "vmovups 0x3E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm9              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm10             \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x20, %[input1]                                         \n\t"
        "add $0x400, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        "vmovups %%ymm6, 0x40(%[output])                             \n\t"
        "vmovups %%ymm9, 0x60(%[output])                            \n\t"
        "vmovups %%ymm1, 0x80(%[output])                             \n\t"
        "vmovups %%ymm4, 0xA0(%[output])                             \n\t"
        "vmovups %%ymm7, 0xC0(%[output])                             \n\t"
        "vmovups %%ymm10, 0xE0(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel1x32(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6                             \n\t"
        "vxorps %%ymm9, %%ymm9, %%ymm9                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"

        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x20(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x40(%[filter]), %%ymm13                         \n\t"
        "vmovups 0x60(%[filter]), %%ymm14                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vmovups 0xA0(%[filter]), %%ymm10                         \n\t"
        "vmovups 0xC0(%[filter]), %%ymm13                         \n\t"
        "vmovups 0xE0(%[filter]), %%ymm14                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x100(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x120(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x140(%[filter]), %%ymm13                         \n\t"
        "vmovups 0x160(%[filter]), %%ymm14                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm13                         \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm14                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x200(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x220(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x240(%[filter]), %%ymm13                         \n\t"
        "vmovups 0x260(%[filter]), %%ymm14                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x280(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x2A0(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x2C0(%[filter]), %%ymm13                         \n\t"
        "vmovups 0x2E0(%[filter]), %%ymm14                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x300(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x320(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x340(%[filter]), %%ymm13                         \n\t"
        "vmovups 0x360(%[filter]), %%ymm14                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x380(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x3A0(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x3C0(%[filter]), %%ymm13                         \n\t"
        "vmovups 0x3E0(%[filter]), %%ymm14                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm14, %%ymm12, %%ymm9              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x400, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        "vmovups %%ymm6, 0x40(%[output])                             \n\t"
        "vmovups %%ymm9, 0x60(%[output])                            \n\t"
        :
        : [input0] "r"(c.input[0]), [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel3x24(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                             \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4                             \n\t"
        "vxorps %%ymm5, %%ymm5, %%ymm5                             \n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6                             \n\t"
        "vxorps %%ymm7, %%ymm7, %%ymm7                             \n\t"
        "vxorps %%ymm8, %%ymm8, %%ymm8                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vbroadcastss (%[input1]), %%ymm13                 \n\t"
        "vbroadcastss (%[input2]), %%ymm14              \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x4(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x4(%[input2]), %%ymm14              \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x8(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x8(%[input2]), %%ymm14              \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x100(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0xC(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0xC(%[input2]), %%ymm14              \n\t"
        "vmovups 0x120(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x140(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x160(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x10(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x10(%[input2]), %%ymm14              \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x14(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x14(%[input2]), %%ymm14              \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x200(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x220(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x18(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x18(%[input2]), %%ymm14              \n\t"
        "vmovups 0x240(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x260(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x280(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x1C(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x1C(%[input2]), %%ymm14              \n\t"
        "vmovups 0x2A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x2C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"
        "vmovups 0x2E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm8              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x20, %[input1]                                         \n\t"
        "add $0x20, %[input2]                                         \n\t"
        "add $0x300, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        "vmovups %%ymm6, 0x40(%[output])                             \n\t"
        "vmovups %%ymm1, 0x60(%[output])                             \n\t"
        "vmovups %%ymm4, 0x80(%[output])                             \n\t"
        "vmovups %%ymm7, 0xA0(%[output])                             \n\t"
        "vmovups %%ymm2, 0xC0(%[output])                             \n\t"
        "vmovups %%ymm5, 0xE0(%[output])                             \n\t"
        "vmovups %%ymm8, 0x100(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel2x24(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4                             \n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6                             \n\t"
        "vxorps %%ymm7, %%ymm7, %%ymm7                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vbroadcastss (%[input1]), %%ymm13                 \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "prefetcht0 0x100(%[filter])                              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x4(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x180(%[filter])                              \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x8(%[input1]), %%ymm13                 \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x200(%[filter])                              \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x100(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0xC(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x120(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x280(%[filter])                              \n\t"
        "vmovups 0x140(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x160(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x10(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x300(%[filter])                              \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x14(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x380(%[filter])                              \n\t"
        "vmovups 0x200(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x220(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x18(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x240(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x400(%[filter])                              \n\t"
        "vmovups 0x260(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x280(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x1C(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x2A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "prefetcht0 0x480(%[filter])                              \n\t"
        "vmovups 0x2C0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vmovups 0x2E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm6              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm7              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x20, %[input1]                                         \n\t"
        "add $0x300, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        "vmovups %%ymm6, 0x40(%[output])                             \n\t"
        "vmovups %%ymm1, 0x60(%[output])                             \n\t"
        "vmovups %%ymm4, 0x80(%[output])                             \n\t"
        "vmovups %%ymm7, 0xA0(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel1x24(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"
        "vxorps %%ymm6, %%ymm6, %%ymm6                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"

        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x20(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x40(%[filter]), %%ymm13                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x80(%[filter]), %%ymm10                         \n\t"
        "vmovups 0xA0(%[filter]), %%ymm13                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                          \n\t"
        "vmovups 0xE0(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x100(%[filter]), %%ymm13                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x120(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x140(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x160(%[filter]), %%ymm13                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm13                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x200(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x220(%[filter]), %%ymm13                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x240(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x260(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x280(%[filter]), %%ymm13                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x2A0(%[filter]), %%ymm15                          \n\t"
        "vmovups 0x2C0(%[filter]), %%ymm10                         \n\t"
        "vmovups 0x2E0(%[filter]), %%ymm13                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm10, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm13, %%ymm12, %%ymm6              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x300, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        "vmovups %%ymm6, 0x40(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel3x16(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                             \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4                             \n\t"
        "vxorps %%ymm5, %%ymm5, %%ymm5                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vbroadcastss (%[input1]), %%ymm13                 \n\t"
        "vbroadcastss (%[input2]), %%ymm14              \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x4(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x4(%[input2]), %%ymm14              \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x8(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x8(%[input2]), %%ymm14              \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0xC(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0xC(%[input2]), %%ymm14              \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x10(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x10(%[input2]), %%ymm14              \n\t"
        "vmovups 0x100(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x120(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x14(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x14(%[input2]), %%ymm14              \n\t"
        "vmovups 0x140(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x160(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x18(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x18(%[input2]), %%ymm14              \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x1C(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x1C(%[input2]), %%ymm14              \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm5              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x20, %[input1]                                         \n\t"
        "add $0x20, %[input2]                                         \n\t"
        "add $0x200, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        "vmovups %%ymm1, 0x40(%[output])                             \n\t"
        "vmovups %%ymm4, 0x60(%[output])                             \n\t"
        "vmovups %%ymm2, 0x80(%[output])                             \n\t"
        "vmovups %%ymm5, 0xA0(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel2x16(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"
        "vxorps %%ymm4, %%ymm4, %%ymm4                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vbroadcastss (%[input1]), %%ymm13                 \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x4(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x8(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0xC(%[input1]), %%ymm13                 \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x10(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x100(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0x120(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x14(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x140(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0x160(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x18(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x1C(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm4              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x20, %[input1]                                         \n\t"
        "add $0x200, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        "vmovups %%ymm1, 0x40(%[output])                             \n\t"
        "vmovups %%ymm4, 0x60(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel1x16(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm3, %%ymm3, %%ymm3                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x100(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x120(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x140(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x160(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x180(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x1A0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x1C0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vmovups 0x1E0(%[filter]), %%ymm15                         \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm3              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x200, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm3, 0x20(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel3x8(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                             \n\t"
        "vxorps %%ymm2, %%ymm2, %%ymm2                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vbroadcastss (%[input1]), %%ymm13                 \n\t"
        "vbroadcastss (%[input2]), %%ymm14              \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x4(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x4(%[input2]), %%ymm14              \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x8(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x8(%[input2]), %%ymm14              \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0xC(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0xC(%[input2]), %%ymm14              \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x10(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x10(%[input2]), %%ymm14              \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x14(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x14(%[input2]), %%ymm14              \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x18(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x18(%[input2]), %%ymm14              \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x1C(%[input1]), %%ymm13                 \n\t"
        "vbroadcastss 0x1C(%[input2]), %%ymm14              \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %%ymm2              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x20, %[input1]                                         \n\t"
        "add $0x20, %[input2]                                         \n\t"
        "add $0x100, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm1, 0x20(%[output])                             \n\t"
        "vmovups %%ymm2, 0x40(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel2x8(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"
        "vxorps %%ymm1, %%ymm1, %%ymm1                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vbroadcastss (%[input1]), %%ymm13                 \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x4(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x8(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0xC(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x10(%[input1]), %%ymm13                 \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x14(%[input1]), %%ymm13                 \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x18(%[input1]), %%ymm13                 \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vbroadcastss 0x1C(%[input1]), %%ymm13                 \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %%ymm1              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x20, %[input1]                                         \n\t"
        "add $0x100, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        "vmovups %%ymm1, 0x20(%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

void winoKernel1x8(ConvController &c)
{
    __asm__ __volatile__(
        "vxorps %%ymm0, %%ymm0, %%ymm0                             \n\t"

        ".align 16                                         \n\t"
        "1:                                                \n\t"
        "vbroadcastss (%[input0]), %%ymm12                        \n\t"
        "vmovups 0x0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

        "vbroadcastss 0x4(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x20(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

        "vbroadcastss 0x8(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x40(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

        "vbroadcastss 0xC(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x60(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

        "vbroadcastss 0x10(%[input0]), %%ymm12                        \n\t"
        "vmovups 0x80(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

        "vbroadcastss 0x14(%[input0]), %%ymm12                        \n\t"
        "vmovups 0xA0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

        "vbroadcastss 0x18(%[input0]), %%ymm12                        \n\t"
        "vmovups 0xC0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

        "vbroadcastss 0x1C(%[input0]), %%ymm12                        \n\t"
        "vmovups 0xE0(%[filter]), %%ymm15                          \n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %%ymm0              \n\t"

        "add $0x20, %[input0]                                         \n\t"
        "add $0x100, %[filter]                                         \n\t"
        "dec %%rcx                                         \n\t"
        "jg 1b                                             \n\t"

        "vmovups %%ymm0, (%[output])                             \n\t"
        :
        : [input0] "r"(c.input[0]), [input1] "r"(c.input[1]), [input2] "r"(c.input[2]),
        [filter] "r"(c.filter), [output] "r"(c.output), [ic] "c"(c.ic)
        : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8",
        "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15", "memory", "cc");
}

EE convolution_winograd(TensorDesc inputDesc,
    F32 *inArray,
    F32 *eltwiseInput,
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

    if ((fdf != DF_NCHWCxN32 && fdf != DF_NCHWCxN24) || (idf != DF_NCHWC8) || (ic % 8 != 0)) {
        CHECK_STATUS(NOT_MATCH);
    }

    if (activationDesc.mode != ACTIVATION_RELU && activationDesc.mode != ACTIVATION_NULL) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    // get kernels
    const kernelFunc wino[4][3] = {
        {winoKernel1x8, winoKernel2x8,   winoKernel3x8},
        {winoKernel1x16, winoKernel2x16, winoKernel3x16},
        {winoKernel1x24, winoKernel2x24, winoKernel3x24},
        {winoKernel1x32, winoKernel2x32, winoKernel3x32}
    };

    // get computing params
    I32 strideH = convParamSpec.stride_h;
    I32 strideW = convParamSpec.stride_w;
    I32 paddingT = convParamSpec.pad_top;
    I32 paddingB = convParamSpec.pad_bottom;
    I32 paddingL = convParamSpec.pad_left;
    I32 paddingR = convParamSpec.pad_right;
    I32 dilateH = convParamSpec.dilatedRate_h;
    I32 dilateW = convParamSpec.dilatedRate_w;
    I32 ih_pad = ih + paddingT + paddingB;
    I32 iw_pad = iw + paddingL + paddingR;
    I32 ohow = oh * ow;

    I32 oPaddingR = (ow % 4 == 0) ? 0 : (4 - ow % 4);
    I32 oPaddingB = (oh % 4 == 0) ? 0 : (4 - oh % 4);
    I32 oh_pad = oh + oPaddingB;
    I32 ow_pad = ow + oPaddingR;
    paddingR += oPaddingR;
    paddingB += oPaddingB;

    // infer block params
    I32 ocBlockSizes[] = {8, 16, 24, 32};
    I32 wSizes[] = {1, 2, 3};
    I32 hLoop = oh_pad / 4;
    I32 ocLoop = (oc + BLOCK_OC_DIM - 1) / BLOCK_OC_DIM;
    I32 hOcLoop = hLoop * ocLoop;

    // infer kernel params
    bool noPadI = (paddingT == 0 && paddingB == 0 && paddingL == 0 && paddingR == 0);
    bool noPadO = (oPaddingB == 0 && oPaddingR == 0);

#ifdef _USE_OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS) if (hLoop > (OMP_NUM_THREADS * 2))
#endif
        {
    for (U32 n = 0; n < in; ++n) {
        F32 *bInArray = inArray + n * ic * ih * iw;
        F32 *bOutArray = outArray + n * oc * oh * ow;

        I32 icSize = 0;
        bool addF = false;
        ActivationMode mode = ACTIVATION_NULL;
        for (I32 icb = 0; icb < (int)ic; icb += icSize) {
            icSize = UNI_MIN(BLOCK_IC_DIM, (int)ic - icb);
            addF = (icb > 0);
            if (icb == (int)ic - icSize) {
                mode = activationDesc.mode;
            }

#ifdef _USE_OPENMP
#pragma omp for schedule(static)
#endif
            for (I32 l = 0; l < hLoop; ++l) {
                I32 h = l * 4;

#ifdef _USE_OPENMP
                U32 nTmpBytes = (36 * 32 + 36 * 36) * 3 + 36 * icSize * (ow_pad / 4 + 1);
                F32 *outer_tmp = (F32 *)tmp + nTmpBytes * omp_get_thread_num();
                F32 *thread_in_tmp = (F32 *)outer_tmp + 36 * icSize * (ow_pad / 4 + 1);
                F32 *ibuff = (F32 *)outer_tmp;
#else
                F32 *thread_in_tmp = (F32 *)tmp + 36 * icSize * (ow_pad / 4 + 1);
                F32 *ibuff = (F32 *)tmp;
#endif

                for (I32 ocl = 0; ocl < ocLoop; ++ocl) {
                    I32 ocb = ocl * BLOCK_OC_DIM;
                    I32 ocSize = UNI_MIN(BLOCK_OC_DIM, (int)oc - ocb);
                    ocSize = ocBlockSizes[(ocSize >> 3) - 1];
                    const F32 *bias = biasArray + ocb;

                    I32 wSize = 0;
                    for (I32 w = 0; w < ow_pad; w += 4 * wSize) {
                        wSize = UNI_MIN((int)ow_pad - w, 12);
                        wSize = wSize >> 2;
                        I32 in_w = w * strideW;
                        I32 in_h = h * strideH;
                        F32 *curI;
                        F32 *curO = bOutArray + ocb * oh * ow + (h * ow + w) * 8;
                        F32 *tmpI = ibuff + 36 * icSize * w / 4;
                        F32 *buff = thread_in_tmp;
                        F32 *tmpO = (F32 *)buff + 36 * 36 * wSize;

                        if (ocb == 0) {
                            if (noPadI) {
                                curI = bInArray + icb * ih * iw + (in_h * iw + in_w) * 8;
                                transformInput4x4_3x3(
                                    curI, tmpI, buff, iw, ih, icSize, wSize, icSize);
                            } else {
                                in_w = (in_w > paddingL) ? (in_w - paddingL) : 0;
                                in_h = (in_h > paddingT) ? (in_h - paddingT) : 0;
                                curI = bInArray + icb * ih * iw + (in_h * iw + in_w) * 8;
                                transformInputWithPad4x4_3x3(curI, tmpI, buff, iw, ih, icSize,
                                    wSize, icSize, paddingL, paddingR, paddingT, paddingB, h, w,
                                    oh_pad, ow_pad);
                            }
                        }

                        ConvController convCtl;
                        convCtl.eltwise = nullptr;
                        F32 *iaddr[3];
                        convCtl.input = iaddr;
                        for (I32 i = 0; i < 36; ++i) {
                            convCtl.ic = icSize / 8;
                            convCtl.input[0] = tmpI + i * icSize;
                            convCtl.input[1] = tmpI + icSize * 36 * 1 + i * icSize;
                            convCtl.input[2] = tmpI + icSize * 36 * 2 + i * icSize;
                            convCtl.output = tmpO + i * ocSize * wSize;
                            convCtl.filter = filterArray + icb * fn * 36 + ocb * icSize * 36 +
                                i * ocSize * icSize;
                            wino[(ocSize >> 3) - 1][wSize - 1](convCtl);
                        }
                        if (noPadO) {
                            transformOutput4x4_3x3(
                                tmpO, curO, buff, bias, ow, oh, ocSize, wSize, addF, mode);
                        } else {
                            transformOutputWithPad4x4_3x3(tmpO, curO, buff, bias, ow, oh, ocSize,
                                wSize, addF, oPaddingR, oPaddingB, h, w, mode);
                        }
                    }

                }
            }
        }
    }
    }
    return SUCCESS;
}
