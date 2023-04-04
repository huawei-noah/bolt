// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_MATRIX_TRANSPOSE
#define _H_BLAS_MATRIX_TRANSPOSE

#include "data_type.h"

#define LD_F16_SR_BF16_4(src, dst)                           \
    asm volatile("ldr d0, [%[in]]\n"                         \
                 "fcvtl v1.4s, v0.4h\n"                      \
                 ".inst 0xea16820  // bfcvtn v0.4h, v1.4s\n" \
                 "str d0, [%[out]]\n"                        \
                 : [in] "+r"(src), [out] "+r"(dst)           \
                 :                                           \
                 : "memory", "cc", "v0", "v1");

// Trans from NK to NKn(size)k4
template <int kk = 4>
void matrix1_trans_fp16(int size, int blockK, int K, F16 *src, F16 *dst)
{
    CHECK_REQUIREMENT(kk == 4);
    int prefetch = 64 / kk;
    int i = 0;
    for (; i < blockK - kk + 1; i += kk) {
        F16 *src1 = src + i;
        for (int j = 0; j < size; j++) {
            if (i % prefetch == 0) {
                asm volatile("prfm pldl2keep, [%0, %1]\n" : "+r"(src1) : "r"((size_t)64) : "memory", "cc");
            }
            LD_F16_SR_BF16_4(src1, dst);
            src1 += K;
            dst += kk;
        }
    }
    if (i < blockK) {
        int kTail = blockK - i;
        F16 buffer[kk];
        F16 *p = buffer;
        UNI_MEMSET(p + kTail, 0, sizeof(BF16) * (kk - kTail));
        F16 *src1 = src + i;
        int validSize = kTail * sizeof(F16);
        for (int j = 0; j < size; j++) {
            UNI_MEMCPY(p, src1, validSize);
            LD_F16_SR_BF16_4(p, dst);
            src1 += K;
            dst += kk;
        }
    }
}

// Trans from KM to MKm(size)k4
template <int kk = 4>
inline void matrix2_trans_fp16(int size, int blockK, int M, F16 *src, F16 *dst)
{
    CHECK_REQUIREMENT(kk == 4);
    int offset = kk * M;
    F16 buffer[kk];
    F16 *p = buffer;
    int i = 0;
    for (; i < blockK - kk + 1; i += kk) {
        F16 *src1 = src + i * M;
        asm volatile("prfm pldl2keep, [%0, %1]\n" : "+r"(src1) : "r"((I64)offset) : "memory", "cc");
        for (int j = 0; j < size; j++) {
            F16 *src1 = src + i * M + j;
            for (int k = 0; k < kk; k++) {
                p[k] = *src1;
                src1 += M;
            }
            LD_F16_SR_BF16_4(p, dst);
            dst += kk;
        }
    }
    if (i < blockK) {
        int kTail = blockK - i;
        UNI_MEMSET(p + kTail, 0, sizeof(BF16) * (kk - kTail));
        for (int j = 0; j < size; j++) {
            F16 *src1 = src + i * M + j;
            for (int k = 0; k < kTail; k++) {
                p[k] = *src1;
                src1 += M;
            }
            LD_F16_SR_BF16_4(p, dst);
            dst += kk;
        }
    }
}
#endif
