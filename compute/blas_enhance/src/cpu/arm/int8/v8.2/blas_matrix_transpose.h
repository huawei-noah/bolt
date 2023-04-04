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

inline void matrix1_trans_n8(int blockK, int K, INT8 *src, INT8 *dst)
{
    // Move k4 as one I32
    I32 *dst1 = (I32 *)dst;

    I32 *in[8];
    for (int i = 0; i < 8; i++) {
        in[i] = (I32 *)(src + i * K);
    }
    int k = 0;
    for (; k < blockK - 7; k += 8) {
        if (k % 64 == 0) {
            asm volatile(
                "prfm pldl2keep, [%[in0], 64]\n"
                "prfm pldl2keep, [%[in1], 64]\n"
                "prfm pldl2keep, [%[in2], 64]\n"
                "prfm pldl2keep, [%[in3], 64]\n"
                "prfm pldl2keep, [%[in4], 64]\n"
                "prfm pldl2keep, [%[in5], 64]\n"
                "prfm pldl2keep, [%[in6], 64]\n"
                "prfm pldl2keep, [%[in7], 64]\n"
                : [in0] "+r"(in[0]), [in1] "+r"(in[1]), [in2] "+r"(in[2]), [in3] "+r"(in[3]),
                [in4] "+r"(in[4]), [in5] "+r"(in[5]), [in6] "+r"(in[6]), [in7] "+r"(in[7])
                :
                : "memory", "cc");
        }
        asm volatile("ldr d0, [%[in0]], 8\n"
                     "ldr d1, [%[in1]], 8\n"
                     "ldr d2, [%[in2]], 8\n"
                     "ldr d3, [%[in3]], 8\n"
                     "ldr d4, [%[in4]], 8\n"
                     "ldr d5, [%[in5]], 8\n"
                     "ldr d6, [%[in6]], 8\n"
                     "ldr d7, [%[in7]], 8\n"

                     "zip1 v8.2s, v0.2s, v1.2s\n"
                     "zip2 v12.2s, v0.2s, v1.2s\n"
                     "zip1 v9.2s, v2.2s, v3.2s\n"
                     "zip2 v13.2s, v2.2s, v3.2s\n"
                     "zip1 v10.2s, v4.2s, v5.2s\n"
                     "zip2 v14.2s, v4.2s, v5.2s\n"
                     "zip1 v11.2s, v6.2s, v7.2s\n"
                     "zip2 v15.2s, v6.2s, v7.2s\n"

                     "str d8, [%[out]]\n"
                     "str d9, [%[out], 8]\n"
                     "str d10, [%[out], 16]\n"
                     "str d11, [%[out], 24]\n"
                     "str d12, [%[out], 32]\n"
                     "str d13, [%[out], 40]\n"
                     "str d14, [%[out], 48]\n"
                     "str d15, [%[out], 56]\n"
                     : [in0] "+r"(in[0]), [in1] "+r"(in[1]), [in2] "+r"(in[2]), [in3] "+r"(in[3]),
                     [in4] "+r"(in[4]), [in5] "+r"(in[5]), [in6] "+r"(in[6]), [in7] "+r"(in[7])
                     : [out] "r"(dst1)
                     : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                     "v10", "v11", "v12", "v13", "v14", "v15");
        dst1 += 16;
    }

    if (k < blockK - 3) {
        for (int i = 0; i < 8; i++) {
            dst1[0] = in[i][0];
            dst1++;
            in[i]++;
        }
        k += 4;
    }

    if (k < blockK) {
        int kTail = blockK - k;
        INT8 *dstI8 = (INT8 *)dst1;
        INT8 *inI[8];
        for (int i = 0; i < 8; i++) {
            inI[i] = (INT8 *)in[i];
        }
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                if (j < kTail) {
                    dstI8[i * 4 + j] = inI[i][j];
                } else {
                    dstI8[i * 4 + j] = 0;
                }
            }
        }
    }
}

// Trans from NK to NKn(size)k4
template <int kk = 4>
inline void matrix1_trans_int8(int size, int blockK, int K, INT8 *src, INT8 *dst)
{
    CHECK_REQUIREMENT(kk == 4 || kk == 8 || kk == 16);
    I32 *dst1 = (I32 *)dst;
    int prefetch = 64 / kk;
    int i = 0;
    for (; i < blockK / kk; i++) {
        int ii = kk / 4 * i;
        for (int j = 0; j < size; j++) {
            I32 *src1 = (I32 *)(src + j * K);
            if (i % prefetch == 0) {
                asm volatile("prfm pldl2keep, [%0, %1]\n" : "+r"(src1) : "r"((size_t)64) : "memory", "cc");
            }
            *dst1++ = *(src1 + ii);
            if (kk >= 8) {
                *dst1++ = *(src1 + ii + 1);
            }
            if (kk >= 16) {
                *dst1++ = *(src1 + ii + 2);
                *dst1++ = *(src1 + ii + 3);
            }
        }
    }
    int kTail = blockK % kk;
    if (kTail > 0) {
        INT8 *srcI8 = src + i * kk;
        INT8 *dstI8 = (INT8 *)dst1;
        int zeroSize = kk - kTail;
        for (int j = 0; j < size; j++) {
            UNI_MEMCPY(dstI8, srcI8, kTail);
            UNI_MEMSET(dstI8 + kTail, 0, zeroSize);
            srcI8 += K;
            dstI8 += kk;
        }
    }
}

inline void matrix2_trans_m12(int blockK, int M, INT8 *src, INT8 *dst)
{
    INT8 *src1 = src;
    INT8 *dst1 = dst;
    int offset = 4 * M;

    int i = 0;
    for (; i < blockK - 3; i += 4) {
        // Prefetch for the next iteration
        asm volatile("prfm pldl2keep, [%0, %1]\n" : "+r"(src1) : "r"((I64)offset) : "memory", "cc");

        INT8 *in12[4];
        for (int j = 0; j < 4; j++) {
            in12[j] = src1 + j * M;
        }
        src1 += offset;

        asm volatile(
            "ldr d0, [%[in0]]\n"
            "ldr d1, [%[in1]]\n"
            "ldr d2, [%[in2]]\n"
            "ldr d3, [%[in3]]\n"
            "zip1 v4.8b, v0.8b, v1.8b\n"
            "zip2 v5.8b, v0.8b, v1.8b\n"
            "zip1 v6.8b, v2.8b, v3.8b\n"
            "zip2 v7.8b, v2.8b, v3.8b\n"

            "zip1 v0.4h, v4.4h, v6.4h\n"
            "zip2 v1.4h, v4.4h, v6.4h\n"
            "zip1 v2.4h, v5.4h, v7.4h\n"
            "zip2 v3.4h, v5.4h, v7.4h\n"
            "str d0, [%[out]]\n"
            "str d1, [%[out], 8]\n"
            "str d2, [%[out], 16]\n"
            "str d3, [%[out], 24]\n"
            :
            : [in0] "r"(in12[0]), [in1] "r"(in12[1]), [in2] "r"(in12[2]), [in3] "r"(in12[3]), [out] "r"(dst1)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                dst1[32 + j * 4 + k] = in12[k][8 + j];
            }
        }

        dst1 += 48;
    }
    if (i < blockK) {
        int kTail = blockK - i;

        INT8 *in12[4];
        INT8 zero[12] = {0};
        for (int j = 0; j < 4; j++) {
            if (j < kTail) {
                in12[j] = src1 + j * M;
            } else {
                in12[j] = zero;
            }
        }

        asm volatile(
            "ldr d0, [%[in0]]\n"
            "ldr d1, [%[in1]]\n"
            "ldr d2, [%[in2]]\n"
            "ldr d3, [%[in3]]\n"
            "zip1 v4.8b, v0.8b, v1.8b\n"
            "zip2 v5.8b, v0.8b, v1.8b\n"
            "zip1 v6.8b, v2.8b, v3.8b\n"
            "zip2 v7.8b, v2.8b, v3.8b\n"

            "zip1 v0.4h, v4.4h, v6.4h\n"
            "zip2 v1.4h, v4.4h, v6.4h\n"
            "zip1 v2.4h, v5.4h, v7.4h\n"
            "zip2 v3.4h, v5.4h, v7.4h\n"
            "str d0, [%[out]]\n"
            "str d1, [%[out], 8]\n"
            "str d2, [%[out], 16]\n"
            "str d3, [%[out], 24]\n"
            :
            : [in0] "r"(in12[0]), [in1] "r"(in12[1]), [in2] "r"(in12[2]), [in3] "r"(in12[3]), [out] "r"(dst1)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                dst1[32 + j * 4 + k] = in12[k][8 + j];
            }
        }
    }
}

// Trans from KM to MKm(size)k4
template <int kk = 4>
inline void matrix2_trans_int8(int size, int blockK, int M, INT8 *src, INT8 *dst)
{
    INT8 *src1 = src;
    INT8 *dst1 = dst;
    int offset = kk * M;
    int i = 0;
    for (; i < blockK - kk + 1; i += kk) {
        src1 = src + i * M;
        asm volatile("prfm pldl2keep, [%0, %1]\n" : "+r"(src1) : "r"((I64)offset) : "memory", "cc");
        for (int j = 0; j < size; j++) {
            src1 = src + i * M + j;
            for (int k = 0; k < kk; k++) {
                *dst1 = *src1;
                dst1++;
                src1 += M;
            }
        }
    }
    if (i < blockK) {
        int kTail = blockK - i;
        for (int j = 0; j < size; j++) {
            src1 = src + i * M + j;
            for (int k = 0; k < kk; k++) {
                if (k < kTail) {
                    *dst1 = *src1;
                    src1 += M;
                } else {
                    *dst1 = 0;
                }
                dst1++;
            }
        }
    }
}
#endif
