// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstring>
#include "cpu/arm/fp32/blas_fp32.h"
#include "error.h"

void matrix_matrix_multiply_tmp_bytes_fp32(
    U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes)
{
    *bytes = row1 * col1 + row2 * col2;
    *bytes *= bytesOf(dt);
    *bytes += 32;
}

EE matrix_matrix_multiply_transform_rhsN_fp32(TensorDesc desc, F32 *src, F32 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    int i = 0;
    for (; i < (int)N - 7; i += 8) {
        matrix2_trans(8, K, N, src + i, dst + i * K);
    }
    for (; i < (int)N - 3; i += 4) {
        matrix2_trans(4, K, N, src + i, dst + i * K);
    }
    if ((int)N > i) {
        matrix2_trans(N - i, K, N, src + i, dst + i * K);
    }
    return SUCCESS;
}

EE matrix_matrix_multiply_transform_rhsT_fp32(TensorDesc desc, F32 *src, F32 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    int i = 0;
    for (; i < (int)N - 7; i += 8) {
        matrix1_trans(8, K, K, src + i * K, dst + i * K);
    }
    for (; i < (int)N - 3; i += 4) {
        matrix1_trans(4, K, K, src + i * K, dst + i * K);
    }
    if ((int)N > i) {
        matrix1_trans(N - i, K, K, src + i * K, dst + i * K);
    }
    return SUCCESS;
}

void mmm_NTail_M8(U32 M, U32 N, U32 K, F32 *matrix1, F32 *matrix2, F32 *result)
{
    float32x4x2_t mat2, res;
    for (U32 i = 0; i < N; i++) {
        res = vld2q_f32(result + i * M);
        for (U32 q = 0; q < K; q++) {
            mat2 = vld2q_f32(matrix2 + q * 8);
            res.val[0] = vfmaq_n_f32(res.val[0], mat2.val[0], matrix1[q * N + i]);
            res.val[1] = vfmaq_n_f32(res.val[1], mat2.val[1], matrix1[q * N + i]);
        }
        vst2q_f32(result + i * M, res);
    }
}

void mmm_NTail_M4(U32 M, U32 N, U32 K, F32 *matrix1, F32 *matrix2, F32 *result)
{
    float32x4_t mat2, res;
    for (U32 i = 0; i < N; i++) {
        res = vld1q_f32(result + i * M);
        for (U32 q = 0; q < K; q++) {
            mat2 = vld1q_f32(matrix2 + q * 4);
            res = vfmaq_n_f32(res, mat2, matrix1[q * N + i]);
        }
        vst1q_f32(result + i * M, res);
    }
}

void mmm_NTail_M(U32 MInner, U32 M, U32 N, U32 K, F32 *matrix1, F32 *matrix2, F32 *result)
{
    for (U32 i = 0; i < N; i++) {
        for (U32 j = 0; j < MInner; j++) {
            for (U32 k = 0; k < K; k++) {
                result[i * M + j] += *(matrix1 + k * N + i) * *(matrix2 + k * MInner + j);
            }
        }
    }
}

void mmm_N6_MTail(U32 MInner, U32 M, U32 K, F32 *matrix1, F32 *matrix2, F32 *result)
{
    float32x2_t mat1[3] = {0}, res[4][3] = {{0}};
    F32 tmp[6] = {0};
    CHECK_REQUIREMENT(MInner < 4);

    for (U32 i = 0; i < K; i++) {
        mat1[0] = vld1_f32(matrix1 + i * 6);
        mat1[1] = vld1_f32(matrix1 + i * 6 + 2);
        mat1[2] = vld1_f32(matrix1 + i * 6 + 4);
        for (U32 j = 0; j < MInner; j++) {
            res[j][0] = vmla_n_f32(res[j][0], mat1[0], matrix2[j + i * MInner]);
            res[j][1] = vmla_n_f32(res[j][1], mat1[1], matrix2[j + i * MInner]);
            res[j][2] = vmla_n_f32(res[j][2], mat1[2], matrix2[j + i * MInner]);
        }
    }
    for (U32 p = 0; p < MInner; p++) {
        vst1_f32(tmp, res[p][0]);
        vst1_f32(tmp + 2, res[p][1]);
        vst1_f32(tmp + 4, res[p][2]);
        for (U32 q = 0; q < 6; q++) {
            result[q * M + p] += tmp[q];
        }
    }
}

void mmm_N4_MTail(U32 MInner, U32 M, U32 K, F32 *matrix1, F32 *matrix2, F32 *result)
{
    float32x4_t mat1 = {0}, res[4] = {0};
    F32 tmp[4] = {0};
    CHECK_REQUIREMENT(MInner < 4);

    for (U32 i = 0; i < K; i++) {
        mat1 = vld1q_f32(matrix1 + i * 4);
        for (U32 j = 0; j < MInner; j++) {
            res[j] = vfmaq_n_f32(res[j], mat1, matrix2[j + i * MInner]);
        }
    }
    for (U32 p = 0; p < MInner; p++) {
        vst1q_f32(tmp, res[p]);
        for (U32 q = 0; q < 4; q++) {
            result[q * M + p] += tmp[q];
        }
    }
}

void mmm_4x4(U32 offset, U32 K, F32 *in, F32 *w, F32 *out)
{
    asm volatile("vld1.f32 {d0-d1}, [%[in]]!\n"

                 "vld1.f32 {d4-d5}, [%[w]]!\n"

                 // K- > r2
                 "mov r2, %[K]\n"

                 // give out address to r1
                 "mov r1, %[out]\n"

                 // load in bias
                 "vld1.f32  {d8-d9}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d12-d13}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d16-d17}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d20-d21}, [r1]\n"

                 // Computation loop
                 "0:\n"

                 "vmla.f32  q4, q2, d0[0]\n"
                 "vmla.f32  q6, q2, d0[1]\n"
                 "vmla.f32  q8, q2, d1[0]\n"
                 "vmla.f32  q10, q2, d1[1]\n"

                 "vld1.f32  {d4-d5}, [%[w]]!\n"
                 "subs r2, r2, #1\n"

                 "vld1.f32 {d0-d1}, [%[in]]!\n"
                 "bne 0b\n"

                 // give out address to r1
                 "mov r1, %[out]\n"

                 "vst1.f32  {d8-d9}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d12-d13}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d16-d17}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d20-d21}, [r1]\n"

                 : [in] "+r"(in), [w] "+r"(w), [out] "+r"(out)
                 : [K] "r"(K), [offset] "r"(offset)
                 : "memory", "cc", "q0", "q2", "q4", "q6", "q8", "q10", "r1", "r2");
}

void mmm_6x4(U32 offset, U32 K, F32 *in, F32 *w, F32 *out)
{
    asm volatile(
        "vld1.f32 {d0-d2}, [%[in]]!\n"

        "vld1.f32 {d4-d5}, [%[w]]!\n"

        // K- > r2
        "mov r2, %[K]\n"

        // give out address to r1
        "mov r1, %[out]\n"

        // load in bias
        "vld1.f32  {d8-d9}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vld1.f32  {d12-d13}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vld1.f32  {d16-d17}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vld1.f32  {d20-d21}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vld1.f32  {d24-d25}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vld1.f32  {d28-d29}, [r1]\n"

        // Computation loop
        "0:\n"

        "vmla.f32  q4, q2, d0[0]\n"
        "vmla.f32  q6, q2, d0[1]\n"
        "vmla.f32  q8, q2, d1[0]\n"
        "vmla.f32  q10, q2, d1[1]\n"
        "vmla.f32  q12, q2, d2[0]\n"
        "vmla.f32  q14, q2, d2[1]\n"

        "vld1.f32  {d4-d5}, [%[w]]!\n"
        "subs r2, r2, #1\n"

        "vld1.f32 {d0-d2}, [%[in]]!\n"
        "bne 0b\n"

        // give out address to r1
        "mov r1, %[out]\n"

        "vst1.f32  {d8-d9}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vst1.f32  {d12-d13}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vst1.f32  {d16-d17}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vst1.f32  {d20-d21}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vst1.f32  {d24-d25}, [r1]\n"
        "add r1, r1, %[offset]\n"
        "vst1.f32  {d28-d29}, [r1]\n"

        : [in] "+r"(in), [w] "+r"(w), [out] "+r"(out)
        : [K] "r"(K), [offset] "r"(offset)
        : "memory", "cc", "q0", "q1", "q2", "q4", "q6", "q8", "q10", "q12", "q14", "r1", "r2");
}

void mmm_4x8(U32 offset, U32 K, F32 *in, F32 *w, F32 *out)
{
    asm volatile("vld1.f32 {d0-d1}, [%[in]]!\n"

                 "vld1.f32  {d4-d7}, [%[w]]!\n"

                 // K- > r2
                 "mov r2, %[K]\n"

                 // give out address to r1
                 "mov r1, %[out]\n"

                 // load in bias
                 "vld1.f32  {d8-d11}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d12-d15}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d16-d19}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d20-d23}, [r1]\n"

                 // Computation loop
                 "0:\n"

                 "vmla.f32  q4, q2, d0[0]\n"
                 "vmla.f32  q6, q2, d0[1]\n"
                 "vmla.f32  q8, q2, d1[0]\n"
                 "vmla.f32  q10, q2, d1[1]\n"

                 "vld1.f32  {d4-d5}, [%[w]]!\n"

                 "vmla.f32  q5, q3, d0[0]\n"
                 "vmla.f32  q7, q3, d0[1]\n"
                 "vmla.f32  q9, q3, d1[0]\n"
                 "vmla.f32  q11, q3, d1[1]\n"

                 "vld1.f32  {d6-d7}, [%[w]]!\n"
                 "subs r2, r2, #1\n"

                 "vld1.f32 {d0-d1}, [%[in]]!\n"
                 "bne 0b\n"

                 // give out address to r1
                 "mov r1, %[out]\n"

                 "vst1.f32  {d8-d11}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d12-d15}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d16-d19}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d20-d23}, [r1]\n"

                 : [in] "+r"(in), [w] "+r"(w), [out] "+r"(out)
                 : [K] "r"(K), [offset] "r"(offset)
                 : "memory", "cc", "q0", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                 "q11", "r1", "r2");
}

void mmm_6x8(U32 offset, U32 K, F32 *in, F32 *w, F32 *out)
{
    asm volatile("vld1.f32 {d0-d2}, [%[in]]!\n"

                 "vld1.f32  {d4-d7}, [%[w]]!\n"

                 // K- > r2
                 "mov r2, %[K]\n"

                 // give out address to r1
                 "mov r1, %[out]\n"

                 // load in bias
                 "vld1.f32  {d8-d11}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d12-d15}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d16-d19}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d20-d23}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d24-d27}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.f32  {d28-d31}, [r1]\n"

                 // Computation loop
                 "0:\n"

                 "vmla.f32  q4, q2, d0[0]\n"
                 "vmla.f32  q6, q2, d0[1]\n"
                 "vmla.f32  q8, q2, d1[0]\n"
                 "vmla.f32  q10, q2, d1[1]\n"
                 "vmla.f32  q12, q2, d2[0]\n"
                 "vmla.f32  q14, q2, d2[1]\n"

                 "vld1.f32  {d4-d5}, [%[w]]!\n"

                 "vmla.f32  q5, q3, d0[0]\n"
                 "vmla.f32  q7, q3, d0[1]\n"
                 "vmla.f32  q9, q3, d1[0]\n"
                 "vmla.f32  q11, q3, d1[1]\n"
                 "vmla.f32  q13, q3, d2[0]\n"
                 "vmla.f32  q15, q3, d2[1]\n"

                 "vld1.f32  {d6-d7}, [%[w]]!\n"
                 "subs r2, r2, #1\n"

                 "vld1.f32 {d0-d2}, [%[in]]!\n"
                 "bne 0b\n"

                 // give out address to r1
                 "mov r1, %[out]\n"

                 "vst1.f32  {d8-d11}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d12-d15}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d16-d19}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d20-d23}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d24-d27}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.f32  {d28-d31}, [r1]\n"

                 : [in] "+r"(in), [w] "+r"(w), [out] "+r"(out)
                 : [K] "r"(K), [offset] "r"(offset)
                 : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                 "q10", "q11", "q12", "q13", "q14", "q15", "r1", "r2");
}

EE mmm_fp32_V7(
    int M, int N, int K, bool transposeA, F32 *matrix1, F32 *matrix2, F32 *tmp, F32 *result)
{
    int blockK = K;
    int blockM = 96;
    F32 *matrix1Trans = tmp;
    F32 *resultCurrent = result;
    int KInner, MInner, m, n;
    for (int k = 0; k < K; k += blockK) {
        KInner = UNI_MIN(blockK, K - k);
        for (int i = 0; i < M; i += blockM) {
            MInner = UNI_MIN(blockM, M - i);
            for (n = 0; n <= N - 6; n += 6) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans(6, KInner, N, matrix1 + n, matrix1Trans + n * KInner);
                    } else {
                        matrix1_trans(6, KInner, K, matrix1 + n * K + k, matrix1Trans + n * KInner);
                    }
                }
                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_6x8(M * 4, KInner, matrix1Trans + n * KInner, matrix2 + (i + m) * KInner,
                        resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_6x4(M * 4, KInner, matrix1Trans + n * KInner, matrix2 + (i + m) * KInner,
                        resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_N6_MTail(MInner - m, M, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }
            }

            if ((N - n) >= 4) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans(4, KInner, N, matrix1 + n, matrix1Trans + n * KInner);
                    } else {
                        matrix1_trans(4, KInner, K, matrix1 + n * K + k, matrix1Trans + n * KInner);
                    }
                }

                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x8(M * 4, KInner, matrix1Trans + n * KInner, matrix2 + (i + m) * KInner,
                        resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x4(M * 4, KInner, matrix1Trans + n * KInner, matrix2 + (i + m) * KInner,
                        resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_N4_MTail(MInner - m, M, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }

                n += 4;
            }

            if (N - n) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans(N - n, KInner, N, matrix1 + n, matrix1Trans + n * KInner);
                    } else {
                        matrix1_trans(
                            N - n, KInner, K, matrix1 + n * K + k, matrix1Trans + n * KInner);
                    }
                }

                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M8(M, N - n, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M4(M, N - n, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                    m += 4;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M(MInner - m, M, N - n, KInner, matrix1Trans + n * KInner,
                        matrix2 + (i + m) * KInner, resultCurrent);
                }
            }
        }
    }
    return SUCCESS;
}
