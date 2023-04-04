// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/arm/int8/blas_int8.h"
#include "cpu/arm/int8/v7/blas_matrix_transpose.h"

EE matrix_matrix_multiply_transform_rhsN_int8(TensorDesc desc, INT8 *src, INT8 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    U32 K4 = UNI_ALIGN(K, 4);
    int i = 0;
    for (; i < (int)N - 7; i += 8) {
        matrix2_trans_int8(8, K, N, src + i, dst + i * K4);
    }
    if ((int)N > i) {
        matrix2_trans_int8(N - i, K, N, src + i, dst + i * K4);
    }
    return SUCCESS;
}

EE matrix_matrix_multiply_transform_rhsT_int8(TensorDesc desc, INT8 *src, INT8 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    U32 K4 = UNI_ALIGN(K, 4);
    int i = 0;
    for (; i < (int)N - 7; i += 8) {
        matrix1_trans_int8(8, K, K, src + i * K, dst + i * K4);
    }
    if ((int)N > i) {
        matrix1_trans_int8(N - i, K, K, src + i * K, dst + i * K4);
    }
    return SUCCESS;
}

void mmm_4x8(U32 offset, U32 K, INT8 *in, INT8 *w, I32 *out)
{
    asm volatile("vld1.s8 {d0[]}, [%[in]]!\n"
                 "vld1.s8 {d1[]}, [%[in]]!\n"
                 "vld1.s8 {d2[]}, [%[in]]!\n"
                 "vld1.s8 {d3[]}, [%[in]]!\n"

                 "vld1.s8  {d4-d5}, [%[w]]!\n"

                 // K- > r2
                 "mov r2, %[K]\n"

                 // give out address to r1
                 "mov r1, %[out]\n"

                 // load in bias
                 "vld1.s32  {d8-d11}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.s32  {d12-d15}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.s32  {d16-d19}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vld1.s32  {d20-d23}, [r1]\n"

                 // Computation loop
                 "0:\n"

                 "vmull.s8 q12, d4, d0\n"
                 "vld1.s8 {d0[]}, [%[in]]!\n"
                 "vmull.s8 q13, d4, d1\n"
                 "vld1.s8 {d1[]}, [%[in]]!\n"
                 "vmull.s8 q14, d4, d2\n"
                 "vld1.s8 {d2[]}, [%[in]]!\n"
                 "vmull.s8 q15, d4, d3\n"
                 "vld1.s8 {d3[]}, [%[in]]!\n"
                 "vld1.s8  {d4}, [%[w]]!\n"

                 // "vaddw.s16 q4, q4, d24\n"
                 // "vaddw.s16 q5, q5, d25\n"
                 // "vaddw.s16 q6, q6, d26\n"
                 // "vaddw.s16 q7, q7, d27\n"
                 // "vaddw.s16 q8, q8, d28\n"
                 // "vaddw.s16 q9, q9, d29\n"
                 // "vaddw.s16 q10, q10, d30\n"
                 // "vaddw.s16 q11, q11, d31\n"
                 // "vmov.s32 q12, #0\n"
                 // "vmov.s32 q13, #0\n"
                 // "vmov.s32 q14, #0\n"
                 // "vmov.s32 q15, #0\n"

                 "vmlal.s8 q12, d5, d0\n"
                 "vmlal.s8 q13, d5, d1\n"
                 "vld1.s8 {d0[]}, [%[in]]!\n"
                 "vmlal.s8 q14, d5, d2\n"
                 "vld1.s8 {d1[]}, [%[in]]!\n"
                 "vmlal.s8 q15, d5, d3\n"

                 // "vaddw.s16 q4, q4, d24\n"
                 // "vaddw.s16 q5, q5, d25\n"
                 // "vaddw.s16 q6, q6, d26\n"
                 // "vaddw.s16 q7, q7, d27\n"
                 // "vaddw.s16 q8, q8, d28\n"
                 // "vaddw.s16 q9, q9, d29\n"
                 // "vaddw.s16 q10, q10, d30\n"
                 // "vaddw.s16 q11, q11, d31\n"
                 // "vmov.s32 q12, #0\n"
                 // "vmov.s32 q13, #0\n"
                 // "vmov.s32 q14, #0\n"
                 // "vmov.s32 q15, #0\n"

                 "vld1.s8 {d2[]}, [%[in]]!\n"
                 "vmlal.s8 q12, d4, d0\n"
                 "vld1.s8 {d3[]}, [%[in]]!\n"
                 "vld1.s8  {d5}, [%[w]]!\n"
                 "vmlal.s8 q13, d4, d1\n"
                 "vld1.s8 {d0[]}, [%[in]]!\n"
                 "vmlal.s8 q14, d4, d2\n"
                 "vld1.s8 {d1[]}, [%[in]]!\n"
                 "vmlal.s8 q15, d4, d3\n"
                 "vld1.s8 {d2[]}, [%[in]]!\n"

                 // "vaddw.s16 q4, q4, d24\n"
                 // "vaddw.s16 q5, q5, d25\n"
                 // "vaddw.s16 q6, q6, d26\n"
                 // "vaddw.s16 q7, q7, d27\n"
                 // "vaddw.s16 q8, q8, d28\n"
                 // "vaddw.s16 q9, q9, d29\n"
                 // "vaddw.s16 q10, q10, d30\n"
                 // "vaddw.s16 q11, q11, d31\n"
                 // "vmov.s32 q12, #0\n"
                 // "vmov.s32 q13, #0\n"
                 // "vmov.s32 q14, #0\n"
                 // "vmov.s32 q15, #0\n"

                 "vmlal.s8 q12, d5, d0\n"
                 "vld1.s8 {d3[]}, [%[in]]!\n"
                 "vld1.s8  {d4}, [%[w]]!\n"
                 "vmlal.s8 q13, d5, d1\n"
                 "vld1.s8 {d0[]}, [%[in]]!\n"
                 "vmlal.s8 q14, d5, d2\n"
                 "vld1.s8 {d1[]}, [%[in]]!\n"
                 "vmlal.s8 q15, d5, d3\n"
                 "vld1.s8 {d2[]}, [%[in]]!\n"
                 "vld1.s8 {d3[]}, [%[in]]!\n"
                 "vld1.s8  {d5}, [%[w]]!\n"

                 "subs r2, r2, #4\n"

                 "vaddw.s16 q4, q4, d24\n"
                 "vaddw.s16 q5, q5, d25\n"
                 "vaddw.s16 q6, q6, d26\n"
                 "vaddw.s16 q7, q7, d27\n"
                 "vaddw.s16 q8, q8, d28\n"
                 "vaddw.s16 q9, q9, d29\n"
                 "vaddw.s16 q10, q10, d30\n"
                 "vaddw.s16 q11, q11, d31\n"

                 "bne 0b\n"

                 // give out address to r1
                 "mov r1, %[out]\n"

                 "vst1.s32  {d8-d11}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.s32  {d12-d15}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.s32  {d16-d19}, [r1]\n"
                 "add r1, r1, %[offset]\n"
                 "vst1.s32  {d20-d23}, [r1]\n"
                 : [in] "+r"(in), [w] "+r"(w), [out] "+r"(out)
                 : [K] "r"(K), [offset] "r"(offset)
                 : "memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                 "q10", "q11", "q12", "q13", "q14", "q15", "r1", "r2");
}

inline void mmm_NTail_M8(U32 M, U32 N, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    for (U32 i = 0; i < N; i++) {
        int32x4_t res1 = vld1q_s32(result + i * M);
        int32x4_t res2 = vld1q_s32(result + i * M + 4);
        for (U32 q = 0; q < K; q += 1) {
            int8x8_t mat2 = vld1_s8(matrix2 + q * 8);
            int8x8_t mat1 = vdup_n_s8(matrix1[q * N + i]);
            int16x8_t r = vmull_s8(mat1, mat2);
            res1 = vaddw_s16(res1, vget_low_s16(r));
            res2 = vaddw_s16(res2, vget_high_s16(r));
        }
        vst1q_s32(result + i * M, res1);
        vst1q_s32(result + i * M + 4, res2);
    }
}

inline void mmm_NTail_M(U32 MInner, U32 M, U32 N, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    for (U32 i = 0; i < N; i++) {
        for (U32 j = 0; j < MInner; j++) {
            for (U32 k = 0; k < K; k++) {
                result[i * M + j] += ((I32)matrix1[k * N + i]) * matrix2[k * MInner + j];
            }
        }
    }
}

inline void mmm_N4_MTail(U32 MInner, U32 M, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    const int unroll = 4;
    int32x4_t res[4][2] = {0};
    for (U32 k = 0; k < K; k += unroll) {
        int16x8_t res_s16[4] = {0};
        for (U32 kk = 0; kk < unroll; kk++) {
            U32 z = k + kk;
            int8x8_t mat2 = vld1_s8(matrix2 + z * MInner);
            for (int i = 0; i < 4; i++) {
                int8x8_t mat10 = vdup_n_s8(matrix1[z * 4 + i]);
                res_s16[i] = vmlal_s8(res_s16[i], mat10, mat2);
            }
        }
        for (int i = 0; i < 4; i++) {
            res[i][0] = vaddw_s16(res[i][0], vget_low_s16(res_s16[i]));
            res[i][1] = vaddw_s16(res[i][1], vget_high_s16(res_s16[i]));
        }
    }
    I32 tmp[8];
    for (int i = 0; i < 4; i++) {
        vst1q_s32(tmp, res[i][0]);
        vst1q_s32(tmp + 4, res[i][1]);
        for (U32 p = 0; p < MInner; p++) {
            result[i * M + p] += tmp[p];
        }
    }
}

EE mmm_int8(
    int M, int N, int K, bool transposeA, INT8 *matrix1, INT8 *matrix2, INT8 *tmp, I32 *result, Arch arch)
{
    int blockK = K;
    U32 K4 = UNI_ALIGN(K, 4);
    int blockM = 96;
    for (int k = 0; k < K; k += blockK) {
        int KInner = UNI_MIN(blockK, K - k);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (int n = 0; n <= N - 4; n += 4) {
            INT8 *matrix1Trans = tmp + n * K4;
            if (transposeA) {
                matrix2_trans_int8(4, KInner, N, matrix1 + n, matrix1Trans);
            } else {
                matrix1_trans_int8(4, KInner, K, matrix1 + n * K + k, matrix1Trans);
            }
        }
        int n = N / 4 * 4;
        if (N - n > 0) {
            INT8 *matrix1Trans = tmp + n * K4;
            if (transposeA) {
                matrix2_trans_int8(N - n, KInner, N, matrix1 + n, matrix1Trans);
            } else {
                matrix1_trans_int8(N - n, KInner, K, matrix1 + n * K + k, matrix1Trans);
            }
        }

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (int i = 0; i < M; i += blockM) {
            int MInner = UNI_MIN(blockM, M - i);
            I32 *resultCurrent;
            int m, n;
            for (n = 0; n <= N - 4; n += 4) {
                INT8 *matrix1Trans = tmp + n * K4;
                //if (i == 0) {
                //    if (transposeA) {
                //        matrix2_trans_int8(4, KInner, N, matrix1 + n, matrix1Trans + n * K4);
                //    } else {
                //        matrix1_trans_int8(4, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K4);
                //    }
                //}
                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x8(M * 4, K4, matrix1Trans, matrix2 + (i + m) * K4, resultCurrent);
                    //mmm_NTail_M(8, M, 4, K4, matrix1Trans,
                    //    matrix2 + (i + m) * K4, resultCurrent);
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_N4_MTail(
                        MInner - m, M, K4, matrix1Trans, matrix2 + (i + m) * K4, resultCurrent);
                    //mmm_NTail_M(MInner - m, M, 4, K4, matrix1Trans, matrix2 + (i + m) * K4,
                    //    resultCurrent);
                }
            }

            if (N - n) {
                INT8 *matrix1Trans = tmp + n * K4;
                //if (i == 0) {
                //    if (transposeA) {
                //        matrix2_trans_int8(N - n, KInner, N, matrix1 + n, matrix1Trans + n * K4);
                //    } else {
                //        matrix1_trans_int8(
                //            N - n, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K4);
                //    }
                //}

                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M8(
                        M, N - n, KInner, matrix1Trans, matrix2 + (i + m) * K4, resultCurrent);
                    //mmm_NTail_M(8, M, N - n, K4, matrix1Trans,
                    //    matrix2 + (i + m) * K4, resultCurrent);
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_NTail_M(MInner - m, M, N - n, K4, matrix1Trans, matrix2 + (i + m) * K4,
                        resultCurrent);
                }
            }
        }
    }
    return SUCCESS;
}
