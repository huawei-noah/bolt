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
#include "cpu/arm/int8/v8.2/blas_matrix_transpose.h"

static const int maxTileN = 4;
static const int maxTileM = 4;
static const int tileK = 16;

EE matrix_matrix_multiply_transform_rhsN_int8(TensorDesc desc, INT8 *src, INT8 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    U32 K4 = UNI_ALIGN(K, tileK);
    int i = 0;
    for (; i < (int)N - maxTileM + 1; i += maxTileM) {
        matrix2_trans_int8<tileK>(maxTileM, K, N, src + i, dst + i * K4);
    }
    if ((int)N > i) {
        matrix2_trans_int8<tileK>(N - i, K, N, src + i, dst + i * K4);
    }
    return SUCCESS;
}

EE matrix_matrix_multiply_transform_rhsT_int8(TensorDesc desc, INT8 *src, INT8 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    U32 K4 = UNI_ALIGN(K, tileK);
    int i = 0;
    for (; i < (int)N - maxTileM + 1; i += maxTileM) {
        matrix1_trans_int8<tileK>(maxTileM, K, K, src + i * K, dst + i * K4);
    }
    if ((int)N > i) {
        matrix1_trans_int8<tileK>(N - i, K, K, src + i * K, dst + i * K4);
    }
    return SUCCESS;
}

void mmm_template(
    const int tileN, const int tileM, U32 offset, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    int32x4_t ret[maxTileN][maxTileM] = {0};
    int8x16_t a[maxTileN];
    int8x16_t b[maxTileM];
    for (U32 n = 0; n < K; n += tileK) {
        for (int i = 0; i < tileN; i++) {
            a[i] = vld1q_s8(matrix1);
            matrix1 += tileK;
        }
        for (int i = 0; i < tileM; i++) {
            b[i] = vld1q_s8(matrix2);
            matrix2 += tileK;
        }
        for (int i = 0; i < tileN; i++) {
#pragma unroll(4)
            for (int j = 0; j < tileM; j++) {
                int16x8_t tmp = vmull_s8(vget_low_s8(a[i]), vget_low_s8(b[j]));
                tmp = vmlal_s8(tmp, vget_high_s8(a[i]), vget_high_s8(b[j]));
                ret[i][j] = vpadalq_s16(ret[i][j], tmp);
            }
        }
    }
    for (int i = 0, ii = 0; i < tileN; i++, ii += offset) {
        for (int j = 0, jj = ii; j < tileM; j++, jj++) {
            result[jj] += vaddvq_s32(ret[i][j]);
        }
    }
}

void mmm_4x8(U32 offset, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    //return mmm_template(4, 8, offset, K, matrix1, matrix2, result);
    asm volatile("ld1 {v16.16b, v17.16b, v18.16b, v19.16b}, [%[matrix1]]\n"
                 "ld1 {v20.16b, v21.16b, v22.16b, v23.16b}, [%[matrix2]]\n"
                 "add x20, %[matrix1], 0x40\n"
                 "add x21, %[matrix2], 0x40\n"
                 "mov x22, %[K]\n"
                 "mov x26, %[result]\n"
                 "movi v0.4s, #0x0\n"
                 "movi v1.4s, #0x0\n"
                 "movi v2.4s, #0x0\n"
                 "movi v3.4s, #0x0\n"
                 "movi v4.4s, #0x0\n"
                 "movi v5.4s, #0x0\n"
                 "movi v6.4s, #0x0\n"
                 "movi v7.4s, #0x0\n"
                 "movi v8.4s, #0x0\n"
                 "movi v9.4s, #0x0\n"
                 "movi v10.4s, #0x0\n"
                 "movi v11.4s, #0x0\n"
                 "movi v12.4s, #0x0\n"
                 "movi v13.4s, #0x0\n"
                 "movi v14.4s, #0x0\n"
                 "movi v15.4s, #0x0\n"

                 // Computation loop
                 "0:\n"
                 "smull v24.8h, v16.8b, v20.8b\n"
                 "smull v25.8h, v16.8b, v21.8b\n"
                 "smull v26.8h, v16.8b, v22.8b\n"
                 "smull v27.8h, v16.8b, v23.8b\n"
                 "smull v28.8h, v17.8b, v20.8b\n"
                 "smull v29.8h, v17.8b, v21.8b\n"
                 "smull v30.8h, v17.8b, v22.8b\n"
                 "smull v31.8h, v17.8b, v23.8b\n"
                 "smlal2 v24.8h, v16.16b, v20.16b\n"
                 "smlal2 v25.8h, v16.16b, v21.16b\n"
                 "smlal2 v26.8h, v16.16b, v22.16b\n"
                 "smlal2 v27.8h, v16.16b, v23.16b\n"
                 "smlal2 v28.8h, v17.16b, v20.16b\n"
                 "smlal2 v29.8h, v17.16b, v21.16b\n"
                 "smlal2 v30.8h, v17.16b, v22.16b\n"
                 "smlal2 v31.8h, v17.16b, v23.16b\n"
                 "ld1 {v16.16b, v17.16b}, [x20]\n"
                 "sadalp v0.4s, v24.8h\n"
                 "sadalp v1.4s, v25.8h\n"
                 "sadalp v2.4s, v26.8h\n"
                 "sadalp v3.4s, v27.8h\n"
                 "sadalp v4.4s, v28.8h\n"
                 "sadalp v5.4s, v29.8h\n"
                 "sadalp v6.4s, v30.8h\n"
                 "sadalp v7.4s, v31.8h\n"

                 "subs x22, x22, #16\n"
                 "smull v24.8h, v18.8b, v20.8b\n"
                 "smull v25.8h, v18.8b, v21.8b\n"
                 "smull v26.8h, v18.8b, v22.8b\n"
                 "smull v27.8h, v18.8b, v23.8b\n"
                 "smull v28.8h, v19.8b, v20.8b\n"
                 "smull v29.8h, v19.8b, v21.8b\n"
                 "smull v30.8h, v19.8b, v22.8b\n"
                 "smull v31.8h, v19.8b, v23.8b\n"
                 "smlal2 v24.8h, v18.16b, v20.16b\n"
                 "smlal2 v25.8h, v18.16b, v21.16b\n"
                 "smlal2 v26.8h, v18.16b, v22.16b\n"
                 "smlal2 v27.8h, v18.16b, v23.16b\n"
                 "ldr q18, [x20, 0x20]\n"
                 "smlal2 v28.8h, v19.16b, v20.16b\n"
                 "smlal2 v29.8h, v19.16b, v21.16b\n"
                 "smlal2 v30.8h, v19.16b, v22.16b\n"
                 "smlal2 v31.8h, v19.16b, v23.16b\n"
                 "ldr q19, [x20, 0x30]\n"
                 "sadalp v8.4s, v24.8h\n"
                 "ld1 {v20.16b, v21.16b, v22.16b, v23.16b}, [x21]\n"
                 "sadalp v9.4s, v25.8h\n"
                 "sadalp v10.4s, v26.8h\n"
                 "add x20, x20, 0x40\n"
                 "sadalp v11.4s, v27.8h\n"
                 "sadalp v12.4s, v28.8h\n"
                 "add x21, x21, 0x40\n"
                 "sadalp v13.4s, v29.8h\n"
                 "sadalp v14.4s, v30.8h\n"
                 "sadalp v15.4s, v31.8h\n"

                 "bne 0b\n"
                 "ldr q16, [x26]\n"
                 "addp v0.4s, v0.4s, v1.4s\n"
                 "addp v1.4s, v2.4s, v3.4s\n"
                 "addp v2.4s, v4.4s, v5.4s\n"
                 "addp v3.4s, v6.4s, v7.4s\n"
                 "addp v4.4s, v8.4s, v9.4s\n"
                 "addp v5.4s, v10.4s, v11.4s\n"
                 "addp v0.4s, v0.4s, v1.4s\n"
                 "addp v6.4s, v12.4s, v13.4s\n"
                 "addp v1.4s, v2.4s, v3.4s\n"
                 "addp v7.4s, v14.4s, v15.4s\n"
                 "add v0.4s, v0.4s, v16.4s\n"
                 "addp v2.4s, v4.4s, v5.4s\n"
                 "addp v3.4s, v6.4s, v7.4s\n"
                 "str q0, [x26]\n"
                 "add x26, x26, %[offset]\n"
                 "ldr q5, [x26]\n"
                 "add v1.4s, v1.4s, v5.4s\n"
                 "str q1, [x26]\n"
                 "add x26, x26, %[offset]\n"
                 "ldr q6, [x26]\n"
                 "add v2.4s, v2.4s, v6.4s\n"
                 "str q2, [x26]\n"
                 "add x26, x26, %[offset]\n"
                 "ldr q7, [x26]\n"
                 "add v3.4s, v3.4s, v7.4s\n"
                 "str q3, [x26]\n"
                 : [matrix1] "+r"(matrix1), [matrix2] "+r"(matrix2), [result] "+r"(result)
                 : [K] "r"((I64)K), [offset] "r"((I64)offset * 4)
                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                 "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
                 "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x26", "x20",
                 "x21", "x22");
}

inline void mmm_NTail_M(U32 MInner, U32 M, U32 N, U32 K, INT8 *matrix1, INT8 *matrix2, I32 *result)
{
    for (U32 k = 0; k < K / tileK; k++) {
        for (U32 i = 0; i < N; i++) {
            for (U32 j = 0; j < MInner; j++) {
                INT8 *p1 = matrix1 + (k * N + i) * tileK;
                INT8 *p2 = matrix2 + (k * MInner + j) * tileK;
                I32 value = 0;
                for (U32 kk = 0; kk < tileK; kk++) {
                    I32 a = *p1++;
                    I32 b = *p2++;
                    value += a * b;
                }
                result[i * M + j] += value;
            }
        }
    }
}

EE mmm_int8(
    int M, int N, int K, bool transposeA, INT8 *matrix1, INT8 *matrix2, INT8 *tmp, I32 *result, Arch arch)
{
    int blockK = K;
    U32 K4 = UNI_ALIGN(K, tileK);
    int blockM = 96;
    for (int k = 0; k < K; k += blockK) {
        int KInner = UNI_MIN(blockK, K - k);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (int n = 0; n <= N - maxTileN; n += maxTileN) {
            INT8 *matrix1Trans = tmp + n * K4;
            if (transposeA) {
                matrix2_trans_int8<tileK>(maxTileN, KInner, N, matrix1 + n, matrix1Trans);
            } else {
                matrix1_trans_int8<tileK>(maxTileN, KInner, K, matrix1 + n * K + k, matrix1Trans);
            }
        }
        int n = N / maxTileN * maxTileN;
        if (N - n > 0) {
            INT8 *matrix1Trans = tmp + n * K4;
            if (transposeA) {
                matrix2_trans_int8<tileK>(N - n, KInner, N, matrix1 + n, matrix1Trans);
            } else {
                matrix1_trans_int8<tileK>(N - n, KInner, K, matrix1 + n * K + k, matrix1Trans);
            }
        }

#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (int i = 0; i < M; i += blockM) {
            int MInner = UNI_MIN(blockM, M - i);
            I32 *resultCurrent;
            int m, n;
            for (n = 0; n <= N - maxTileN; n += maxTileN) {
                INT8 *matrix1Trans = tmp + n * K4;
                //if (i == 0) {
                //    if (transposeA) {
                //        matrix2_trans_int8(4, KInner, N, matrix1 + n, matrix1Trans + n * K4);
                //    } else {
                //        matrix1_trans_int8(4, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K4);
                //    }
                //}
                for (m = 0; m <= (MInner - maxTileM); m += maxTileM) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x8(M, K4, matrix1Trans, matrix2 + (i + m) * K4, resultCurrent);
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_template(maxTileN, MInner - m, M, KInner, matrix1Trans,
                        matrix2 + (i + m) * K4, resultCurrent);
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

                for (m = 0; m <= (MInner - maxTileM); m += maxTileM) {
                    resultCurrent = result + n * M + m + i;
                    mmm_template(N - n, maxTileM, M, KInner, matrix1Trans, matrix2 + (i + m) * K4,
                        resultCurrent);
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
