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
#include "cpu/arm/blas_arm.h"
#include "cpu/arm/int8/blas_matrix_transpose.h"
#include "uni.h"
#include "thread_affinity.h"

static const int tileN = 8;
EE matrix_matrix_multiply_transform_rhsN_int8(TensorDesc desc, INT8 *src, INT8 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    U32 K4 = pad_to_4_multiple(K);
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
    U32 K4 = pad_to_4_multiple(K);
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
#if 1
    int32x4_t ret[tileN][2];
    for (int i = 0; i < tileN; i++) {
        for (int j = 0; j < 2; j++) {
            ret[i][j] = vld1q_s32(out + i * offset + j * 4);
        }
    }
    int16x8_t c[tileN];
    for (U32 n = 0; n < K; n += 4) {
        int8x8_t b0 = vld1_s8(w);
        w += 8;
        for (int i = 0; i < tileN; i++) {
            int8x8_t a0 = vdup_n_s8(in[0]);
            c[i] = vmull_s8(a0, b0);
            in++;
        }
        for (U32 j = 0; j < 3; j++) {
            int8x8_t b0 = vld1_s8(w);
            w += 8;
            for (int i = 0; i < tileN; i++) {
                int8x8_t a0 = vdup_n_s8(in[0]);
                c[i] = vmlal_s8(c[i], a0, b0);
                in++;
            }
        }
        for (int i = 0; i < tileN; i++) {
            ret[i][0] = vaddw_s16(ret[i][0], vget_low_s16(c[i]));
            ret[i][1] = vaddw_s16(ret[i][1], vget_high_s16(c[i]));
        }
    }
    for (int i = 0; i < tileN; i++) {
        for (int j = 0; j < 2; j++) {
            vst1q_s32(out + i * offset + j * 4, ret[i][j]);
        }
    }
#else
    offset *= 4;
    asm volatile("mov x3, %0\n"
                 "ld1r {v0.8b}, [x3]\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 //"ld1r {v4.8b}, [x3]!\n"
                 //"ld1r {v5.8b}, [x3]!\n"
    
                 "mov x0, %1\n"
                 "ldp d6, d7, [x0]!\n"
    
                 // give out address to x26
                 "mov x26, %2\n"
    
                 // load in bias
                 "ldp  q8, q9, [x26]\n"
                 "add x26, x26, %4\n"
                 "ldp  q10, q11, [x26]\n"
                 "add x26, x26, %4\n"
                 "ldp  q12, q13, [x26]\n"
                 "add x26, x26, %4\n"
                 "ldp  q14, q15, [x26]\n"
                 "add x26, x26, %4\n"
                 "ldp  q24, q25, [x26]\n"
                 "add x26, x26, %4\n"
                 "ldp  q26, q27, [x26]\n"
                 "add x26, x26, %4\n"
                 "ldp  q28, q29, [x26]\n"
                 "add x26, x26, %4\n"
                 "ldp  q30, q31, [x26]\n"
    
                 // K- > x26
                 "mov x26, %3\n"
    
                 // Computation loop
                 "0:\n"
    
                 "smull v16.8h, v0.8b, v6.8b\n"
                 "ld1r {v0.8b}, [x3]!\n"
                 "smull v17.8h, v1.8b, v6.8b\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "smull v18.8h, v2.8b, v6.8b\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "smull v19.8h, v3.8b, v6.8b\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 "smull v20.8h, v0.8b, v6.8b\n"
                 "ld1r {v0.8b}, [x3]!\n"
                 "smull v21.8h, v1.8b, v6.8b\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "smull v22.8h, v2.8b, v6.8b\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "smull v23.8h, v3.8b, v6.8b\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 "ldr d6, [x0]!\n"

                 "smlal v16.8h, v0.8b, v7.8b\n"
                 "ld1r {v0.8b}, [x3]!\n"
                 "smlal v17.8h, v1.8b, v7.8b\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "smlal v18.8h, v2.8b, v7.8b\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "smlal v19.8h, v3.8b, v7.8b\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 "smlal v20.8h, v0.8b, v7.8b\n"
                 "ld1r {v0.8b}, [x3]!\n"
                 "smlal v21.8h, v1.8b, v7.8b\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "smlal v22.8h, v2.8b, v7.8b\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "smlal v23.8h, v3.8b, v7.8b\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 "ldr d7, [x0]!\n"

                 "smlal v16.8h, v0.8b, v6.8b\n"
                 "ld1r {v0.8b}, [x3]!\n"
                 "smlal v17.8h, v1.8b, v6.8b\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "smlal v18.8h, v2.8b, v6.8b\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "smlal v19.8h, v3.8b, v6.8b\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 "smlal v20.8h, v4.8b, v6.8b\n"
                 "ld1r {v0.8b}, [x3]!\n"
                 "smlal v21.8h, v1.8b, v6.8b\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "smlal v22.8h, v2.8b, v6.8b\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "smlal v23.8h, v3.8b, v6.8b\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 "ldr d6, [x0]!\n"

                 "smlal v16.8h, v0.8b, v7.8b\n"
                 "ld1r {v0.8b}, [x3]!\n"
                 "smlal v17.8h, v1.8b, v7.8b\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "smlal v18.8h, v2.8b, v7.8b\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "smlal v19.8h, v3.8b, v7.8b\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 "smlal v20.8h, v0.8b, v7.8b\n"
                 "ld1r {v0.8b}, [x3]!\n"
                 "smlal v21.8h, v1.8b, v7.8b\n"
                 "ld1r {v1.8b}, [x3]!\n"
                 "smlal v22.8h, v2.8b, v7.8b\n"
                 "ld1r {v2.8b}, [x3]!\n"
                 "smlal v23.8h, v3.8b, v7.8b\n"
                 "ld1r {v3.8b}, [x3]!\n"
                 "ldr d7, [x0]!\n"
    
                 "subs x26, x26, #4\n"
    
                 "saddw    v8.4s,  v8.4s, v16.4h\n"
                 "saddw2   v9.4s,  v9.4s, v16.8h\n"
                 "saddw   v10.4s, v10.4s, v17.4h\n"
                 "saddw2  v11.4s, v11.4s, v17.8h\n"
                 "saddw   v12.4s, v12.4s, v18.4h\n"
                 "saddw2  v13.4s, v13.4s, v18.8h\n"
                 "saddw   v14.4s, v14.4s, v19.4h\n"
                 "saddw2  v15.4s, v15.4s, v19.8h\n"
                 "saddw   v24.4s, v24.4s, v20.4h\n"
                 "saddw2  v25.4s, v25.4s, v20.8h\n"
                 "saddw   v26.4s, v26.4s, v21.4h\n"
                 "saddw2  v27.4s, v27.4s, v21.8h\n"
                 "saddw   v28.4s, v28.4s, v22.4h\n"
                 "saddw2  v29.4s, v29.4s, v22.8h\n"
                 "saddw   v30.4s, v30.4s, v23.4h\n"
                 "saddw2  v31.4s, v31.4s, v23.8h\n"
    
                 "bne 0b\n"
    
                 // give out address to x26
                 "mov x26, %2\n"
    
                 "stp  q8, q9, [x26]\n"
                 "add x26, x26, %4\n"
                 "stp  q10, q11, [x26]\n"
                 "add x26, x26, %4\n"
                 "stp  q12, q13, [x26]\n"
                 "add x26, x26, %4\n"
                 "stp  q14, q15, [x26]\n"
                 "add x26, x26, %4\n"
                 "stp  q24, q25, [x26]\n"
                 "add x26, x26, %4\n"
                 "stp  q26, q27, [x26]\n"
                 "add x26, x26, %4\n"
                 "stp  q28, q29, [x26]\n"
                 "add x26, x26, %4\n"
                 "stp  q30, q31, [x26]\n"
                 : "+r"(in), "+r"(w), "+r"(out)
                 : "r"((I64)K), "r"((I64)offset)
                 : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                 "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                 "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                 "v30", "v31", "x26");
#endif
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
    int32x4_t res[tileN][2] = {0};
    for (U32 k = 0; k < K; k += unroll) {
        int16x8_t res_s16[tileN] = {0};
        for (U32 kk = 0; kk < unroll; kk++) {
            U32 z = k + kk;
            int8x8_t mat2 = vld1_s8(matrix2 + z * MInner);
            for (int i = 0; i < tileN; i++) {
                int8x8_t mat10 = vdup_n_s8(matrix1[z * tileN + i]);
                res_s16[i] = vmlal_s8(res_s16[i], mat10, mat2);
            }
        }
        for (int i = 0; i < tileN; i++) {
            res[i][0] = vaddw_s16(res[i][0], vget_low_s16(res_s16[i]));
            res[i][1] = vaddw_s16(res[i][1], vget_high_s16(res_s16[i]));
        }
    }
    int tmp[8];
    for (int i = 0; i < tileN; i++) {
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
    U32 K4 = pad_to_4_multiple(K);
    int blockM = 96;
    for (int k = 0; k < K; k += blockK) {
        int KInner = UNI_MIN(blockK, K - k);
#ifdef _USE_OPENMP
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
#endif
        for (int n = 0; n <= N - tileN; n += tileN) {
            INT8 *matrix1Trans = tmp + n * K4;
            if (transposeA) {
                matrix2_trans_int8(tileN, KInner, N, matrix1 + n, matrix1Trans);
            } else {
                matrix1_trans_int8(tileN, KInner, K, matrix1 + n * K + k, matrix1Trans);
            }
        }
        int n = N / tileN * tileN;
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
            for (n = 0; n <= N - tileN; n += tileN) {
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
                    mmm_4x8(M, K4, matrix1Trans, matrix2 + (i + m) * K4, resultCurrent);
                    //mmm_NTail_M(8, M, 4, K4, matrix1Trans, matrix2 + (i + m) * K4, resultCurrent);
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
