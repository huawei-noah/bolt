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
#include "cpu/arm/int8/v9/kernels/12x8.h"
#include "cpu/arm/int8/v9/kernels/12x4.h"
#include "cpu/arm/int8/v9/kernels/12x2.h"
#include "cpu/arm/int8/v9/kernels/8x8.h"
#include "cpu/arm/int8/v9/kernels/8x4.h"
#include "cpu/arm/int8/v9/kernels/8x2.h"
#include "cpu/arm/int8/v9/kernels/4x8.h"
#include "cpu/arm/int8/v9/kernels/4x4.h"
#include "cpu/arm/int8/v9/kernels/4x2.h"
#include "cpu/arm/int8/v9/kernels/2x8.h"
#include "cpu/arm/int8/v9/kernels/2x4.h"
#include "cpu/arm/int8/v9/kernels/2x2.h"
#include "cpu/arm/int8/v9/kernels/tail.h"

EE matrix_matrix_multiply_transform_rhsN_int8(TensorDesc desc, INT8 *src, INT8 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    U32 K8 = UNI_ALIGN(K, 8);
    int i = 0;
    for (; i < (int)N - 7; i += 8) {
        matrix2_trans_int8<8>(8, K, N, src + i, dst + i * K8);
    }
    for (; i < (int)N - 3; i += 4) {
        matrix2_trans_int8<8>(4, K, N, src + i, dst + i * K8);
    }
    for (; i < (int)N - 1; i += 2) {
        matrix2_trans_int8<8>(2, K, N, src + i, dst + i * K8);
    }
    if (N - i == 1) {
        matrix2_trans_int8<8>(1, K, N, src + i, dst + i * K8);
    }
    return SUCCESS;
}

EE matrix_matrix_multiply_transform_rhsT_int8(TensorDesc desc, INT8 *src, INT8 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    U32 K8 = UNI_ALIGN(K, 8);
    int i = 0;
    for (; i < (int)N - 7; i += 8) {
        matrix1_trans_int8<8>(8, K, K, src + i * K, dst + i * K8);
    }
    for (; i < (int)N - 3; i += 4) {
        matrix1_trans_int8<8>(4, K, K, src + i * K, dst + i * K8);
    }
    for (; i < (int)N - 1; i += 2) {
        matrix1_trans_int8<8>(2, K, K, src + i * K, dst + i * K8);
    }
    if (N - i == 1) {
        matrix1_trans_int8<8>(1, K, K, src + i * K, dst + i * K8);
    }
    return SUCCESS;
}

EE mmm_int8(
    int M, int N, int K, bool transposeA, INT8 *matrix1, INT8 *matrix2, INT8 *tmp, I32 *result, Arch arch)
{
    int blockK = K;
    U32 K8 = UNI_ALIGN(K, 8);
    int blockM = 96;
    INT8 *matrix1Trans = tmp;
    I32 *resultCurrent = result;

    U32 KK8 = K8 / 8;
    int KInner, MInner, m, n;
    for (int k = 0; k < K; k += blockK) {
        KInner = UNI_MIN(blockK, K - k);  // K for this inner iteration
        for (int i = 0; i < M; i += blockM) {
            MInner = UNI_MIN(blockM, M - i);  // M for this inner iteration
            for (n = 0; n <= N - 12; n += 12) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans_int8<8>(12, KInner, N, matrix1 + n, matrix1Trans + n * K8);
                    } else {
                        matrix1_trans_int8<8>(
                            12, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K8);
                    }
                }

                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_12x8(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_12x4(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                    m += 4;
                }

                if ((MInner - m) >= 2) {
                    resultCurrent = result + n * M + m + i;
                    mmm_12x2(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                    m += 2;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_Nx1(12, M, K8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                }
            }

            if ((N - n) >= 8) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans_int8<8>(8, KInner, N, matrix1 + n, matrix1Trans + n * K8);
                    } else {
                        matrix1_trans_int8<8>(
                            8, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K8);
                    }
                }

                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x8(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x4(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                    m += 4;
                }

                if ((MInner - m) >= 2) {
                    resultCurrent = result + n * M + m + i;
                    mmm_8x2(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                    m += 2;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_Nx1(8, M, K8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                }
                n += 8;
            }

            if ((N - n) >= 4) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans_int8<8>(4, KInner, N, matrix1 + n, matrix1Trans + n * K8);
                    } else {
                        matrix1_trans_int8<8>(
                            4, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K8);
                    }
                }

                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x8(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x4(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                    m += 4;
                }

                if ((MInner - m) >= 2) {
                    resultCurrent = result + n * M + m + i;
                    mmm_4x2(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                    m += 2;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_Nx1(4, M, K8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                }
                n += 4;
            }

            if ((N - n) >= 2) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans_int8<8>(2, KInner, N, matrix1 + n, matrix1Trans + n * K8);
                    } else {
                        matrix1_trans_int8<8>(
                            2, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K8);
                    }
                }

                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_2x8(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_2x4(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                    m += 4;
                }

                if ((MInner - m) >= 2) {
                    resultCurrent = result + n * M + m + i;
                    mmm_2x2(
                        M * 4, KK8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                    m += 2;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_Nx1(2, M, K8, matrix1Trans + n * K8, matrix2 + (i + m) * K8, resultCurrent);
                }
                n += 2;
            }

            if (N - n) {
                if (i == 0) {
                    if (transposeA) {
                        matrix2_trans_int8<8>(N - n, KInner, N, matrix1 + n, matrix1Trans + n * K8);
                    } else {
                        matrix1_trans_int8<8>(
                            N - n, KInner, K, matrix1 + n * K + k, matrix1Trans + n * K8);
                    }
                }

                for (m = 0; m <= (MInner - 8); m += 8) {
                    resultCurrent = result + n * M + m + i;
                    mmm_Nx1(8, 1, K8, matrix2 + (i + m) * K8, matrix1Trans + n * K8, resultCurrent);
                }

                if ((MInner - m) >= 4) {
                    resultCurrent = result + n * M + m + i;
                    mmm_Nx1(4, 1, K8, matrix2 + (i + m) * K8, matrix1Trans + n * K8, resultCurrent);
                    m += 4;
                }

                if ((MInner - m) >= 2) {
                    resultCurrent = result + n * M + m + i;
                    mmm_Nx1(2, 1, K8, matrix2 + (i + m) * K8, matrix1Trans + n * K8, resultCurrent);
                    m += 2;
                }

                if (MInner - m) {
                    resultCurrent = result + n * M + m + i;
                    mmm_1x1(K8, matrix2 + (i + m) * K8, matrix1Trans + n * K8, resultCurrent);
                }
            }
        }
    }
    return SUCCESS;
}
