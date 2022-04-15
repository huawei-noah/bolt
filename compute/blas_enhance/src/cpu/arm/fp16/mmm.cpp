// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "error.h"
#include "cpu/arm/fp16/blas_fp16.h"
#include "mmm.h"
#include "mmm_common.h"

void matrix_matrix_multiply_tmp_bytes_fp16(
    U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes)
{
    *bytes = row1 * col1 + row2 * col2;
    *bytes *= bytesOf(dt);
    *bytes += 32;
}

EE matrix_matrix_multiply_transform_rhsN_fp16(TensorDesc desc, F16 *src, F16 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
    int i = 0;
    for (; i < (int)N - 23; i += 24) {
        matrix2_trans(24, K, N, src + i, dst + i * K);
    }
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

EE matrix_matrix_multiply_transform_rhsT_fp16(TensorDesc desc, F16 *src, F16 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
    int i = 0;
    for (; i < (int)N - 23; i += 24) {
        matrix1_trans(24, K, K, src + i * K, dst + i * K);
    }
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

EE mmm_fp16(
    int M, int N, int K, bool transposeA, F16 *matrix1, F16 *matrix2, F16 *tmp, F16 *result, Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
        case ARM_A55:
            mmm_A55(M, N, K, transposeA, matrix1, matrix2, tmp, result);
            break;
        case ARM_A76:
            mmm_A76(M, N, K, transposeA, matrix1, matrix2, tmp, result);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
