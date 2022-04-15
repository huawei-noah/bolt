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
#include "cpu/arm/fp16/mvm.h"
#include "cpu/arm/fp16/mmm_common.h"
#include "cpu/arm/fp16/mvm_common.h"

EE matrix_vector_multiply_transform_weight_fp16(TensorDesc desc, F16 *src, F16 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    EE ret = SUCCESS;
    int i = 0;
    switch (desc.df) {
        case DF_NORMAL: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
            for (; i < (int)N - 63; i += 64) {
                matrix1_trans(64, K, K, src + i * K, dst + i * K);
            }
            if (i < (int)N) {
                memcpy(dst + i * K, src + i * K, (N - i) * K * bytesOf(DT_F16));
            }
            break;
        }
        case DF_TRANSPOSE: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
            for (; i < (int)N - 63; i += 64) {
                matrix2_trans(64, K, N, src + i, dst + i * K);
            }
            if (i < (int)N) {
                int base = i;
                F16 *basePtr = dst + i * K;
                for (int j = 0; j < (int)K; j++) {
                    for (int k = base; k < (int)N; k++) {
                        basePtr[(k - base) * K + j] = src[j * N + k];
                    }
                }
            }
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

void mvm_kernel_fp16(U32 rounds, U32 K, F16 *matrix, F16 *vector, F16 *result)
{
    U32 N = rounds * 64;
    float16x8_t mat[8];
    F16 v;
    float16x8_t res[8];

    for (U32 n = 0; n < N; n += 64) {
        F16 *bufMov = matrix + n * K;
        for (int i = 0; i < 8; i++) {
            res[i] = vld1q_f16(result + n + i * 8);
        }
        for (U32 k = 0; k < K; k++) {
            v = vector[k];
            for (int i = 0; i < 8; i++) {
                mat[i] = vld1q_f16(bufMov + i * 8);
            }
            for (int i = 0; i < 8; i++) {
                res[i] = vfmaq_n_f16(res[i], mat[i], v);
            }
            bufMov += 64;
        }
        for (int i = 0; i < 8; i++) {
            vst1q_f16(result + n + i * 8, res[i]);
        }
    }
}

void mvm_pack(U32 row, U32 col, F16 *matrix, F16 *vector, F16 *result)
{
    U32 rounds = row / 64;
    U32 nTail = row % 64;

    mvm_kernel_fp16(rounds, col, matrix, vector, result);
    if (0 != nTail) {
        mvm_row_tail(nTail, col, matrix + (row - nTail) * col, vector, result + (row - nTail));
    }
}

EE mvm_fp16(U32 row, U32 col, DataFormat df, F16 *matrix, F16 *vector, F16 *result, Arch arch)
{
    EE ret = SUCCESS;
    if (DF_NKN64 == df) {
        mvm_pack(row, col, matrix, vector, result);
        return ret;
    }
    switch (arch) {
        case ARM_A55:
            mvm_A55(row, col, DF_TRANSPOSE == df, matrix, vector, result);
            break;
        case ARM_A76:
            mvm_A76(row, col, DF_TRANSPOSE == df, matrix, vector, result);
            break;
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
