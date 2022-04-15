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
#include "cpu/arm/fp32/blas_fp32.h"

EE matrix_vector_multiply_transform_weight_fp32(TensorDesc desc, F32 *src, F32 *dst)
{
    DataType dt;
    DataFormat df;
    U32 N, K;
    EE ret = SUCCESS;
    int i = 0;
    switch (desc.df) {
        case DF_NORMAL: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &N, &K));
            for (; i < (int)N - 15; i += 16) {
                matrix1_trans(16, K, K, src + i * K, dst + i * K);
            }
            if (i < (int)N) {
                memcpy(dst + i * K, src + i * K, (N - i) * K * bytesOf(DT_F32));
            }
            break;
        }
        case DF_TRANSPOSE: {
            CHECK_STATUS(tensor2dGet(desc, &dt, &df, &K, &N));
            for (; i < (int)N - 15; i += 16) {
                matrix2_trans(16, K, N, src + i, dst + i * K);
            }
            if (i < (int)N) {
                int base = i;
                F32 *basePtr = dst + i * K;
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

void mvm_kernel_fp32(U32 rounds, U32 K, F32 *matrix, F32 *vector, F32 *result)
{
    U32 N = rounds * 16;
    float32x4_t mat[4];
    F32 v;
    float32x4_t res[4];

    for (U32 n = 0; n < N; n += 16) {
        F32 *bufMov = matrix + n * K;
        for (int i = 0; i < 4; i++) {
            res[i] = vld1q_f32(result + n + i * 4);
        }
        for (U32 k = 0; k < K; k++) {
            v = vector[k];
            for (int i = 0; i < 4; i++) {
                mat[i] = vld1q_f32(bufMov + i * 4);
            }
            for (int i = 0; i < 4; i++) {
                res[i] = vfmaq_n_f32(res[i], mat[i], v);
            }
            bufMov += 16;
        }
        for (int i = 0; i < 4; i++) {
            vst1q_f32(result + n + i * 4, res[i]);
        }
    }
}

void mvm_pack_fp32(U32 row, U32 col, F32 *matrix, F32 *vector, F32 *result)
{
    U32 rounds = row / 16;
    U32 nTail = row % 16;

    mvm_kernel_fp32(rounds, col, matrix, vector, result);
    if (0 != nTail) {
        mvm_row_tail(nTail, col, matrix + (row - nTail) * col, vector, result + (row - nTail));
    }
}

EE mvm_fp32(U32 row, U32 col, DataFormat df, F32 *matrix, F32 *vector, F32 *result)
{
    EE ret = SUCCESS;
    switch (df) {
        case DF_NKN16: {
            mvm_pack_fp32(row, col, matrix, vector, result);
            break;
        }
        case DF_NORMAL: {
            mvm_row_fp32(row, col, matrix, vector, result);
            break;
        }
        case DF_TRANSPOSE: {
            mvm_col_fp32(row, col, matrix, vector, result);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
