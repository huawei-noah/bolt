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
#include "cpu/general/blas_general.h"

template <typename T1, typename T2>
inline void mmm(
    U32 N, U32 M, U32 K, bool transposeA, bool transposeB, T1 *matrixA, T1 *matrixB, T2 *matrixC)
{
    for (U32 i = 0; i < M; i++) {
        for (U32 n = 0; n < N; n++) {
            F32 value = 0;
            for (U32 j = 0; j < K; j++) {
                U32 indexA = 0, indexB = 0;
                if (transposeA) {
                    indexA = j * M + i;
                } else {
                    indexA = i * K + j;
                }
                if (transposeB) {
                    indexB = n * K + j;
                } else {
                    indexB = j * N + n;
                }
                value += matrixA[indexA] * matrixB[indexB];
            }
            matrixC[i * N + n] += value;
        }
    }
}

EE mmm_general(U32 matrixC_N,
    U32 matrixC_M,
    U32 matrixA_K,
    bool transposeA,
    bool transposeB,
    DataType dt,
    const void *matrixAData,
    const void *matrixBData,
    void *matrixCData)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16: {
            mmm<F16, F16>(matrixC_N, matrixC_M, matrixA_K, transposeA, transposeB,
                (F16 *)matrixAData, (F16 *)matrixBData, (F16 *)matrixCData);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            mmm<INT8, I32>(matrixC_N, matrixC_M, matrixA_K, transposeA, transposeB,
                (INT8 *)matrixAData, (INT8 *)matrixBData, (I32 *)matrixCData);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            mmm<F32, F32>(matrixC_N, matrixC_M, matrixA_K, transposeA, transposeB,
                (F32 *)matrixAData, (F32 *)matrixBData, (F32 *)matrixCData);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
