// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_ENHANCE
#define _H_BLAS_ENHANCE

#include "sys.h"
#include "tensor_desc.h"

#ifdef __cplusplus
extern "C" {
#endif

// a * x[N] + b = y[N]
EE vector_vector_axpby(
    F32 a, TensorDesc xDesc, const void *x, F32 b, TensorDesc yDesc, void *y, Arch arch);

// A[M x K] * B[K x N] = C[M, N]
// A = Normal     MxK    tensor2df(DF_NORMAL, M, K)
// A = Transpose  KxM    tensor2df(DF_TRANSPOSE, K, M)
// A = ????       MxK    tensor2df(????, M, K)           deprecated
// B = Normal     KxN    tensor2df(DF_NORMAL, K, N)
// B = Transpose  NxK    tensor2df(DF_TRANSPOSE, N, K)
// B = NKNx       NxK    tensor2df(NKNx, N, K)
// If B is weight, B can be prereordered before inference, you can see below.
EE matrix_matrix_multiply_tmp_bytes(
    TensorDesc matrixADesc, TensorDesc matrixBDesc, U32 *bytes, Arch arch);

EE matrix_matrix_multiply(TensorDesc matrixADesc,
    const void *matrixA,
    TensorDesc matrixBDesc,
    const void *matrixB,
    U32 bytes,
    void *tmp,
    TensorDesc matrixCDesc,
    void *matrixC,
    const F32 *scale,
    Arch arch);

// matrix[N x K] * vector[K] = result[N]
// matrix = Normal     NxK    tensor2df(DF_NORMAL, N, K)
// matrix = Transpose  KxN    tensor2df(DF_TRANSPOSE, K, N)
// matrix = NKNx       NxK    tensor2df(NKNx, M, K)
// If matrix is weight, matrix can be prereordered before inference, you can see below.
EE matrix_vector_multiply_tmp_bytes(TensorDesc matrixDesc, TensorDesc vectorDesc, U32 *bytes, Arch);

EE matrix_vector_multiply(TensorDesc matrixDesc,
    const void *matrix,
    TensorDesc vectorDesc,
    const void *vector,
    U32 bytes,
    void *tmp,
    TensorDesc resultDesc,
    void *result,
    const F32 *scale,
    Arch arch);

// If you want to reorder weight matrix for mmm, you can use these functions.
DataFormat matrix_matrix_multiply_rhs_format(DataType dt);

// bytes contains all needed, and rhsBytes only contains packed matrix bytes.
EE matrix_matrix_multiply_transform_rhs_bytes(TensorDesc desc, U32 *bytes, U32 *rhsBytes, Arch arch);

// dst contains transformed matrix and offset array for uint8.
EE matrix_matrix_multiply_transform_rhs(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, Arch arch);

// If you want to reorder weight matrix for mvm, you can use these functions.
DataFormat matrix_vector_multiply_weight_format(DataType dt);

EE matrix_vector_multiply_transform_weight_bytes(TensorDesc desc, U32 *bytes, Arch arch);

// dst contains offset array for uint8 and transformed matrix.
EE matrix_vector_multiply_transform_weight(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, Arch arch);
#ifdef __cplusplus
}
#endif
#endif
