// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_X86
#define _H_BLAS_X86

#include "tensor_desc.h"

EE axpby_x86(I32 len, DataType dt, F32 a, const void *x, F32 b, void *y);

EE matrix_vector_multiply_tmp_bytes_x86(U32 row, U32 col, DataType dt, DataFormat df, U32 *bytes);

EE mvm_x86(U32 row,
    U32 col,
    DataType dt,
    DataFormat df,
    const void *matrix,
    const void *vector,
    void *result,
    void *offsetCBias,
    const F32 *scale);

EE matrix_matrix_multiply_tmp_bytes_x86(U32 matrixC_N,
    U32 matrixC_M,
    U32 matrixA_K,
    DataType adt,
    DataFormat adf,
    DataType bdt,
    DataFormat bdf,
    U32 *bytes);

EE mmm_x86(U32 matrixC_N,
    U32 matrixC_M,
    U32 matrixA_K,
    DataType matrixADataType,
    DataFormat matrixADataFormat,
    const void *matrixAData,
    const void *matrixBData,
    void *tmp,
    void *matrixCData,
    const F32 *scale);

EE matrix_vector_multiply_transform_weight_bytes_x86(
    U32 row, U32 col, DataType dt, DataFormat df, U32 *bytes);

EE matrix_vector_multiply_transform_weight_x86(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, void *offsetCBias);

EE matrix_matrix_multiply_transform_rhs_bytes_x86(
    U32 matrixC_N, U32 matrixA_K, DataType bdt, DataFormat bdf, U32 *bytes, U32 *rhsBytes);

EE matrix_matrix_multiply_transform_rhs_x86(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst);
#endif
