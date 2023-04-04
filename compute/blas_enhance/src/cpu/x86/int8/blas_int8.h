// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_INT8
#define _H_BLAS_INT8

#include "cpu/x86/int8/blas_common_int8.h"

void matrix_matrix_multiply_tmp_bytes_int8(
    U32 M, U32 N, U32 K, DataFormat matrixA_df, DataFormat matrixB_df, U32 *bytes);

EE mmm_avx512_vnni_int8(U32 M,
    U32 N,
    U32 K,
    DataFormat matrixADataFormat,
    UINT8 *matrix1,
    INT8 *matrix2,
    UINT8 *tmp,
    UINT8 *result,
    const F32 *scale);

EE mvm_avx512_int8(U32 numRows,
    U32 numColumns,
    INT8 *packB,
    UINT8 *vector,
    UINT8 *result,
    I32 *offsetCBias,
    const F32 *scale);

EE mvm_avx512_int8_row_i8u8(U32 numRows,
    U32 numColumns,
    DataFormat df,
    UINT8 *packB,
    INT8 *vector,
    UINT8 *result,
    I32 *tmp,
    const F32 *scale);

EE matrix_vector_multiply_transform_weight_int8(
    TensorDesc desc, INT8 *src, INT8 *packB, I32 *offsetCBias);

void matrix_matrix_multiply_transform_rhs_bytes_int8(
    U32 N, U32 K, DataFormat matrixB_df, U32 *bytes, U32 *rhsBytes);

EE matrix_matrix_multiply_transform_rhsN_int8(TensorDesc desc, INT8 *src, INT8 *dst);

EE matrix_matrix_multiply_transform_rhsT_int8(TensorDesc desc, INT8 *src, INT8 *dst);

#endif
