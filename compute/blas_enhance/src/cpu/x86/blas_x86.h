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

#include "error.h"
#include "sys.h"
#include "tensor_desc.h"

EE matrix_vector_multiply_transform_weight_x86(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst);

EE matrix_vector_multiply_tmp_bytes_x86(bool transpose, DataType dt, U32 *bytes);

EE mvm_x86(
    U32 row, U32 col, DataType dt, DataFormat df, const void *matrix, const void *vector, void *result);

EE matrix_matrix_multiply_tmp_bytes_x86(
    U32 matrixA_M, U32 matrixA_K, U32 matrixB_K, U32 matrixB_N, DataType dt, U32 *bytes);

EE matrix_matrix_multiply_transform_rhs_x86(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst);

EE mmm_x86(U32 matrixC_N,
    U32 matrixC_M,
    U32 matrixA_K,
    DataType matrixADataType,
    DataFormat matrixADataFormat,
    const void *matrixAData,
    const void *matrixBData,
    void *tmp,
    void *matrixCData);

#endif
