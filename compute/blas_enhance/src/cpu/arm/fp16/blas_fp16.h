// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_FP16
#ifndef _H_BLAS_FP16
#define _H_BLAS_FP16

#include "sys.h"

#include "error.h"
#include "tensor_desc.h"

EE matrix_vector_multiply_transform_weight_fp16(TensorDesc desc, F16 *src, F16 *dst);

EE mvm_fp16(U32 row, U32 col, DataFormat df, F16 *matrix, F16 *vector, F16 *result, Arch arch);

void matrix_matrix_multiply_tmp_bytes_fp16(
    U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes);

EE matrix_matrix_multiply_transform_rhsN_fp16(TensorDesc desc, F16 *src, F16 *dst);

EE matrix_matrix_multiply_transform_rhsT_fp16(TensorDesc desc, F16 *src, F16 *dst);

EE mmm_fp16(
    int M, int N, int K, bool transposeA, F16 *matrix1, F16 *matrix2, F16 *tmp, F16 *result, Arch arch);

EE axpby_fp16(U32 len, F32 a, const F16 *x, F32 b, F16 *y);

#endif
#endif
