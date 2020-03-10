// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_BLAS_FP32
#define _H_BLAS_FP32

#include "sys.h"
#include "type.h"
#include "error.h"

void mvm_col_V8(U32 row, U32 col, F32* matrix, F32* vector, F32* result);

void mvm_row_V8(U32 row, U32 col, F32* matrix, F32* vector, F32* result);

inline void mvm_fp32(U32 row, U32 col, bool transpose, F32* matrix, F32* vector, F32* result)
{
    if (transpose) {
        mvm_col_V8(row, col, matrix, vector, result);
    } else {
        mvm_row_V8(row, col, matrix, vector, result);
    }
}

void matrix_matrix_multiply_tmp_bytes_fp32(U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes);

void mmm_fp32_V8(int M, int N, int K, F32* matrix1, F32* matrix2, F32* tmp, F32* result);

#endif
