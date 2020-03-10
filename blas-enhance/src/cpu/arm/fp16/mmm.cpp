// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "type.h"
#include "error.h"
#include "cpu/arm/fp16/blas_fp16.h"
#include "mmm.h"


void matrix_matrix_multiply_tmp_bytes_fp16(U32 row1, U32 col1, U32 row2, U32 col2, DataType dt, U32 *bytes)
{
    *bytes = row1 * col1 + row2 * col2;
    *bytes *= bytesOf (dt);
    *bytes += 32;
}

EE mmm_fp16(int M, int N, int K, F16* matrix1, F16* matrix2, F16* tmp, F16* result, Arch arch)
{
    EE ret = SUCCESS;
    switch (arch) {
       case ARM_A55:
           mmm_A55(M, N, K, matrix1, matrix2, tmp, result);
           break;
       case ARM_A76:
           mmm_A76(M, N, K, matrix1, matrix2, tmp, result);
           break;
       default:
           ret = NOT_SUPPORTED;
           break;
    }
    return ret;
}
