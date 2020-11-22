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
#include "types.h"
#include "blas_enhance.h"
#include "cpu/x86/blas_x86.h"
#ifdef _USE_FP32
#include "cpu/x86/fp32/blas_fp32.h"
#endif

EE matrix_vector_multiply_tmp_bytes_x86(bool transpose, DataType dt, U32 *bytes)
{
    if (nullptr == bytes) {
        CHECK_STATUS(NULL_POINTER);
    }
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32:
            *bytes = 0;
            break;
#endif
        default:
            break;
    }
    return SUCCESS;
}

EE mvm_x86(
    U32 row, U32 col, DataType dt, bool transpose, const void *matrix, const void *vector, void *result)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32: {
            ret = mvm_avx2_fp32(row, col, transpose, (F32 *)matrix, (F32 *)vector, (F32 *)result);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
