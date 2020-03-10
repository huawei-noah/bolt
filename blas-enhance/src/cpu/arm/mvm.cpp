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
#include "type.h"
#include "cpu/arm/blas_arm.h"
#ifdef _USE_FP16
#include "cpu/arm/fp16/blas_fp16.h"
#endif
#ifdef _USE_FP32
#include "cpu/arm/fp32/blas_fp32.h"
#endif
#ifdef _USE_INT8
#include "cpu/arm/int8/blas_int8.h"
#endif

EE matrix_vector_multiply_tmp_bytes_arm(bool transpose,
    DataType dt, U32 *bytes)
{
    if (nullptr == bytes)
        CHECK_STATUS(NULL_POINTER);
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            *bytes = 0;
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            *bytes = 0;
            break;
#endif
#ifdef _USE_INT8
        case DT_I8: {
            if (transpose)
                *bytes = 64 * sizeof(I32);
            break;
        }
#endif
        default:
            break;
    }
    return SUCCESS;
}

EE mvm_arm(U32 row, U32 col, DataType dt, bool transpose,
    const void *matrix, const void *vector,
    void *tmp,
    void *result,
    Arch arch)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
            mvm_fp16(row, col, transpose, (F16*)matrix, (F16*)vector, (F16*)result, arch);
            break;
#endif
#ifdef _USE_FP32
        case DT_F32:
            UNUSED(arch);
            mvm_fp32(row, col, transpose, (F32*)matrix, (F32*)vector, (F32*)result);
            break;
#endif
#ifdef _USE_INT8
        case DT_I8:
            UNUSED(arch);
            mvm_int8(row, col, transpose, (INT8*)matrix, (INT8*)vector, (I32*)tmp, (I32*)result);
            break;
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
