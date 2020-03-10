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


EE matrix_matrix_multiply_tmp_bytes_arm(U32 matrixA_M, U32 matrixA_K, U32 matrixB_K, U32 matrixB_N,
    DataType dt, U32 *bytes)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16: {
            matrix_matrix_multiply_tmp_bytes_fp16(matrixA_M, matrixA_K, matrixB_K, matrixB_N, dt, bytes);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            matrix_matrix_multiply_tmp_bytes_fp32(matrixA_M, matrixA_K, matrixB_K, matrixB_N, dt, bytes);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            matrix_matrix_multiply_tmp_bytes_int8(matrixA_M, matrixA_K, matrixB_K, matrixB_N, dt, bytes);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}

EE mmm_arm(U32 matrixC_N, U32 matrixC_M, U32 matrixA_K,
     DataType dt,
     const void* matrixAData, const void* matrixBData,
     void* tmp,
     void* matrixCData,
     Arch arch)
{
    EE ret = SUCCESS;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16: {
            mmm_fp16(matrixC_N, matrixC_M, matrixA_K, (F16*)matrixAData, (F16*)matrixBData, (F16*)tmp, (F16*)matrixCData, arch);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            UNUSED(arch);
            mmm_fp32_V8(matrixC_N, matrixC_M, matrixA_K, (F32*)matrixAData, (F32*)matrixBData, (F32*)tmp, (F32*)matrixCData);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            if (matrixA_K % 4 != 0) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            mmm_int8(matrixC_N, matrixC_M, matrixA_K, (INT8*)matrixAData, (INT8*)matrixBData, (INT8*)tmp, (I32*)matrixCData, arch);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
