// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
#include "blas_enhance.h"

EE matrix_matrix_multiply_tmp_bytes_arm(U32 matrixC_N,
    U32 matrixC_M,
    U32 matrixA_K,
    DataType adt,
    DataFormat adf,
    DataType bdt,
    DataFormat bdf,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (adt) {
#ifdef _USE_FP16
        case DT_F16:
        case DT_BF16: {
            matrix_matrix_multiply_tmp_bytes_fp16(matrixC_N, matrixC_M, matrixA_K, bdf, bytes);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            matrix_matrix_multiply_tmp_bytes_fp32(matrixC_N, matrixC_M, matrixA_K, bdf, bytes);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            matrix_matrix_multiply_tmp_bytes_int8(matrixC_N, matrixC_M, matrixA_K, bdf, bytes);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE mmm_arm(U32 matrixC_N,
    U32 matrixC_M,
    U32 matrixA_K,
    DataType dt,
    bool transposeA,
    const void *matrixAData,
    const void *matrixBData,
    void *tmp,
    void *matrixCData,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    switch (dt) {
#ifdef _USE_FP16
        case DT_F16:
        case DT_BF16: {
            ret = mmm_fp16(matrixC_N, matrixC_M, matrixA_K, transposeA, (F16 *)matrixAData,
                (F16 *)matrixBData, (F16 *)tmp, (F16 *)matrixCData, arch);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = mmm_fp32(matrixC_N, matrixC_M, matrixA_K, transposeA, (F32 *)matrixAData,
                (F32 *)matrixBData, (F32 *)tmp, (F32 *)matrixCData);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = mmm_int8(matrixC_N, matrixC_M, matrixA_K, transposeA, (INT8 *)matrixAData,
                (INT8 *)matrixBData, (INT8 *)tmp, (I32 *)matrixCData, arch);
            break;
        }
#endif
        default:
            break;
    }
    return ret;
}

EE matrix_matrix_multiply_transform_rhs_bytes_arm(
    U32 matrixC_N, U32 matrixA_K, DataType bdt, DataFormat bdf, U32 *bytes, U32 *rhsBytes)
{
    EE ret = SUCCESS;
    switch (bdt) {
#ifdef _USE_FP16
        case DT_F16:
        case DT_BF16: {
            matrix_matrix_multiply_transform_rhs_bytes_fp16(
                matrixC_N, matrixA_K, bdf, bytes, rhsBytes);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            matrix_matrix_multiply_transform_rhs_bytes_fp32(
                matrixC_N, matrixA_K, bdf, bytes, rhsBytes);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            matrix_matrix_multiply_transform_rhs_bytes_int8(
                matrixC_N, matrixA_K, bdf, bytes, rhsBytes);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

static EE matrix_matrix_multiply_transform_rhsN(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst)
{
    EE ret = NOT_SUPPORTED;
    switch (desc.dt) {
#ifdef _USE_FP16
        case DT_F16:
        case DT_BF16: {
            ret = matrix_matrix_multiply_transform_rhsN_fp16(desc, (F16 *)src, (F16 *)dst);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = matrix_matrix_multiply_transform_rhsN_fp32(desc, (F32 *)src, (F32 *)dst);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = matrix_matrix_multiply_transform_rhsN_int8(desc, (INT8 *)src, (INT8 *)dst);
            break;
        }
#endif
        default:
            break;
    }
    (*descTran) = desc;
    (*descTran).df = matrix_matrix_multiply_rhs_format(desc.dt);
    return ret;
}

static EE matrix_matrix_multiply_transform_rhsT(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst)
{
    EE ret = NOT_SUPPORTED;
    switch (desc.dt) {
#ifdef _USE_FP16
        case DT_F16:
        case DT_BF16: {
            ret = matrix_matrix_multiply_transform_rhsT_fp16(desc, (F16 *)src, (F16 *)dst);
            break;
        }
#endif
#ifdef _USE_FP32
        case DT_F32: {
            ret = matrix_matrix_multiply_transform_rhsT_fp32(desc, (F32 *)src, (F32 *)dst);
            break;
        }
#endif
#ifdef _USE_INT8
        case DT_I8: {
            ret = matrix_matrix_multiply_transform_rhsT_int8(desc, (INT8 *)src, (INT8 *)dst);
            break;
        }
#endif
        default:
            break;
    }
    (*descTran) = desc;
    (*descTran).df = matrix_matrix_multiply_rhs_format(desc.dt);
    std::swap(descTran->dims[0], descTran->dims[1]);
    return ret;
}

EE matrix_matrix_multiply_transform_rhs_arm(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst)
{
    if (desc.df == matrix_matrix_multiply_rhs_format(desc.dt)) {
        return SUCCESS;
    }
    EE ret = NOT_SUPPORTED;
    switch (desc.df) {
        case DF_NORMAL: {
            ret = matrix_matrix_multiply_transform_rhsN(desc, src, descTran, dst);
            break;
        }
        case DF_TRANSPOSE: {
            ret = matrix_matrix_multiply_transform_rhsT(desc, src, descTran, dst);
            break;
        }
        default:
            break;
    }
    return ret;
}
