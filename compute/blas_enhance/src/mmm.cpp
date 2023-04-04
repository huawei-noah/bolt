// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "blas_enhance.h"
#include "uni.h"
#ifdef _USE_GENERAL
#include "cpu/general/blas_general.h"
#endif
#ifdef _USE_NEON
#include "cpu/arm/blas_arm.h"
#endif
#ifdef _USE_X86
#include "cpu/x86/blas_x86.h"
#endif

EE matrix_matrix_multiply_tmp_bytes(
    TensorDesc matrixADesc, TensorDesc matrixBDesc, U32 *bytes, Arch arch)
{
    DataType matrixADataType, matrixBDataType;
    DataFormat matrixADataFormat, matrixBDataFormat;
    U32 matrixA_M, matrixA_K, matrixB_K, matrixB_N;
    CHECK_STATUS(
        tensor2dGet(matrixADesc, &matrixADataType, &matrixADataFormat, &matrixA_M, &matrixA_K));
    CHECK_STATUS(
        tensor2dGet(matrixBDesc, &matrixBDataType, &matrixBDataFormat, &matrixB_K, &matrixB_N));
    if (matrixADesc.df == DF_TRANSPOSE) {
        std::swap(matrixA_K, matrixA_M);
    }
    if (matrixBDesc.df == DF_TRANSPOSE) {
        std::swap(matrixB_K, matrixB_N);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = matrix_matrix_multiply_tmp_bytes_x86(matrixB_N, matrixA_M, matrixB_K, matrixADataType,
            matrixADataFormat, matrixBDataType, matrixBDataFormat, bytes);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = matrix_matrix_multiply_tmp_bytes_arm(matrixB_N, matrixA_M, matrixB_K, matrixADataType,
            matrixADataFormat, matrixBDataType, matrixBDataFormat, bytes);
#endif
    }
    return ret;
}

DataFormat matrix_matrix_multiply_rhs_format(DataType dt)
{
    DataFormat ret;
    switch (dt) {
        case DT_F16: {
#ifdef _USE_MATRIX
            ret = DF_NKNxKx;
#else
            ret = DF_NKN24;
#endif
            break;
        }
        case DT_BF16: {
            ret = DF_NKNxKx;
            break;
        }
        case DT_F32: {
#ifdef __aarch64__
            ret = DF_NKN12;
#else
            ret = DF_NKN8;
#endif
            break;
        }
        case DT_I8: {
            ret = DF_NKNxKx;
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
            break;
        }
    }
    return ret;
}

EE matrix_matrix_multiply_transform_rhs_bytes(
    TensorDesc matrixBDesc, U32 *bytes, U32 *rhsBytes, Arch arch)
{
    DataType matrixBDataType;
    DataFormat matrixBDataFormat;
    U32 matrixB_K, matrixB_N;
    CHECK_STATUS(
        tensor2dGet(matrixBDesc, &matrixBDataType, &matrixBDataFormat, &matrixB_K, &matrixB_N));
    if (matrixBDesc.df == DF_TRANSPOSE) {
        std::swap(matrixB_K, matrixB_N);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        *bytes = tensorNumBytes(matrixBDesc);
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = matrix_matrix_multiply_transform_rhs_bytes_x86(
            matrixB_N, matrixB_K, matrixBDataType, matrixBDataFormat, bytes, rhsBytes);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = matrix_matrix_multiply_transform_rhs_bytes_arm(
            matrixB_N, matrixB_K, matrixBDataType, matrixBDataFormat, bytes, rhsBytes);
#endif
    }
    return ret;
}

EE matrix_matrix_multiply_transform_rhs(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret = matrix_matrix_multiply_transform_rhs_arm(desc, src, descTran, dst);
#endif
#ifdef _USE_GENERAL
    } else if (IS_GENERAL(arch)) {
        UNI_MEMCPY(dst, src, tensorNumBytes(desc));
        (*descTran) = desc;
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = matrix_matrix_multiply_transform_rhs_x86(desc, src, descTran, dst);
#endif
    }
    return ret;
}

EE matrix_matrix_multiply(TensorDesc matrixADesc,
    const void *matrixAData,
    TensorDesc matrixBDesc,
    const void *matrixBData,
    U32 bytes,
    void *tmp,
    TensorDesc matrixCDesc,
    void *matrixCData,
    const F32 *scale,
    Arch arch)
{
    if (bytes != 0 && tmp == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (nullptr == matrixAData || nullptr == matrixBData || nullptr == matrixCData) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (tensorNumElements(matrixCDesc) == 0) {
        return SUCCESS;
    }

    DataType matrixADataType, matrixBDataType, matrixCDataType;
    DataFormat matrixADataFormat, matrixBDataFormat, matrixCDataFormat;
    U32 matrixA_M, matrixA_K, matrixB_K, matrixB_N, matrixC_M, matrixC_N;
    CHECK_STATUS(
        tensor2dGet(matrixADesc, &matrixADataType, &matrixADataFormat, &matrixA_M, &matrixA_K));
    CHECK_STATUS(
        tensor2dGet(matrixBDesc, &matrixBDataType, &matrixBDataFormat, &matrixB_K, &matrixB_N));
    CHECK_STATUS(
        tensor2dGet(matrixCDesc, &matrixCDataType, &matrixCDataFormat, &matrixC_M, &matrixC_N));

    if (matrixADataType != matrixBDataType) {
        if (!(matrixADataType == DT_U8_Q && matrixBDataType == DT_I8)) {
            CHECK_STATUS(NOT_MATCH);
        }
    }
    bool transposeA = false, transposeB = false;
    if (matrixADataFormat == DF_TRANSPOSE) {
        std::swap(matrixA_M, matrixA_K);
        transposeA = true;
    }
    if (matrixBDataFormat == DF_TRANSPOSE) {
        std::swap(matrixB_K, matrixB_N);
        transposeB = true;
    }
    if (matrixA_M != matrixC_M || matrixB_N != matrixC_N || matrixA_K != matrixB_K) {
        CHECK_STATUS(NOT_MATCH);
    }

    EE ret = NOT_SUPPORTED;
    if (IS_CPU(arch)) {
        if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
            ret = mmm_general(matrixC_N, matrixC_M, matrixA_K, transposeA, transposeB,
                matrixADataType, matrixAData, matrixBData, matrixCData);
#endif
        } else {
            auto transB = matrixBData;
            if (matrixBDataFormat != matrix_matrix_multiply_rhs_format(matrixBDataType)) {
                U32 transBBytes = 0;
                CHECK_STATUS(matrix_matrix_multiply_transform_rhs_bytes(
                    matrixBDesc, nullptr, &transBBytes, arch));
                CHECK_REQUIREMENT(
                    transBBytes >= tensorNumBytes(matrixBDesc) && bytes >= transBBytes);
                TensorDesc transBDesc;
                CHECK_STATUS(matrix_matrix_multiply_transform_rhs(
                    matrixBDesc, matrixBData, &transBDesc, tmp, arch));
                transB = tmp;
                tmp = (U8 *)tmp + transBBytes;
            }
#ifdef _USE_X86
            if (IS_X86(arch)) {
                if (matrixCDataType == DT_I32) {
                    scale = nullptr;
                }
                ret = mmm_x86(matrixC_N, matrixC_M, matrixA_K, matrixBDataType, matrixADataFormat,
                    matrixAData, transB, tmp, matrixCData, scale);
            }
#endif
#ifdef _USE_NEON
            if (IS_ARM(arch)) {
                ret = mmm_arm(matrixC_N, matrixC_M, matrixA_K, matrixADataType, transposeA,
                    matrixAData, transB, tmp, matrixCData, arch);
            }
#endif
        }
    }
    return ret;
}
