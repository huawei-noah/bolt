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
    if (matrixBDesc.df == DF_TRANSPOSE) {
        std::swap(matrixB_K, matrixB_N);
    }

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = matrix_matrix_multiply_tmp_bytes_x86(
            matrixA_M, matrixA_K, matrixB_K, matrixB_N, matrixADataType, bytes);
#endif
#ifdef _USE_NEON
    } else {
        ret = matrix_matrix_multiply_tmp_bytes_arm(
            matrixA_M, matrixA_K, matrixB_K, matrixB_N, matrixADataType, bytes);
#endif
    }
    return ret;
}

EE matrix_matrix_multiply_transform_rhs(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, Arch arch)
{
    EE ret = NOT_SUPPORTED;
#ifdef _USE_NEON
    if (IS_ARM(arch)) {
        ret = matrix_matrix_multiply_transform_rhs_arm(desc, src, descTran, dst);
    }
#endif
#ifdef _USE_GENERAL
    if (IS_GENERAL(arch)) {
        memcpy(dst, src, tensorNumBytes(desc));
        (*descTran) = desc;
        ret = SUCCESS;
    }
#endif
#ifdef _USE_X86
    if (IS_X86_AVX2(arch)) {
        ret = matrix_matrix_multiply_transform_rhs_x86(desc, src, descTran, dst);
    }
#endif
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
    Arch arch)
{
    if (bytes != 0 && tmp == nullptr) {
        CHECK_STATUS(NULL_POINTER);
        return NULL_POINTER;
    }
    if (nullptr == matrixAData || nullptr == matrixBData || nullptr == matrixCData) {
        CHECK_STATUS(NULL_POINTER);
        return NULL_POINTER;
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
        CHECK_STATUS(NOT_MATCH);
    }
    if (matrixADataType != matrixCDataType) {
        if (matrixADataType != DT_I8 || matrixCDataType != DT_I32) {
            CHECK_STATUS(NOT_MATCH);
        }
    }

    bool transposeA = false, transposeB = false;
    if (matrixADataFormat == DF_TRANSPOSE || matrixADataFormat == DF_NKN8) {
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
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = mmm_general(matrixC_N, matrixC_M, matrixA_K, transposeA, transposeB, matrixADataType,
            matrixAData, matrixBData, matrixCData);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        TensorDesc tranDescB;
        U8 *dataB = (U8 *)matrixBData;
        if (matrixBDataFormat != targetFormat4MatrixB(matrixBDataType)) {
            dataB = ((U8 *)tmp) + matrixA_M * matrixA_K * bytesOf(matrixADataType);
            ret = matrix_matrix_multiply_transform_rhs_x86(
                matrixBDesc, matrixBData, &tranDescB, dataB);
        }
        ret = mmm_x86(matrixC_N, matrixC_M, matrixA_K, matrixADataType, matrixADataFormat,
            matrixAData, dataB, tmp, matrixCData);
#endif
#ifdef _USE_NEON
    } else {
        TensorDesc tranDescB;
        U8 *dataB = (U8 *)matrixBData;
        if (matrixBDataFormat != targetFormat4MatrixB(matrixBDataType)) {
            U32 K = matrixA_K;
            if (DT_I8 == matrixADataType) {
                K = pad_to_4_multiple(K);
            }
            dataB = ((U8 *)tmp) + matrixA_M * K * bytesOf(matrixADataType);
            ret = matrix_matrix_multiply_transform_rhs_arm(
                matrixBDesc, matrixBData, &tranDescB, dataB);
        }
        ret = mmm_arm(matrixC_N, matrixC_M, matrixA_K, matrixADataType, transposeA, matrixAData,
            dataB, tmp, matrixCData, arch);
#endif
    }
    return ret;
}
