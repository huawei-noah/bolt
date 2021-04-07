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
#include "uni.h"

EE matrix_vector_multiply_tmp_bytes(
    TensorDesc matrixDesc, TensorDesc vectorDesc, U32 *bytes, Arch arch)
{
    UNUSED(vectorDesc);

    bool transpose = (matrixDesc.df == DF_TRANSPOSE);
    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = matrix_vector_multiply_tmp_bytes_x86(transpose, matrixDesc.dt, bytes);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = matrix_vector_multiply_tmp_bytes_arm(transpose, matrixDesc.dt, bytes);
#endif
    }
    return ret;
}

EE matrix_vector_multiply_transform_weight(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, Arch arch)
{
    EE ret = NOT_SUPPORTED;
#ifdef _USE_NEON
    if (IS_ARM(arch)) {
        ret = matrix_vector_multiply_transform_weight_arm(desc, src, descTran, dst);
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
        ret = matrix_vector_multiply_transform_weight_x86(desc, src, descTran, dst);
    }
#endif
    return ret;
}

EE matrix_vector_multiply(TensorDesc matrixDesc,
    const void *matrix,
    TensorDesc vectorDesc,
    const void *vector,
    U32 bytes,
    void *tmp,
    TensorDesc resultDesc,
    void *result,
    Arch arch)
{
    if (bytes != 0 && tmp == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (nullptr == matrix || nullptr == vector || nullptr == result) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType matrixDataType, vectorDataType, resultDataType;
    DataFormat matrixDataFormat, vectorDataFormat, resultDataFormat;
    U32 matrixRow, matrixColumn, vectorColumn, resultColumn;
    CHECK_STATUS(
        tensor2dGet(matrixDesc, &matrixDataType, &matrixDataFormat, &matrixRow, &matrixColumn));
    CHECK_STATUS(tensor1dGet(vectorDesc, &vectorDataType, &vectorDataFormat, &vectorColumn));
    CHECK_STATUS(tensor1dGet(resultDesc, &resultDataType, &resultDataFormat, &resultColumn));

    if (matrixDataType != vectorDataType) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (matrixDataType != resultDataType) {
        if (matrixDataType != DT_I8 || resultDataType != DT_I32) {
            CHECK_STATUS(NOT_MATCH);
        }
    }

    bool transpose = (matrixDataFormat == DF_TRANSPOSE);
    if (transpose) {
        std::swap(matrixRow, matrixColumn);
    }
    if (matrixRow != resultColumn || matrixColumn != vectorColumn) {
        CHECK_STATUS(NOT_MATCH);
    }

    EE ret = NOT_SUPPORTED;
    if (IS_GENERAL(arch)) {
#ifdef _USE_GENERAL
        ret =
            mvm_general(matrixRow, matrixColumn, matrixDataType, transpose, matrix, vector, result);
#endif
#ifdef _USE_X86
    } else if (IS_X86_AVX2(arch)) {
        ret = mvm_x86(
            matrixRow, matrixColumn, matrixDataType, matrixDataFormat, matrix, vector, result);
#endif
#ifdef _USE_NEON
    } else {
        ret = mvm_arm(matrixRow, matrixColumn, matrixDataType, matrixDataFormat, matrix, vector,
            tmp, result, arch);
#endif
    }
    return ret;
}
