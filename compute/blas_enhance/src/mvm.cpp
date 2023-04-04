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
    DataType matrixDataType;
    DataFormat matrixDataFormat;
    U32 matrixRow, matrixColumn;
    CHECK_STATUS(
        tensor2dGet(matrixDesc, &matrixDataType, &matrixDataFormat, &matrixRow, &matrixColumn));
    bool transpose = (matrixDataFormat == DF_TRANSPOSE);
    if (transpose) {
        std::swap(matrixRow, matrixColumn);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_X86(arch)) {
#ifdef _USE_X86
        ret = matrix_vector_multiply_tmp_bytes_x86(
            matrixRow, matrixColumn, matrixDataType, matrixDataFormat, bytes);
#endif
    } else {
        *bytes = 0;
        ret = SUCCESS;
    }
    return ret;
}

DataFormat matrix_vector_multiply_weight_format(DataType dt)
{
    DataFormat ret;
    switch (dt) {
        case DT_I8: {
            ret = DF_NKN32K4;
            break;
        }
        case DT_F16: {
            ret = DF_NKN64;
            break;
        }
        case DT_F32: {
            ret = DF_NKN16;
            break;
        }
        case DT_U8_Q: {
            ret = DF_NK;
            break;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
            break;
        }
    }
    return ret;
}

EE matrix_vector_multiply_transform_weight_bytes(TensorDesc matrixDesc, U32 *bytes, Arch arch)
{
    DataType matrixDataType;
    DataFormat matrixDataFormat;
    U32 matrixRow, matrixColumn;
    CHECK_STATUS(
        tensor2dGet(matrixDesc, &matrixDataType, &matrixDataFormat, &matrixRow, &matrixColumn));
    bool transpose = (matrixDataFormat == DF_TRANSPOSE);
    if (transpose) {
        std::swap(matrixRow, matrixColumn);
    }
    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret = matrix_vector_multiply_transform_weight_bytes_arm(
            matrixRow, matrixColumn, matrixDataType, matrixDataFormat, bytes);
#endif
#ifdef _USE_GENERAL
    } else if (IS_GENERAL(arch)) {
        *bytes = tensorNumBytes(matrixDesc);
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = matrix_vector_multiply_transform_weight_bytes_x86(
            matrixRow, matrixColumn, matrixDataType, matrixDataFormat, bytes);
#endif
    }
    return ret;
}

EE matrix_vector_multiply_transform_weight(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (IS_ARM(arch)) {
#ifdef _USE_NEON
        ret = matrix_vector_multiply_transform_weight_arm(desc, src, descTran, dst);
#endif
#ifdef _USE_GENERAL
    } else if (IS_GENERAL(arch)) {
        UNI_MEMCPY(dst, src, tensorNumBytes(desc));
        (*descTran) = desc;
        ret = SUCCESS;
#endif
#ifdef _USE_X86
    } else if (IS_X86(arch)) {
        ret = matrix_vector_multiply_transform_weight_x86(desc, src, descTran, dst, nullptr);
#endif
    }
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
    const F32 *scale,
    Arch arch)
{
    if ((bytes != 0 && tmp == nullptr) || nullptr == matrix || nullptr == vector ||
        nullptr == result) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType matrixDataType, vectorDataType, resultDataType;
    DataFormat matrixDataFormat, vectorDataFormat, resultDataFormat;
    U32 matrixRow, matrixColumn, vectorColumn, resultColumn;
    CHECK_STATUS(
        tensor2dGet(matrixDesc, &matrixDataType, &matrixDataFormat, &matrixRow, &matrixColumn));
    CHECK_STATUS(tensor1dGet(vectorDesc, &vectorDataType, &vectorDataFormat, &vectorColumn));
    CHECK_STATUS(tensor1dGet(resultDesc, &resultDataType, &resultDataFormat, &resultColumn));
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
    } else if (IS_X86(arch)) {
        U8 *dataB = (U8 *)matrix;
#ifdef _USE_INT8
        if (matrixDataType == DT_U8_Q &&
            matrixDataFormat == matrix_vector_multiply_weight_format(matrixDataType)) {
            return NOT_SUPPORTED;
        }
        if (matrixDataType == DT_I8 &&
            matrixDataFormat != matrix_vector_multiply_weight_format(matrixDataType)) {
            dataB = ((U8 *)tmp);
            TensorDesc tranDescB;
            CHECK_STATUS(matrix_vector_multiply_transform_weight_x86(
                matrixDesc, matrix, &tranDescB, dataB + resultColumn * bytesOf(DT_I32), dataB));
        }
#endif
        ret = mvm_x86(matrixRow, matrixColumn, matrixDataType, matrixDataFormat, dataB, vector,
            result, tmp, scale);
#endif
#ifdef _USE_NEON
    } else if (IS_ARM(arch)) {
        ret = mvm_arm(matrixRow, matrixColumn, matrixDataType, matrixDataFormat, matrix, vector,
            tmp, result, arch);
#endif
    }
    return ret;
}
