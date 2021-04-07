// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_BLAS_ENHANCE
#define _H_BLAS_ENHANCE

#include "sys.h"
#include "tensor_desc.h"

#ifdef __cplusplus
extern "C" {
#endif

EE matrix_matrix_multiply_tmp_bytes(
    TensorDesc matrixADesc, TensorDesc matrixBDesc, U32 *bytes, Arch arch);

EE matrix_matrix_multiply(TensorDesc matrixADesc,
    const void *matrixA,
    TensorDesc matrixBDesc,
    const void *matrixB,
    U32 bytes,
    void *tmp,
    TensorDesc matrixCDesc,
    void *matrixC,
    Arch arch);

EE matrix_vector_multiply_tmp_bytes(TensorDesc matrixDesc, TensorDesc vectorDesc, U32 *bytes, Arch);

EE matrix_vector_multiply(TensorDesc matrixDesc,
    const void *matrix,
    TensorDesc vectorDesc,
    const void *vector,
    U32 bytes,
    void *tmp,
    TensorDesc resultDesc,
    void *result,
    Arch arch);

inline DataFormat targetFormat4MatrixB(DataType dt)
{
    switch (dt) {
        case DT_F16: {
            return DF_NKN24;
        }
        case DT_F32: {
#ifdef __aarch64__
            return DF_NKN12;
#else
            return DF_NKN8;
#endif
        }
        case DT_I8: {
            return DF_NKN12K4;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
            return DF_NCHWC8;
        }
    }
}

inline DataFormat targetFormat4mvmMatrix(DataType dt)
{
    switch (dt) {
        case DT_I8: {
            return DF_NKN32K4;
        }
        case DT_F16: {
            return DF_NKN64;
        }
        case DT_F32: {
            return DF_NKN16;
        }
        default: {
            CHECK_STATUS(NOT_SUPPORTED);
            return DF_NCHWC8;
        }
    }
}

EE matrix_matrix_multiply_transform_rhs(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, Arch arch);

EE matrix_vector_multiply_transform_weight(
    TensorDesc desc, const void *src, TensorDesc *descTran, void *dst, Arch arch);

EE vector_vector_axpby(
    F32 a, TensorDesc xDesc, const void *x, F32 b, TensorDesc yDesc, void *y, Arch arch);

#ifdef __cplusplus
}
#endif

#endif
