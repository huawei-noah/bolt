// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "tensor_computing.h"
#include "blas-enhance.h"
#include <string.h>

EE matmul_infer_output_size(TensorDesc matrixADesc, bool transposeA,
    TensorDesc matrixBDesc, bool transposeB,
    TensorDesc *matrixCDesc)
{
    if (matrixCDesc == nullptr)
        CHECK_STATUS(NULL_POINTER);

    if (matrixADesc.dt != matrixBDesc.dt
       || matrixADesc.nDims < 2) {
        CHECK_STATUS(NOT_MATCH);
    }

    int i = 0;
    int j = 0;
    int dimA = matrixADesc.nDims;
    int dimB = matrixBDesc.nDims;
    while (i < dimA-2 || j < dimB-2) {
        if (matrixADesc.dims[dimA-1-i] != matrixBDesc.dims[dimB-1-j]) {
            if (matrixADesc.dims[dimA-1-i] == 1) {
                i++;
                continue;
            }
            if(matrixBDesc.dims[dimB-1-j] == 1) {
                j++;
                continue;
            }
            CHECK_STATUS(NOT_MATCH);
        }
        else {
            i++;
            j++;
        }
    }
    if (i != dimA-2 || j != dimB-2)
        CHECK_STATUS(NOT_MATCH);

    U32 kDimA, kDimB;
    if (transposeA) {
        kDimA = 1;
    } else {
        kDimA = 0;
    }
    if (transposeB) {
        kDimB = 0;
    } else {
        kDimB = 1;
    }

    if (matrixADesc.dims[kDimA] != matrixBDesc.dims[kDimB]) {
        CHECK_STATUS(NOT_MATCH);
    }

    *matrixCDesc = matrixADesc;
    (*matrixCDesc).dims[kDimA] = matrixBDesc.dims[1-kDimB];
    if (transposeA) {
        U32 tmp = (*matrixCDesc).dims[0];
        (*matrixCDesc).dims[0] = (*matrixCDesc).dims[1];
        (*matrixCDesc).dims[1] = tmp;
    }
    return SUCCESS;
}


EE matmul_infer_forward_tmp_bytes(TensorDesc matrixADesc, bool transposeA,
    TensorDesc matrixBDesc, bool transposeB,
    U32 *bytes, Arch arch)
{
    if (bytes == nullptr)
        CHECK_STATUS(NULL_POINTER);

    EE ret = SUCCESS;
    U32 kDimA, kDimB;
    DataFormat dataFormatA, dataFormatB;
    if (transposeA) {
        kDimA = 1;
        dataFormatA = DF_TRANSPOSE;
    } else {
        kDimA = 0;
        dataFormatA = DF_NORMAL;
    }
    if (transposeB) {
        kDimB = 0;
        dataFormatB = DF_TRANSPOSE;
    } else {
        kDimB = 1;
        dataFormatB = DF_NORMAL;
    }
    if (matrixADesc.dims[1-kDimA] == 1 || matrixBDesc.dims[1-kDimB] == 1) {
        TensorDesc matrixDesc, vectorDesc;
        if (matrixADesc.dims[1-kDimA] == 1) {
            matrixDesc = tensor2df(matrixBDesc.dt, dataFormatB, matrixBDesc.dims[1], matrixBDesc.dims[0]);
            vectorDesc = tensor1d(matrixADesc.dt, matrixADesc.dims[kDimA]);
        } else {
            matrixDesc = tensor2df(matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
            vectorDesc = tensor1d(matrixBDesc.dt, matrixBDesc.dims[kDimB]);
        }
        ret = matrix_vector_multiply_tmp_bytes(matrixDesc, vectorDesc, bytes, arch);
    } else {
        TensorDesc matrixA2DDesc = tensor2df(matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
        TensorDesc matrixB2Ddesc = tensor2df(matrixBDesc.dt, dataFormatB, matrixBDesc.dims[1], matrixBDesc.dims[0]);
        ret = matrix_matrix_multiply_tmp_bytes(matrixA2DDesc, matrixB2Ddesc, bytes, arch);
    }
    return ret;
}

EE matmul(TensorDesc matrixADesc, bool transposeA, const void* matrixA,
    TensorDesc matrixBDesc, bool transposeB, const void* matrixB,
    void* tmp, U32 bytes,
    TensorDesc matrixCDesc, void* matrixC,
    Arch arch)
{
    if (matrixA == nullptr || matrixB == nullptr || matrixC == nullptr)
        CHECK_STATUS(NULL_POINTER);

    U32 sizeA = tensorNumElements(matrixADesc);
    U32 loops = sizeA / (matrixADesc.dims[1] * matrixADesc.dims[0]);
    U32 kDimA, kDimB;
    if (transposeA) {
        kDimA = 1;
    } else {
        kDimA = 0;
    }
    if (transposeB) {
        kDimB = 0;
    } else {
        kDimB = 1;
    }

    U32 matrixA2DBytes = (matrixADesc.dims[1] * matrixADesc.dims[0]) * bytesOf(matrixADesc.dt);
    U32 matrixB2DBytes = (matrixBDesc.dims[1] * matrixBDesc.dims[0]) * bytesOf(matrixBDesc.dt);
    U32 matrixC2DBytes = (matrixCDesc.dims[1] * matrixCDesc.dims[0]) * bytesOf(matrixCDesc.dt);
    U8* matrixAPtr = (U8 *)matrixA;
    U8* matrixBPtr = (U8 *)matrixB;
    U8* matrixCPtr = (U8 *)matrixC;
    memset(matrixC, 0, tensorNumBytes(matrixCDesc));
    for (U32 i = 0; i < loops; i++) {
        if (matrixADesc.dims[1-kDimA] == 1) {
            TensorDesc matrixA1DDesc = tensor1d(matrixADesc.dt, matrixADesc.dims[kDimA]);
            TensorDesc matrixB2DDesc;
            if (transposeB) {
                matrixB2DDesc = tensor2df(matrixBDesc.dt, DF_NORMAL, matrixBDesc.dims[1], matrixBDesc.dims[0]);
            } else {
                matrixB2DDesc = tensor2df(matrixBDesc.dt, DF_TRANSPOSE, matrixBDesc.dims[0], matrixBDesc.dims[1]);
            }
            TensorDesc matrixC1DDesc = tensor1d(matrixCDesc.dt, matrixCDesc.dims[0]);
            CHECK_STATUS(matrix_vector_multiply(
                                            matrixB2DDesc, matrixBPtr,
                                            matrixA1DDesc, matrixAPtr,
                                            bytes, tmp,
                                            matrixC1DDesc, matrixCPtr, arch));
        } else {
            if (matrixBDesc.dims[1-kDimB] == 1) {
                TensorDesc matrixA2DDesc;
                if (transposeA) {
                    matrixA2DDesc = tensor2df(matrixADesc.dt, DF_TRANSPOSE, matrixADesc.dims[0], matrixADesc.dims[1]);
                } else {
                    matrixA2DDesc = tensor2df(matrixADesc.dt, DF_NORMAL, matrixADesc.dims[1], matrixADesc.dims[0]);
                }
                TensorDesc matrixB1DDesc = tensor1d(matrixBDesc.dt, matrixBDesc.dims[kDimB]);
                TensorDesc matrixC1DDesc = tensor1d(matrixCDesc.dt, matrixCDesc.dims[1]);
                CHECK_STATUS(matrix_vector_multiply(matrixA2DDesc, matrixAPtr,
                                                matrixB1DDesc, matrixBPtr,
                                                bytes, tmp,
                                                matrixC1DDesc, matrixCPtr, arch));
            } else {
                DataFormat dataFormatA, dataFormatB;
                if (transposeA) {
                    dataFormatA = DF_TRANSPOSE;
                } else {
                    dataFormatA = DF_NORMAL;
                }
                if (transposeB) {
                    dataFormatB = DF_TRANSPOSE;
                } else {
                    dataFormatB = DF_NORMAL;
                }
                TensorDesc matrixA2DDesc = tensor2df(matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
                TensorDesc matrixB2DDesc = tensor2df(matrixBDesc.dt, dataFormatB, matrixBDesc.dims[1], matrixBDesc.dims[0]);
                TensorDesc matrixC2DDesc = tensor2df(matrixCDesc.dt, DF_NORMAL, matrixCDesc.dims[1], matrixCDesc.dims[0]);
                CHECK_STATUS(matrix_matrix_multiply(matrixA2DDesc, matrixAPtr,
                                                matrixB2DDesc, matrixBPtr,
                                                bytes, tmp,
                                                matrixC2DDesc, matrixCPtr, arch));
            }
        }
        matrixAPtr += matrixA2DBytes;
        matrixBPtr += matrixB2DBytes;
        matrixCPtr += matrixC2DBytes;
    }
    
    return SUCCESS;
}
