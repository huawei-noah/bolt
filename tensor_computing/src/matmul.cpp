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


EE matmul_infer_output_size(TensorDesc matrixADesc, TensorDesc matrixBDesc, TensorDesc *matrixCDesc)
{
    if (matrixCDesc == nullptr)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);

    if (matrixADesc.dt != matrixBDesc.dt
       || matrixADesc.df != matrixBDesc.df
       || matrixADesc.nDims != matrixBDesc.nDims
       || matrixADesc.nDims < 2) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }

    int dim = matrixADesc.nDims;
    for (I32 i = 0; i < dim-2; i++) {
        if (matrixADesc.dims[dim-i] != matrixBDesc.dims[dim-i]) {
            CHECK_STATUS_WITH_RETURN(NOT_MATCH);
        }
    }
    
    if (matrixADesc.dims[0] != matrixBDesc.dims[1]) {
        CHECK_STATUS_WITH_RETURN(NOT_MATCH);
    }

    *matrixCDesc = matrixADesc;
    (*matrixCDesc).dims[0] = matrixBDesc.dims[0];
    return SUCCESS;
}


EE matmul_infer_forward_tmp_bytes(TensorDesc matrixADesc, TensorDesc matrixBDesc, U32 *bytes, Arch arch)
{
    if (bytes == nullptr)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);

    EE ret = SUCCESS;
    TensorDesc matrixA2DDesc = tensor2df(matrixADesc.dt, DF_NORMAL, matrixADesc.dims[1], matrixADesc.dims[0]);
    TensorDesc matrixB2Ddesc = tensor2df(matrixBDesc.dt, DF_NORMAL, matrixBDesc.dims[1], matrixBDesc.dims[0]);
    ret = matrix_matrix_multiply_tmp_bytes(matrixA2DDesc, matrixB2Ddesc, bytes, arch);
    return ret;
}

EE  matmul(TensorDesc matrixADesc, const void* matrixA,
    TensorDesc matrixBDesc, const void* matrixB,
    void* tmp, U32 bytes,
    TensorDesc matrixCDesc, void* matrixC,
    Arch arch)
{
    if (matrixA == nullptr || matrixB == nullptr || matrixC == nullptr)
        CHECK_STATUS_WITH_RETURN(NULL_POINTER);

    U32 sizeA = tensorNumElements(matrixADesc);
    U32 loops = sizeA / (matrixADesc.dims[1] * matrixADesc.dims[0]);
    TensorDesc matrixA2DDesc = tensor2df(matrixADesc.dt, DF_NORMAL, matrixADesc.dims[1], matrixADesc.dims[0]);
    TensorDesc matrixB2DDesc = tensor2df(matrixBDesc.dt, DF_NORMAL, matrixBDesc.dims[1], matrixBDesc.dims[0]);
    TensorDesc matrixC2DDesc = tensor2df(matrixCDesc.dt, DF_NORMAL, matrixCDesc.dims[1], matrixCDesc.dims[0]);
    U32 matrixA2DBytes = tensorNumBytes(matrixA2DDesc);
    U32 matrixB2DBytes = tensorNumBytes(matrixB2DDesc);
    U32 matrixC2DBytes = tensorNumBytes(matrixC2DDesc);
    
    U8* matrixAPtr = (U8 *)matrixA;
    U8* matrixBPtr = (U8 *)matrixB;
    U8* matrixCPtr = (U8 *)matrixC;
    for (U32 i = 0; i < loops; i++) {
        CHECK_STATUS_WITH_RETURN(matrix_matrix_multiply(matrixA2DDesc, matrixAPtr,
                                        matrixB2DDesc, matrixBPtr,
                                        bytes, tmp,
                                        matrixC2DDesc, matrixCPtr, arch));

        matrixAPtr += matrixA2DBytes;
        matrixBPtr += matrixB2DBytes;
        matrixCPtr += matrixC2DBytes;
    }
    
    return SUCCESS;
}
