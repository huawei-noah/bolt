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
#include "blas_enhance.h"
#include <string.h>
#ifdef _USE_MALI
#include "gpu/mali/tensor_computing_mali.h"
#endif

EE matmul_infer_output_size_cpu(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc *matrixCDesc)
{
    if (matrixCDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }

    if (DT_I8 == matrixADesc.dt || DT_I8 == matrixBDesc.dt) {
        matrixADesc.dt = DT_I8;
        matrixBDesc.dt = DT_I8;
    }

    if (matrixADesc.dt != matrixBDesc.dt || matrixADesc.nDims < 2) {
        CHECK_STATUS(NOT_MATCH);
    }

    if (DF_NCHWC8 == matrixADesc.df && 4 == matrixADesc.nDims) {
        CHECK_REQUIREMENT(1 == matrixADesc.dims[1] && 1 == matrixADesc.dims[0]);
    }

    if (DF_NCHWC8 == matrixBDesc.df && 4 == matrixBDesc.nDims) {
        CHECK_REQUIREMENT(1 == matrixBDesc.dims[1] && 1 == matrixBDesc.dims[0]);
    }

    int i = 0;
    int j = 0;
    int dimA = matrixADesc.nDims;
    int dimB = matrixBDesc.nDims;

    if (dimA > 2 && dimB > 2 && dimA == dimB && matrixBDesc.dims[1] == matrixADesc.dims[0]) {
        (*matrixCDesc) = matrixADesc;
        (*matrixCDesc).dims[0] = matrixBDesc.dims[0];
        (*matrixCDesc).dims[1] = matrixADesc.dims[1];
        for (int i = 2; i < dimA; i++) {
            if (matrixADesc.dims[i] == 1 || matrixBDesc.dims[i] == 1 ||
                matrixBDesc.dims[i] == matrixADesc.dims[i]) {
                (*matrixCDesc).dims[i] = matrixADesc.dims[i] > matrixBDesc.dims[i]
                    ? matrixADesc.dims[i]
                    : matrixBDesc.dims[i];
            }
        }
        return SUCCESS;
    }

    while (i < dimA - 2 || j < dimB - 2) {
        if (matrixADesc.dims[dimA - 1 - i] != matrixBDesc.dims[dimB - 1 - j]) {
            if (matrixADesc.dims[dimA - 1 - i] == 1) {
                i++;
                continue;
            }
            if (matrixBDesc.dims[dimB - 1 - j] == 1) {
                j++;
                continue;
            }
            CHECK_STATUS(NOT_MATCH);
        } else {
            i++;
            j++;
        }
    }
    if (i != dimA - 2 || j != dimB - 2) {
        CHECK_STATUS(NOT_MATCH);
    }

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
    (*matrixCDesc).dims[kDimA] = matrixBDesc.dims[1 - kDimB];
    if (transposeA) {
        U32 tmp = (*matrixCDesc).dims[0];
        (*matrixCDesc).dims[0] = (*matrixCDesc).dims[1];
        (*matrixCDesc).dims[1] = tmp;
    }
    return SUCCESS;
}

EE matmul_infer_output_size(Tensor *matrixATensor,
    bool transposeA,
    Tensor *matrixBTensor,
    bool transposeB,
    Tensor *matrixCTensor,
    ArchInfo_t archInfo)
{
    if (matrixATensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (matrixBTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (matrixCTensor == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    TensorDesc matrixADesc = matrixATensor->get_desc();
    TensorDesc matrixBDesc = matrixBTensor->get_desc();
    TensorDesc matrixCDesc = matrixCTensor->get_desc();
    if (IS_MALI_GPU(archInfo->arch)) {
#ifdef _USE_MALI
        GCLMemDesc gclmemMatrixADesc = ocl_get_desc(*matrixATensor);
        GCLMemDesc gclmemMatrixBDesc = ocl_get_desc(*matrixBTensor);
        GCLMemDesc gclmemMatrixCDesc = ocl_get_desc(*matrixCTensor);
        CHECK_STATUS(matmul_infer_output_size_mali(matrixADesc, transposeA, matrixBDesc, transposeB,
            &matrixCDesc, &gclmemMatrixADesc, &gclmemMatrixBDesc, &gclmemMatrixCDesc));
        ocl_set_desc(matrixATensor, gclmemMatrixADesc);
        ocl_set_desc(matrixBTensor, gclmemMatrixBDesc);
        ocl_set_desc(matrixCTensor, gclmemMatrixCDesc);
#endif
    } else {
        CHECK_STATUS(matmul_infer_output_size_cpu(
            matrixADesc, transposeA, matrixBDesc, transposeB, &matrixCDesc));
    }
    matrixCTensor->resize(matrixCDesc);
    return SUCCESS;
}

EE matmul_infer_forward_algorithm(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    Tensor matrixCTensor,
    ArchInfo_t archInfo)
{
#ifdef _USE_MALI
    if (IS_MALI_GPU(archInfo->arch)) {
        TensorDesc matrixADesc = matrixATensor.get_desc();
        TensorDesc matrixBDesc = matrixBTensor.get_desc();
        TensorDesc matrixCDesc = matrixCTensor.get_desc();
        GCLMemDesc gclmemMatrixADesc = ocl_get_desc(matrixATensor);
        GCLMemDesc gclmemMatrixBDesc = ocl_get_desc(matrixBTensor);
        GCLMemDesc gclmemMatrixCDesc = ocl_get_desc(matrixCTensor);
        CHECK_STATUS(matmul_infer_forward_algorithm_mali(((MaliPara_t)(archInfo->archPara))->handle,
            matrixADesc, transposeA, matrixBDesc, transposeB, matrixCDesc, gclmemMatrixADesc,
            gclmemMatrixBDesc, gclmemMatrixCDesc,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo));
    } else {
#endif
        return NOT_SUPPORTED;
#ifdef _USE_MALI
    }
#endif
    return SUCCESS;
}

EE matmul_infer_forward_tmp_bytes(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    TensorDesc matrixADesc = matrixATensor.get_desc();
    TensorDesc matrixBDesc = matrixBTensor.get_desc();

    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
#ifdef _USE_MALI
    if (IS_MALI_GPU(archInfo->arch)) {
        CHECK_STATUS(matmul_infer_forward_tmp_bytes_mali(matrixADesc, transposeA, matrixBDesc,
            transposeB, bytes, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo));
        return SUCCESS;
    }
#endif
    bool quantA = false;
    bool quantB = false;
    if (DT_I8 == matrixADesc.dt || DT_I8 == matrixBDesc.dt) {
        if (DT_F16 == matrixADesc.dt) {
            quantA = true;
            matrixADesc.dt = DT_I8;
        }

        if (DT_F16 == matrixBDesc.dt) {
            quantB = true;
            matrixBDesc.dt = DT_I8;
        }
    }

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
    if (matrixADesc.dims[1 - kDimA] == 1 || matrixBDesc.dims[1 - kDimB] == 1) {
        TensorDesc matrixDesc, vectorDesc;
        if (matrixADesc.dims[1 - kDimA] == 1) {
            matrixDesc =
                tensor2df(matrixBDesc.dt, dataFormatB, matrixBDesc.dims[1], matrixBDesc.dims[0]);
            vectorDesc = tensor1d(matrixADesc.dt, matrixADesc.dims[kDimA]);
        } else {
            matrixDesc =
                tensor2df(matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
            vectorDesc = tensor1d(matrixBDesc.dt, matrixBDesc.dims[kDimB]);
        }
        ret = matrix_vector_multiply_tmp_bytes(matrixDesc, vectorDesc, bytes, archInfo->arch);
    } else {
        TensorDesc matrixA2DDesc =
            tensor2df(matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
        TensorDesc matrixB2Ddesc =
            tensor2df(matrixBDesc.dt, dataFormatB, matrixBDesc.dims[1], matrixBDesc.dims[0]);
        ret = matrix_matrix_multiply_tmp_bytes(matrixA2DDesc, matrixB2Ddesc, bytes, archInfo->arch);
    }

    if (quantA) {
        *bytes += tensorNumBytes(matrixADesc);
    }
    if (quantB) {
        *bytes += tensorNumBytes(matrixBDesc);
    }
    return ret;
}

EE matmul(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    Tensor tmpTensor,
    Tensor matrixCTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    U32 tmpBytes = tmpTensor.bytes();
    void *tmp = get_ptr_from_tensor(tmpTensor, arch);
    TensorDesc matrixADesc = matrixATensor.get_desc();
    void *matrixA = get_ptr_from_tensor(matrixATensor, arch);
    TensorDesc matrixBDesc = matrixBTensor.get_desc();
    void *matrixB = get_ptr_from_tensor(matrixBTensor, arch);
    TensorDesc matrixCDesc = matrixCTensor.get_desc();
    void *matrixC = get_ptr_from_tensor(matrixCTensor, arch);

    if (matrixA == nullptr || matrixB == nullptr || matrixC == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
#ifdef _USE_MALI
    if (IS_MALI_GPU(arch)) {
        CHECK_STATUS(matmul_mali(((MaliPara_t)(archInfo->archPara))->handle, matrixADesc,
            transposeA, (GCLMem_t)matrixA, matrixBDesc, transposeB, (GCLMem_t)matrixB, (GCLMem_t)tmp,
            matrixCDesc, (GCLMem_t)matrixC, ((MaliPara_t)(archInfo->archPara))->forwardRunInfo));
        return SUCCESS;
    }
#endif

#ifdef _USE_INT8
    F32 scaleO = 1;
    if (DT_I8 == matrixADesc.dt || DT_I8 == matrixBDesc.dt) {
        if (DT_F16 == matrixADesc.dt) {
            F16 *inD = (F16 *)matrixA;
            INT8 *inQ = (INT8 *)tmp;
            F16 scale = matrixATensor.get_scale();
            quantize_tensor(matrixADesc, inD, &matrixADesc, inQ, &scale);
            scaleO *= scale;
            matrixA = (U8 *)tmp;
            tmp = (U8 *)tmp + tensorNumBytes(matrixADesc);
        } else {
            scaleO *= matrixATensor.get_scale();
        }
        if (DT_F16 == matrixBDesc.dt) {
            F16 *inD = (F16 *)matrixB;
            INT8 *inQ = (INT8 *)tmp;
            F16 scale = matrixBTensor.get_scale();
            quantize_tensor(matrixBDesc, inD, &matrixBDesc, inQ, &scale);
            scaleO *= scale;
            matrixB = (U8 *)tmp;
            tmp = (U8 *)tmp + tensorNumBytes(matrixBDesc);
        } else {
            scaleO *= matrixBTensor.get_scale();
        }
        matrixCDesc.dt = DT_I32;
        matrixC = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(matrixCDesc);
    }
#endif

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
    U8 *matrixAPtr = (U8 *)matrixA;
    U8 *matrixBPtr = (U8 *)matrixB;
    U8 *matrixCPtr = (U8 *)matrixC;
    memset(matrixC, 0, tensorNumBytes(matrixCDesc));
    for (U32 i = 0; i < loops; i++) {
        if (matrixADesc.dims[1 - kDimA] == 1) {
            TensorDesc matrixA1DDesc = tensor1d(matrixADesc.dt, matrixADesc.dims[kDimA]);
            TensorDesc matrixB2DDesc = tensor2df(matrixBDesc.dt,
                transposeB ? DF_NORMAL : DF_TRANSPOSE, matrixBDesc.dims[1], matrixBDesc.dims[0]);
            TensorDesc matrixC1DDesc = tensor1d(matrixCDesc.dt, matrixCDesc.dims[0]);
            CHECK_STATUS(matrix_vector_multiply(matrixB2DDesc, matrixBPtr, matrixA1DDesc,
                matrixAPtr, tmpBytes, tmp, matrixC1DDesc, matrixCPtr, archInfo->arch));
        } else {
            if (matrixBDesc.dims[1 - kDimB] == 1) {
                TensorDesc matrixA2DDesc;
                if (transposeA) {
                    matrixA2DDesc = tensor2df(
                        matrixADesc.dt, DF_TRANSPOSE, matrixADesc.dims[1], matrixADesc.dims[0]);
                } else {
                    matrixA2DDesc = tensor2df(
                        matrixADesc.dt, DF_NORMAL, matrixADesc.dims[1], matrixADesc.dims[0]);
                }
                TensorDesc matrixB1DDesc = tensor1d(matrixBDesc.dt, matrixBDesc.dims[kDimB]);
                TensorDesc matrixC1DDesc = tensor1d(matrixCDesc.dt, matrixCDesc.dims[1]);
                CHECK_STATUS(matrix_vector_multiply(matrixA2DDesc, matrixAPtr, matrixB1DDesc,
                    matrixBPtr, tmpBytes, tmp, matrixC1DDesc, matrixCPtr, archInfo->arch));
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
                TensorDesc matrixA2DDesc = tensor2df(
                    matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
                TensorDesc matrixB2DDesc = tensor2df(
                    matrixBDesc.dt, dataFormatB, matrixBDesc.dims[1], matrixBDesc.dims[0]);
                TensorDesc matrixC2DDesc =
                    tensor2df(matrixCDesc.dt, DF_NORMAL, matrixCDesc.dims[1], matrixCDesc.dims[0]);
                CHECK_STATUS(matrix_matrix_multiply(matrixA2DDesc, matrixAPtr, matrixB2DDesc,
                    matrixBPtr, tmpBytes, tmp, matrixC2DDesc, matrixCPtr, archInfo->arch));
            }
        }
        matrixAPtr += matrixA2DBytes;
        matrixBPtr += matrixB2DBytes;
        matrixCPtr += matrixC2DBytes;
    }
#ifdef _USE_INT8
    if (DT_I8 == matrixADesc.dt || DT_I8 == matrixBDesc.dt) {
        if (DT_I8 == matrixCTensor.get_desc().dt) {
            CHECK_STATUS(quantize_tensor(matrixCDesc, matrixC, &matrixCDesc,
                get_ptr_from_tensor(matrixCTensor, arch), &scaleO));
            matrixCTensor.set_scale(scaleO);
        } else {
            CHECK_REQUIREMENT(DT_F16 == matrixCTensor.get_desc().dt);
            F16 *output = (F16 *)get_ptr_from_tensor(matrixCTensor, arch);
            dequantize_int32_to_fp16(tensorNumElements(matrixCDesc), (I32 *)matrixC, scaleO, output);
        }
    }
#endif
    return SUCCESS;
}
