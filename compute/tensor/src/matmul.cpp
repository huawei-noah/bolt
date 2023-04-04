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

#ifdef _USE_GPU
#include "gpu/mali/tensor_computing_mali.h"
#endif
#ifdef _USE_CPU
#include "cpu/tensor_computing_cpu.h"
#endif

static void align_input_desc(TensorDesc *matrixADesc, TensorDesc *matrixBDesc)
{
    if (matrixADesc->nDims > matrixBDesc->nDims) {
        for (unsigned int i = matrixBDesc->nDims; i < matrixADesc->nDims; i++) {
            matrixBDesc->dims[i] = 1;
        }
        matrixBDesc->nDims = matrixADesc->nDims;
    }
    if (matrixADesc->nDims < matrixBDesc->nDims) {
        for (unsigned int i = matrixADesc->nDims; i < matrixBDesc->nDims; i++) {
            matrixADesc->dims[i] = 1;
        }
        matrixADesc->nDims = matrixBDesc->nDims;
    }
}

EE matmul_infer_output_size_cpu(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc *matrixCDesc)
{
    if (transposeA) {
        std::swap(matrixADesc.dims[0], matrixADesc.dims[1]);
    }
    if (transposeB) {
        std::swap(matrixBDesc.dims[0], matrixBDesc.dims[1]);
    }

    if (DF_NCHWC8 == matrixADesc.df && 4 == matrixADesc.nDims) {
        CHECK_REQUIREMENT(1 == matrixADesc.dims[1] && 1 == matrixADesc.dims[0]);
    }
    if (DF_NCHWC8 == matrixBDesc.df && 4 == matrixBDesc.nDims) {
        CHECK_REQUIREMENT(1 == matrixBDesc.dims[1] && 1 == matrixBDesc.dims[0]);
    }
    if (matrixADesc.dims[0] != matrixBDesc.dims[1]) {
        CHECK_STATUS(NOT_MATCH);
    }

    // case1: A(1, 16, 24, 33) x B(1, 16, 33, 8) = C(1, 16, 24, 8)
    // case2: A(2, 16, 24, 33) x B(1, 16, 33, 8) = C(2, 16, 24, 8)
    // case3: A(2, 16, 24, 33) x B(16, 33, 8) = C(2, 16, 24, 8)
    align_input_desc(&matrixADesc, &matrixBDesc);
    int dimA = matrixADesc.nDims;
    int dimB = matrixBDesc.nDims;
    *matrixCDesc = matrixADesc;
    (*matrixCDesc).dims[0] = matrixBDesc.dims[0];
    if (dimA >= 2 && dimB >= 2 && dimA == dimB) {
        for (int i = 2; i < dimA; i++) {
            matrixCDesc->dims[i] = UNI_MAX(matrixADesc.dims[i], matrixBDesc.dims[i]);
        }
        return SUCCESS;
    }

    int i = 0;
    int j = 0;
    int k = UNI_MIN(matrixADesc.nDims, 2);
    while (i < dimA - k || j < dimB - 2) {
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
    if (i != dimA - k || j != dimB - 2) {
        CHECK_STATUS(NOT_MATCH);
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
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        OclMemory *inputAMem = (OclMemory *)matrixATensor->get_memory();
        OclMemory *inputBMem = (OclMemory *)matrixBTensor->get_memory();
        OclMemory *outputCMem = (OclMemory *)matrixCTensor->get_memory();
        ret = matmul_padding_input_mali(matrixADesc, transposeA, matrixBDesc, transposeB,
            &matrixCDesc, inputAMem, inputBMem, outputCMem);
#endif
    } else {
        ret = matmul_infer_output_size_cpu(
            matrixADesc, transposeA, matrixBDesc, transposeB, &matrixCDesc);
    }
    matrixCTensor->resize(matrixCDesc);
    return ret;
}

EE matmul_infer_forward_algorithm(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    Tensor matrixCTensor,
    ArchInfo_t archInfo)
{
    EE ret = NOT_SUPPORTED;
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        TensorDesc matrixADesc = matrixATensor.get_desc();
        TensorDesc matrixBDesc = matrixBTensor.get_desc();
        TensorDesc matrixCDesc = matrixCTensor.get_desc();
        GCLMemDesc gclmemMatrixADesc = ocl_get_desc(matrixATensor);
        GCLMemDesc gclmemMatrixBDesc = ocl_get_desc(matrixBTensor);
        GCLMemDesc gclmemMatrixCDesc = ocl_get_desc(matrixCTensor);
        ret = matmul_infer_forward_algorithm_mali(((MaliPara_t)(archInfo->archPara))->handle,
            matrixADesc, transposeA, matrixBDesc, transposeB, matrixCDesc, gclmemMatrixADesc,
            gclmemMatrixBDesc, gclmemMatrixCDesc,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#endif
    } else {
        ret = SUCCESS;
    }
    return ret;
}

inline bool useINT8Type(DataType aDt, DataType bDt, DataType cDt, I32 flag)
{
    return (DT_I8 == aDt || DT_I8 == bDt || DT_U8_Q == aDt || DT_U8_Q == bDt || DT_U8_Q == cDt ||
        DT_I8 == cDt || flag != 0);
}

EE mmm_infer_forward_tmp_bytes(U32 *bytes,
    U32 kDimA,
    U32 kDimB,
    DataFormat dataFormatA,
    DataFormat dataFormatB,
    TensorDesc matrixADesc,
    TensorDesc matrixBDesc,
    Arch arch)
{
    EE ret = NOT_SUPPORTED;
    if (matrixADesc.dims[1 - kDimA] == 1) {
        TensorDesc matrixA1DDesc = tensor1d(matrixADesc.dt, matrixADesc.dims[kDimA]);
        TensorDesc matrixB2DDesc = tensor2df(matrixBDesc.dt,
            (dataFormatB == DF_TRANSPOSE) ? DF_NORMAL : DF_TRANSPOSE, matrixBDesc.dims[1],
            matrixBDesc.dims[0]);
        ret = matrix_vector_multiply_tmp_bytes(matrixB2DDesc, matrixA1DDesc, bytes, arch);
    } else if (matrixBDesc.dims[1 - kDimB] == 1) {
        TensorDesc matrixA2DDesc =
            tensor2df(matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
        TensorDesc matrixB1DDesc = tensor1d(matrixBDesc.dt, matrixBDesc.dims[kDimB]);
        ret = matrix_vector_multiply_tmp_bytes(matrixA2DDesc, matrixB1DDesc, bytes, arch);
    } else {
        TensorDesc matrixA2DDesc =
            tensor2df(matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
        TensorDesc matrixB2Ddesc =
            tensor2df(matrixBDesc.dt, dataFormatB, matrixBDesc.dims[1], matrixBDesc.dims[0]);
        ret = matrix_matrix_multiply_tmp_bytes(matrixA2DDesc, matrixB2Ddesc, bytes, arch);
    }
    return ret;
}

EE matmul_infer_forward_tmp_bytes(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    Tensor matrixCTensor,
    U32 *bytes,
    ArchInfo_t archInfo)
{
    if (bytes == nullptr) {
        return NULL_POINTER;
    }
    TensorDesc matrixADesc = matrixATensor.get_desc();
    TensorDesc matrixBDesc = matrixBTensor.get_desc();
    TensorDesc matrixCDesc = matrixCTensor.get_desc();
    if (IS_GPU(archInfo->arch)) {
#ifdef _USE_GPU
        GCLMemDesc gclmemMatrixADesc = ocl_get_desc(matrixATensor);
        GCLMemDesc gclmemMatrixBDesc = ocl_get_desc(matrixBTensor);
        GCLMemDesc gclmemMatrixCDesc = ocl_get_desc(matrixCTensor);
        return matmul_infer_forward_tmp_bytes_mali(matrixADesc, transposeA, matrixBDesc, transposeB,
            matrixCDesc, gclmemMatrixADesc, gclmemMatrixBDesc, gclmemMatrixCDesc, bytes,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#else
        return NOT_SUPPORTED;
#endif
    }
    bool quantA = false;
    bool quantB = false;
    bool quantC = false;
#ifdef _USE_INT8
    if (useINT8Type(matrixADesc.dt, matrixBDesc.dt, matrixCDesc.dt, matrixCTensor.get_scale())) {
        DataType qAType, qBType, qCType;
        if (IS_X86(archInfo->arch)) {
            bool isMvm = ((matrixCDesc.dims[0] == 1) || (matrixCDesc.dims[1] == 1));
            if (isMvm) {
                qAType = DT_I8;
                qBType = DT_U8_Q;
            } else {
                qAType = DT_U8_Q;
                qBType = DT_I8;
            }
            qCType = DT_F32;
        } else {
            qAType = DT_I8;
            qBType = DT_I8;
            qCType = DT_I32;
        }
        if (qAType != matrixADesc.dt) {
            quantA = true;
            matrixADesc.dt = qAType;
        }
        if (qBType != matrixBDesc.dt) {
            quantB = true;
            matrixBDesc.dt = qBType;
        }
        if (qCType != matrixCDesc.dt) {
            quantC = true;
            matrixCDesc.dt = qCType;
        }
    }
#endif

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
    mmm_infer_forward_tmp_bytes(
        bytes, kDimA, kDimB, dataFormatA, dataFormatB, matrixADesc, matrixBDesc, archInfo->arch);
#ifdef _USE_OPENMP
    U32 loopsC = tensorNumElements(matrixCDesc) / (matrixCDesc.dims[1] * matrixCDesc.dims[0]);
    *bytes *= loopsC;
#endif

    if (quantA || !isSameDataFormat(matrixADesc.df, DF_NCHW)) {
        *bytes += tensorNumBytes(matrixADesc);
    }
    if (quantB || !isSameDataFormat(matrixBDesc.df, DF_NCHW)) {
        *bytes += tensorNumBytes(matrixBDesc);
    }
    if (quantC) {
        *bytes += tensorNumBytes(matrixCDesc);
    }

    return ret;
}

EE matmul(Tensor matrixATensor,
    bool transposeA,
    Tensor matrixBTensor,
    bool transposeB,
    Tensor biasTensor,
    std::vector<Tensor> tmpTensors,
    Tensor matrixCTensor,
    ArchInfo_t archInfo)
{
    auto arch = archInfo->arch;
    U32 tmpBytes = tmpTensors[0].bytes();
    void *tmp = get_ptr_from_tensor(tmpTensors[0], arch);
    TensorDesc matrixADesc = matrixATensor.get_desc();
    void *matrixA = get_ptr_from_tensor(matrixATensor, arch);
    TensorDesc matrixBDesc = matrixBTensor.get_desc();
    void *matrixB = get_ptr_from_tensor(matrixBTensor, arch);
    TensorDesc matrixCDesc = matrixCTensor.get_desc();
    void *matrixC = get_ptr_from_tensor(matrixCTensor, arch);
    F32 *scalePtr = nullptr;
    bool useINT8 =
        useINT8Type(matrixADesc.dt, matrixBDesc.dt, matrixCDesc.dt, matrixCTensor.get_scale());
    if (matrixA == nullptr || matrixB == nullptr || matrixC == nullptr) {
        return NULL_POINTER;
    }
    if (IS_GPU(arch)) {
#ifdef _USE_GPU
        void *bias = get_ptr_from_tensor(biasTensor, arch);
        TensorDesc biasDesc;
        if (bias) {
            biasDesc = biasTensor.get_desc();
        }
        std::vector<GCLMem_t> tmpVec(3, NULL);
        for (U32 i = 0; i < tmpTensors.size(); i++) {
            tmpVec[i] = (GCLMem_t)get_ptr_from_tensor(tmpTensors[i], arch);
        }
        return matmul_mali(((MaliPara_t)(archInfo->archPara))->handle, matrixADesc, transposeA,
            (GCLMem_t)matrixA, matrixBDesc, transposeB, (GCLMem_t)matrixB, biasDesc, (GCLMem_t)bias,
            tmpVec, matrixCDesc, (GCLMem_t)matrixC,
            ((MaliPara_t)(archInfo->archPara))->forwardRunInfo);
#else
        return NOT_SUPPORTED;
#endif
    }
    if (!isSameDataFormat(matrixADesc.df, DF_NCHW)) {
        TensorDesc desc = matrixADesc;
        desc.df = DF_NCHW;
        transformToNCHW(matrixADesc, matrixA, desc, tmp);
        matrixA = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(matrixADesc);
        matrixADesc.df = DF_NCHW;
    }
    if (!isSameDataFormat(matrixBDesc.df, DF_NCHW)) {
        TensorDesc desc = matrixBDesc;
        desc.df = DF_NCHW;
        transformToNCHW(matrixBDesc, matrixB, desc, tmp);
        matrixB = tmp;
        tmp = (U8 *)tmp + tensorNumBytes(matrixBDesc);
        matrixBDesc.df = DF_NCHW;
    }
    if (matrixADesc.nDims == 1) {
        matrixADesc.nDims = 2;
        matrixADesc.dims[1] = 1;
        matrixCDesc.nDims = 2;
        matrixCDesc.dims[1] = 1;
    }
#ifdef _USE_INT8
    F32 scaleO = 1;
    F32 scaleArray[2] = {-1, -1};
    if (useINT8) {
        TensorDesc qADesc = matrixADesc;
        TensorDesc qBDesc = matrixBDesc;
        TensorDesc qCDesc = matrixCDesc;

        if (IS_X86(arch)) {
            bool isMvm = ((qCDesc.dims[0] == 1) || (qCDesc.dims[1] == 1));
            if (isMvm) {
                qADesc.dt = DT_I8;
                qBDesc.dt = DT_U8_Q;
            } else {
                qADesc.dt = DT_U8_Q;
                qBDesc.dt = DT_I8;
            }
            if (matrixCDesc.dt == DT_F32) {
                scalePtr = &scaleO;
            } else if (matrixCDesc.dt == DT_U8_Q) {
                if (matrixCTensor.get_scale() > 0) {
                    scalePtr = scaleArray;
                    scalePtr[1] = matrixCTensor.get_scale();
                } else {
                    qCDesc.dt = DT_F32;
                    scalePtr = &scaleO;
                }
            }
        } else {
            qADesc.dt = DT_I8;
            qBDesc.dt = DT_I8;
            qCDesc.dt = DT_I32;
        }

        if (qADesc.dt != matrixADesc.dt) {
            F32 scale = matrixATensor.get_scale();
            CHECK_STATUS(quantize_cpu(matrixADesc, matrixA, &qADesc, tmp, &scale, arch));
            matrixADesc = qADesc;
            scaleO *= scale;
            matrixA = (U8 *)tmp;
            tmp = (U8 *)tmp + tensorNumBytes(matrixADesc);
        } else {
            scaleO *= matrixATensor.get_scale();
        }

        if (qBDesc.dt != matrixBDesc.dt) {
            F32 scale = matrixBTensor.get_scale();
            CHECK_STATUS(quantize_cpu(matrixBDesc, matrixB, &qBDesc, tmp, &scale, arch));
            matrixBDesc = qBDesc;
            scaleO *= scale;
            matrixB = (U8 *)tmp;
            tmp = (U8 *)tmp + tensorNumBytes(matrixBDesc);
        } else {
            scaleO *= matrixBTensor.get_scale();
        }

        if (qCDesc.dt != matrixCDesc.dt) {
            matrixC = tmp;
            matrixCDesc = qCDesc;
            tmp = (U8 *)tmp + tensorNumBytes(matrixCDesc);
        }

        if (matrixCDesc.dt == DT_U8_Q && matrixCTensor.get_scale() > 0) {
            scaleArray[1] = scaleArray[1] / scaleO;
        }
    }
#endif

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
    align_input_desc(&matrixADesc, &matrixBDesc);
    std::vector<U8 *> p = {(U8 *)matrixA, (U8 *)matrixB, (U8 *)matrixC, (U8 *)tmp};

#if defined(_USE_OPENMP) && defined(_USE_CPU)
#pragma omp parallel num_threads(OMP_NUM_THREADS)
#endif
    {
        if (biasTensor.bytes() > 0) {
            U8 *bias = (U8 *)get_ptr_from_tensor(biasTensor, arch);
#if defined(_USE_OPENMP)
#pragma omp for
#endif
            for (U32 i = 0; i < tensorNumBytes(matrixCDesc) / biasTensor.bytes(); i++) {
                UNI_MEMCPY((U8 *)matrixC + i * biasTensor.bytes(), bias, biasTensor.bytes());
            }
        } else {
            U32 allBytes = tensorNumBytes(matrixCDesc);
            U32 blockBytes = allBytes;
#if defined(_USE_OPENMP)
            blockBytes = 128;
#pragma omp for nowait
#endif
            for (U32 i = 0; i < allBytes; i += blockBytes) {
                UNI_MEMSET((U8 *)matrixC + i, 0, UNI_MIN(blockBytes, allBytes - i));
            }
        }
    }

    U32 mmmBytes = 0;
    // #if defined(_USE_OPENMP) && defined(_USE_CPU)
    //     CHECK_STATUS(mmm_infer_forward_tmp_bytes(&mmmBytes, kDimA, kDimB, dataFormatA, dataFormatB,
    //         matrixADesc, matrixBDesc, archInfo->arch));
    // #pragma omp parallel num_threads(OMP_NUM_THREADS)
    // #endif
    {
        U32 matrixA2DBytes = (matrixADesc.dims[1] * matrixADesc.dims[0]) * bytesOf(matrixADesc.dt);
        U32 matrixB2DBytes = (matrixBDesc.dims[1] * matrixBDesc.dims[0]) * bytesOf(matrixBDesc.dt);
        U32 matrixC2DBytes = (matrixCDesc.dims[1] * matrixCDesc.dims[0]) * bytesOf(matrixCDesc.dt);
        U32 loopsA = tensorNumElements(matrixADesc) / (matrixADesc.dims[1] * matrixADesc.dims[0]);
        U32 loopsB = tensorNumElements(matrixBDesc) / (matrixBDesc.dims[1] * matrixBDesc.dims[0]);
        U32 loopsC = tensorNumElements(matrixCDesc) / (matrixCDesc.dims[1] * matrixCDesc.dims[0]);
        // #if defined(_USE_OPENMP)
        // #pragma omp for
        // #endif
        for (U32 ic = 0; ic < loopsC; ic++) {
            U32 ia, ib;
            std::vector<U32> ADims, BDims, CDims;
            U8 *tmpPtr = p[3] + ic * mmmBytes;
            CDims = calculateLocalIndex(ic, matrixCDesc.dims + 2, matrixCDesc.nDims - 2);
            if (loopsA == loopsC) {
                ia = ic;
            } else {
                ADims = CDims;
                for (U32 i = 2; i < matrixADesc.nDims; i++) {
                    if (ADims[i - 2] >= matrixADesc.dims[i]) {
                        ADims[i - 2] = 0;
                    }
                }
                ia = calculateGlobalIndex(ADims.data(), matrixADesc.dims + 2, matrixADesc.nDims - 2);
            }
            if (loopsB == loopsC) {
                ib = ic;
            } else {
                BDims = CDims;
                for (U32 i = 2; i < matrixBDesc.nDims; i++) {
                    if (BDims[i - 2] >= matrixBDesc.dims[i]) {
                        BDims[i - 2] = 0;
                    }
                }
                ib = calculateGlobalIndex(BDims.data(), matrixBDesc.dims + 2, matrixBDesc.nDims - 2);
            }

            U8 *matrixAPtr = p[0] + ia * matrixA2DBytes;
            U8 *matrixBPtr = p[1] + ib * matrixB2DBytes;
            U8 *matrixCPtr = p[2] + ic * matrixC2DBytes;
            if (matrixADesc.dims[1 - kDimA] == 1) {
                TensorDesc matrixA1DDesc = tensor1d(matrixADesc.dt, matrixADesc.dims[kDimA]);
                TensorDesc matrixB2DDesc = tensor2df(matrixBDesc.dt,
                    (dataFormatB == DF_TRANSPOSE) ? DF_NORMAL : DF_TRANSPOSE, matrixBDesc.dims[1],
                    matrixBDesc.dims[0]);
                TensorDesc matrixC1DDesc = tensor1d(matrixCDesc.dt, matrixCDesc.dims[0]);
                CHECK_STATUS(
                    matrix_vector_multiply(matrixB2DDesc, matrixBPtr, matrixA1DDesc, matrixAPtr,
                        tmpBytes, tmpPtr, matrixC1DDesc, matrixCPtr, scalePtr, archInfo->arch));
            } else if (matrixBDesc.dims[1 - kDimB] == 1) {
                TensorDesc matrixA2DDesc = tensor2df(
                    matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
                TensorDesc matrixB1DDesc = tensor1d(matrixBDesc.dt, matrixBDesc.dims[kDimB]);
                TensorDesc matrixC1DDesc = tensor1d(matrixCDesc.dt, matrixCDesc.dims[1]);
                CHECK_STATUS(
                    matrix_vector_multiply(matrixA2DDesc, matrixAPtr, matrixB1DDesc, matrixBPtr,
                        tmpBytes, tmpPtr, matrixC1DDesc, matrixCPtr, scalePtr, archInfo->arch));
            } else {
                TensorDesc matrixA2DDesc = tensor2df(
                    matrixADesc.dt, dataFormatA, matrixADesc.dims[1], matrixADesc.dims[0]);
                TensorDesc matrixB2DDesc = tensor2df(
                    matrixBDesc.dt, dataFormatB, matrixBDesc.dims[1], matrixBDesc.dims[0]);
                TensorDesc matrixC2DDesc =
                    tensor2df(matrixCDesc.dt, DF_NORMAL, matrixCDesc.dims[1], matrixCDesc.dims[0]);
                CHECK_STATUS(
                    matrix_matrix_multiply(matrixA2DDesc, matrixAPtr, matrixB2DDesc, matrixBPtr,
                        tmpBytes, tmpPtr, matrixC2DDesc, matrixCPtr, scalePtr, archInfo->arch));
            }
        }
    }
#ifdef _USE_INT8
    if (useINT8 && (matrixCTensor.get_desc().dt != matrixCDesc.dt)) {
        if (DT_I8 == matrixCTensor.get_desc().dt || DT_U8_Q == matrixCTensor.get_desc().dt) {
            F32 scales[2] = {-1, -1};  // 0 is outputScale, 1 is computeScale
            scales[0] = matrixCTensor.get_scale();
            scales[1] = scaleO;
            TensorDesc qDesc = matrixCTensor.get_desc();
            CHECK_STATUS(quantize_cpu(matrixCDesc, matrixC, &qDesc,
                get_ptr_from_tensor(matrixCTensor, arch), scales, arch));
            matrixCTensor.set_scale(scales[0]);
        } else {
            Tensor tmpOutput, biasTensor;
            tmpOutput.resize(matrixCDesc);
            std::shared_ptr<U8> shared_data((U8 *)matrixC, [](U8 *ptr) {});
            ((CpuMemory *)(tmpOutput.get_memory()))->set_shared_ptr(shared_data);
            CHECK_STATUS(dequantize(tmpOutput, &scaleO, biasTensor, matrixCTensor, archInfo));
        }
    }
#endif
    return SUCCESS;
}
