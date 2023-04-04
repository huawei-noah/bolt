// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "sys.h"

#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/matmul_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"
inline void matmul_produce_algos_paras(bool transposeA,
    TensorDesc matrixADesc,
    bool transposeB,
    TensorDesc matrixBDesc,
    std::vector<ConvolutionForwardAlgorithm> *matmulAlgorithms,
    std::vector<U32> *vecH,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    if (matmulAlgorithms) {
        (*matmulAlgorithms).push_back(CONVOLUTION_ALGORITHM_GEMM);
    }
    bool useImg = check_qualcomm_device();
    GCLMemType mt = (useImg) ? GCL_MEM_IMG_3D : GCL_MEM_BUF;
    CHECK_STATUS(get_gemm_tn_cal_scheme(vecH, vecC, vecK, mt, mt, GCL_MEM_BUF));
}

inline EE matmul_checkpara_mali(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    const GCLMem_t matrixA,
    TensorDesc matrixBDesc,
    bool transposeB,
    const GCLMem_t matrixB,
    TensorDesc matrixCDesc,
    GCLMem_t matrixC)
{
    if (nullptr == handle || nullptr == matrixA || nullptr == matrixB || nullptr == matrixC) {
        return NULL_POINTER;
    }
    U32 ah, aw;
    U32 bh, bw;
    U32 ch, cw;
    tensorSelectGet(matrixADesc, NULL, NULL, NULL, NULL, &ah, &aw);
    tensorSelectGet(matrixBDesc, NULL, NULL, NULL, NULL, &bh, &bw);
    tensorSelectGet(matrixCDesc, NULL, NULL, NULL, NULL, &ch, &cw);
    U32 m, n, ra, rb;
    if (!transposeA) {
        m = ah;
        ra = aw;
    } else {
        m = aw;
        ra = ah;
    }
    if (!transposeB) {
        n = bw;
        rb = bh;
    } else {
        n = bh;
        rb = bw;
    }
    if (ra != rb) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (n != cw || m != ch) {
        CHECK_STATUS(NOT_MATCH);
    }

    if (matrixC->desc.memFormat != DF_NCHW) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE matmul_padding_input_mali(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc *matrixCDesc,
    OclMemory *inputAMem,
    OclMemory *inputBMem,
    OclMemory *outputCMem)
{
    if (matrixCDesc == nullptr || inputAMem == nullptr || inputBMem == nullptr ||
        outputCMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 aDims = matrixADesc.nDims;
    U32 bDims = matrixBDesc.nDims;
    DataType adt = matrixADesc.dt;
    DataType bdt = matrixBDesc.dt;
    if (adt != bdt) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 aw, ah;
    U32 bw, bh;
    U32 ch, cw;
    aw = matrixADesc.dims[0];
    ah = (aDims > 1) ? matrixADesc.dims[1] : 1;
    bw = matrixBDesc.dims[0];
    bh = (bDims > 1) ? matrixBDesc.dims[1] : 1;
    bool needReshapeA, needReshapeB;
    get_reshaped_desc(
        matrixADesc, matrixBDesc, transposeA, transposeB, &needReshapeA, &needReshapeB, NULL, NULL);
    GCLMemType amt = inputAMem->gclMemType();
    GCLMemType bmt = inputBMem->gclMemType();
    bool needProcessA = need_process_matmul_input(matrixADesc, amt, needReshapeA, transposeA, true);
    bool needProcessB = need_process_matmul_input(matrixBDesc, bmt, needReshapeB, transposeB, false);

    std::vector<ConvolutionForwardAlgorithm> matmulAlgorithms;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    if (!needProcessA || !needProcessB) {
        GCLMemType cmt = outputCMem->gclMemType();
        matmul_produce_algos_paras(transposeA, matrixADesc, transposeB, matrixBDesc,
            &matmulAlgorithms, &vecH, &vecC, &vecK);
    }

    U32 aw_align = aw;
    U32 bw_align = bw;
    U32 ar, br;
    if (transposeA) {
        if (!needProcessA) {
            for (auto item_k : vecK) {
                U32 i = UNI_ALIGN(aw, item_k);
                aw_align = (aw_align < i) ? i : aw_align;
            }
        }
        ch = aw;
        ar = ah;  //reduce axis len for matrix A
    } else {
        ch = ah;
        ar = aw;
    }

    if (!transposeB) {
        if (!needProcessB) {
            for (auto item_h : vecH) {
                U32 i = UNI_ALIGN(bw, item_h);
                bw_align = (bw_align < i) ? i : bw_align;
            }
        }
        cw = bw;
        br = bh;  //reduce axis len for matrix B
    } else {
        cw = bh;
        br = bw;
    }
    if (ar != br) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 cDims = (aDims > bDims) ? aDims : bDims;
    if (cDims < 2) {
        CHECK_STATUS(NOT_MATCH);
    }
    DataFormat cdf = getTensorDefaultDataFormat(cDims);
    TensorDesc cDesc;
    cDesc.dt = adt;
    cDesc.df = cdf;
    cDesc.nDims = cDims;
    cDesc.dims[0] = cw;
    cDesc.dims[1] = ch;
    for (U32 i = 2; i < cDims; i++) {
        U32 av = (i < aDims) ? matrixADesc.dims[i] : 1;
        U32 bv = (i < bDims) ? matrixBDesc.dims[i] : 1;
        cDesc.dims[i] = (av > bv) ? av : bv;
    }
    (*matrixCDesc) = cDesc;
    U32 pr = aw_align - aw;
    inputAMem->padding(0, pr, 0, 0);
    pr = bw_align - bw;
    inputBMem->padding(0, pr, 0, 0);
    return SUCCESS;
}

EE matmul_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc matrixCDesc,
    GCLMemDesc gclmemMatrixADesc,
    GCLMemDesc gclmemMatrixBDesc,
    GCLMemDesc gclmemMatrixCDesc,
    ForwardRunInfoMali_t forwardRunInfo)
{
    if (forwardRunInfo == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if (algorithm != CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }
    GCLMemType amt = gclmemMatrixADesc.memType;
    GCLMemType bmt = gclmemMatrixBDesc.memType;
    GCLMemType cmt = gclmemMatrixCDesc.memType;
    std::vector<I32> flag = build_matmul_forward_algorithm_flag(
        matrixADesc, transposeA, matrixBDesc, transposeB, amt, bmt, cmt);
    if (gcl_get_runInfo_from_cache(handle, flag, forwardRunInfo)) {
        return SUCCESS;
    }
    std::vector<ConvolutionForwardAlgorithm> matmulAlgorithms;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    matmul_produce_algos_paras(
        transposeA, matrixADesc, transposeB, matrixBDesc, &matmulAlgorithms, &vecH, &vecC, &vecK);

    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    GCLMem_t matrixA = gcl_create_gclmem();
    GCLMem_t matrixB = gcl_create_gclmem();
    GCLMem_t matrixC = gcl_create_gclmem();
    GCLMem_t tmpbuf = gcl_create_gclmem();
    GCLMem_t tmpImgA = gcl_create_gclmem();
    GCLMem_t tmpImgB = gcl_create_gclmem();
    std::vector<ForwardRunInfoMali> runInfos;
    U32 stride[3] = {0};
    U32 offset[3] = {0};
    U32 bytes[7] = {0};
    U32 maxBytes[7] = {0};
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = matmulAlgorithms[0];

    for (U32 i = 0; i < vecH.size(); i++) {
        runInfo.best_h[0] = vecH[i];
        runInfo.best_c[0] = vecC[i];
        runInfo.best_k[0] = vecK[i];
        if (matmul_infer_forward_tmp_bytes_mali(matrixADesc, transposeA, matrixBDesc, transposeB,
                matrixCDesc, gclmemMatrixADesc, gclmemMatrixBDesc, gclmemMatrixCDesc, bytes,
                &runInfo) != SUCCESS) {
            continue;
        }
        for (U32 i = 0; i < 7; i++) {
            maxBytes[i] = (maxBytes[i] < bytes[i]) ? bytes[i] : maxBytes[i];
        }
        runInfos.push_back(runInfo);
    }
    U32 algosNum = runInfos.size();
    if (algosNum == 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    gclmemMatrixCDesc.need_pad = false;
    matrixA->desc = gclmemMatrixADesc;
    matrixB->desc = gclmemMatrixBDesc;
    matrixC->desc = gclmemMatrixCDesc;
    gcl_create_memory(handle, matrixA);
    gcl_create_memory(handle, matrixB);
    gcl_create_memory(handle, matrixC);
    std::vector<GCLMem_t> tmp(3, NULL);
    maxBytes[0] += 1;
    tmpbuf->desc.byteSize = maxBytes[0];
    gcl_create_memory(handle, tmpbuf);
    tmp[0] = tmpbuf;
    if (maxBytes[1] > 0 && maxBytes[2] > 0 && maxBytes[3] > 0) {
        tmpImgA->desc.memType = GCL_MEM_IMG_3D;
        tmpImgA->desc.stride[0] = maxBytes[1];
        tmpImgA->desc.stride[1] = maxBytes[2];
        tmpImgA->desc.stride[2] = maxBytes[3];
        gcl_create_memory(handle, tmpImgA);
        tmp[1] = tmpImgA;
    }
    if (maxBytes[4] > 0 && maxBytes[5] > 0 && maxBytes[6] > 0) {
        tmpImgB->desc.memType = GCL_MEM_IMG_3D;
        tmpImgB->desc.stride[0] = maxBytes[4];
        tmpImgB->desc.stride[1] = maxBytes[5];
        tmpImgB->desc.stride[2] = maxBytes[6];
        gcl_create_memory(handle, tmpImgB);
        tmp[2] = tmpImgB;
    }

    double minTime = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    for (U32 i = 0; i < algosNum; i++) {
        if (matmul_mali(handle, matrixADesc, transposeA, matrixA, matrixBDesc, transposeB, matrixB,
                matrixADesc, NULL, tmp, matrixCDesc, matrixC, &runInfos[i]) == SUCCESS) {
            U32 kernelVecNum = handle->kernelVec->size();
            gcl_run_kernelVec_timing(handle, kernelVecNum - 1, kernelVecNum);
            if (minTime > handle->t_execute) {
                minTime = handle->t_execute;
                bestRunInfo = runInfos[i];
            }
        }
    }

    if (minTime == DBL_MAX) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    *forwardRunInfo = bestRunInfo;
    gcl_set_runInfo_to_cache(handle, flag, bestRunInfo);
    CHECK_STATUS(gcl_finish(handle));
    gcl_destroy_gclmem(matrixA);
    gcl_destroy_gclmem(matrixB);
    gcl_destroy_gclmem(matrixC);
    gcl_destroy_gclmem(tmpbuf);
    gcl_destroy_gclmem(tmpImgA);
    gcl_destroy_gclmem(tmpImgB);
    runInfos.clear();
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_clean_programMap(handle));
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}

EE matmul_infer_forward_tmp_bytes_mali(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc matrixCDesc,
    GCLMemDesc gclmemMatrixADesc,
    GCLMemDesc gclmemMatrixBDesc,
    GCLMemDesc gclmemMatrixCDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = NOT_SUPPORTED;
    switch (matrixADesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = matmul_infer_forward_tmp_bytes_mali_fp16(matrixADesc, transposeA, matrixBDesc,
                transposeB, matrixCDesc, gclmemMatrixADesc, gclmemMatrixBDesc, gclmemMatrixCDesc,
                bytes, forwardRunInfo);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE matmul_mali(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    GCLMem_t matrixA,
    TensorDesc matrixBDesc,
    bool transposeB,
    GCLMem_t matrixB,
    TensorDesc biasDesc,
    GCLMem_t bias,
    std::vector<GCLMem_t> tmp,
    TensorDesc matrixCDesc,
    GCLMem_t matrixC,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(matmul_checkpara_mali(handle, matrixADesc, transposeA, matrixA, matrixBDesc,
        transposeB, matrixB, matrixCDesc, matrixC));
    EE ret = NOT_SUPPORTED;
    switch (matrixADesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = matmul_mali_fp16(handle, matrixADesc, transposeA, matrixA, matrixBDesc,
                transposeB, matrixB, biasDesc, bias, tmp, matrixCDesc, matrixC, forwardRunInfo);
            break;
        }
        default:
            break;
    }
    return ret;
}
