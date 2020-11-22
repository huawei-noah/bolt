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
#include "types.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/matmul_mali_fp16.h"
inline void matmul_produce_algos_paras(bool transposeA,
    TensorDesc matrixADesc,
    bool transposeB,
    TensorDesc matrixBDesc,
    std::vector<ConvolutionForwardAlgorithm> *matmulAlgorithms,
    std::vector<U32> *vecW,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    U32 configInfo[3][192];
    U32 configNum = 0;
    if (matmulAlgorithms) {
        (*matmulAlgorithms).push_back(CONVOLUTION_ALGORITHM_GEMM);
    }
    if (transposeA && !transposeB) {  //TN
        for (U32 i = 1; i <= 8; ++i) {
            for (U32 j = 1; j <= 8; ++j) {
                if (i * j <= 2) {
                    continue;
                }
                configInfo[0][configNum] = j;  // w
                configInfo[1][configNum] = 1;  // c
                configInfo[2][configNum] = i;  // k
                configNum++;
            }
        }
    } else if (!transposeA && transposeB) {
        for (U32 ii = 1; ii <= 2; ++ii) {
            for (U32 i = 1; i <= 8; ++i) {
                for (U32 j = 1; j <= 8; ++j) {
                    if (i * j <= 2) {
                        continue;
                    }
                    if (i == 5 && j > 6 && ii == 2) {
                        continue;
                    }
                    if (i == 6) {
                        if ((j > 7 && ii == 1) || (j > 5 && ii == 2)) {
                            continue;
                        }
                    }
                    if (i == 7) {
                        if ((j > 6 && ii == 1) || (j > 4 && ii == 2)) {
                            continue;
                        }
                    }
                    if (i == 8) {
                        if ((j > 5 && ii == 1) || (j > 4 && ii == 2)) {
                            continue;
                        }
                    }
                    configInfo[0][configNum] = j;       // w
                    configInfo[1][configNum] = 2 * ii;  // c
                    configInfo[2][configNum] = i;       // k
                    configNum++;
                }
            }
        }
    } else if (transposeA && transposeB) {
        for (U32 ii = 1; ii <= 2; ++ii) {
            for (U32 i = 1; i <= 8; ++i) {
                if (i <= 2) {
                    continue;
                }
                configInfo[0][configNum] = i;       // w
                configInfo[1][configNum] = 2 * ii;  // c
                configInfo[2][configNum] = 1;       // k
                configNum++;
            }
        }
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    for (U32 i = 0; i < configNum; i++) {
        if (vecW) {
            (*vecW).push_back(configInfo[0][i]);
        }
        if (vecC) {
            (*vecC).push_back(configInfo[1][i]);
        }
        if (vecK) {
            (*vecK).push_back(configInfo[2][i]);
        }
    }
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
    if ((transposeA && !transposeB) || (!transposeA && transposeB)) {
        if (matrixADesc.df != matrixBDesc.df || matrixADesc.df != matrixCDesc.df ||
            matrixADesc.df != DF_NCHW) {
            return NOT_SUPPORTED;
        }
        if (matrixA->desc.memFormat != DF_NCHW || matrixB->desc.memFormat != DF_NCHW ||
            matrixC->desc.memFormat != DF_NCHW) {
            return NOT_SUPPORTED;
        }
    }
    if (!transposeA && !transposeB) {
        return NOT_SUPPORTED;
    }
    if (matrixA->desc.stride[2] != matrixB->desc.stride[2]) {
        return NOT_MATCH;
    }
    if (matrixA->desc.offset[0] != 0 || matrixA->desc.offset[1] != 0) {
        return NOT_SUPPORTED;
    }
    if (matrixB->desc.offset[0] != 0 || matrixB->desc.offset[1] != 0) {
        return NOT_SUPPORTED;
    }
    if (matrixC->desc.offset[0] != 0 || matrixC->desc.offset[1] != 0) {
        return NOT_SUPPORTED;
    }
    return SUCCESS;
}

EE matmul_infer_output_size_mali(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc *matrixCDesc,
    GCLMemDesc_t gclmemMatrixADesc,
    GCLMemDesc_t gclmemMatrixBDesc,
    GCLMemDesc_t gclmemMatrixCDesc)
{
    U32 adims = matrixADesc.nDims;
    U32 bdims = matrixBDesc.nDims;
    DataType adt = matrixADesc.dt;
    DataType bdt = matrixBDesc.dt;
    if (adims < 2 || bdims < 2) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (adt != bdt) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 ac = (adims > 2) ? matrixADesc.dims[2] : 1;
    U32 ah = matrixADesc.dims[1];
    U32 aw = matrixADesc.dims[0];
    U32 bc = (bdims > 2) ? matrixBDesc.dims[2] : 1;
    U32 bh = matrixBDesc.dims[1];
    U32 bw = matrixBDesc.dims[0];
    bool need_pad_a = false;
    bool need_pad_b = false;
    if (ac != bc) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    std::vector<U32> vecW;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    matmul_produce_algos_paras(
        transposeA, matrixADesc, transposeB, matrixBDesc, NULL, &vecW, NULL, &vecK);

    if (transposeA && !transposeB) {
        /*TN*/
        if (ah != bh) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (matrixCDesc) {
            *matrixCDesc = matrixADesc;
            (*matrixCDesc).dims[0] = bw;
            (*matrixCDesc).dims[1] = aw;
        }
        U32 aw_align = aw;
        U32 bw_align = bw;
        for (auto item_k : vecK) {
            U32 i = ALIGN(aw, item_k);
            aw_align = (aw_align < i) ? i : aw_align;
        }
        for (auto item_w : vecW) {
            U32 i = ALIGN(bw, item_w);
            bw_align = (bw_align < i) ? i : bw_align;
        }
        if (aw_align != aw) {
            need_pad_a = true;
        }
        if (bw_align != bw) {
            need_pad_b = true;
        }
        CHECK_STATUS(infer_gclmem_desc_nchw(aw_align, ah, ac, 0, 0, bw_align, aw_align, ac, adt,
            adt, gclmemMatrixADesc, gclmemMatrixCDesc, need_pad_a));
        CHECK_STATUS(infer_gclmem_desc_nchw(
            bw_align, bh, bc, 0, 0, 0, 0, 0, adt, adt, gclmemMatrixBDesc, NULL, need_pad_b));
        return SUCCESS;
    }
    if (!transposeA && transposeB) {
        /*NT*/
        if (aw != bw) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (matrixCDesc) {
            *matrixCDesc = matrixADesc;
            (*matrixCDesc).dims[0] = bh;
            (*matrixCDesc).dims[1] = ah;
        }
        U32 ah_align = ah;
        U32 bh_align = bh;
        U32 aw_align = aw;
        for (auto item_k : vecK) {
            U32 i = ALIGN(ah, item_k);
            ah_align = (ah_align < i) ? i : ah_align;
        }
        for (auto item_w : vecW) {
            U32 i = ALIGN(bh, item_w);
            bh_align = (bh_align < i) ? i : bh_align;
        }
        for (auto item_c : vecC) {
            U32 i = ALIGN(aw, item_c);
            aw_align = (aw_align < i) ? i : aw_align;
        }
        if (aw_align != aw || ah_align != ah) {
            need_pad_a = true;
        }
        if (aw_align != aw || bh_align != bh) {
            need_pad_b = true;
        }
        CHECK_STATUS(infer_gclmem_desc_nchw(aw_align, ah_align, ac, 0, 0, bh_align, ah_align, ac,
            adt, adt, gclmemMatrixADesc, gclmemMatrixCDesc, need_pad_a));
        CHECK_STATUS(infer_gclmem_desc_nchw(
            aw_align, bh_align, bc, 0, 0, 0, 0, 0, adt, adt, gclmemMatrixBDesc, NULL, need_pad_b));
        return SUCCESS;
    }

    if (transposeA && transposeB) {
        /*TT*/
        if (ah != bw) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (matrixCDesc) {
            *matrixCDesc = matrixADesc;
            (*matrixCDesc).dims[0] = bh;
            (*matrixCDesc).dims[1] = aw;
        }
        U32 aw_align = aw;
        U32 ah_align = ah;
        U32 bh_align = bh;
        for (auto item_k : vecK) {
            U32 i = ALIGN(aw, item_k);
            aw_align = (aw_align < i) ? i : ah_align;
        }
        for (auto item_c : vecC) {
            U32 i = ALIGN(ah, item_c);
            ah_align = (ah_align < i) ? i : ah_align;
        }
        for (auto item_w : vecW) {
            U32 i = ALIGN(bh, item_w);
            bh_align = (bh_align < i) ? i : bh_align;
        }
        if (aw_align != aw || ah_align != ah) {
            need_pad_a = true;
        }
        if (ah_align != ah || bh_align != bh) {
            need_pad_b = true;
        }
        if (matrixADesc.df == DF_MKT) {
            U32 m, k, t;
            U32 gw, gh, gc;
            get_nlp_mkt_val(matrixADesc, NULL, &m, &k, &t);
            if (t != 1) {
                CHECK_STATUS(NOT_SUPPORTED);
            }
            if (ah != k) {
                CHECK_STATUS(NOT_MATCH);
            }
            if (aw != t) {
                CHECK_STATUS(NOT_MATCH)
                t = aw_align;
            }
            k = ah_align;
            map_nlp_mkt_to_ncwhc4(m, k, t, &gw, &gh, &gc);
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(
                gw, gh, gc * 4, 0, 0, 0, 0, 0, adt, adt, gclmemMatrixADesc, NULL, need_pad_a));
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        CHECK_STATUS(infer_gclmem_desc_nchw(
            ah_align, bh_align, bc, 0, 0, 0, 0, 0, adt, adt, gclmemMatrixBDesc, NULL, need_pad_b));
        CHECK_STATUS(infer_gclmem_desc_nchw(
            0, 0, 0, 0, 0, bh_align, aw_align, ac, adt, adt, NULL, gclmemMatrixCDesc));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE matmul_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    TensorDesc matrixCDesc,
    ForwardRunInfoMali_t forwardRunInfo)
{
    if (forwardRunInfo == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if (algorithm != CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }
    std::vector<ConvolutionForwardAlgorithm> matmulAlgorithms;
    std::vector<U32> vecW;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    matmul_produce_algos_paras(
        transposeA, matrixADesc, transposeB, matrixBDesc, &matmulAlgorithms, &vecW, &vecC, &vecK);
    if (vecW.size() == 1) {
        forwardRunInfo->best_w[0] = vecW[0];
        forwardRunInfo->best_k[0] = vecK[0];
        forwardRunInfo->best_c[0] = vecC[0];
        forwardRunInfo->algorithm = matmulAlgorithms[0];
        return SUCCESS;
    }

    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    GCLMem_t matrixA = gcl_create_gclmem();
    GCLMem_t matrixB = gcl_create_gclmem();
    GCLMem_t matrixC = gcl_create_gclmem();
    GCLMem_t tmpbuf = gcl_create_gclmem();
    std::vector<ForwardRunInfoMali> runInfos;
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};
    GCLMemDesc matrixAMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
    GCLMemDesc matrixBMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
    GCLMemDesc matrixCMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
    U32 bytes;
    U32 maxBytes = 0;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = matmulAlgorithms[0];
    CHECK_STATUS(matmul_infer_output_size_mali(matrixADesc, transposeA, matrixBDesc, transposeB,
        NULL, &matrixAMemDesc, &matrixBMemDesc, &matrixCMemDesc));

    for (U32 i = 0; i < vecW.size(); i++) {
        runInfo.best_w[0] = vecW[i];
        runInfo.best_c[0] = vecC[i];
        runInfo.best_k[0] = vecK[i];
        if (matmul_infer_forward_tmp_bytes_mali(
                matrixADesc, transposeA, matrixBDesc, transposeB, &bytes, &runInfo) != SUCCESS) {
            continue;
        }
        maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
        runInfos.push_back(runInfo);
    }

    U32 algosNum = runInfos.size();
    if (algosNum == 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    matrixA->desc = matrixAMemDesc;
    matrixB->desc = matrixBMemDesc;
    matrixC->desc = matrixCMemDesc;
    tmpbuf->desc.byteSize = maxBytes;
    gcl_create_memory(handle, matrixA);
    gcl_create_memory(handle, matrixB);
    gcl_create_memory(handle, matrixC);
    if (maxBytes) {
        gcl_create_memory(handle, tmpbuf);
    }

    U32 runKernelBe = 0;
    U32 runKernelEnd = 0;
    double minTime = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    for (U32 i = 0; i < algosNum; i++) {
        if (matmul_mali(handle, matrixADesc, transposeA, matrixA, matrixBDesc, transposeB, matrixB,
                tmpbuf, matrixCDesc, matrixC, &runInfos[i]) == SUCCESS) {
            runKernelEnd = handle->kernelVec->size();
            gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
            runKernelBe = runKernelEnd;
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
    CHECK_STATUS(gcl_finish(handle));
    gcl_destroy_gclmem(matrixA);
    gcl_destroy_gclmem(matrixB);
    gcl_destroy_gclmem(matrixC);
    gcl_destroy_gclmem(tmpbuf);
    runInfos.clear();
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}

EE matmul_infer_forward_tmp_bytes_mali(TensorDesc matrixADesc,
    bool transposeA,
    TensorDesc matrixBDesc,
    bool transposeB,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (matrixADesc.dt) {
        case DT_F16: {
            ret = matmul_infer_forward_tmp_bytes_mali_fp16(
                matrixADesc, transposeA, matrixBDesc, transposeB, bytes, forwardRunInfo);
            break;
        }
        case DT_I8: {
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE matmul_mali(GCLHandle_t handle,
    TensorDesc matrixADesc,
    bool transposeA,
    const GCLMem_t matrixA,
    TensorDesc matrixBDesc,
    bool transposeB,
    const GCLMem_t matrixB,
    GCLMem_t tmp,
    TensorDesc matrixCDesc,
    GCLMem_t matrixC,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    ret = matmul_checkpara_mali(handle, matrixADesc, transposeA, matrixA, matrixBDesc, transposeB,
        matrixB, matrixCDesc, matrixC);
    switch (matrixADesc.dt) {
        case DT_F16: {
            ret = matmul_mali_fp16(handle, matrixADesc, transposeA, matrixA, matrixBDesc,
                transposeB, matrixB, tmp, matrixCDesc, matrixC, forwardRunInfo);
            break;
        }
        case DT_I8: {
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
