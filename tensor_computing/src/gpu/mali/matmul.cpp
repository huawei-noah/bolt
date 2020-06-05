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
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/matmul_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"
inline EE matmul_checkpara_mali(GCLHandle_t    handle,
                                TensorDesc     matrixADesc, 
                                bool           transposeA,
                                const GCLMem_t matrixA,
                                TensorDesc     matrixBDesc, 
                                bool           transposeB,
                                const GCLMem_t matrixB,
                                TensorDesc     matrixCDesc,
                                GCLMem_t       matrixC) {
    if(nullptr == handle || nullptr == matrixA || nullptr == matrixB || nullptr == matrixC) return NULL_POINTER;
    if(matrixADesc.df != matrixBDesc.df || matrixADesc.df != matrixCDesc.df || matrixADesc.df != DF_NCHW) return NOT_SUPPORTED;
    if(matrixA->desc.memFormat != DF_NCHW || matrixB->desc.memFormat != DF_NCHW || matrixC->desc.memFormat != DF_NCHW) return NOT_SUPPORTED;
    if(transposeA && transposeB)   return NOT_SUPPORTED;
    if(!transposeA && !transposeB) return NOT_SUPPORTED;
    if(matrixA->desc.stride[2] != matrixB->desc.stride[2]) return NOT_MATCH;
    if(matrixA->desc.offset[0] != 0 || matrixA->desc.offset[1] != 0) return NOT_SUPPORTED;
    if(matrixB->desc.offset[0] != 0 || matrixB->desc.offset[1] != 0) return NOT_SUPPORTED;
    if(matrixC->desc.offset[0] != 0 || matrixC->desc.offset[1] != 0) return NOT_SUPPORTED;
    return SUCCESS;
}

EE matmul_infer_output_size_mali(TensorDesc           matrixADesc,
                                 bool                 transposeA,
                                 TensorDesc           matrixBDesc,
                                 bool                 transposeB,
                                 TensorDesc*          matrixCDesc,
                                 GCLMemDesc_t         gclmemMatrixADesc,
                                 GCLMemDesc_t         gclmemMatrixBDesc,
                                 GCLMemDesc_t         gclmemMatrixCDesc,
                                 ForwardRunInfoMali_t forwardRunInfo) {
    U32 adims = matrixADesc.nDims;
    U32 bdims = matrixBDesc.nDims;
    DataType adt = matrixADesc.dt;
    DataType bdt = matrixBDesc.dt;
    if(adims < 2 || bdims < 2) CHECK_STATUS(NOT_MATCH);
    if(adt != bdt) CHECK_STATUS(NOT_MATCH);
    U32 ac = (adims > 2) ? matrixADesc.dims[2] : 1;
    U32 ah = matrixADesc.dims[1];
    U32 aw = matrixADesc.dims[0];
    U32 bc = (bdims > 2) ? matrixBDesc.dims[2] : 1;
    U32 bh = matrixBDesc.dims[1];
    U32 bw = matrixBDesc.dims[0];
    if(ac != bc) CHECK_STATUS(NOT_SUPPORTED);
    if(transposeA && transposeB)   CHECK_STATUS(NOT_SUPPORTED);
    if(!transposeA && !transposeB) CHECK_STATUS(NOT_SUPPORTED);
    if(transposeA && !transposeB) {
        /*TN*/   
        if(ah != bh) CHECK_STATUS(NOT_SUPPORTED);
        if(matrixCDesc) {
            *matrixCDesc = matrixADesc;
            (*matrixCDesc).dims[0] = bw;
            (*matrixCDesc).dims[1] = aw;
        }
        U32 item_w = forwardRunInfo->best_w[0];
        U32 item_k = forwardRunInfo->best_k[0];
        U32 aw_align = (aw + item_k - 1) / item_k * item_k;
        U32 bw_align = (bw + item_w - 1) / item_w * item_w;
        CHECK_STATUS(infer_gclmem_desc_nchw(aw_align, ah, ac, 0, 0, bw_align, aw_align, ac, adt, adt, gclmemMatrixADesc, gclmemMatrixCDesc));
        CHECK_STATUS(infer_gclmem_desc_nchw(bw_align, bh, bc, 0, 0, 0, 0, 0, adt, adt, gclmemMatrixBDesc, NULL));
        return SUCCESS;
    }
    if(!transposeA && transposeB) {
        /*NT*/
        if(aw != bw) CHECK_STATUS(NOT_SUPPORTED);
        if(matrixCDesc) {
            *matrixCDesc = matrixADesc;
            (*matrixCDesc).dims[0] = bh;
            (*matrixCDesc).dims[1] = ah;
        }
        U32 item_w = forwardRunInfo->best_w[0];
        U32 item_c = forwardRunInfo->best_c[0];
        U32 item_k = forwardRunInfo->best_k[0];
        U32 ah_align = (ah + item_k - 1) / item_k * item_k;
        U32 bh_align = (bh + item_w - 1) / item_w * item_w;
        U32 aw_align = (aw + item_c - 1) / item_c * item_c;
        CHECK_STATUS(infer_gclmem_desc_nchw(aw_align, ah_align, ac, 0, 0, bh_align, ah_align, ac, adt, adt, gclmemMatrixADesc, gclmemMatrixCDesc));
        CHECK_STATUS(infer_gclmem_desc_nchw(aw_align, bh_align, bc, 0, 0, 0, 0, 0, adt, adt, gclmemMatrixBDesc, NULL));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE matmul_infer_forward_algorithm_mali(GCLHandle_t          handle,
                                       TensorDesc           matrixADesc,
                                       bool                 transposeA,
                                       TensorDesc           matrixBDesc,
                                       bool                 transposeB,
                                       TensorDesc           matrixCDesc,
                                       ForwardRunInfoMali_t forwardRunInfo) {
    if(forwardRunInfo == nullptr) CHECK_STATUS(NULL_POINTER);
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if(algorithm != CONVOLUTION_ALGORITHM_NULL) return SUCCESS;
    GCLHandle_t handle_tun;
    CHECK_STATUS(gcl_create_handle_profiling(&handle_tun));
    handle_tun->binMapPtr = handle->binMapPtr;
    GCLMem_t matrixA = gcl_create_gclmem();
    GCLMem_t matrixB = gcl_create_gclmem();
    GCLMem_t matrixC = gcl_create_gclmem();
    GCLMem_t tmpbuf  = gcl_create_gclmem();
    std::vector<ForwardRunInfoMali> runInfos;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)CONVOLUTION_ALGORITHM_GEMM;
    std::vector<GCLMemDesc> matrixAMemDescs;
    std::vector<GCLMemDesc> matrixBMemDescs;
    std::vector<GCLMemDesc> matrixCMemDescs;
    U32 configInfo[3][192];
    U32 configNum = 0;
    U32 bytes;
    U32 maxBytes = 0;
    U32 maxASize = 0;
    U32 maxBSize = 0;
    U32 maxCSize = 0;
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};
    for(U32 i = 1; i <= 8; ++i) {
        for(U32 j = 1; j <= 8; ++j) {
            if(i * j <= 2) continue;
            configInfo[0][configNum] = j;//w
            configInfo[1][configNum] = 1;//c
            configInfo[2][configNum] = i;//k
            configNum++;
        }
    }

    if(!transposeA && transposeB) {
        for(U32 i = 1; i <= 8; ++i) {
            for(U32 j = 1; j <= 8; ++j) {
                if(i * j <= 2) continue;
                if(i == 6 && j > 7) continue;
                if(i == 7 && j > 6) continue;
                if(i == 8 && j > 5) continue;
                configInfo[0][configNum] = j;//w
                configInfo[1][configNum] = 2;//c
                configInfo[2][configNum] = i;//k
                configNum++;
            }
        }

        for(U32 i = 1; i <= 8; ++i) {
            for(U32 j = 1; j <= 8; ++j) {
                if(i * j <= 2) continue;
                if(i == 5 && j > 6) continue;
                if(i == 6 && j > 5) continue;
                if(i == 7 && j > 4) continue;
                if(i == 8 && j > 3) continue;
                configInfo[0][configNum] = j;//w
                configInfo[1][configNum] = 4;//c
                configInfo[2][configNum] = i;//k
                configNum++;
            }
        }
    }

    for(U32 i = 0; i < configNum; ++i) {
        GCLMemDesc matrixAMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc matrixBMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc matrixCMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        runInfo.best_w[0] = configInfo[0][i];
        runInfo.best_c[0] = configInfo[1][i];
        runInfo.best_k[0] = configInfo[2][i];
        if(matmul_infer_output_size_mali(matrixADesc, transposeA, matrixBDesc, transposeB, NULL, &matrixAMemDesc, &matrixBMemDesc, &matrixCMemDesc, &runInfo) != SUCCESS) continue;
        if(matmul_infer_forward_tmp_bytes_mali(matrixADesc, transposeA, matrixBDesc, transposeB, &bytes, &runInfo) != SUCCESS) continue;
        if(maxBytes < bytes) maxBytes= bytes;
        if(maxASize < matrixAMemDesc.byteSize) maxASize = matrixAMemDesc.byteSize;
        if(maxBSize < matrixBMemDesc.byteSize) maxBSize = matrixBMemDesc.byteSize;
        if(maxCSize < matrixCMemDesc.byteSize) maxCSize = matrixCMemDesc.byteSize;
        matrixAMemDescs.push_back(matrixAMemDesc);
        matrixBMemDescs.push_back(matrixBMemDesc);
        matrixCMemDescs.push_back(matrixCMemDesc);
        runInfos.push_back(runInfo);
    }
    U32 algosNum = runInfos.size();
    if(algosNum == 0) CHECK_STATUS(NOT_SUPPORTED);
    matrixAMemDescs[0].byteSize = maxASize;
    matrixBMemDescs[0].byteSize = maxBSize;
    matrixCMemDescs[0].byteSize = maxCSize;
    matrixA->desc = matrixAMemDescs[0];
    matrixB->desc = matrixBMemDescs[0];
    matrixC->desc = matrixCMemDescs[0];
    tmpbuf->desc.byteSize = maxBytes;
    gcl_create_memory(handle_tun, matrixA);
    gcl_create_memory(handle_tun, matrixB);
    gcl_create_memory(handle_tun, matrixC);
    if(maxBytes) gcl_create_memory(handle_tun, tmpbuf);

    U32 runKernelBe = 0;
    U32 runKernelEnd = 0;
    double minTime = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    for(U32 i = 0; i < algosNum; i++) {
        matrixA->desc = matrixAMemDescs[i];
        matrixB->desc = matrixBMemDescs[i];
        matrixC->desc = matrixCMemDescs[i];
        if(matmul_mali(handle_tun, matrixADesc, transposeA, matrixA, matrixBDesc, transposeB, matrixB, tmpbuf, 
            matrixCDesc, matrixC, &runInfos[i]) == SUCCESS) {
            runKernelEnd = handle_tun->kernelVec.size();
            gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
            runKernelBe = runKernelEnd;
            if(minTime > handle_tun->t_execute) {
                minTime = handle_tun->t_execute;
                bestRunInfo = runInfos[i];
            }
        }
    }
    if(minTime == DBL_MAX) CHECK_STATUS(NOT_SUPPORTED);
    *forwardRunInfo = bestRunInfo;
    CHECK_STATUS(gcl_finish(handle_tun));
    gcl_destroy_gclmem(matrixA);
    gcl_destroy_gclmem(matrixB);
    gcl_destroy_gclmem(matrixC);
    gcl_destroy_gclmem(tmpbuf);
    runInfos.clear();
    matrixAMemDescs.clear();
    matrixBMemDescs.clear();
    matrixCMemDescs.clear();
    gcl_destroy_handle(handle_tun);
    return SUCCESS;
}


EE matmul_infer_forward_tmp_bytes_mali(TensorDesc           matrixADesc,
                                       bool                 transposeA,
                                       TensorDesc           matrixBDesc,
                                       bool                 transposeB,
                                       U32*                 bytes,
                                       ForwardRunInfoMali_t forwardRunInfo) {
    EE ret = SUCCESS;
    switch(matrixADesc.dt) {
        case DT_F16:{
            ret = matmul_infer_forward_tmp_bytes_mali_fp16(matrixADesc, transposeA, matrixBDesc, transposeB, bytes, forwardRunInfo);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE matmul_mali(GCLHandle_t          handle,
               TensorDesc           matrixADesc, 
               bool                 transposeA,
               const GCLMem_t       matrixA,
               TensorDesc           matrixBDesc, 
               bool                 transposeB,
               const GCLMem_t       matrixB,
               GCLMem_t             tmp,
               TensorDesc           matrixCDesc,
               GCLMem_t             matrixC,
               ForwardRunInfoMali_t forwardRunInfo) {
    EE ret = SUCCESS;
    ret = matmul_checkpara_mali(handle, matrixADesc, transposeA, matrixA, matrixBDesc, transposeB, matrixB, matrixCDesc, matrixC);
    switch(matrixADesc.dt) {
        case DT_F16:{
            ret = matmul_mali_fp16(handle, matrixADesc, transposeA, matrixA, matrixBDesc, transposeB, matrixB, tmp, matrixCDesc, matrixC, forwardRunInfo);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

