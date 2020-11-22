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
#include "gpu/mali/fp16/rnncell_mali_fp16.h"
#include "gpu/mali/fp16/rnn_mali_fp16.h"

inline EE rnncell_checkpara_mali(GCLHandle_t handle,
    TensorDesc xDesc,
    GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    GCLMem_t bias,
    GCLMem_t state,
    RNNParamSpec rnncellDesc,
    GCLMem_t tmpBuf,
    TensorDesc hDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == currentX || nullptr == filter || nullptr == output ||
        nullptr == state || nullptr == bias || nullptr == tmpBuf) {
        return NULL_POINTER;
    }
    DataFormat df;
    DataType dt;
    U32 iB, iX;
    if (xDesc.nDims == 2) {
        CHECK_STATUS(tensor2dGet(xDesc, &dt, &df, &iB, &iX));
    }
    if (xDesc.nDims == 3) {
        if (xDesc.df != DF_MTK && xDesc.df != DF_MKT) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        U32 m, k, t;
        get_nlp_mkt_val(xDesc, &dt, &m, &k, &t);
        iB = m;
        iX = k;
        if (t != 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    if (iB != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 hDim = rnncellDesc.numOutput;
    U32 col = (rnncellDesc.numProjection > 0) ? rnncellDesc.numProjection : hDim;
    U32 filterRow, filterCol;
    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, &filterRow, &filterCol);
    if (filterCol != hDim + iX) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (filterRow != col * 4) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (hDesc.df != xDesc.df) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (hDesc.dims[0] != hDim && hDesc.dims[1] != hDim) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

EE rnncell_infer_output_size_mali(TensorDesc inputDesc,
    RNNParamSpec rnncellDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemStateDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    DataType dt;
    DataFormat df;
    U32 iB, iX;
    U32 hDim = rnncellDesc.numOutput;
    if (inputDesc.nDims == 2) {
        CHECK_STATUS(tensor2dGet(inputDesc, &dt, &df, &iB, &iX));
    } else if (inputDesc.nDims == 3) {
        if (inputDesc.df != DF_MTK && inputDesc.df != DF_MKT) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        U32 m, k, t;
        get_nlp_mkt_val(inputDesc, &dt, &m, &k, &t);
        iB = m;
        iX = k;
    } else {
        return NOT_SUPPORTED;
    }

    if (outputDesc) {
        *outputDesc = inputDesc;
        if (inputDesc.nDims == 2) {
            (*outputDesc).dims[0] = hDim;
        }
        if (inputDesc.df == DF_MTK) {
            (*outputDesc).dims[0] = hDim;
        }
        if (inputDesc.df == DF_MKT) {
            (*outputDesc).dims[1] = hDim;
        }
    }

    // U32 item_c = forwardRunInfo->best_c[0];
    // U32 iX_align = (iX + item_c - 1) / item_c * item_c;
    CHECK_STATUS(infer_gclmem_desc_ncwhc4(
        1, 1, iX, 0, 0, 1, 1, hDim, dt, dt, gclmemInputDesc, gclmemOutputDesc));
    U32 hdim = rnncellDesc.numOutput;
    U32 col = (rnncellDesc.numProjection > 0) ? rnncellDesc.numProjection : hDim;
    U32 numState = col + (hdim + 3) / 4 * 4;
    CHECK_STATUS(
        infer_gclmem_desc_nchw(1, 1, numState, 0, 0, 0, 0, 0, dt, dt, gclmemStateDesc, NULL));
    return SUCCESS;
}

EE rnncell_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc xDesc,
    TensorDesc filterDesc,
    TensorDesc biasDesc,
    RNNParamSpec rnncellDesc,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    ForwardRunInfoMali_t forwardRunInfo)
{
    if (forwardRunInfo == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if (algorithm != CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    GCLMem_t currentX = gcl_create_gclmem();
    GCLMem_t state = gcl_create_gclmem();
    GCLMem_t filter0 = gcl_create_gclmem();
    GCLMem_t filter1 = gcl_create_gclmem();
    GCLMem_t bias = gcl_create_gclmem();
    GCLMem_t tmpbuf = gcl_create_gclmem();
    GCLMem_t currentH = gcl_create_gclmem();

    std::vector<ForwardRunInfoMali> runInfos;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)CONVOLUTION_ALGORITHM_DIRECT;
    std::vector<GCLMemDesc> currentXMemDescs;
    std::vector<GCLMemDesc> stateMemDescs;
    std::vector<GCLMemDesc> currentHMemDescs;
    std::vector<GCLMemDesc> filterMemDescs;
    std::vector<GCLMemDesc> filterMemProDescs;
    U32 configInfo[3][64];
    U32 configNum = 3;
    U32 bytes = 0;
    U32 maxBytes = 0;
    U32 maxCurrentXSize = 0;
    U32 maxStateSize = 0;
    U32 maxCurrentHSize = 0;
    U32 maxFilterSize = 0;
    U32 hDim = rnncellDesc.numOutput;
    U32 col = (rnncellDesc.numProjection > 0) ? rnncellDesc.numProjection : hDim;
    U32 biasNum = col * 4;
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};
    DataType dt = xDesc.dt;
    bool useProject = (rnncellDesc.numProjection > 0) ? true : false;
    for (U32 i = 0; i < configNum; ++i) {
        configInfo[0][i] = 1;
        configInfo[1][i] = 1 << (2 + i);
        configInfo[2][i] = 0;
        configInfo[0][i + configNum] = 1;
        configInfo[1][i + configNum] = 1 << (2 + i);
        configInfo[2][i + configNum] = 0;
    }

    for (U32 i = 0; i < configNum; ++i) {
        GCLMemDesc currentXMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc stateMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCHW);
        GCLMemDesc currentHMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc filterMemDesc[2];
        filterMemDesc[0] = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        filterMemDesc[1] = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        runInfo.best_w[0] = configInfo[0][i];
        runInfo.best_c[0] = configInfo[1][i];
        runInfo.best_k[0] = configInfo[2][i];
        runInfo.best_w[1] = configInfo[0][i + configNum];
        runInfo.best_c[1] = configInfo[1][i + configNum];
        runInfo.best_k[1] = configInfo[2][i + configNum];
        if (rnncell_infer_output_size_mali(xDesc, rnncellDesc, NULL, &currentXMemDesc,
                &stateMemDesc, &currentHMemDesc) != SUCCESS) {
            continue;
        }
        if (rnn_transform_filter_bytes_mali(
                filterDesc, rnncellDesc, filterMemDesc, &bytes, &runInfo) != SUCCESS) {
            continue;
        }
        if (maxBytes < bytes) {
            maxBytes = bytes;
        }
        if (rnncell_infer_forward_tmp_bytes_mali(
                xDesc, filterDesc, hDesc, rnncellDesc, &bytes, &runInfo) != SUCCESS) {
            continue;
        }
        if (maxBytes < bytes) {
            maxBytes = bytes;
        }
        if (maxCurrentXSize < currentXMemDesc.byteSize) {
            maxCurrentXSize = currentXMemDesc.byteSize;
        }
        if (maxStateSize < stateMemDesc.byteSize) {
            maxStateSize = stateMemDesc.byteSize;
        }
        if (maxCurrentHSize < currentHMemDesc.byteSize) {
            maxCurrentHSize = currentHMemDesc.byteSize;
        }
        if (maxFilterSize < filterMemDesc[0].byteSize) {
            maxFilterSize = filterMemDesc[0].byteSize;
        }
        if (maxFilterSize < filterMemDesc[1].byteSize) {
            maxFilterSize = filterMemDesc[1].byteSize;
        }
        currentXMemDescs.push_back(currentXMemDesc);
        stateMemDescs.push_back(stateMemDesc);
        currentHMemDescs.push_back(currentHMemDesc);
        filterMemDescs.push_back(filterMemDesc[0]);
        filterMemProDescs.push_back(filterMemDesc[1]);
        runInfos.push_back(runInfo);
    }

    U32 algosNum = runInfos.size();
    if (algosNum == 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    currentXMemDescs[0].byteSize = maxCurrentXSize;
    stateMemDescs[0].byteSize = maxStateSize;
    currentHMemDescs[0].byteSize = maxCurrentHSize;
    filterMemDescs[0].byteSize = maxFilterSize;
    filterMemProDescs[0].byteSize = maxFilterSize;

    currentX->desc = currentXMemDescs[0];
    state->desc = stateMemDescs[0];
    currentH->desc = currentHMemDescs[0];
    filter0->desc = filterMemDescs[0];
    filter1->desc = filterMemProDescs[0];
    bias->desc.stride[0] = biasNum;
    bias->desc.stride[1] = 1;
    bias->desc.stride[2] = 1;
    bias->desc.offset[0] = 0;
    bias->desc.offset[1] = 0;
    bias->desc.offset[2] = 0;
    bias->desc.num = biasNum;
    bias->desc.memFormat = DF_NHWC;
    bias->desc.byteSize = biasNum * bytesOf(dt);
    bias->desc.memType = GCL_MEM_BUF;
    tmpbuf->desc.byteSize = maxBytes;
    gcl_create_memory(handle, currentX);
    gcl_create_memory(handle, state);
    gcl_create_memory(handle, currentH);
    gcl_create_memory(handle, filter0);
    gcl_create_memory(handle, filter1);
    gcl_create_memory(handle, bias);
    if (maxBytes) {
        gcl_create_memory(handle, tmpbuf);
    }

    U32 runKernelBe = 0;
    U32 runKernelEnd = 0;
    double minTime = DBL_MAX;
    double minTimePro = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    for (U32 i = 0; i < algosNum; i++) {
        currentX->desc = currentXMemDescs[i];
        state->desc = currentXMemDescs[i];
        currentH->desc = currentHMemDescs[i];
        filter0->desc = filterMemDescs[i];
        filter1->desc = filterMemProDescs[i];
        GCLMem filter[2];
        filter[0] = *filter0;
        filter[1] = *filter1;

        runKernelBe = handle->kernelVec->size() + 1;
        runKernelEnd = handle->kernelVec->size() + 2;
        if (rnncell_mali(handle, xDesc, currentX, filterDesc, filter, biasDesc, bias, state,
                rnncellDesc, batchStrideX, batchStrideH, maxBytes, tmpbuf, hDesc, currentH,
                &runInfos[i]) == SUCCESS) {
            gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
            if (minTime > handle->t_execute) {
                minTime = handle->t_execute;
                bestRunInfo.algorithm = runInfos[i].algorithm;
                bestRunInfo.best_w[0] = runInfos[i].best_w[0];
                bestRunInfo.best_c[0] = runInfos[i].best_c[0];
                bestRunInfo.best_k[0] = runInfos[i].best_k[0];
            }
            if (useProject) {
                runKernelBe += 2;
                runKernelEnd += 2;
                gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
                if (minTimePro > handle->t_execute) {
                    minTime = handle->t_execute;
                    bestRunInfo.algorithm = runInfos[i].algorithm;
                    bestRunInfo.best_w[1] = runInfos[i].best_w[1];
                    bestRunInfo.best_c[1] = runInfos[i].best_c[1];
                    bestRunInfo.best_k[1] = runInfos[i].best_k[1];
                }
            }
        }
    }
    if (minTime == DBL_MAX) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (useProject && minTimePro == DBL_MAX) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    *forwardRunInfo = bestRunInfo;
    CHECK_STATUS(gcl_finish(handle));
    gcl_destroy_gclmem(currentX);
    gcl_destroy_gclmem(state);
    gcl_destroy_gclmem(currentH);
    gcl_destroy_gclmem(filter0);
    gcl_destroy_gclmem(filter1);
    gcl_destroy_gclmem(bias);
    runInfos.clear();
    currentXMemDescs.clear();
    stateMemDescs.clear();
    currentHMemDescs.clear();
    filterMemDescs.clear();
    filterMemProDescs.clear();
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}

EE rnn_transform_filter_bytes_mali(TensorDesc filterDesc,
    RNNParamSpec rnnParamSpec,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = rnn_transform_filter_bytes_mali_fp16(
                filterDesc, rnnParamSpec, gclmemFilterDesc, bytes, forwardRunInfo);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE rnn_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    RNNParamSpec rnnParamSpec,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = rnn_transform_filter_mali_fp16(
                handle, filterDesc, filter, rnnParamSpec, fltmemDesc, fltmem, forwardRunInfo);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE rnncell_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnncellDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = rnncell_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, rnncellDesc, bytes, forwardRunInfo);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE rnncell_mali(GCLHandle_t handle,
    TensorDesc xDesc,
    const GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    GCLMem_t state,
    RNNParamSpec rnncellDesc,
    U32 batchStrideX,
    U32 batchStrideH,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc hDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    ret = rnncell_checkpara_mali(handle, xDesc, currentX, filterDesc, filter, bias, state,
        rnncellDesc, tmpBuf, hDesc, output);
    switch (xDesc.dt) {
        case DT_F16: {
            ret = rnncell_mali_fp16(handle, xDesc, currentX, filterDesc, filter, biasDesc, bias,
                state, tmpBytes, tmpBuf, rnncellDesc, batchStrideX, batchStrideH, hDesc, output,
                forwardRunInfo);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
