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
#include "gpu/mali/fp16/rnncell_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/gemv_opt.h"

inline void rnncell_produce_algos_paras(RNNParamSpec rnnPara,
    std::vector<ConvolutionForwardAlgorithm> *rnncellAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecH,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK,
    std::vector<U32> *algoNumIndexP,
    std::vector<U32> *vecHP,
    std::vector<U32> *vecCP,
    std::vector<U32> *vecKP)
{
    rnncellAlgorithms->push_back(CONVOLUTION_ALGORITHM_GEMM);
    CHECK_STATUS(get_gemv_cal_scheme(vecH, vecC, vecK));
    algoNumIndex->push_back(vecH->size());
    if (rnnPara.num_projection) {
        CHECK_STATUS(get_gemv_cal_scheme(vecHP, vecCP, vecKP));
        algoNumIndexP->push_back(vecHP->size());
    }
}

inline EE rnncell_checkpara_mali(GCLHandle_t handle,
    TensorDesc xDesc,
    GCLMem_t currentX,
    TensorDesc filterDesc,
    GCLMem_t filter,
    GCLMem_t bias,
    GCLMem_t state,
    RNNParamSpec rnnPara,
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
    CHECK_STATUS(tensor2dGet(xDesc, &dt, &df, &iB, &iX));
    if (iB != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 hDim = rnnPara.num_outputs;
    if (hDesc.dims[0] != hDim && hDesc.dims[1] != hDim) {
        CHECK_STATUS(NOT_MATCH);
    }
    return SUCCESS;
}

EE rnncell_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc xDesc,
    TensorDesc filterDesc,
    TensorDesc biasDesc,
    RNNParamSpec rnnPara,
    U32 batchStrideX,
    U32 batchStrideH,
    TensorDesc hDesc,
    GCLMemDesc inputMemDesc,
    GCLMemDesc stateMemDesc,
    GCLMemDesc outputMemDesc,
    ForwardRunInfoMali_t forwardRunInfo)
{
    if (forwardRunInfo == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if (algorithm != CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }
    std::vector<TensorDesc> filterDescVec(1, filterDesc);
    std::vector<I32> flag = build_rnn_forward_algorithm_flag(xDesc, filterDescVec, rnnPara);
    if (gcl_get_runInfo_from_cache(handle, flag, forwardRunInfo)) {
        return SUCCESS;
    }
    std::vector<ConvolutionForwardAlgorithm> rnncellAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    std::vector<U32> algoNumIndexP;
    std::vector<U32> vecHP;
    std::vector<U32> vecCP;
    std::vector<U32> vecKP;
    rnncell_produce_algos_paras(rnnPara, &rnncellAlgorithms, &algoNumIndex, &vecH, &vecC, &vecK,
        &algoNumIndexP, &vecHP, &vecCP, &vecKP);

    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    GCLMem_t currentX = gcl_create_gclmem();
    GCLMem_t state = gcl_create_gclmem();
    GCLMem_t filter0 = gcl_create_gclmem();
    GCLMem_t filter1 = gcl_create_gclmem();
    GCLMem_t bias0 = gcl_create_gclmem();
    GCLMem_t bias1 = gcl_create_gclmem();
    GCLMem_t tmpbuf = gcl_create_gclmem();
    GCLMem_t currentH = gcl_create_gclmem();

    std::vector<ForwardRunInfoMali> runInfos;
    U32 bytes = 0;
    U32 maxBytes = 0;
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};
    U32 maxFilterSize[2] = {0, 0};
    TensorDesc ftmDesc[2];
    bool useProject = (rnnPara.num_projection > 0) ? true : false;
    U32 filterNum = (useProject) ? 2 : 1;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = rnncellAlgorithms[0];
    for (U32 i = 0; i < algoNumIndex[0]; ++i) {
        runInfo.best_h[0] = vecH[i];
        runInfo.best_c[0] = vecC[i];
        runInfo.best_k[0] = vecK[i];
        if (useProject) {
            runInfo.best_h[1] = vecHP[i];
            runInfo.best_c[1] = vecCP[i];
            runInfo.best_k[1] = vecKP[i];
        }
        TensorDesc desc[2];
        if (rnncell_transform_filter_bytes_mali(filterDesc, rnnPara, &runInfo, desc) != SUCCESS) {
            continue;
        }
        if (rnncell_infer_forward_tmp_bytes_mali(
                xDesc, filterDesc, hDesc, rnnPara, &bytes, &runInfo) != SUCCESS) {
            continue;
        }
        if (maxBytes < bytes) {
            maxBytes = bytes;
        }
        for (U32 i = 0; i < filterNum; i++) {
            if (tensorNumBytes(desc[i]) > maxFilterSize[i]) {
                ftmDesc[i] = desc[i];
                maxFilterSize[i] = tensorNumBytes(desc[i]);
            }
        }
        runInfos.push_back(runInfo);
    }

    U32 algosNum = runInfos.size();
    if (algosNum == 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 col = (useProject) ? rnnPara.num_projection : rnnPara.num_outputs;
    stride[0] = col * 4;
    stride[1] = 1;
    stride[2] = 1;
    DataType dt = xDesc.dt;
    CHECK_STATUS(gclmem_set_desc_padding(
        &bias0->desc, stride, offset, dt, DF_NHWC, GCL_MEM_BUF, CL_MEM_READ_WRITE));
    stride[0] = ftmDesc[0].dims[0];
    stride[1] = ftmDesc[0].dims[1];
    stride[2] = ftmDesc[0].dims[2];
    CHECK_STATUS(gclmem_set_desc_padding(
        &filter0->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));
    gcl_create_memory(handle, filter0);
    gcl_create_memory(handle, bias0);

    if (useProject) {
        stride[0] = ftmDesc[1].dims[0];
        stride[1] = ftmDesc[1].dims[1];
        stride[2] = ftmDesc[1].dims[2];
        CHECK_STATUS(gclmem_set_desc_padding(
            &filter1->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));
        stride[0] = rnnPara.num_outputs;
        CHECK_STATUS(gclmem_set_desc_padding(
            &bias1->desc, stride, offset, dt, DF_NHWC, GCL_MEM_BUF, CL_MEM_READ_WRITE));
        gcl_create_memory(handle, filter1);
        gcl_create_memory(handle, bias1);
    }

    outputMemDesc.need_pad = false;
    currentX->desc = inputMemDesc;
    state->desc = stateMemDesc;
    currentH->desc = outputMemDesc;
    tmpbuf->desc.byteSize = maxBytes;
    gcl_create_memory(handle, currentX);
    gcl_create_memory(handle, state);
    gcl_create_memory(handle, currentH);
    if (maxBytes) {
        gcl_create_memory(handle, tmpbuf);
    }

    U32 runKernelBe = 0;
    double minTime = DBL_MAX;
    double minTimePro = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    UNI_MEMSET(&bestRunInfo, 0, sizeof(ForwardRunInfoMali));
    for (U32 i = 0; i < algosNum; i++) {
        GCLMem filter[2];
        GCLMem bias[2];
        filter[0] = *filter0;
        bias[0] = *bias0;
        if (useProject) {
            filter[1] = *filter1;
            bias[1] = *bias1;
        }

        if (rnncell_mali(handle, xDesc, currentX, filterDesc, filter, biasDesc, bias, state,
                rnnPara, batchStrideX, batchStrideH, maxBytes, tmpbuf, hDesc, currentH,
                &runInfos[i]) == SUCCESS) {
            gcl_run_kernelVec_timing(handle, runKernelBe + 1, runKernelBe + 2);
            if (minTime > handle->t_execute) {
                minTime = handle->t_execute;
                bestRunInfo.algorithm = runInfos[i].algorithm;
                bestRunInfo.best_h[0] = runInfos[i].best_h[0];
                bestRunInfo.best_c[0] = runInfos[i].best_c[0];
                bestRunInfo.best_k[0] = runInfos[i].best_k[0];
            }
            if (useProject) {
                gcl_run_kernelVec_timing(handle, runKernelBe + 3, runKernelBe + 4);
                if (minTimePro > handle->t_execute) {
                    minTimePro = handle->t_execute;
                    bestRunInfo.algorithm = runInfos[i].algorithm;
                    bestRunInfo.best_h[1] = runInfos[i].best_h[1];
                    bestRunInfo.best_c[1] = runInfos[i].best_c[1];
                    bestRunInfo.best_k[1] = runInfos[i].best_k[1];
                }
            }
            runKernelBe = handle->kernelVec->size();
        }
    }
    if (minTime == DBL_MAX) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (useProject && minTimePro == DBL_MAX) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    *forwardRunInfo = bestRunInfo;
    gcl_set_runInfo_to_cache(handle, flag, bestRunInfo);
    CHECK_STATUS(gcl_finish(handle));
    gcl_destroy_gclmem(currentX);
    gcl_destroy_gclmem(state);
    gcl_destroy_gclmem(currentH);
    gcl_destroy_gclmem(filter0);
    gcl_destroy_gclmem(bias0);
    gcl_destroy_gclmem(filter1);
    gcl_destroy_gclmem(bias1);
    runInfos.clear();
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_clean_programMap(handle));
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}

EE rnncell_transform_filter_bytes_mali(TensorDesc filterDesc,
    RNNParamSpec rnnParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *ftmDesc)
{
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = rnncell_transform_filter_bytes_mali_fp16(
                filterDesc, rnnParamSpec, forwardRunInfo, ftmDesc);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE rnncell_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    RNNParamSpec rnnParamSpec,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = rnncell_transform_filter_mali_fp16(
                handle, filterDesc, filter, rnnParamSpec, fltmemDesc, fltmem, forwardRunInfo);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE rnncell_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    RNNParamSpec rnnPara,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = rnncell_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, rnnPara, bytes, forwardRunInfo);
            break;
        }
        default:
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
    RNNParamSpec rnnPara,
    U32 batchStrideX,
    U32 batchStrideH,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc hDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(rnncell_checkpara_mali(
        handle, xDesc, currentX, filterDesc, filter, bias, state, rnnPara, tmpBuf, hDesc, output));
    EE ret = NOT_SUPPORTED;
    switch (xDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = rnncell_mali_fp16(handle, xDesc, currentX, filterDesc, filter, biasDesc, bias,
                state, tmpBytes, tmpBuf, rnnPara, batchStrideX, batchStrideH, hDesc, output,
                forwardRunInfo);
            break;
        }
        default:
            break;
    }
    return ret;
}
