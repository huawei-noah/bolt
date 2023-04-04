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
#include "gpu/mali/fp16/rnn_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"
#include "gpu/mali/cl/kernel_option/gemv_opt.h"

inline void rnn_produce_algos_paras(U32 filterRow,
    U32 filterCol,
    U32 filterRowPro,
    U32 filterColPro,
    bool useProjection,
    std::vector<ConvolutionForwardAlgorithm> *rnnAlgorithms,
    std::vector<U32> *algoNumIndexGemm,
    std::vector<U32> *vecHGemm,
    std::vector<U32> *vecCGemm,
    std::vector<U32> *vecKGemm,
    std::vector<U32> *algoNumIndexGemv,
    std::vector<U32> *vecHGemv,
    std::vector<U32> *vecCGemv,
    std::vector<U32> *vecKGemv,
    std::vector<U32> *algoNumIndexGemvPro,
    std::vector<U32> *vecHGemvPro,
    std::vector<U32> *vecCGemvPro,
    std::vector<U32> *vecKGemvPro)
{
    rnnAlgorithms->push_back(CONVOLUTION_ALGORITHM_GEMM);
    GCLMemType mt = (check_qualcomm_device()) ? GCL_MEM_IMG_3D : GCL_MEM_BUF;
    CHECK_STATUS(get_gemm_tn_cal_scheme(vecHGemm, vecCGemm, vecKGemm, mt, mt, GCL_MEM_BUF));
    algoNumIndexGemm->push_back(vecHGemm->size());
    CHECK_STATUS(get_gemv_cal_scheme(vecHGemv, vecCGemv, vecKGemv));
    algoNumIndexGemv->push_back(vecHGemv->size());
    if (useProjection) {
        CHECK_STATUS(get_gemv_cal_scheme(vecHGemvPro, vecCGemvPro, vecKGemvPro));
        algoNumIndexGemvPro->push_back(vecHGemvPro->size());
    }
}

inline EE rnn_checkpara_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDescs,
    GCLMem_t input,
    std::vector<TensorDesc> filterDescs,
    GCLMem_t filter,
    std::vector<TensorDesc> biasDescs,
    GCLMem_t bias,
    RNNParamSpec rnnPara,
    GCLMem_t tmpBuf,
    std::vector<TensorDesc> outputDescs,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == filter || nullptr == bias ||
        nullptr == tmpBuf || nullptr == output) {
        return NULL_POINTER;
    }
    U32 inDims = inputDescs[0].nDims;
    U32 batch = inputDescs[0].dims[inDims - 1];
    if (batch != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    return SUCCESS;
}

EE rnn_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    std::vector<TensorDesc> filterDescs,
    std::vector<TensorDesc> biasDescs,
    RNNParamSpec rnnPara,
    TensorDesc outputDesc,
    GCLMemDesc inputMemDesc,
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
    std::vector<I32> flag = build_rnn_forward_algorithm_flag(inputDesc, filterDescs, rnnPara);
    if (gcl_get_runInfo_from_cache(handle, flag, forwardRunInfo)) {
        return SUCCESS;
    }
    std::vector<ConvolutionForwardAlgorithm> rnnAlgorithms;
    std::vector<U32> algoNumIndexGemm;
    std::vector<U32> vecHGemm;
    std::vector<U32> vecCGemm;
    std::vector<U32> vecKGemm;
    std::vector<U32> algoNumIndexGemv;
    std::vector<U32> vecHGemv;
    std::vector<U32> vecCGemv;
    std::vector<U32> vecKGemv;
    std::vector<U32> algoNumIndexGemvPro;
    std::vector<U32> vecHGemvPro;
    std::vector<U32> vecCGemvPro;
    std::vector<U32> vecKGemvPro;
    bool useProjection = (rnnPara.num_projection > 0) ? true : false;
    U32 filterCol = filterDescs[0].dims[0];
    U32 filterRow = filterDescs[0].dims[1];
    U32 filterColPro = (useProjection) ? filterDescs[1].dims[0] : filterCol;
    U32 filterRowPro = (useProjection) ? filterDescs[1].dims[1] : filterRow;
    rnn_produce_algos_paras(filterCol, filterRow, filterColPro, filterRowPro, useProjection,
        &rnnAlgorithms, &algoNumIndexGemm, &vecHGemm, &vecCGemm, &vecKGemm, &algoNumIndexGemv,
        &vecHGemv, &vecCGemv, &vecKGemv, &algoNumIndexGemvPro, &vecHGemvPro, &vecCGemvPro,
        &vecKGemvPro);

    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    GCLMem_t input = gcl_create_gclmem();
    GCLMem_t filterX = gcl_create_gclmem();
    GCLMem_t filterH = gcl_create_gclmem();
    GCLMem_t filterPro = gcl_create_gclmem();
    GCLMem_t bias = gcl_create_gclmem();
    GCLMem_t biasPro = gcl_create_gclmem();
    GCLMem_t tmpBuf = gcl_create_gclmem();
    GCLMem_t tmpImg = gcl_create_gclmem();
    GCLMem_t output = gcl_create_gclmem();
    std::vector<ForwardRunInfoMali> runInfos;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)CONVOLUTION_ALGORITHM_GEMM;
    U32 ni = (useProjection) ? 3 : 2;
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};
    U32 bytes[4] = {0};
    U32 maxBytes[4] = {0};
    U32 gemmAlgoNum = 0;
    U32 gemvAlgoNum = 0;
    U32 gemvProAlgoNum = 0;
    TensorDesc ftmDesc[3];
    U32 maxFilterBytes[3] = {0, 0, 0};
    U32 filterNum = (useProjection) ? 3 : 2;
    for (U32 i = 0; i < ni; i++) {
        U32 algoNum = algoNumIndexGemm[0];
        if (i == 1) {
            algoNum = algoNumIndexGemv[0];
        } else if (i == 2) {
            algoNum = algoNumIndexGemvPro[0];
        }
        U32 gemmIndex = 0;
        U32 gemvIndex = 0;
        U32 gemvProIndex = 0;
        TensorDesc desc[3];
        for (U32 j = 0; j < algoNum; j++) {
            if (i == 0) {
                gemmIndex = j;
            } else if (i == 1) {
                gemvIndex = j;
            } else if (i == 2) {
                gemvProIndex = j;
            }
            runInfo.best_h[0] = vecHGemm[gemmIndex];
            runInfo.best_c[0] = vecCGemm[gemmIndex];
            runInfo.best_k[0] = vecKGemm[gemmIndex];
            runInfo.best_h[1] = vecHGemv[gemvIndex];
            runInfo.best_c[1] = vecCGemv[gemvIndex];
            runInfo.best_k[1] = vecKGemv[gemvIndex];
            if (useProjection) {
                runInfo.best_h[2] = vecHGemvPro[gemvProIndex];
                runInfo.best_c[2] = vecCGemvPro[gemvProIndex];
                runInfo.best_k[2] = vecKGemvPro[gemvProIndex];
            }
            if (rnn_transform_filter_bytes_mali(filterDescs[0], rnnPara, &runInfo, desc) != SUCCESS) {
                continue;
            }
            if (rnn_infer_forward_tmp_bytes_mali(inputDesc, inputMemDesc, filterDescs[0],
                    outputDesc, rnnPara, bytes, &runInfo) != SUCCESS) {
                continue;
            }
            for (U32 i = 0; i < 4; i++) {
                maxBytes[i] = (maxBytes[i] < bytes[i]) ? bytes[i] : maxBytes[i];
            }
            for (U32 i = 0; i < filterNum; i++) {
                if (maxFilterBytes[i] < tensorNumBytes(desc[i])) {
                    ftmDesc[i] = desc[i];
                    maxFilterBytes[i] = tensorNumBytes(desc[i]);
                }
            }
            runInfos.push_back(runInfo);
        }
        if (i == 0) {
            gemmAlgoNum = runInfos.size();
        } else if (i == 1) {
            gemvAlgoNum = runInfos.size();
        } else if (i == 2) {
            gemvProAlgoNum = runInfos.size();
        }
    }
    if (gemmAlgoNum == 0 || gemvAlgoNum == 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    DataType dt = inputDesc.dt;
    stride[0] = (biasDescs[0].dims[0] + 7) / 8 * 8;
    stride[1] = 1;
    stride[2] = 1;
    CHECK_STATUS(gclmem_set_desc_padding(
        &bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_BUF, CL_MEM_READ_WRITE));

    stride[0] = ftmDesc[0].dims[0];
    stride[1] = ftmDesc[0].dims[1];
    stride[2] = ftmDesc[0].dims[2];
    GCLMemType mt = GCL_MEM_BUF;
    bool useImg = check_qualcomm_device();
    if (useImg) {
        if (CHECK_MEET_IMAGE_LIMITS(stride[0] / 4, stride[1], stride[2])) {
            stride[0] = stride[0] / 4;
            mt = GCL_MEM_IMG_3D;
        }
    }
    CHECK_STATUS(
        gclmem_set_desc_padding(&filterX->desc, stride, offset, dt, DF_NCHW, mt, CL_MEM_READ_WRITE));

    stride[0] = ftmDesc[1].dims[0];
    stride[1] = ftmDesc[1].dims[1];
    stride[2] = ftmDesc[1].dims[2];
    CHECK_STATUS(gclmem_set_desc_padding(
        &filterH->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));
    gcl_create_memory(handle, filterX);
    gcl_create_memory(handle, filterH);

    if (useProjection) {
        stride[0] = (biasDescs[1].dims[0] + 3) / 4 * 4;
        CHECK_STATUS(gclmem_set_desc_padding(
            &biasPro->desc, stride, offset, dt, DF_NHWC, GCL_MEM_BUF, CL_MEM_READ_WRITE));
        stride[0] = ftmDesc[2].dims[0];
        stride[1] = ftmDesc[2].dims[1];
        stride[2] = ftmDesc[2].dims[2];
        CHECK_STATUS(gclmem_set_desc_padding(
            &filterPro->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));
        gcl_create_memory(handle, filterPro);
        gcl_create_memory(handle, biasPro);
    }
    outputMemDesc.need_pad = false;
    input->desc = inputMemDesc;
    output->desc = outputMemDesc;
    gcl_create_memory(handle, input);
    gcl_create_memory(handle, output);
    gcl_create_memory(handle, bias);
    std::vector<GCLMem_t> tmp(2, NULL);
    if (maxBytes[0]) {
        tmpBuf->desc.byteSize = maxBytes[0];
        gcl_create_memory(handle, tmpBuf);
        tmp[0] = tmpBuf;
    }
    if (maxBytes[1] > 0 && maxBytes[2] > 0 && maxBytes[3] > 0) {
        tmpImg->desc.memType = GCL_MEM_IMG_3D;
        tmpImg->desc.stride[0] = maxBytes[1];
        tmpImg->desc.stride[1] = maxBytes[2];
        tmpImg->desc.stride[2] = maxBytes[3];
        gcl_create_memory(handle, tmpImg);
        tmp[1] = tmpImg;
    }

    std::vector<TensorDesc> inputDescs;
    std::vector<TensorDesc> outputDescs;
    inputDescs.push_back(inputDesc);
    outputDescs.push_back(outputDesc);
    std::vector<GCLMem> filters;
    std::vector<GCLMem> biases;
    U32 biDirNum = (rnnPara.bi_direction) ? 2 : 1;
    for (U32 i = 0; i < biDirNum; i++) {
        filters.push_back(*filterX);
        filters.push_back(*filterH);
        biases.push_back(*bias);
        if (useProjection) {
            filters.push_back(*filterPro);
            biases.push_back(*biasPro);
        }
    }
    U32 runKernelBeBase = (needReshapeInput(inputMemDesc)) ? 2 : 1;
    double minTimeGemm = DBL_MAX;
    double minTimeGemv = DBL_MAX;
    double minTimeGemvPro = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    UNI_MEMSET(&bestRunInfo, 0, sizeof(bestRunInfo));
    for (U32 i = 0; i < gemmAlgoNum; i++) {
        U32 runKernelBe = handle->kernelVec->size() + runKernelBeBase;
        if (rnn_mali_fp16(handle, inputDescs, input, filterDescs, filters.data(), biasDescs,
                biases.data(), rnnPara, tmp, outputDescs, output, &runInfos[i]) == SUCCESS) {
            gcl_run_kernelVec_timing(handle, runKernelBe, runKernelBe + 1);
            if (minTimeGemm > handle->t_execute) {
                minTimeGemm = handle->t_execute;
                bestRunInfo.algorithm = runInfos[i].algorithm;
                bestRunInfo.best_h[0] = runInfos[i].best_h[0];
                bestRunInfo.best_c[0] = runInfos[i].best_c[0];
                bestRunInfo.best_k[0] = runInfos[i].best_k[0];
            }
        }
    }
    for (U32 i = gemmAlgoNum; i < gemvAlgoNum; i++) {
        U32 runKernelBe = handle->kernelVec->size() + runKernelBeBase;
        if (rnn_mali_fp16(handle, inputDescs, input, filterDescs, filters.data(), biasDescs,
                biases.data(), rnnPara, tmp, outputDescs, output, &runInfos[i]) == SUCCESS) {
            gcl_run_kernelVec_timing(handle, runKernelBe + 3, runKernelBe + 4);
            if (minTimeGemv > handle->t_execute) {
                minTimeGemv = handle->t_execute;
                bestRunInfo.algorithm = runInfos[i].algorithm;
                bestRunInfo.best_h[1] = runInfos[i].best_h[1];
                bestRunInfo.best_c[1] = runInfos[i].best_c[1];
                bestRunInfo.best_k[1] = runInfos[i].best_k[1];
            }
        }
    }
    for (U32 i = gemvAlgoNum; i < gemvProAlgoNum; i++) {
        U32 runKernelBe = handle->kernelVec->size() + runKernelBeBase;
        if (rnn_mali_fp16(handle, inputDescs, input, filterDescs, filters.data(), biasDescs,
                biases.data(), rnnPara, tmp, outputDescs, output, &runInfos[i]) == SUCCESS) {
            gcl_run_kernelVec_timing(handle, runKernelBe + 5, runKernelBe + 6);
            if (minTimeGemvPro > handle->t_execute) {
                minTimeGemvPro = handle->t_execute;
                bestRunInfo.algorithm = runInfos[i].algorithm;
                bestRunInfo.best_h[2] = runInfos[i].best_h[2];
                bestRunInfo.best_c[2] = runInfos[i].best_c[2];
                bestRunInfo.best_k[2] = runInfos[i].best_k[2];
            }
        }
    }
    if (minTimeGemm == DBL_MAX || minTimeGemv == DBL_MAX ||
        (useProjection && minTimeGemvPro == DBL_MAX)) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    *forwardRunInfo = bestRunInfo;
    gcl_set_runInfo_to_cache(handle, flag, bestRunInfo);
    CHECK_STATUS(gcl_finish(handle));
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(filterX);
    gcl_destroy_gclmem(filterH);
    gcl_destroy_gclmem(filterPro);
    gcl_destroy_gclmem(bias);
    gcl_destroy_gclmem(biasPro);
    gcl_destroy_gclmem(tmpBuf);
    gcl_destroy_gclmem(tmpImg);
    gcl_destroy_gclmem(output);
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_clean_programMap(handle));
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}

EE rnn_transform_filter_bytes_mali(TensorDesc filterDesc,
    RNNParamSpec rnnParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *ftmDesc)
{
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = rnn_transform_filter_bytes_mali_fp16(
                filterDesc, rnnParamSpec, forwardRunInfo, ftmDesc);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE rnn_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    GCLMem_t tmpBuf,
    RNNParamSpec rnnParamSpec,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = rnn_transform_filter_mali_fp16(handle, filterDesc, filter, tmpBuf, rnnParamSpec,
                fltmemDesc, fltmem, forwardRunInfo);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE rnn_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    GCLMemDesc gclmemInputDesc,
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
            ret = rnn_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, gclmemInputDesc, filterDesc, outputDesc, rnnPara, bytes, forwardRunInfo);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE rnn_mali(GCLHandle_t handle,
    std::vector<TensorDesc> inputDescs,
    GCLMem_t input,
    std::vector<TensorDesc> filterDescs,
    GCLMem_t filter,
    std::vector<TensorDesc> biasDescs,
    GCLMem_t bias,
    RNNParamSpec rnnPara,
    std::vector<GCLMem_t> tmp,
    std::vector<TensorDesc> outputDescs,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(rnn_checkpara_mali(handle, inputDescs, input, filterDescs, filter, biasDescs, bias,
        rnnPara, tmp[0], outputDescs, output));
    EE ret = NOT_SUPPORTED;
    switch (inputDescs[0].dt) {
        case DT_F16:
        case DT_F32: {
            ret = rnn_mali_fp16(handle, inputDescs, input, filterDescs, filter, biasDescs, bias,
                rnnPara, tmp, outputDescs, output, forwardRunInfo);
            break;
        }
        default:
            break;
    }
    return ret;
}
