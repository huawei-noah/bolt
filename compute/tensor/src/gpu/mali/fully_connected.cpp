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
#include "gpu/mali/fp16/fully_connected_mali_fp16.h"

inline void fully_connected_produce_algos_paras(U32 row,
    U32 fc,
    std::vector<ConvolutionForwardAlgorithm> *fcAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecW,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    U32 configInfo[3][128];
    U32 configNums[1];
    ConvolutionForwardAlgorithm algo[1];
    U32 algoNum = 1;
    algo[0] = CONVOLUTION_ALGORITHM_DIRECT;
    U32 configNum = 0;
    if (row == 1) {
        U32 j = 8;
        for (U32 i = 0; i < 3; i++) {
            configInfo[0][configNum] = 1;
            configInfo[1][configNum] = 1 << (2 + i);
            configInfo[2][configNum] = 0;
            configNum++;
            if (fc % j != 0) {
                break;
            }
            j = j << 1;
        }
    } else {
        for (U32 i = 1; i <= 8; i++) {
            for (U32 j = 1; j <= 8; j++) {
                if (i * j < 3) {
                    continue;
                }
                configInfo[0][configNum] = i;
                configInfo[1][configNum] = 1;
                configInfo[2][configNum] = j;
                configNum++;
            }
        }
    }
    configNums[0] = configNum;
    for (U32 i = 0; i < algoNum; i++) {
        (*fcAlgorithms).push_back(algo[i]);
        (*algoNumIndex).push_back(configNums[i]);
        U32 be = (i == 0) ? 0 : configNums[i - 1];
        U32 end = configNums[i];
        for (U32 j = be; j < end; j++) {
            if (vecW) {
                (*vecW).push_back(configInfo[0][j]);
            }
            if (vecC) {
                (*vecC).push_back(configInfo[1][j]);
            }
            if (vecK) {
                (*vecK).push_back(configInfo[2][j]);
            }
        }
    }
}
inline EE fully_connected_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    GCLMem_t bias,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == filter || nullptr == output ||
        nullptr == bias) {
        return NULL_POINTER;
    }
    if (!tensorIs2d(filterDesc)) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    U32 fn, fc;
    U32 in, ic, ih, iw;
    CHECK_STATUS(tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw));
    fc = filterDesc.dims[0];
    fn = filterDesc.dims[1];
    if (tensorNumElements(inputDesc) % fc != 0) {
        CHECK_STATUS(NOT_MATCH);
    }
    U32 row = tensorNumElements(inputDesc) / fc;
    if (row > 1) {
        if (iw != fc) {
            CHECK_STATUS(NOT_MATCH);
        }
        if (in * ic > 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    return SUCCESS;
}

EE fully_connected_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 fn, fc;
    fc = filterDesc.dims[0];
    fn = filterDesc.dims[1];
    DataType dt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    U32 ow, oh, oc, on;
    tensorSelectGet(inputDesc, &dt, &idf, &in, &ic, &ih, &iw);
    U32 row = tensorNumElements(inputDesc) / fc;
    *outputDesc = inputDesc;
    outputDesc->dims[0] = fn;
    outputDesc->dims[1] = row;
    for (U32 i = 2; i < inputDesc.nDims; i++) {
        outputDesc->dims[i] = 1;
    }

    DataFormat imf = gclmemInputDesc->memFormat;
    if (imf == DF_NCHW || gclmemInputDesc->byteSize == 0) {
        CHECK_STATUS(
            infer_gclmem_desc_nchw(iw, ih, ic, 0, 0, 0, 0, 0, dt, dt, gclmemInputDesc, NULL));
    } else if (imf == DF_NCWHC4 && row == 1) {
        CHECK_STATUS(
            infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, 0, 0, 0, dt, dt, gclmemInputDesc, NULL));
    } else {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    CHECK_STATUS(infer_gclmem_desc_nchw(0, 0, 0, 0, 0, fn, row, 1, dt, dt, NULL, gclmemOutputDesc));
    return SUCCESS;
}

EE fully_connected_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
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
    DataType dt = inputDesc.dt;
    U32 fc = filterDesc.dims[0];
    U32 fn = filterDesc.dims[1];
    U32 row = tensorNumElements(inputDesc) / filterDesc.dims[0];
    std::vector<ConvolutionForwardAlgorithm> fcAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecW;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    fully_connected_produce_algos_paras(row, fc, &fcAlgorithms, &algoNumIndex, &vecW, &vecC, &vecK);
    if (vecW.size() == 1) {
        forwardRunInfo->best_w[0] = vecW[0];
        forwardRunInfo->best_k[0] = vecK[0];
        forwardRunInfo->best_c[0] = vecC[0];
        forwardRunInfo->algorithm = fcAlgorithms[0];
        return SUCCESS;
    }

    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    GCLMem_t input = gcl_create_gclmem();
    GCLMem_t tmpbuf = gcl_create_gclmem();
    GCLMem_t filter = gcl_create_gclmem();
    GCLMem_t bias = gcl_create_gclmem();
    GCLMem_t output = gcl_create_gclmem();

    std::vector<ForwardRunInfoMali> runInfos;
    std::vector<GCLMemDesc> filterMemDescs;
    U32 maxBytes = 0;
    U32 maxFilterSize = 0;
    for (U32 i = 0; i < algoNumIndex.size(); i++) {
        U32 bytes = 0;
        ForwardRunInfoMali runInfo;
        runInfo.algorithm = fcAlgorithms[i];
        U32 be = (i == 0) ? 0 : algoNumIndex[i - 1];
        U32 end = algoNumIndex[i];
        for (U32 j = be; j < end; j++) {
            GCLMemDesc filterMemDesc = gclmem_build_desc();
            runInfo.best_w[0] = vecW[j];
            runInfo.best_c[0] = vecC[j];
            runInfo.best_k[0] = vecK[j];
            if (fully_connected_transform_filter_bytes_mali(
                    filterDesc, &filterMemDesc, &bytes, &runInfo) != SUCCESS) {
                continue;
            }
            maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
            if (fully_connected_infer_forward_tmp_bytes_mali(
                    inputDesc, filterDesc, &bytes, &runInfo) != SUCCESS) {
                continue;
            }
            maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
            maxFilterSize = (maxFilterSize < filterMemDesc.byteSize) ? filterMemDesc.byteSize
                                                                     : maxFilterSize;
            filterMemDescs.push_back(filterMemDesc);
            runInfos.push_back(runInfo);
        }
    }

    MemFlags flags = CL_MEM_READ_WRITE;
    U32 fn_align = fn;
    for (U32 i = 0; i < vecW.size(); ++i) {
        U32 j = ALIGN(fn, vecW[i]);
        if (fn_align < j) {
            fn_align = j;
        }
    }
    U32 stride[3] = {fn_align, 1, 1};
    U32 offset[3] = {0, 0, 0};
    CHECK_STATUS(
        gclmem_set_desc_padding(&bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_BUF, flags));

    U32 algosNum = runInfos.size();
    if (algosNum == 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    TensorDesc biasDesc = tensor1d(dt, fn);
    filterMemDescs[0].byteSize = maxFilterSize;
    outputMemDesc.need_pad = false;
    input->desc = inputMemDesc;
    output->desc = outputMemDesc;
    filter->desc = filterMemDescs[0];
    tmpbuf->desc.byteSize = maxBytes;
    gcl_create_memory(handle, input);
    gcl_create_memory(handle, filter);
    gcl_create_memory(handle, bias);
    gcl_create_memory(handle, output);
    if (maxBytes) {
        gcl_create_memory(handle, tmpbuf);
    }

    U32 runKernelBe = 0;
    U32 runKernelEnd = 0;
    double minTime = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    for (U32 i = 0; i < algosNum; i++) {
        filter->desc = filterMemDescs[i];
        if (fully_connected_mali(handle, inputDesc, input, filterDesc, filter, biasDesc, bias,
                maxBytes, tmpbuf, outputDesc, output, &runInfos[i]) == SUCCESS) {
            runKernelEnd = handle->kernelVec->size();
            if (runKernelEnd == runKernelBe + 2) {
                runKernelBe += 1;
            }
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
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(tmpbuf);
    gcl_destroy_gclmem(filter);
    gcl_destroy_gclmem(output);
    gcl_destroy_gclmem(bias);
    runInfos.clear();
    filterMemDescs.clear();
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_clean_programMap(handle));
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}

EE fully_connected_transform_filter_bytes_mali(TensorDesc filterDesc,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = fully_connected_transform_filter_bytes_mali_fp16(
                filterDesc, gclmemFilterDesc, bytes, forwardRunInfo);
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

EE fully_connected_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = fully_connected_transform_filter_mali_fp16(
                handle, filterDesc, filter, fltmemDesc, fltmem, forwardRunInfo);
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

EE fully_connected_infer_forward_tmp_bytes_mali(
    TensorDesc inputDesc, TensorDesc filterDesc, U32 *bytes, ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = fully_connected_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, bytes, forwardRunInfo);
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

EE fully_connected_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    ret = fully_connected_checkpara_mali(
        handle, inputDesc, input, filterDesc, filter, bias, outputDesc, output);
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = fully_connected_mali_fp16(handle, inputDesc, input, filterDesc, filter, biasDesc,
                bias, tmpBytes, tmpBuf, outputDesc, output, forwardRunInfo);
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
