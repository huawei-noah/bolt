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
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"
#include "gpu/mali/cl/kernel_option/gemv_opt.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"

inline void fully_connected_produce_algos_paras(U32 row,
    U32 fc,
    U32 fn,
    GCLMemType outputMemType,
    std::vector<ConvolutionForwardAlgorithm> *fcAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecH,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    fcAlgorithms->push_back(CONVOLUTION_ALGORITHM_GEMM);
    if (row == 1) {
        CHECK_STATUS(get_gemv_cal_scheme(vecH, vecC, vecK));
    } else {  //input need to trans
        GCLMemType mt = (check_qualcomm_device()) ? GCL_MEM_IMG_3D : GCL_MEM_BUF;
        CHECK_STATUS(get_gemm_tn_cal_scheme(vecH, vecC, vecK, mt, mt, outputMemType));
    }
    algoNumIndex->push_back(vecH->size());
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
    if (nullptr == handle || nullptr == input || nullptr == filter || nullptr == output) {
        CHECK_STATUS(NULL_POINTER);
    }
    return SUCCESS;
}

EE fully_connected_padding_input_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (outputDesc == nullptr || inputMem == nullptr || outputMem == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 fn, fc;
    fc = filterDesc.dims[0];
    fn = filterDesc.dims[1];
    DataType dt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    tensorSelectGet(inputDesc, &dt, &idf, &in, &ic, &ih, &iw);
    U32 row = tensorNumElements(inputDesc) / fc;
    *outputDesc = inputDesc;
    outputDesc->dims[0] = fn;
    outputDesc->dims[1] = row;
    for (U32 i = 2; i < inputDesc.nDims; i++) {
        outputDesc->dims[i] = 1;
    }
    if (outputDesc->df == DF_NCHWC4) {
        outputDesc->df = DF_NCHW;
    }
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
    GCLMemType imt = inputMemDesc.memType;
    GCLMemType omt = outputMemDesc.memType;
    std::vector<I32> flag =
        build_fully_connected_forward_algorithm_flag(inputDesc, filterDesc, imt, omt);
    if (gcl_get_runInfo_from_cache(handle, flag, forwardRunInfo)) {
        return SUCCESS;
    }
    DataType dt = inputDesc.dt;
    U32 fc = filterDesc.dims[0];
    U32 fn = filterDesc.dims[1];
    U32 row = outputDesc.dims[1];
    std::vector<ConvolutionForwardAlgorithm> fcAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    fully_connected_produce_algos_paras(
        row, fc, fn, outputMemDesc.memType, &fcAlgorithms, &algoNumIndex, &vecH, &vecC, &vecK);
    if (vecH.size() == 1) {
        forwardRunInfo->best_h[0] = vecH[0];
        forwardRunInfo->best_k[0] = vecK[0];
        forwardRunInfo->best_c[0] = vecC[0];
        forwardRunInfo->algorithm = fcAlgorithms[0];
        return SUCCESS;
    }

    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    GCLMem_t input = gcl_create_gclmem();
    GCLMem_t tmpBuf = gcl_create_gclmem();
    GCLMem_t tmpImg = gcl_create_gclmem();
    GCLMem_t filter = gcl_create_gclmem();
    GCLMem_t bias = gcl_create_gclmem();
    GCLMem_t output = gcl_create_gclmem();

    std::vector<ForwardRunInfoMali> runInfos;
    U32 maxBytes[4] = {0};
    U32 maxFilterSize = 0;
    TensorDesc ftmDesc;
    for (U32 i = 0; i < algoNumIndex.size(); i++) {
        U32 bytes[4] = {0};
        ForwardRunInfoMali runInfo;
        runInfo.algorithm = fcAlgorithms[i];
        U32 be = (i == 0) ? 0 : algoNumIndex[i - 1];
        U32 end = algoNumIndex[i];
        for (U32 j = be; j < end; j++) {
            runInfo.best_h[0] = vecH[j];
            runInfo.best_c[0] = vecC[j];
            runInfo.best_k[0] = vecK[j];
            TensorDesc desc;
            if (fully_connected_transform_filter_bytes_mali(filterDesc, &runInfo, &desc) != SUCCESS) {
                continue;
            }
            if (fully_connected_infer_forward_tmp_bytes_mali(
                    inputDesc, filterDesc, outputDesc, inputMemDesc, bytes, &runInfo) != SUCCESS) {
                continue;
            }
            for (U32 i = 0; i < 4; i++) {
                maxBytes[i] = (maxBytes[i] < bytes[i]) ? bytes[i] : maxBytes[i];
            }
            if (maxFilterSize < tensorNumBytes(desc)) {
                ftmDesc = desc;
                maxFilterSize = tensorNumBytes(desc);
            }
            runInfos.push_back(runInfo);
        }
    }

    MemFlags flags = CL_MEM_READ_WRITE;
    U32 fn_align = fn;
    for (U32 i = 0; i < vecH.size(); ++i) {
        U32 j = UNI_ALIGN(fn, vecH[i]);
        if (fn_align < j) {
            fn_align = j;
        }
    }
    U32 stride[3] = {fn_align, 1, 1};
    U32 offset[3] = {0, 0, 0};
    CHECK_STATUS(
        gclmem_set_desc_padding(&bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_BUF, flags));

    stride[0] = ftmDesc.dims[0];
    stride[1] = ftmDesc.dims[1];
    stride[2] = ftmDesc.dims[2];
    GCLMemType mt = GCL_MEM_BUF;
    bool useImg = check_qualcomm_device();
    if (useImg && row > 1) {
        if (CHECK_MEET_IMAGE_LIMITS(stride[0] / 4, stride[1], stride[2])) {
            stride[0] = stride[0] / 4;
            mt = GCL_MEM_IMG_3D;
        }
    }
    CHECK_STATUS(
        gclmem_set_desc_padding(&filter->desc, stride, offset, dt, DF_NCHW, mt, CL_MEM_READ_WRITE));

    U32 algosNum = runInfos.size();
    if (algosNum == 0) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    TensorDesc biasDesc = tensor1d(dt, fn);
    outputMemDesc.need_pad = false;
    input->desc = inputMemDesc;
    output->desc = outputMemDesc;
    gcl_create_memory(handle, input);
    gcl_create_memory(handle, filter);
    gcl_create_memory(handle, bias);
    gcl_create_memory(handle, output);
    std::vector<GCLMem_t> tmp(2, NULL);
    maxBytes[0] += 1;
    tmpBuf->desc.byteSize = maxBytes[0];
    tmp[0] = tmpBuf;
    gcl_create_memory(handle, tmpBuf);
    if (maxBytes[1] > 0 && maxBytes[2] > 0 && maxBytes[3] > 0) {
        tmpImg->desc.memType = GCL_MEM_IMG_3D;
        tmpImg->desc.stride[0] = maxBytes[1];
        tmpImg->desc.stride[1] = maxBytes[2];
        tmpImg->desc.stride[2] = maxBytes[3];
        gcl_create_memory(handle, tmpImg);
        tmp[1] = tmpImg;
    }

    double minTime = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    for (U32 i = 0; i < algosNum; i++) {
        if (fully_connected_mali(handle, inputDesc, input, filterDesc, filter, biasDesc, bias,
                maxBytes[0], tmp, outputDesc, output, &runInfos[i]) == SUCCESS) {
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
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(tmpBuf);
    gcl_destroy_gclmem(tmpImg);
    gcl_destroy_gclmem(filter);
    gcl_destroy_gclmem(output);
    gcl_destroy_gclmem(bias);
    runInfos.clear();
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_clean_programMap(handle));
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}

EE fully_connected_transform_filter_bytes_mali(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = fully_connected_transform_filter_bytes_mali_fp16(
                filterDesc, forwardRunInfo, ftmDesc);
            break;
        }
        default:
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
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = fully_connected_transform_filter_mali_fp16(
                handle, filterDesc, filter, fltmemDesc, fltmem, forwardRunInfo);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE fully_connected_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    GCLMemDesc gclmemInputDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    return fully_connected_infer_forward_tmp_bytes_mali_fp16(
        inputDesc, filterDesc, outputDesc, gclmemInputDesc, bytes, forwardRunInfo);
}

EE fully_connected_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    GCLMem_t filter,
    TensorDesc biasDesc,
    GCLMem_t bias,
    U32 tmpBytes,
    std::vector<GCLMem_t> tmp,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    CHECK_STATUS(fully_connected_checkpara_mali(
        handle, inputDesc, input, filterDesc, filter, bias, outputDesc, output));
    return fully_connected_mali_fp16(handle, inputDesc, input, filterDesc, filter, biasDesc, bias,
        tmpBytes, tmp, outputDesc, output, forwardRunInfo);
}
