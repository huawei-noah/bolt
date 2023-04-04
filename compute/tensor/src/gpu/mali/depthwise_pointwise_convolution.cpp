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

#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/depthwise_pointwise_convolution_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_depthwise_opt.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"

inline void depthwise_pointwise_convolution_produce_algos_paras(U32 ow,
    U32 pointwiseFilterNum,
    U32 dw,
    U32 dh,
    U32 in,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    std::vector<DepthwiseConvolutionForwardAlgorithm> *depthwisePointwiseConvAlgorithms,
    std::vector<U32> *algoNumIndexD,
    std::vector<U32> *vecHD,
    std::vector<U32> *vecCD,
    std::vector<U32> *vecKD,
    std::vector<U32> *algoNumIndexP,
    std::vector<U32> *vecHP,
    std::vector<U32> *vecCP,
    std::vector<U32> *vecKP)
{
    depthwisePointwiseConvAlgorithms->push_back(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT);
    if (!check_qualcomm_device()) {
        depthwisePointwiseConvAlgorithms->push_back(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM);
    }
    for (auto algo : (*depthwisePointwiseConvAlgorithms)) {
        if (dw == 1 && dh == 1) {
            CHECK_STATUS(get_conv_depthwise_cal_scheme(vecHD, vecCD, vecKD));
        } else {
            CHECK_STATUS(get_conv_depthwise_dila_cal_scheme(dh, vecHD, vecCD, vecKD));
        }
        algoNumIndexD->push_back(vecHD->size());
        if (algo == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) {
            CHECK_STATUS(get_conv_direct_cal_scheme(vecHP, vecCP, vecKP, 1, 1, pointwiseFilterNum));
            if (!check_qualcomm_device()) {
                CHECK_STATUS(get_conv_direct_reuse_w_cal_scheme(
                    vecHP, vecCP, vecKP, ow, pointwiseFilterNum, inputMemType));
            }
        } else if (algo == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM) {
            CHECK_STATUS(get_gemm_tn_pointwise_cal_scheme(
                vecHP, vecCP, vecKP, pointwiseFilterNum, outputMemType));
        }
        algoNumIndexP->push_back(vecHP->size());
    }
}

EE depthwise_pointwise_convolution_padding_input_mali(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (inputMem == nullptr || outputMem == nullptr || outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 sh = convParamSpec.stride_h;
    U32 dw = convParamSpec.dilatedRate_w;
    U32 dh = convParamSpec.dilatedRate_h;
    U32 ih = inputDesc.dims[1];
    U32 in = inputDesc.dims[inputDesc.nDims - 1];
    U32 fh = dwFilterDesc.dims[1];
    U32 ow = (*outputDesc).dims[0];
    U32 oh = (*outputDesc).dims[1];
    U32 pfn = pwFilterDesc.dims[pwFilterDesc.nDims - 1];
    (*outputDesc).df = DF_NCHWC4;
    if ((pfn & 3) != 0) {
        return NOT_SUPPORTED;
    }

    if (inputDesc.df == DF_NCHWC4) {
        std::vector<DepthwiseConvolutionForwardAlgorithm> depthwisePointwiseConvAlgorithms;
        std::vector<U32> algoNumIndexD;
        std::vector<U32> vecHD;
        std::vector<U32> vecCD;
        std::vector<U32> vecKD;
        std::vector<U32> algoNumIndexP;
        std::vector<U32> vecHP;
        std::vector<U32> vecCP;
        std::vector<U32> vecKP;
        GCLMemType imt = inputMem->gclMemType();
        GCLMemType omt = outputMem->gclMemType();
        depthwise_pointwise_convolution_produce_algos_paras(ow, pfn, dw, dh, in, imt, omt,
            &depthwisePointwiseConvAlgorithms, &algoNumIndexD, &vecHD, &vecCD, &vecKD,
            &algoNumIndexP, &vecHP, &vecCP, &vecKP);

        U32 ih_align = oh;
        for (auto item_h : vecHD) {
            U32 i = UNI_ALIGN(oh, item_h);
            ih_align = (ih_align < i) ? i : ih_align;
        }
        ih_align *= sh;
        U32 fhd = (fh - 1) * dh + 1;
        U32 pl = convParamSpec.pad_left;
        U32 pr = convParamSpec.pad_right;
        U32 pt = convParamSpec.pad_top;
        U32 pb = ih_align + (fhd / 2 * 2) - pt - ih;
        if (pb < convParamSpec.pad_bottom) {
            pb = convParamSpec.pad_bottom;
        }
        inputMem->padding(pl, pr, pt, pb);
    }
    return SUCCESS;
}

EE depthwise_pointwise_convolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    GCLMemDesc inputMemDesc,
    GCLMemDesc outputMemDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec,
    ForwardRunInfoMali_t forwardRunInfo)
{
    if (forwardRunInfo == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    DepthwiseConvolutionForwardAlgorithm algorithm =
        (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if (algorithm != DEPTHWISE_CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }
    if (policy == CONVOLUTION_LIBRARY_SEARCH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (policy == CONVOLUTION_FASTEST) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    GCLMemType imt = inputMemDesc.memType;
    GCLMemType omt = outputMemDesc.memType;
    std::vector<TensorDesc> filterDescVec = {dwFilterDesc, pwFilterDesc};
    std::vector<I32> flag = build_conv_forward_algorithm_flag(
        inputDesc, filterDescVec, OT_Conv, imt, omt, convParamSpec);
    if (gcl_get_runInfo_from_cache(handle, flag, forwardRunInfo)) {
        return SUCCESS;
    }
    std::vector<DepthwiseConvolutionForwardAlgorithm> depthwisePointwiseConvAlgorithms;
    std::vector<U32> algoNumIndexD;
    std::vector<U32> vecHD;
    std::vector<U32> vecCD;
    std::vector<U32> vecKD;
    std::vector<U32> algoNumIndexP;
    std::vector<U32> vecHP;
    std::vector<U32> vecCP;
    std::vector<U32> vecKP;
    DataType dt;
    U32 in, ic, ow, pfn, dw, dh;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, NULL, NULL);
    tensorSelectGet(outputDesc, NULL, NULL, NULL, NULL, NULL, &ow);
    tensorSelectGet(pwFilterDesc, &dt, NULL, &pfn, NULL, NULL, NULL);
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    depthwise_pointwise_convolution_produce_algos_paras(ow, pfn, dw, dh, in, inputMemDesc.memType,
        outputMemDesc.memType, &depthwisePointwiseConvAlgorithms, &algoNumIndexD, &vecHD, &vecCD,
        &vecKD, &algoNumIndexP, &vecHP, &vecCP, &vecKP);

    if (policy == CONVOLUTION_TUNNING) {
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_enable_queue_profiling(handle));
        GCLMem_t input = gcl_create_gclmem();
        GCLMem_t dwFilter = gcl_create_gclmem();
        GCLMem_t pwFilter = gcl_create_gclmem();
        GCLMem_t output = gcl_create_gclmem();
        GCLMem_t tmpbuf = gcl_create_gclmem();
        GCLMem_t tmpImgA = gcl_create_gclmem();
        GCLMem_t tmpImgB = gcl_create_gclmem();
        GCLMem_t dwBias = gcl_create_gclmem();
        GCLMem_t pwBiasBuf = gcl_create_gclmem();
        GCLMem_t pwBiasImg = gcl_create_gclmem();
        U32 maxDwFilterSize = 0;
        U32 maxPwFilterSize = 0;
        U32 maxBytes[7] = {0};
        U32 algosNum = 0;
        std::vector<ForwardRunInfoMali> runInfos;
        if (algoNumIndexD.size() != algoNumIndexP.size()) {
            CHECK_STATUS(NOT_MATCH);
        }

        U32 runInfoBe[2][2];
        U32 runInfoEnd[2][2];
        U32 runInfoCount = 0;
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        TensorDesc dwFtmDesc;
        TensorDesc pwFtmDesc;
        for (U32 i = 0; i < algoNumIndexD.size(); i++) {
            U32 bytes[7] = {0};
            ForwardRunInfoMali runInfo;
            U32 be = (i == 0) ? 0 : algoNumIndexD[i - 1];
            U32 end = algoNumIndexD[i];
            runInfo.algorithm = depthwisePointwiseConvAlgorithms[i];
            for (U32 j = 0; j < 2; j++) {
                runInfoBe[i][j] = runInfoCount;
                U32 depthwiseIndex = 0;
                U32 pointwiseIndex = 0;
                for (U32 k = be; k < end; k++) {
                    GCLMemDesc dwFilterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCHWC4);
                    GCLMemDesc pwFilterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCHWC4);
                    if (j == 0) {
                        depthwiseIndex = k;
                    }
                    if (j == 1) {
                        pointwiseIndex = k;
                    }
                    runInfo.best_h[0] = vecHD[depthwiseIndex];
                    runInfo.best_c[0] = vecCD[depthwiseIndex];
                    runInfo.best_k[0] = vecKD[depthwiseIndex];
                    runInfo.best_h[1] = vecHP[pointwiseIndex];
                    runInfo.best_c[1] = vecCP[pointwiseIndex];
                    runInfo.best_k[1] = vecKP[pointwiseIndex];
                    runInfoCount++;
                    TensorDesc dwDesc, pwDesc;
                    if (depthwise_pointwise_convolution_transform_filter_bytes_mali(
                            dwFilterDesc, pwFilterDesc, &runInfo, &dwDesc, &pwDesc) != SUCCESS) {
                        continue;
                    }
                    if (depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali(inputDesc,
                            dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, &runInfo,
                            bytes) != SUCCESS) {
                        continue;
                    }
                    for (U32 i = 0; i < 7; i++) {
                        maxBytes[i] = (maxBytes[i] < bytes[i]) ? bytes[i] : maxBytes[i];
                    }
                    if (maxDwFilterSize < tensorNumBytes(dwDesc)) {
                        dwFtmDesc = dwDesc;
                        maxDwFilterSize = tensorNumBytes(dwDesc);
                    }
                    if (maxPwFilterSize < tensorNumBytes(pwDesc)) {
                        pwFtmDesc = pwDesc;
                        maxPwFilterSize = tensorNumBytes(pwDesc);
                    }
                    runInfos.push_back(runInfo);
                }
                runInfoEnd[i][j] = runInfoCount;
                be = (i == 0) ? 0 : algoNumIndexP[i - 1];
                end = algoNumIndexP[i];
            }
        }

        TensorDesc dwBiasDesc = tensor1d(dt, ic);
        TensorDesc pwBiasDesc = tensor1d(dt, pfn);
        U32 dwStride[3] = {(ic + 3) / 4, 1, 1};
        CHECK_STATUS(gclmem_set_desc_padding(
            &dwBias->desc, dwStride, offset, dt, DF_NHWC, GCL_MEM_IMG_1D, CL_MEM_READ_WRITE));
        U32 pwStride[3] = {(pfn + 3) / 4, 1, 1};
        CHECK_STATUS(gclmem_set_desc_padding(
            &pwBiasImg->desc, pwStride, offset, dt, DF_NHWC, GCL_MEM_IMG_1D, CL_MEM_READ_WRITE));
        pwStride[0] = (pfn + 7) / 8 * 8;
        CHECK_STATUS(gclmem_set_desc_padding(
            &pwBiasBuf->desc, pwStride, offset, dt, DF_NHWC, GCL_MEM_BUF, CL_MEM_READ_WRITE));

        stride[0] = dwFtmDesc.dims[0];
        stride[1] = dwFtmDesc.dims[1];
        stride[2] = dwFtmDesc.dims[2];
        CHECK_STATUS(gclmem_set_desc_padding(
            &dwFilter->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));

        stride[0] = pwFtmDesc.dims[0];
        stride[1] = pwFtmDesc.dims[1];
        stride[2] = pwFtmDesc.dims[2];
        CHECK_STATUS(gclmem_set_desc_padding(
            &pwFilter->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));

        algosNum = runInfos.size();
        if (algosNum == 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        outputMemDesc.need_pad = false;
        input->desc = inputMemDesc;
        output->desc = outputMemDesc;
        gcl_create_memory(handle, input);
        gcl_create_memory(handle, output);
        gcl_create_memory(handle, dwFilter);
        gcl_create_memory(handle, pwFilter);
        gcl_create_memory(handle, dwBias);
        gcl_create_memory(handle, pwBiasImg);
        gcl_create_memory(handle, pwBiasBuf);
        std::vector<GCLMem_t> tmp(3, NULL);
        if (maxBytes[0]) {
            tmpbuf->desc.byteSize = maxBytes[0];
            gcl_create_memory(handle, tmpbuf);
            tmp[0] = tmpbuf;
        }
        if (check_qualcomm_device() && maxBytes[1] > 0 && maxBytes[2] > 0 && maxBytes[3] > 0) {
            tmpImgA->desc.memType = GCL_MEM_IMG_3D;
            tmpImgA->desc.stride[0] = maxBytes[1];
            tmpImgA->desc.stride[1] = maxBytes[2];
            tmpImgA->desc.stride[2] = maxBytes[3];
            gcl_create_memory(handle, tmpImgA);
            tmp[1] = tmpImgA;
        }
        if (check_qualcomm_device() && maxBytes[4] > 0 && maxBytes[5] > 0 && maxBytes[6] > 0) {
            tmpImgB->desc.memType = GCL_MEM_IMG_3D;
            tmpImgB->desc.stride[0] = maxBytes[4];
            tmpImgB->desc.stride[1] = maxBytes[5];
            tmpImgB->desc.stride[2] = maxBytes[6];
            gcl_create_memory(handle, tmpImgB);
            tmp[2] = tmpImgB;
        }

        double minTimeDepthwise[2] = {DBL_MAX, DBL_MAX};
        double minTimePointwise[2] = {DBL_MAX, DBL_MAX};
        ForwardRunInfoMali bestRunInfo[2];

        for (U32 i = 0; i < depthwisePointwiseConvAlgorithms.size(); i++) {
            U32 depthwiseBe = runInfoBe[i][0];
            U32 depthwiseEnd = runInfoEnd[i][0];
            U32 pointwiseBe = runInfoBe[i][1];
            U32 pointwiseEnd = runInfoEnd[i][1];
            GCLMem_t pwBias = pwBiasBuf;
            if (depthwisePointwiseConvAlgorithms[i] ==
                DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) {
                pwBias = pwBiasImg;
            }
            for (U32 j = depthwiseBe; j < depthwiseEnd; j++) {
                if (depthwise_pointwise_convolution_mali(handle, inputDesc, input, dwFilterDesc,
                        pwFilterDesc, dwFilter, pwFilter, convParamSpec, &runInfos[j], dwBiasDesc,
                        pwBiasDesc, dwBias, pwBias, maxBytes[0], tmp, outputDesc, output,
                        depthwiseActivationParamSpec, pointwiseActivationParamSpec) == SUCCESS) {
                    U32 kernelVecSize = handle->kernelVec->size();
                    gcl_run_kernelVec_timing(handle, kernelVecSize - 2, kernelVecSize - 1);
                    if (minTimeDepthwise[i] > handle->t_execute) {
                        minTimeDepthwise[i] = handle->t_execute;
                        bestRunInfo[i].algorithm = runInfos[j].algorithm;
                        bestRunInfo[i].best_h[0] = runInfos[j].best_h[0];
                        bestRunInfo[i].best_c[0] = runInfos[j].best_c[0];
                        bestRunInfo[i].best_k[0] = runInfos[j].best_k[0];
                    }
                }
            }
            for (U32 j = pointwiseBe; j < pointwiseEnd; j++) {
                if (depthwise_pointwise_convolution_mali(handle, inputDesc, input, dwFilterDesc,
                        pwFilterDesc, dwFilter, pwFilter, convParamSpec, &runInfos[j], dwBiasDesc,
                        pwBiasDesc, dwBias, pwBias, maxBytes[0], tmp, outputDesc, output,
                        depthwiseActivationParamSpec, pointwiseActivationParamSpec) == SUCCESS) {
                    U32 kernelVecSize = handle->kernelVec->size();
                    gcl_run_kernelVec_timing(handle, kernelVecSize - 1, kernelVecSize);
                    if (minTimePointwise[i] > handle->t_execute) {
                        minTimePointwise[i] = handle->t_execute;
                        bestRunInfo[i].algorithm = runInfos[j].algorithm;
                        bestRunInfo[i].best_h[1] = runInfos[j].best_h[1];
                        bestRunInfo[i].best_c[1] = runInfos[j].best_c[1];
                        bestRunInfo[i].best_k[1] = runInfos[j].best_k[1];
                    }
                }
            }
        }

        double minTimeDirect = minTimeDepthwise[0] + minTimePointwise[0];
        double minTimeGemm = minTimeDepthwise[1] + minTimePointwise[1];
        if (minTimeDirect == DBL_MAX && minTimeGemm == DBL_MAX) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        if (minTimeDirect > minTimeGemm) {
            bestRunInfo[0] = bestRunInfo[1];
        }

        *forwardRunInfo = bestRunInfo[0];
        gcl_set_runInfo_to_cache(handle, flag, bestRunInfo[0]);
        CHECK_STATUS(gcl_finish(handle));
        gcl_destroy_gclmem(input);
        gcl_destroy_gclmem(dwFilter);
        gcl_destroy_gclmem(pwFilter);
        gcl_destroy_gclmem(output);
        gcl_destroy_gclmem(dwBias);
        gcl_destroy_gclmem(pwBiasImg);
        gcl_destroy_gclmem(pwBiasBuf);
        gcl_destroy_gclmem(tmpbuf);
        gcl_destroy_gclmem(tmpImgA);
        gcl_destroy_gclmem(tmpImgB);
        runInfos.clear();
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_clean_programMap(handle));
        CHECK_STATUS(gcl_off_queue_profiling(handle));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE depthwise_pointwise_convolution_transform_filter_bytes_mali(TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *dwFtmDesc,
    TensorDesc *pwFtmDesc)
{
    EE ret = NOT_SUPPORTED;
    switch (dwFilterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = depthwise_pointwise_convolution_transform_filter_bytes_mali_fp16(
                dwFilterDesc, pwFilterDesc, forwardRunInfo, dwFtmDesc, pwFtmDesc);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE depthwise_pointwise_convolution_transform_filter_mali(GCLHandle_t handle,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    GCLMem_t dwFilter,
    GCLMem_t pwFilter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *dwfltmemDesc,
    TensorDesc *pwfltmemDesc,
    GCLMem_t dwfltmem,
    GCLMem_t pwfltmem)
{
    EE ret = NOT_SUPPORTED;
    switch (dwFilterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = depthwise_pointwise_convolution_transform_filter_mali_fp16(handle, dwFilterDesc,
                pwFilterDesc, dwFilter, pwFilter, forwardRunInfo, dwfltmemDesc, pwfltmemDesc,
                dwfltmem, pwfltmem);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali_fp16(inputDesc,
                dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE depthwise_pointwise_convolution_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    const GCLMem_t dwFilter,
    const GCLMem_t pwFilter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc dwBiasDesc,
    TensorDesc pwBiasDesc,
    const GCLMem_t dwBias,
    const GCLMem_t pwBias,
    U32 tmpBytes,
    std::vector<GCLMem_t> tmp,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationParamSpec depthwiseActivationParamSpec,
    ActivationParamSpec pointwiseActivationParamSpec)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = depthwise_pointwise_convolution_mali_fp16(handle, inputDesc, input, dwFilterDesc,
                pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo, dwBiasDesc,
                pwBiasDesc, dwBias, pwBias, tmpBytes, tmp, outputDesc, output,
                depthwiseActivationParamSpec, pointwiseActivationParamSpec);
            break;
        }
        default:
            break;
    }
    return ret;
}
