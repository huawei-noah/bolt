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
inline void depthwise_pointwise_convolution_produce_algos_paras(U32 oh,
    U32 pointwiseFilterNum,
    U32 dw,
    std::vector<DepthwiseConvolutionForwardAlgorithm> *depthwisePointwiseConvAlgorithms,
    std::vector<U32> *algoNumIndexD,
    std::vector<U32> *vecWD,
    std::vector<U32> *vecCD,
    std::vector<U32> *vecKD,
    std::vector<U32> *algoNumIndexP,
    std::vector<U32> *vecWP,
    std::vector<U32> *vecCP,
    std::vector<U32> *vecKP)
{
    U32 algoNum = 2;
    DepthwiseConvolutionForwardAlgorithm algo[2];
    algo[0] = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT;
    algo[1] = DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM;
    U32 configNumsD[2];
    U32 configNumsP[2];
    U32 configNumD = 0;
    U32 configNumP = 0;
    U32 configInfo[3][128];
    for (U32 ii = 0; ii < algoNum; ii++) {
        U32 j = (dw == 2) ? 2 : 0;
        for (U32 i = j; i < 3; i++) {
            if (vecWD) {
                (*vecWD).push_back(i + 1);
            }
            if (vecCD) {
                (*vecCD).push_back(1);
            }
            if (vecKD) {
                (*vecKD).push_back(4);
            }
            configNumD++;
        }
        configNumsD[ii] = configNumD;
        U32 c = (algo[ii] == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) ? 4 : 1;
        U32 k = 4;
        U32 nj = 8;
        for (U32 i = 0; i < 2; i++) {
            for (U32 j = 0; j < nj; j++) {
                configInfo[0][configNumP] = j + 1;
                configInfo[1][configNumP] = c;
                configInfo[2][configNumP] = k;
                configNumP++;
            }
            k = k << 1;
            if (pointwiseFilterNum % k != 0) {
                break;
            }
            if (algo[ii] == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) {
                nj = 4;
            }
        }
        if (algo[ii] == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) {
            if (pointwiseFilterNum % 16 == 0) {
                for (U32 i = 0; i < 3; i++) {
                    configInfo[0][configNumP] = i + 1;
                    configInfo[1][configNumP] = 4;
                    configInfo[2][configNumP] = 16;
                    configNumP++;
                }
            }
            U32 k = 4;
            U32 nj = 2;
            for (U32 i = 0; i < 3; i++) {
                U32 w = 2;
                if (i == 2) {
                    nj = 1;
                }
                for (U32 j = 0; j < nj; j++) {
                    if (oh % w != 0) {
                        continue;
                    }
                    configInfo[0][configNumP] = w << 8;
                    configInfo[1][configNumP] = 4;
                    configInfo[2][configNumP] = k;
                    configNumP++;
                    w = w << 1;
                }
                k = k << 1;
                if (pointwiseFilterNum % k != 0) {
                    break;
                }
            }
        }
        configNumsP[ii] = configNumP;
    }

    for (U32 i = 0; i < algoNum; i++) {
        if (depthwisePointwiseConvAlgorithms) {
            (*depthwisePointwiseConvAlgorithms).push_back(algo[i]);
        }
        if (algoNumIndexD) {
            (*algoNumIndexD).push_back(configNumsD[i]);
        }
        if (algoNumIndexP) {
            (*algoNumIndexP).push_back(configNumsP[i]);
        }
    }
    for (U32 i = 0; i < configNumP; i++) {
        if (vecWP) {
            (*vecWP).push_back(configInfo[0][i]);
        }
        if (vecCP) {
            (*vecCP).push_back(configInfo[1][i]);
        }
        if (vecKP) {
            (*vecKP).push_back(configInfo[2][i]);
        }
    }
}

EE depthwise_pointwise_convolution_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    U32 fw, fh;
    U32 pfn;
    int ow, oh;
    U32 sw, sh, pl, pt, dw, dh, pr, pb;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    tensorSelectGet(dwFilterDesc, NULL, NULL, NULL, NULL, &fh, &fw);
    tensorSelectGet(pwFilterDesc, NULL, NULL, &pfn, NULL, NULL, NULL);
    pl = convParamSpec.padding_left;
    pr = convParamSpec.padding_right;
    pt = convParamSpec.padding_top;
    pb = convParamSpec.padding_bottom;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    if (fw < 1 || fh < 1) {
        return NOT_SUPPORTED;
    }
    if ((pfn & 3) != 0) {
        return NOT_SUPPORTED;
    }
    U32 fwd = (fw - 1) * dw + 1;
    U32 fhd = (fh - 1) * dh + 1;
    ow = (iw + pl + pr - fwd) / sw + 1;
    oh = (ih + pt + pb - fhd) / sh + 1;
    if (ow <= 0 || oh <= 0) {
        CHECK_STATUS(NOT_MATCH);
    }
    if (outputDesc) {
        *outputDesc = tensor4df(idt, idf, in, pfn, oh, ow);
    }
    bool need_pad = false;

    std::vector<U32> vecW;
    depthwise_pointwise_convolution_produce_algos_paras(
        oh, pfn, dw, NULL, NULL, &vecW, NULL, NULL, NULL, NULL, NULL, NULL);
    U32 iw_align = ow;
    for (auto item_w : vecW) {
        U32 i = ALIGN(ow, item_w);
        iw_align = (iw_align < i) ? i : iw_align;
    }
    U32 ext_w = (fwd / 2 < pl) ? pl : fwd / 2;
    iw_align = iw_align * sw;
    if (pl < ext_w) {
        iw_align = iw_align + 2 * (ext_w - pl);
        ext_w = pl;
    }
    if (iw_align != iw) {
        need_pad = true;
    }
    if (ext_w != 0 || pt != 0) {
        need_pad = true;
    }
    CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih, ic, ext_w, pt, ow, oh, pfn, idt, idt,
        gclmemInputDesc, gclmemOutputDesc, need_pad));
    return SUCCESS;
}

EE depthwise_pointwise_convolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode,
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
    std::vector<DepthwiseConvolutionForwardAlgorithm> depthwisePointwiseConvAlgorithms;
    std::vector<U32> algoNumIndexD;
    std::vector<U32> vecWD;
    std::vector<U32> vecCD;
    std::vector<U32> vecKD;
    std::vector<U32> algoNumIndexP;
    std::vector<U32> vecWP;
    std::vector<U32> vecCP;
    std::vector<U32> vecKP;
    DataType dt;
    U32 ic, oh, pfn, dw;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, &ic, NULL, NULL);
    tensorSelectGet(outputDesc, NULL, NULL, NULL, NULL, &oh, NULL);
    tensorSelectGet(pwFilterDesc, &dt, NULL, &pfn, NULL, NULL, NULL);
    dw = convParamSpec.dilatedRate_w;
    depthwise_pointwise_convolution_produce_algos_paras(oh, pfn, dw,
        &depthwisePointwiseConvAlgorithms, &algoNumIndexD, &vecWD, &vecCD, &vecKD, &algoNumIndexP,
        &vecWP, &vecCP, &vecKP);

    if (policy == CONVOLUTION_TUNNING) {
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_enable_queue_profiling(handle));
        GCLMem_t input = gcl_create_gclmem();
        GCLMem_t dwFilter = gcl_create_gclmem();
        GCLMem_t pwFilter = gcl_create_gclmem();
        GCLMem_t output = gcl_create_gclmem();
        GCLMem_t tmpbuf = gcl_create_gclmem();
        GCLMem_t dwBias = gcl_create_gclmem();
        GCLMem_t pwBiasBuf = gcl_create_gclmem();
        GCLMem_t pwBiasImg = gcl_create_gclmem();
        U32 maxDwFilterSize = 0;
        U32 maxPwFilterSize = 0;
        U32 maxBytes = 0;
        U32 algosNum = 0;
        std::vector<ForwardRunInfoMali> runInfos;
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc inputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc outputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        CHECK_STATUS(depthwise_pointwise_convolution_infer_output_size_mali(inputDesc, dwFilterDesc,
            pwFilterDesc, convParamSpec, NULL, &inputMemDesc, &outputMemDesc));
        std::vector<GCLMemDesc> dwFilterMemDescs;
        std::vector<GCLMemDesc> pwFilterMemDescs;
        if (algoNumIndexD.size() != algoNumIndexP.size()) {
            CHECK_STATUS(NOT_MATCH);
        }

        U32 runInfoBe[2][2];
        U32 runInfoEnd[2][2];
        U32 runInfoCount = 0;
        for (U32 i = 0; i < algoNumIndexD.size(); i++) {
            U32 bytes = 0;
            ForwardRunInfoMali runInfo;
            U32 be = (i == 0) ? 0 : algoNumIndexD[i - 1];
            U32 end = algoNumIndexD[i];
            runInfo.algorithm = depthwisePointwiseConvAlgorithms[i];
            for (U32 j = 0; j < 2; j++) {
                runInfoBe[i][j] = runInfoCount;
                U32 depthwiseIndex = 0;
                U32 pointwiseIndex = 0;
                for (U32 k = be; k < end; k++) {
                    GCLMemDesc dwFilterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                    GCLMemDesc pwFilterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                    if (j == 0) {
                        depthwiseIndex = k;
                    }
                    if (j == 1) {
                        pointwiseIndex = k;
                    }
                    runInfo.best_w[0] = vecWD[depthwiseIndex];
                    runInfo.best_c[0] = vecCD[depthwiseIndex];
                    runInfo.best_k[0] = vecKD[depthwiseIndex];
                    runInfo.best_w[1] = vecWP[pointwiseIndex];
                    runInfo.best_c[1] = vecCP[pointwiseIndex];
                    runInfo.best_k[1] = vecKP[pointwiseIndex];
                    runInfoCount++;
                    if (depthwise_pointwise_convolution_transform_filter_bytes_mali(dwFilterDesc,
                            pwFilterDesc, &runInfo, &dwFilterMemDesc, &pwFilterMemDesc,
                            &bytes) != SUCCESS) {
                        continue;
                    }
                    maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
                    if (depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali(inputDesc,
                            dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, &runInfo,
                            &bytes) != SUCCESS) {
                        continue;
                    }
                    maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
                    maxDwFilterSize = (maxDwFilterSize < dwFilterMemDesc.byteSize)
                        ? dwFilterMemDesc.byteSize
                        : maxDwFilterSize;
                    maxPwFilterSize = (maxPwFilterSize < pwFilterMemDesc.byteSize)
                        ? pwFilterMemDesc.byteSize
                        : maxPwFilterSize;
                    dwFilterMemDescs.push_back(dwFilterMemDesc);
                    pwFilterMemDescs.push_back(pwFilterMemDesc);
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

        algosNum = runInfos.size();
        if (algosNum == 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        dwFilterMemDescs[0].byteSize = maxDwFilterSize;
        pwFilterMemDescs[0].byteSize = maxPwFilterSize;
        input->desc = inputMemDesc;
        output->desc = outputMemDesc;
        dwFilter->desc = dwFilterMemDescs[0];
        pwFilter->desc = pwFilterMemDescs[0];
        tmpbuf->desc.byteSize = maxBytes;
        gcl_create_memory(handle, input);
        gcl_create_memory(handle, output);
        gcl_create_memory(handle, dwFilter);
        gcl_create_memory(handle, pwFilter);
        gcl_create_memory(handle, dwBias);
        gcl_create_memory(handle, pwBiasImg);
        gcl_create_memory(handle, pwBiasBuf);
        if (maxBytes) {
            gcl_create_memory(handle, tmpbuf);
        }

        double minTimeDepthwise[2] = {DBL_MAX, DBL_MAX};
        double minTimePointwise[2] = {DBL_MAX, DBL_MAX};
        ForwardRunInfoMali bestRunInfo[2];
        U32 runKernelBe = 0;
        U32 runKernelEnd = 0;

        for (U32 i = 0; i < 2; i++) {
            U32 depthwiseBe = runInfoBe[i][0];
            U32 depthwiseEnd = runInfoEnd[i][0];
            U32 pointwiseBe = runInfoBe[i][1];
            U32 pointwiseEnd = runInfoEnd[i][1];
            GCLMem_t pwBias = (i == 0) ? pwBiasImg : pwBiasBuf;
            for (U32 j = depthwiseBe; j < depthwiseEnd; j++) {
                if (depthwise_pointwise_convolution_mali(handle, inputDesc, input, dwFilterDesc,
                        pwFilterDesc, dwFilter, pwFilter, convParamSpec, &runInfos[j], dwBiasDesc,
                        pwBiasDesc, dwBias, pwBias, maxBytes, tmpbuf, outputDesc, output,
                        depthwiseActivationMode, pointwiseActivationMode) == SUCCESS) {
                    runKernelEnd = handle->kernelVec->size();
                    gcl_run_kernelVec_timing(handle, runKernelBe, runKernelBe + 1);
                    if (minTimeDepthwise[i] > handle->t_execute) {
                        minTimeDepthwise[i] = handle->t_execute;
                        bestRunInfo[i].algorithm = runInfos[j].algorithm;
                        bestRunInfo[i].best_w[0] = runInfos[j].best_w[0];
                        bestRunInfo[i].best_c[0] = runInfos[j].best_c[0];
                        bestRunInfo[i].best_k[0] = runInfos[j].best_k[0];
                    }
                    runKernelBe = runKernelEnd;
                }
            }
            for (U32 j = pointwiseBe; j < pointwiseEnd; j++) {
                if (depthwise_pointwise_convolution_mali(handle, inputDesc, input, dwFilterDesc,
                        pwFilterDesc, dwFilter, pwFilter, convParamSpec, &runInfos[j], dwBiasDesc,
                        pwBiasDesc, dwBias, pwBias, maxBytes, tmpbuf, outputDesc, output,
                        depthwiseActivationMode, pointwiseActivationMode) == SUCCESS) {
                    runKernelEnd = handle->kernelVec->size();
                    gcl_run_kernelVec_timing(handle, runKernelEnd - 1, runKernelEnd);
                    if (minTimePointwise[i] > handle->t_execute) {
                        minTimePointwise[i] = handle->t_execute;
                        bestRunInfo[i].algorithm = runInfos[j].algorithm;
                        bestRunInfo[i].best_w[1] = runInfos[j].best_w[1];
                        bestRunInfo[i].best_c[1] = runInfos[j].best_c[1];
                        bestRunInfo[i].best_k[1] = runInfos[j].best_k[1];
                    }
                    runKernelBe = runKernelEnd;
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
        CHECK_STATUS(gcl_finish(handle));
        gcl_destroy_gclmem(input);
        gcl_destroy_gclmem(dwFilter);
        gcl_destroy_gclmem(pwFilter);
        gcl_destroy_gclmem(output);
        gcl_destroy_gclmem(dwBias);
        gcl_destroy_gclmem(pwBiasImg);
        gcl_destroy_gclmem(pwBiasBuf);
        gcl_destroy_gclmem(tmpbuf);
        runInfos.clear();
        dwFilterMemDescs.clear();
        pwFilterMemDescs.clear();
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
    GCLMemDesc_t gclmemDwFilterDesc,
    GCLMemDesc_t gclmemPwFilterDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (dwFilterDesc.dt) {
        case DT_F16: {
            ret = depthwise_pointwise_convolution_transform_filter_bytes_mali_fp16(dwFilterDesc,
                pwFilterDesc, forwardRunInfo, gclmemDwFilterDesc, gclmemPwFilterDesc, bytes);
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
    EE ret = SUCCESS;
    switch (dwFilterDesc.dt) {
        case DT_F16: {
            ret = depthwise_pointwise_convolution_transform_filter_mali_fp16(handle, dwFilterDesc,
                pwFilterDesc, dwFilter, pwFilter, forwardRunInfo, dwfltmemDesc, pwfltmemDesc,
                dwfltmem, pwfltmem);
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

EE depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc dwFilterDesc,
    TensorDesc pwFilterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = depthwise_pointwise_convolution_infer_forward_tmp_bytes_mali_fp16(inputDesc,
                dwFilterDesc, pwFilterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        }
        default:
            ret = NOT_SUPPORTED;
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
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode,
    ActivationMode pointwiseActivationMode)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = depthwise_pointwise_convolution_mali_fp16(handle, inputDesc, input, dwFilterDesc,
                pwFilterDesc, dwFilter, pwFilter, convParamSpec, forwardRunInfo, dwBiasDesc,
                pwBiasDesc, dwBias, pwBias, tmpBytes, tmpBuf, outputDesc, output,
                depthwiseActivationMode, pointwiseActivationMode);
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
