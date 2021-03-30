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
#include "gpu/mali/fp16/depthwise_convolution_mali_fp16.h"

inline void depthwise_convolution_produce_algos_paras(U32 dw,
    std::vector<DepthwiseConvolutionForwardAlgorithm> *depthwiseConvAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecW,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    U32 configNum = 8;
    U32 j = (dw == 2) ? 2 : 0;
    for (U32 i = j; i < configNum; i++) {
        if (vecW) {
            (*vecW).push_back(i + 1);
        }
        if (vecC) {
            (*vecC).push_back(1);
        }
        if (vecK) {
            (*vecK).push_back(4);
        }
    }
    if (depthwiseConvAlgorithms) {
        (*depthwiseConvAlgorithms).push_back(DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT);
    }
    if (algoNumIndex) {
        (*algoNumIndex).push_back(configNum - j);
    }
}

EE depthwise_convolution_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    DataType idt;
    DataFormat idf;
    U32 iw, ih, ic, in;
    U32 fw, fh;
    I32 ow, oh;
    U32 sw, sh, pl, pt, dw, dh, pr, pb;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, NULL, NULL, &fh, &fw);
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
    if ((ic & 3) != 0) {
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
        *outputDesc = tensor4df(idt, idf, in, ic, oh, ow);
    }
    bool need_pad = false;

    std::vector<U32> vecW;
    depthwise_convolution_produce_algos_paras(dw, NULL, NULL, &vecW, NULL, NULL);
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
    CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih, ic, ext_w, pt, ow, oh, ic, idt, idt,
        gclmemInputDesc, gclmemOutputDesc, need_pad));
    return SUCCESS;
}

EE depthwise_convolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ActivationMode depthwiseActivationMode,
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
    U32 dw = convParamSpec.dilatedRate_w;
    std::vector<DepthwiseConvolutionForwardAlgorithm> depthwiseConvAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecW;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    depthwise_convolution_produce_algos_paras(
        dw, &depthwiseConvAlgorithms, &algoNumIndex, &vecW, &vecC, &vecK);

    if (policy == CONVOLUTION_TUNNING) {
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_enable_queue_profiling(handle));
        GCLMem_t input = gcl_create_gclmem();
        GCLMem_t filter = gcl_create_gclmem();
        GCLMem_t output = gcl_create_gclmem();
        GCLMem_t bias = gcl_create_gclmem();
        GCLMem_t tmpbuf = gcl_create_gclmem();
        U32 maxFilterSize = 0;
        U32 maxBytes = 0;
        U32 algosNum = 0;
        std::vector<ForwardRunInfoMali> runInfos;
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc inputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc outputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        CHECK_STATUS(depthwise_convolution_infer_output_size_mali(
            inputDesc, filterDesc, convParamSpec, NULL, &inputMemDesc, &outputMemDesc));
        std::vector<GCLMemDesc> filterMemDescs;
        U32 ic;
        DataType dt;
        tensorSelectGet(inputDesc, &dt, NULL, NULL, &ic, NULL, NULL);
        for (U32 i = 0; i < algoNumIndex.size(); i++) {
            U32 bytes = 0;
            ForwardRunInfoMali runInfo;
            U32 be = (i == 0) ? 0 : algoNumIndex[i - 1];
            U32 end = algoNumIndex[i];
            runInfo.algorithm = depthwiseConvAlgorithms[i];
            for (U32 j = be; j < end; j++) {
                GCLMemDesc filterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                runInfo.best_w[0] = vecW[j];
                runInfo.best_c[0] = vecC[j];
                runInfo.best_k[0] = vecK[j];
                if (depthwise_convolution_transform_filter_bytes_mali(
                        filterDesc, &runInfo, &filterMemDesc, &bytes) != SUCCESS) {
                    continue;
                }
                maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
                if (depthwise_convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc,
                        outputDesc, convParamSpec, &runInfo, &bytes) != SUCCESS) {
                    continue;
                }
                maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
                maxFilterSize = (maxFilterSize < filterMemDesc.byteSize) ? filterMemDesc.byteSize
                                                                         : maxFilterSize;
                filterMemDescs.push_back(filterMemDesc);
                runInfos.push_back(runInfo);
            }
        }

        TensorDesc biasDesc = tensor1d(dt, ic);
        stride[0] = (ic + 3) / 4;
        CHECK_STATUS(gclmem_set_desc_padding(
            &bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_IMG_1D, CL_MEM_READ_WRITE));
        algosNum = runInfos.size();
        if (algosNum == 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        filterMemDescs[0].byteSize = maxFilterSize;
        input->desc = inputMemDesc;
        output->desc = outputMemDesc;
        filter->desc = filterMemDescs[0];
        tmpbuf->desc.byteSize = maxBytes;
        gcl_create_memory(handle, input);
        gcl_create_memory(handle, output);
        gcl_create_memory(handle, filter);
        gcl_create_memory(handle, bias);
        if (maxBytes) {
            gcl_create_memory(handle, tmpbuf);
        }

        double minTime = DBL_MAX;
        U32 runKernelBe = 0;
        U32 runKernelEnd = 0;
        ForwardRunInfoMali bestRunInfo;
        for (U32 i = 0; i < algosNum; i++) {
            filter->desc = filterMemDescs[i];
            if (depthwise_convolution_mali(handle, inputDesc, input, filterDesc, filter,
                    convParamSpec, &runInfos[i], biasDesc, bias, maxBytes, tmpbuf, outputDesc,
                    output, depthwiseActivationMode) == SUCCESS) {
                runKernelEnd = handle->kernelVec->size();
                gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
                if (minTime > handle->t_execute) {
                    minTime = handle->t_execute;
                    bestRunInfo = runInfos[i];
                }
                runKernelBe = runKernelEnd;
            }
        }
        if (minTime == DBL_MAX) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        *forwardRunInfo = bestRunInfo;
        CHECK_STATUS(gcl_finish(handle));
        gcl_destroy_gclmem(input);
        gcl_destroy_gclmem(filter);
        gcl_destroy_gclmem(output);
        gcl_destroy_gclmem(bias);
        gcl_destroy_gclmem(tmpbuf);
        depthwiseConvAlgorithms.clear();
        runInfos.clear();
        filterMemDescs.clear();
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_clean_programMap(handle));
        CHECK_STATUS(gcl_off_queue_profiling(handle));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE depthwise_convolution_transform_filter_bytes_mali(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = depthwise_convolution_transform_filter_bytes_mali_fp16(
                filterDesc, forwardRunInfo, gclmemFilterDesc, bytes);
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

EE depthwise_convolution_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = depthwise_convolution_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem);
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

EE depthwise_convolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = depthwise_convolution_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
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

EE depthwise_convolution_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationMode depthwiseActivationMode)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = depthwise_convolution_mali_fp16(handle, inputDesc, input, filterDesc, filter,
                convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
                depthwiseActivationMode);
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
