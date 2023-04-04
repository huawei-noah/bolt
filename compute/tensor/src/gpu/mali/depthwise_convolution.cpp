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
#include "gpu/mali/cl/kernel_option/conv_depthwise_opt.h"

inline void depthwise_convolution_produce_algos_paras(U32 dw,
    U32 dh,
    std::vector<DepthwiseConvolutionForwardAlgorithm> *depthwiseConvAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecH,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    if (dw == 1 && dh == 1) {
        CHECK_STATUS(get_conv_depthwise_cal_scheme(vecH, vecC, vecK));
    } else {
        CHECK_STATUS(get_conv_depthwise_dila_cal_scheme(dh, vecH, vecC, vecK));
    }
    depthwiseConvAlgorithms->push_back(DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT);
    algoNumIndex->push_back(vecH->size());
}

EE depthwise_convolution_padding_input_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (inputMem == nullptr || outputMem == nullptr || outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 dw = convParamSpec.dilatedRate_w;
    U32 dh = convParamSpec.dilatedRate_h;
    U32 ic = inputDesc.dims[inputDesc.nDims - 2];
    if (inputDesc.df != DF_NCHW && inputDesc.df != DF_NCHWC4) {
        return NOT_SUPPORTED;
    }
    if ((ic & 3) != 0) {
        return NOT_SUPPORTED;
    }
    (*outputDesc).df = DF_NCHWC4;

    if (inputDesc.df == DF_NCHWC4) {
        std::vector<DepthwiseConvolutionForwardAlgorithm> depthwiseConvAlgorithms;
        std::vector<U32> algoNumIndex;
        std::vector<U32> vecH;
        std::vector<U32> vecC;
        std::vector<U32> vecK;
        depthwise_convolution_produce_algos_paras(
            dw, dh, &depthwiseConvAlgorithms, &algoNumIndex, &vecH, &vecC, &vecK);
        U32 oh = (*outputDesc).dims[1];
        U32 ih_align = oh;
        for (auto item_h : vecH) {
            U32 i = UNI_ALIGN(oh, item_h);
            ih_align = (ih_align < i) ? i : ih_align;
        }
        U32 pl, pr, pt, pb;
        calDepthwisePaddingVal(inputDesc, convParamSpec, ih_align, &pl, &pr, &pt, &pb);
        inputMem->padding(pl, pr, pt, pb);
    }
    return SUCCESS;
}

EE depthwise_convolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    GCLMemDesc inputMemDesc,
    GCLMemDesc outputMemDesc,
    ConvolutionParamSpec convParamSpec,
    ConvolutionPolicy policy,
    ActivationParamSpec depthwiseActivationParamSpec,
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
    std::vector<TensorDesc> filterDescVec(1, filterDesc);
    std::vector<I32> flag = build_conv_forward_algorithm_flag(
        inputDesc, filterDescVec, OT_Conv, imt, omt, convParamSpec);
    if (gcl_get_runInfo_from_cache(handle, flag, forwardRunInfo)) {
        return SUCCESS;
    }
    U32 dw = convParamSpec.dilatedRate_w;
    U32 dh = convParamSpec.dilatedRate_h;
    std::vector<DepthwiseConvolutionForwardAlgorithm> depthwiseConvAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    depthwise_convolution_produce_algos_paras(
        dw, dh, &depthwiseConvAlgorithms, &algoNumIndex, &vecH, &vecC, &vecK);

    if (policy == CONVOLUTION_TUNNING) {
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_enable_queue_profiling(handle));
        GCLMem_t input = gcl_create_gclmem();
        GCLMem_t filter = gcl_create_gclmem();
        GCLMem_t output = gcl_create_gclmem();
        GCLMem_t bias = gcl_create_gclmem();
        GCLMem_t tmpbuf = gcl_create_gclmem();
        GCLMem_t tmpImg = gcl_create_gclmem();
        U32 maxFilterSize = 0;
        U32 maxBytes[4] = {0};
        U32 algosNum = 0;
        std::vector<ForwardRunInfoMali> runInfos;
        U32 ic;
        DataType dt;
        tensorSelectGet(inputDesc, &dt, NULL, NULL, &ic, NULL, NULL);
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        TensorDesc ftmDesc;
        for (U32 i = 0; i < algoNumIndex.size(); i++) {
            U32 bytes[4] = {0};
            ForwardRunInfoMali runInfo;
            U32 be = (i == 0) ? 0 : algoNumIndex[i - 1];
            U32 end = algoNumIndex[i];
            runInfo.algorithm = depthwiseConvAlgorithms[i];
            for (U32 j = be; j < end; j++) {
                runInfo.best_h[0] = vecH[j];
                runInfo.best_c[0] = vecC[j];
                runInfo.best_k[0] = vecK[j];
                TensorDesc desc;
                if (depthwise_convolution_transform_filter_bytes_mali(
                        filterDesc, &runInfo, &desc) != SUCCESS) {
                    continue;
                }
                if (depthwise_convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc,
                        outputDesc, convParamSpec, &runInfo, bytes) != SUCCESS) {
                    continue;
                }
                if (tensorNumBytes(desc) > maxFilterSize) {
                    ftmDesc = desc;
                    maxFilterSize = tensorNumBytes(desc);
                }
                for (U32 i = 0; i < 4; i++) {
                    maxBytes[i] = (maxBytes[i] < bytes[i]) ? bytes[i] : maxBytes[i];
                }
                runInfos.push_back(runInfo);
            }
        }

        TensorDesc biasDesc = tensor1d(dt, ic);
        stride[0] = (ic + 3) / 4;
        CHECK_STATUS(gclmem_set_desc_padding(
            &bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_IMG_1D, CL_MEM_READ_WRITE));

        stride[0] = ftmDesc.dims[0];
        stride[1] = ftmDesc.dims[1];
        stride[2] = ftmDesc.dims[2];
        CHECK_STATUS(gclmem_set_desc_padding(
            &filter->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));

        algosNum = runInfos.size();
        if (algosNum == 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        outputMemDesc.need_pad = false;
        input->desc = inputMemDesc;
        output->desc = outputMemDesc;
        tmpbuf->desc.byteSize = maxBytes[0];
        gcl_create_memory(handle, input);
        gcl_create_memory(handle, output);
        gcl_create_memory(handle, filter);
        gcl_create_memory(handle, bias);
        GCLMem_t tmp = NULL;
        if (maxBytes[0]) {
            gcl_create_memory(handle, tmpbuf);
            tmp = tmpbuf;
        }
        if (check_qualcomm_device() && maxBytes[1] > 0 && maxBytes[2] > 0 && maxBytes[3] > 0) {
            tmpImg->desc.memType = GCL_MEM_IMG_3D;
            tmpImg->desc.stride[0] = maxBytes[1];
            tmpImg->desc.stride[1] = maxBytes[2];
            tmpImg->desc.stride[2] = maxBytes[3];
            gcl_create_memory(handle, tmpImg);
            tmp = tmpImg;
        }

        double minTime = DBL_MAX;
        ForwardRunInfoMali bestRunInfo;
        for (U32 i = 0; i < algosNum; i++) {
            if (depthwise_convolution_mali(handle, inputDesc, input, filterDesc, filter,
                    convParamSpec, &runInfos[i], biasDesc, bias, maxBytes[0], tmp, outputDesc,
                    output, depthwiseActivationParamSpec) == SUCCESS) {
                U32 kernelVecSize = handle->kernelVec->size();
                gcl_run_kernelVec_timing(handle, kernelVecSize - 1, kernelVecSize);
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
        gcl_destroy_gclmem(filter);
        gcl_destroy_gclmem(output);
        gcl_destroy_gclmem(bias);
        gcl_destroy_gclmem(tmpbuf);
        depthwiseConvAlgorithms.clear();
        runInfos.clear();
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_clean_programMap(handle));
        CHECK_STATUS(gcl_off_queue_profiling(handle));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE depthwise_convolution_transform_filter_bytes_mali(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = depthwise_convolution_transform_filter_bytes_mali_fp16(
                filterDesc, forwardRunInfo, ftmDesc);
            break;
        }
        default:
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
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = depthwise_convolution_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem);
            break;
        }
        default:
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
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = depthwise_convolution_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        }
        default:
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
    ActivationParamSpec depthwiseActivationParamSpec)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = depthwise_convolution_mali_fp16(handle, inputDesc, input, filterDesc, filter,
                convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
                depthwiseActivationParamSpec);
            break;
        }
        default:
            break;
    }
    return ret;
}
