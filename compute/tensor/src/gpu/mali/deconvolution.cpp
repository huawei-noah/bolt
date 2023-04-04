// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>
#include "sys.h"

#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/deconvolution_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/deconv_opt.h"

inline void deconvolution_produce_algos_paras(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    std::vector<ConvolutionForwardAlgorithm> *deconvAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecH,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    DataFormat idf;
    U32 ic, ih, iw, fn, fc, fh, fw, sh, sw;
    tensorSelectGet(inputDesc, NULL, &idf, NULL, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    sh = convParamSpec.stride_h;
    sw = convParamSpec.stride_w;

    deconvAlgorithms->push_back(CONVOLUTION_ALGORITHM_GEMM);
    if (fw == 2 && fh == 2 && sw == 2 && sh == 2) {
        CHECK_STATUS(get_deconv_gemm_f2s2_scheme(vecH, vecC, vecK, iw));
    } else {
        CHECK_STATUS(get_deconv_gemm_scheme(vecH, vecC, vecK, iw, fw, fh, fc));
    }
    algoNumIndex->push_back(vecH->size());
}

EE deconvolution_padding_input_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (inputMem == nullptr || outputMem == nullptr || outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    if (inputDesc.df != DF_NCHWC4) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    std::vector<ConvolutionForwardAlgorithm> deconvAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    deconvolution_produce_algos_paras(inputDesc, filterDesc, convParamSpec, &deconvAlgorithms,
        &algoNumIndex, &vecH, &vecC, &vecK);
    (*outputDesc).df = inputDesc.df;
    U32 ow = (*outputDesc).dims[0];
    U32 iw = inputDesc.dims[0];
    U32 iw_align = iw;
    for (U32 i = 0; i < algoNumIndex[0]; i++) {
        U32 item_h = vecH[i];
        item_h = ((item_h >> 8) > 0) ? 1 : item_h;
        U32 j = UNI_ALIGN(ow, item_h);
        iw_align = (iw_align < j) ? j : iw_align;
    }
    U32 pr = iw_align - iw;
    inputMem->padding(0, pr, 0, 0);
    return SUCCESS;
}

EE deconvolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc outputDesc,
    ConvolutionPolicy policy,
    ActivationParamSpec activationMode,
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
    std::vector<TensorDesc> filterDescVec(1, filterDesc);
    std::vector<I32> flag = build_conv_forward_algorithm_flag(
        inputDesc, filterDescVec, OT_Deconvolution, imt, omt, convParamSpec);
    if (gcl_get_runInfo_from_cache(handle, flag, forwardRunInfo)) {
        return SUCCESS;
    }
    DataType dt;
    U32 ih, iw, fc, fh, fw;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw);
    tensorSelectGet(filterDesc, &dt, NULL, NULL, &fc, &fh, &fw);
    std::vector<ConvolutionForwardAlgorithm> deconvAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    deconvolution_produce_algos_paras(inputDesc, filterDesc, convParamSpec, &deconvAlgorithms,
        &algoNumIndex, &vecH, &vecC, &vecK);
    if (vecH.size() == 1) {
        forwardRunInfo->best_h[0] = vecH[0];
        forwardRunInfo->best_k[0] = vecK[0];
        forwardRunInfo->best_c[0] = vecC[0];
        forwardRunInfo->algorithm = deconvAlgorithms[0];
        return SUCCESS;
    }

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
        TensorDesc ftmDesc;
        for (U32 i = 0; i < algoNumIndex.size(); i++) {
            U32 bytes = 0;
            ForwardRunInfoMali runInfo;
            U32 be = (i == 0) ? 0 : algoNumIndex[i - 1];
            U32 end = algoNumIndex[i];
            runInfo.algorithm = deconvAlgorithms[i];
            for (U32 j = be; j < end; j++) {
                TensorDesc desc;
                runInfo.best_h[0] = vecH[j];
                runInfo.best_c[0] = vecC[j];
                runInfo.best_k[0] = vecK[j];
                if (deconvolution_transform_filter_bytes_mali(filterDesc, &runInfo, &desc) !=
                    SUCCESS) {
                    continue;
                }
                if (deconvolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc,
                        convParamSpec, &runInfo, &bytes) != SUCCESS) {
                    continue;
                }
                maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
                if (maxFilterSize < tensorNumBytes(desc)) {
                    ftmDesc = desc;
                    maxFilterSize = tensorNumBytes(desc);
                }
                runInfos.push_back(runInfo);
            }
        }

        algosNum = runInfos.size();
        TensorDesc biasDesc = tensor1d(dt, fc);
        stride[0] = ftmDesc.dims[0];
        stride[1] = ftmDesc.dims[1];
        stride[2] = ftmDesc.dims[2];
        CHECK_STATUS(gclmem_set_desc_padding(
            &filter->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));

        outputMemDesc.need_pad = false;
        input->desc = inputMemDesc;
        output->desc = outputMemDesc;
        stride[0] = (fc + 3) / 4;
        stride[1] = 1;
        stride[2] = 1;
        MemFlags flags = CL_MEM_READ_WRITE;
        CHECK_STATUS(gclmem_set_desc_padding(
            &bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_IMG_1D, flags));
        tmpbuf->desc.byteSize = maxBytes;
        gcl_create_memory(handle, input);
        gcl_create_memory(handle, output);
        gcl_create_memory(handle, filter);
        gcl_create_memory(handle, bias);
        if (maxBytes) {
            gcl_create_memory(handle, tmpbuf);
        }

        double minTimeGemm = DBL_MAX;
        double minTime = DBL_MAX;
        ForwardRunInfoMali bestRunInfo, bestRunInfoGemm;
        UNI_MEMSET(&bestRunInfo, 0, sizeof(bestRunInfo));
        for (U32 i = 0; i < algosNum; i++) {
            U32 runKernelBe = handle->kernelVec->size();
            if (deconvolution_mali(handle, inputDesc, input, filterDesc, filter, convParamSpec,
                    &runInfos[i], biasDesc, NULL, biasDesc, bias, maxBytes, tmpbuf, outputDesc,
                    output, activationMode) == SUCCESS) {
                if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_GEMM) {
                    gcl_run_kernelVec_timing(handle, runKernelBe, runKernelBe + 1);
                    if (minTimeGemm > handle->t_execute) {
                        minTimeGemm = handle->t_execute;
                        bestRunInfoGemm = runInfos[i];
                    }
                }
            }
        }
        minTime = minTimeGemm;
        bestRunInfo = bestRunInfoGemm;
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
        deconvAlgorithms.clear();
        runInfos.clear();
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_clean_programMap(handle));
        CHECK_STATUS(gcl_off_queue_profiling(handle));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE deconvolution_transform_filter_bytes_mali(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret =
                deconvolution_transform_filter_bytes_mali_fp16(filterDesc, forwardRunInfo, ftmDesc);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE deconvolution_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMem_t tmp,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem)
{
    EE ret = NOT_SUPPORTED;
    switch (filterDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = deconvolution_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem, tmp);
            break;
        }
        default:
            break;
    }
    return ret;
}

EE deconvolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = deconvolution_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        }
        default:
            break;
    }
    return ret;
}
EE deconvolution_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    TensorDesc filterDesc,
    const GCLMem_t filter,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    TensorDesc scaleDesc,
    const GCLMem_t scale,
    TensorDesc biasDesc,
    const GCLMem_t bias,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationParamSpec activationMode)
{
    UNUSED(scaleDesc);
    UNUSED(scale);
    return deconvolution_mali_fp16(handle, inputDesc, input, filterDesc, filter, convParamSpec,
        forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode);
}
