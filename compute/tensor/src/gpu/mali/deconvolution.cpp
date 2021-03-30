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

inline void deconvolution_produce_algos_paras(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    std::vector<ConvolutionForwardAlgorithm> *deconvAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecW,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    DataFormat idf;
    U32 ic, ih, iw, fn, fc, fh, fw, sh, sw;
    tensorSelectGet(inputDesc, NULL, &idf, NULL, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw);
    sh = convParamSpec.stride_h;
    sw = convParamSpec.stride_w;
    U32 configInfo[3][128];
    U32 configNums[2];
    ConvolutionForwardAlgorithm algo[2];
    U32 algoNum = 1;
    algo[0] = CONVOLUTION_ALGORITHM_DIRECT;
    if (fw != 2 || fh != 2 || sw != 2 || sh != 2) {
        configInfo[0][0] = 1;
        configInfo[1][0] = 4;
        configInfo[2][0] = 4;
        configNums[0] = 1;
    } else {
        algo[0] = CONVOLUTION_ALGORITHM_GEMM;
        U32 configNum = 0;
        U32 c = 8;
        U32 ni = 4;
        for (U32 ii = 0; ii < 2; ii++) {
            for (U32 i = 0; i < ni; i++) {
                configInfo[0][configNum] = i + 1;
                configInfo[1][configNum] = c;
                configInfo[2][configNum] = 4;
                configNum++;
            }
            c = c << 1;
            ni = 3;
        }

        ni = 2;
        U32 w = 2;
        for (U32 ii = 0; ii < 2; ii++) {
            c = 8;
            if (ih % w == 0) {
                for (U32 i = 0; i < ni; i++) {
                    configInfo[0][configNum] = w << 8;
                    configInfo[1][configNum] = c;
                    configInfo[2][configNum] = 4;
                    configNum++;
                    c = c << 1;
                }
            }
            w = w << 1;
            ni = 1;
        }
        configNums[0] = configNum;
    }

    for (U32 i = 0; i < algoNum; i++) {
        (*deconvAlgorithms).push_back(algo[i]);
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
EE deconvolution_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 iw, ih, ic, in;
    U32 fw, fh, fc, fn;
    U32 ow, oh;
    U32 sw, sh, dw, dh;
    U32 pt, pb, pl, pr;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    if (in != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (fw < 1 || fh < 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (dw != 1 || dh != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }

    sw = convParamSpec.stride_h;
    sh = convParamSpec.stride_w;
    pt = convParamSpec.padding_top;
    pb = convParamSpec.padding_bottom;
    pl = convParamSpec.padding_left;
    pr = convParamSpec.padding_right;

    oh = fh + sh * (ih - 1) - pt - pb;
    ow = fw + sw * (iw - 1) - pl - pr;

    bool need_pad = false;
    std::vector<ConvolutionForwardAlgorithm> deconvAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecW;
    deconvolution_produce_algos_paras(
        inputDesc, filterDesc, convParamSpec, &deconvAlgorithms, &algoNumIndex, &vecW, NULL, NULL);

    if (idf == DF_NCHW) {
        if (outputDesc) {
            *outputDesc = tensor4df(idt, DF_NCHW, in, fc, oh, ow);
        }
        if (fw == 2 && fh == 2 && sw == 2 && sh == 2) {
            U32 iw_align, item_w;
            iw_align = ow;
            U32 tmp_align = 0;
            for (U32 i = 0; i < algoNumIndex[0]; i++) {
                item_w = vecW[i];
                item_w = ((item_w >> 8) > 0) ? 1 : item_w;
                U32 j = ALIGN(ow, item_w);
                tmp_align = (tmp_align < j) ? j : tmp_align;
            }
            iw_align = (iw_align < tmp_align) ? tmp_align : iw_align;
            if (iw_align != iw) {
                need_pad = true;
            }
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih, ic, 0, 0, ow, oh, fc, idt, idt,
                gclmemInputDesc, gclmemOutputDesc, need_pad));
        } else {
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, ow, oh, fc, idt, idt,
                gclmemInputDesc, gclmemOutputDesc, need_pad));
        }
        return SUCCESS;
    }

    return NOT_SUPPORTED;
}

EE deconvolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc outputDesc,
    ConvolutionPolicy policy,
    ActivationMode activationMode,
    ForwardRunInfoMali_t forwardRunInfo)
{
    if (forwardRunInfo == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if (algorithm != CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }
    DataType dt;
    U32 ih, iw, fc, fh, fw;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw);
    tensorSelectGet(filterDesc, &dt, NULL, NULL, &fc, &fh, &fw);
    std::vector<ConvolutionForwardAlgorithm> deconvAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecW;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    deconvolution_produce_algos_paras(inputDesc, filterDesc, convParamSpec, &deconvAlgorithms,
        &algoNumIndex, &vecW, &vecC, &vecK);
    if (vecW.size() == 1) {
        forwardRunInfo->best_w[0] = vecW[0];
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
        GCLMemDesc inputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc outputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        CHECK_STATUS(deconvolution_infer_output_size_mali(
            inputDesc, filterDesc, convParamSpec, NULL, &inputMemDesc, &outputMemDesc));
        std::vector<GCLMemDesc> filterMemDescs;
        for (U32 i = 0; i < algoNumIndex.size(); i++) {
            U32 bytes = 0;
            ForwardRunInfoMali runInfo;
            U32 be = (i == 0) ? 0 : algoNumIndex[i - 1];
            U32 end = algoNumIndex[i];
            runInfo.algorithm = deconvAlgorithms[i];
            for (U32 j = be; j < end; j++) {
                GCLMemDesc filterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                runInfo.best_w[0] = vecW[j];
                runInfo.best_c[0] = vecC[j];
                runInfo.best_k[0] = vecK[j];
                if (deconvolution_transform_filter_bytes_mali(
                        filterDesc, &runInfo, &filterMemDesc, &bytes) != SUCCESS) {
                    continue;
                }
                maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
                if (deconvolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc,
                        convParamSpec, &runInfo, &bytes) != SUCCESS) {
                    continue;
                }
                maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
                maxFilterSize = (maxFilterSize < filterMemDesc.byteSize) ? filterMemDesc.byteSize
                                                                         : maxFilterSize;
                filterMemDescs.push_back(filterMemDesc);
                runInfos.push_back(runInfo);
            }
        }

        algosNum = runInfos.size();
        TensorDesc biasDesc = tensor1d(dt, fc);
        filterMemDescs[0].byteSize = maxFilterSize;
        input->desc = inputMemDesc;
        output->desc = outputMemDesc;
        filter->desc = filterMemDescs[0];
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
        U32 runKernelBe = 0;
        U32 runKernelEnd = 0;
        ForwardRunInfoMali bestRunInfo;
        ForwardRunInfoMali bestRunInfoGemm;
        for (U32 i = 0; i < algosNum; i++) {
            filter->desc = filterMemDescs[i];
            if (deconvolution_mali(handle, inputDesc, input, filterDesc, filter, convParamSpec,
                    &runInfos[i], biasDesc, NULL, biasDesc, bias, maxBytes, tmpbuf, outputDesc,
                    output, activationMode) == SUCCESS) {
                if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_GEMM) {
                    runKernelEnd = handle->kernelVec->size();
                    gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
                    runKernelBe = runKernelEnd;
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
        CHECK_STATUS(gcl_finish(handle));
        gcl_destroy_gclmem(input);
        gcl_destroy_gclmem(filter);
        gcl_destroy_gclmem(output);
        gcl_destroy_gclmem(bias);
        gcl_destroy_gclmem(tmpbuf);
        deconvAlgorithms.clear();
        runInfos.clear();
        filterMemDescs.clear();
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_clean_programMap(handle));
        CHECK_STATUS(gcl_off_queue_profiling(handle));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE deconvolution_transform_filter_bytes_mali(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = deconvolution_transform_filter_bytes_mali_fp16(
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

EE deconvolution_transform_filter_mali(GCLHandle_t handle,
    TensorDesc filterDesc,
    GCLMem_t filter,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMem_t tmp,
    TensorDesc *fltmemDesc,
    GCLMem_t fltmem)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = deconvolution_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem, tmp);
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

EE deconvolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = deconvolution_infer_forward_tmp_bytes_mali_fp16(
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
    ActivationMode activationMode)
{
    UNUSED(scaleDesc);
    UNUSED(scale);
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = deconvolution_mali_fp16(handle, inputDesc, input, filterDesc, filter,
                convParamSpec, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
                activationMode);
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
