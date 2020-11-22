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
#include "types.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/convolution_mali_fp16.h"

inline void convolution_produce_algos_paras(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    DataFormat inputGclmemFormat,
    std::vector<ConvolutionForwardAlgorithm> *convolutionAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecW,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    DataFormat idf;
    U32 ic, ih, iw, fn, fh, fw, sh, sw;
    tensorSelectGet(inputDesc, NULL, &idf, NULL, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, NULL, NULL, &fn, NULL, &fh, &fw);
    sh = convParamSpec.stride_h;
    sw = convParamSpec.stride_w;
    U32 configInfo[3][128];
    U32 configNums[2];
    ConvolutionForwardAlgorithm algo[2];
    U32 algoNum = 1;
    algo[0] = CONVOLUTION_ALGORITHM_DIRECT;
    if (inputGclmemFormat == DF_NCHW && (ih != 1 || iw != 1 || fw != 1 || fh != 1)) {
        configInfo[0][0] = (8 * sw - (fw - 1)) / sw;
        configInfo[1][0] = 1;
        configInfo[2][0] = 4;
        configNums[0] = 1;
    } else if (fn == 1 && sw == 1 && (fw == fh) && (fw == 1 || fw == 3 || fw == 5 || fw == 7)) {
        configInfo[0][0] = (fw == 7) ? 6 : 8;
        configInfo[1][0] = 4;
        configInfo[2][0] = 1;
        configNums[0] = 1;
    } else {
        if (fw == 3 && fh == 3 && sw == 1 && sh == 1) {
            algo[1] = CONVOLUTION_ALGORITHM_WINOGRAD;
            algoNum = 2;
        }
        U32 configNum = 0;
        for (U32 ii = 0; ii < algoNum; ii++) {
            if (algo[ii] == CONVOLUTION_ALGORITHM_DIRECT) {
                if (ih == 1 && iw == 1 && fh == 1 && fw == 1) {
                    U32 j = 8;
                    for (U32 i = 0; i < 3; i++) {
                        configInfo[0][configNum] = 1;
                        configInfo[1][configNum] = 1 << (2 + i);
                        configInfo[2][configNum] = 0;
                        configNum++;
                        if (ic % j != 0) {
                            break;
                        }
                        j = j << 1;
                    }
                } else {
                    U32 k = 4;
                    U32 nj = 8;
                    for (U32 i = 0; i < 2; i++) {
                        for (U32 j = 0; j < nj; j++) {
                            configInfo[0][configNum] = j + 1;
                            configInfo[1][configNum] = 4;
                            configInfo[2][configNum] = k;
                            configNum++;
                        }
                        k = k << 1;
                        if (fn % k != 0) {
                            break;
                        }
                        nj = 4;
                    }
                    if (fw == 1 && fh == 1 && sw == 1 && sh == 1) {
                        U32 k = 4;
                        U32 nj = 2;
                        for (U32 i = 0; i < 3; i++) {
                            U32 w = 2;
                            if (i == 2) {
                                nj = 1;
                            }
                            for (U32 j = 0; j < nj; j++) {
                                if (ih % w != 0) {
                                    continue;
                                }
                                configInfo[0][configNum] = w << 8;
                                configInfo[1][configNum] = 4;
                                configInfo[2][configNum] = k;
                                configNum += 1;
                                w = w << 1;
                            }
                            k = k << 1;
                            if (fn % k != 0) {
                                break;
                            }
                        }
                        if (fn % 16 == 0) {
                            for (U32 i = 0; i < 3; i++) {
                                configInfo[0][configNum] = i + 1;
                                configInfo[1][configNum] = 4;
                                configInfo[2][configNum] = 16;
                                configNum++;
                            }
                        }
                    }
                }
            }

            if (algo[ii] == CONVOLUTION_ALGORITHM_WINOGRAD) {
                for (U32 i = 1; i <= 8; i++) {
                    for (U32 j = 4; j <= 8; j += 4) {
                        if (i * j <= 2) {
                            continue;
                        }
                        configInfo[0][configNum] = i;
                        configInfo[1][configNum] = 1;
                        configInfo[2][configNum] = j;
                        configNum++;
                    }
                }
            }
            configNums[ii] = configNum;
        }
    }

    for (U32 i = 0; i < algoNum; i++) {
        (*convolutionAlgorithms).push_back(algo[i]);
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

EE convolution_infer_output_size_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc)
{
    if (outputDesc == nullptr || gclmemInputDesc == nullptr || gclmemOutputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    DataType idt, fdt;
    DataFormat idf, fdf;
    U32 iw, ih, ic, in, it;
    U32 fw, fh, fc, fn, ft;
    U32 ow, oh, ot;
    U32 sw, sh, st, dw, dh, fdw, fdh;
    U32 pl, pr, pt, pb, pt_b, pt_a;
    tensorSelectGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw, &ft);
    pl = convParamSpec.padding_left;
    pr = convParamSpec.padding_right;
    pt = convParamSpec.padding_top;
    pb = convParamSpec.padding_bottom;
    sw = convParamSpec.stride_w;
    sh = convParamSpec.stride_h;
    dw = convParamSpec.dilatedRate_w;
    dh = convParamSpec.dilatedRate_h;
    pt_b = convParamSpec.padding_before;
    pt_a = convParamSpec.padding_after;
    st = convParamSpec.stride_t;

    if (fw > 7 || fw == 6) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (in != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (fw < 1 || fh < 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (dw != 1 || dh != 1) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (sw != 1 && sw != 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (sh != 1 && sh != 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    fdw = (fw - 1) * dw + 1;
    fdh = (fh - 1) * dh + 1;
    ow = (iw + pl + pr - fdw) / sw + 1;
    oh = (ih + pt + pb - fdh) / sh + 1;
    ot = (inputDesc.df == DF_NCTHW) ? (it + pt_b + pt_a - ft) / st + 1 : 1;

    U32 iw_align, ih_align, item_w, ext_w, ext_h;
    bool need_pad = false;

    if (inputDesc.df == DF_NCTHW) {
        *outputDesc = tensor5df(idt, idf, in, fn, ot, oh, ow);
    } else {
        *outputDesc = tensor4df(idt, idf, in, fn, oh, ow);
    }
    ext_w = (fw / 2 < pl) ? pl : fw / 2;  // if fw / 2 < pl, use pl as offset
    ext_h = pt;

    std::vector<ConvolutionForwardAlgorithm> convolutionAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecW;
    DataFormat inputGclmemFormat = DF_NCWHC4;
    if (gclmemInputDesc->byteSize == 0) {
        inputGclmemFormat = DF_NCHW;
    }
    convolution_produce_algos_paras(inputDesc, filterDesc, convParamSpec, inputGclmemFormat,
        &convolutionAlgorithms, &algoNumIndex, &vecW, NULL, NULL);
    iw_align = ow;
    for (auto p : convolutionAlgorithms) {
        U32 tmp_align = 0;
        if (p == CONVOLUTION_ALGORITHM_WINOGRAD) {
            tmp_align = ALIGN(ow, 16);
        } else {
            for (U32 i = 0; i < algoNumIndex[0]; i++) {
                item_w = vecW[i];
                item_w = ((item_w >> 8) > 0) ? 1 : item_w;
                U32 j = ALIGN(ow, item_w);
                tmp_align = (tmp_align < j) ? j : tmp_align;
            }
        }
        iw_align = (iw_align < tmp_align) ? tmp_align : iw_align;
    }
    iw_align = iw_align * sw;
    ih_align = ih + pt + pb;
    ih_align = ih_align - ext_h * 2;

    if (pl < ext_w) {  // if fw / 2 > pl, use pl as offset, and pad (ext_w - pl) * 2 in the end
        iw_align = iw_align + 2 * (ext_w - pl);
        ext_w = pl;
    }
    if (iw_align != iw || ih_align != ih) {
        need_pad = true;
    }
    if (ext_w != 0 || ext_h != 0) {
        need_pad = true;
    }

    if (fw == 1 && fh == 1 && ft == 1 && iw == 1 && ih == 1 && it == 1) {
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih_align, ic, ext_w, ext_h, ow, oh, fn, idt,
            idt, gclmemInputDesc, gclmemOutputDesc, need_pad));
        return SUCCESS;
    }

    if (inputGclmemFormat == DF_NCHW) {
        if (fw == fh && (fw == 1 || fw == 3 || fw == 5 || fw == 7)) {
            CHECK_STATUS(infer_gclmem_desc_nchw(iw_align, ih_align, ic * it, ext_w, ext_h, 0, 0, 0,
                idt, idt, gclmemInputDesc, NULL, need_pad));
        } else {
            ic = ALIGN(ic, 4);
            CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih_align, ic * it, ext_w, ext_h, 0, 0,
                0, idt, idt, gclmemInputDesc, NULL, need_pad));
        }
        fn = ALIGN(fn, 4);
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            0, 0, 0, 0, 0, ow, oh, fn * ot, idt, idt, NULL, gclmemOutputDesc));
        return SUCCESS;
    }

    ic = ALIGN(ic, 4);
    CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih_align, ic * it, ext_w, ext_h, 0, 0, 0, idt,
        idt, gclmemInputDesc, NULL, need_pad));
    if (fn == 1 && sw == 1 && (fw == fh) && ft == 1 && (fw == 1 || fw == 3 || fw == 5 || fw == 7)) {
        CHECK_STATUS(infer_gclmem_desc_nchw(
            0, 0, 0, 0, 0, ow, oh, fn * ot, idt, idt, NULL, gclmemOutputDesc));
    } else {
        fn = ALIGN(fn, 4);
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            0, 0, 0, 0, 0, ow, oh, fn * ot, idt, idt, NULL, gclmemOutputDesc));
    }
    return SUCCESS;
}

EE convolution_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc outputDesc,
    GCLMemDesc inputMemDesc,
    GCLMemDesc outputMemDesc,
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
    if (policy == CONVOLUTION_LIBRARY_SEARCH) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (policy == CONVOLUTION_FASTEST) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    DataType dt;
    U32 ih, iw, fn, fh, fw;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, NULL, &ih, &iw);
    tensorSelectGet(filterDesc, &dt, NULL, &fn, NULL, &fh, &fw);

    std::vector<ConvolutionForwardAlgorithm> convolutionAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecW;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    DataFormat inputGclmemFormat = inputMemDesc.memFormat;
    convolution_produce_algos_paras(inputDesc, filterDesc, convParamSpec, inputGclmemFormat,
        &convolutionAlgorithms, &algoNumIndex, &vecW, &vecC, &vecK);
    if (vecW.size() == 1) {
        forwardRunInfo->best_w[0] = vecW[0];
        forwardRunInfo->best_k[0] = vecK[0];
        forwardRunInfo->best_c[0] = vecC[0];
        forwardRunInfo->algorithm = convolutionAlgorithms[0];
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
        std::vector<GCLMemDesc> filterMemDescs;
        for (U32 i = 0; i < algoNumIndex.size(); i++) {
            U32 bytes = 0;
            ForwardRunInfoMali runInfo;
            U32 be = (i == 0) ? 0 : algoNumIndex[i - 1];
            U32 end = algoNumIndex[i];
            runInfo.algorithm = convolutionAlgorithms[i];
            for (U32 j = be; j < end; j++) {
                GCLMemDesc filterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                runInfo.best_w[0] = vecW[j];
                runInfo.best_c[0] = vecC[j];
                runInfo.best_k[0] = vecK[j];
                if (convolution_transform_filter_bytes_mali(
                        filterDesc, &runInfo, &filterMemDesc, &bytes) != SUCCESS) {
                    continue;
                }
                maxBytes = (maxBytes < bytes) ? bytes : maxBytes;
                if (convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc,
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

        if (ih == 1 && iw == 1 && fh == 1 && fw == 1) {
            U32 stride[3] = {fn, 1, 1};
            U32 offset[3] = {0, 0, 0};
            CHECK_STATUS(gclmem_set_desc_padding(
                &bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_BUF, CL_MEM_READ_WRITE));
        } else {
            U32 stride[3] = {(fn + 3) / 4, 1, 1};
            U32 offset[3] = {0, 0, 0};
            CHECK_STATUS(gclmem_set_desc_padding(
                &bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_IMG_1D, CL_MEM_READ_WRITE));
        }
        algosNum = runInfos.size();
        if (algosNum == 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        TensorDesc scaleDesc = tensor1d(DT_F32, 0);
        TensorDesc biasDesc = tensor1d(dt, fn);
        filterMemDescs[0].byteSize = maxFilterSize;
        outputMemDesc.need_pad = false;
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

        double minTimeDirect = DBL_MAX;
        double minTimeWinograd = DBL_MAX;
        double minTime = DBL_MAX;
        double winogradPicTranTime = DBL_MAX;
        double winogradOutTranTime = DBL_MAX;
        U32 runKernelBe = 0;
        U32 runKernelEnd = 0;
        ForwardRunInfoMali bestRunInfo;
        ForwardRunInfoMali bestRunInfoDirect;
        ForwardRunInfoMali bestRunInfoWinograd;
        for (U32 i = 0; i < algosNum; i++) {
            filter->desc = filterMemDescs[i];
            if (convolution_mali(handle, inputDesc, input, filterDesc, filter, convParamSpec,
                    &runInfos[i], scaleDesc, NULL, biasDesc, bias, maxBytes, tmpbuf, outputDesc,
                    output, activationMode) == SUCCESS) {
                if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_DIRECT) {
                    runKernelEnd = handle->kernelVec->size();
                    gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
                    runKernelBe = runKernelEnd;
                    if (minTimeDirect > handle->t_execute) {
                        minTimeDirect = handle->t_execute;
                        bestRunInfoDirect = runInfos[i];
                    }
                }

                if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_WINOGRAD) {
                    if (winogradPicTranTime == DBL_MAX) {
                        runKernelEnd = runKernelBe + 2;
                        gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
                        winogradPicTranTime = handle->t_execute;
                    }
                    runKernelBe += 2;
                    runKernelEnd = runKernelBe + 1;
                    gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
                    if (minTimeWinograd > handle->t_execute) {
                        minTimeWinograd = handle->t_execute;
                        bestRunInfoWinograd = runInfos[i];
                    }
                    runKernelBe += 36;
                    if (winogradOutTranTime == DBL_MAX) {
                        runKernelEnd = runKernelBe + 1;
                        gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
                        winogradOutTranTime = handle->t_execute;
                    }
                    runKernelBe = handle->kernelVec->size();
                }
            }
        }

        if (minTimeWinograd != DBL_MAX) {
            minTimeWinograd = 36 * minTimeWinograd + winogradPicTranTime + winogradOutTranTime;
        }
        minTime = minTimeDirect;
        bestRunInfo = bestRunInfoDirect;
        if (minTimeWinograd < minTime) {
            minTime = minTimeWinograd;
            bestRunInfo = bestRunInfoWinograd;
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
        convolutionAlgorithms.clear();
        runInfos.clear();
        filterMemDescs.clear();
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_off_queue_profiling(handle));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE convolution_transform_filter_bytes_mali(TensorDesc filterDesc,
    ForwardRunInfoMali_t forwardRunInfo,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (filterDesc.dt) {
        case DT_F16: {
            ret = convolution_transform_filter_bytes_mali_fp16(
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

EE convolution_transform_filter_mali(GCLHandle_t handle,
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
            ret = convolution_transform_filter_mali_fp16(
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

EE convolution_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    TensorDesc outputDesc,
    ConvolutionParamSpec convParamSpec,
    ForwardRunInfoMali_t forwardRunInfo,
    U32 *bytes)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = convolution_infer_forward_tmp_bytes_mali_fp16(
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
EE convolution_mali(GCLHandle_t handle,
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
            ret = convolution_mali_fp16(handle, inputDesc, input, filterDesc, filter, convParamSpec,
                forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
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
