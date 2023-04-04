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
#include "gpu/mali/fp16/convolution_mali_fp16.h"
#include "gpu/mali/cl/kernel_option/conv_direct_opt.h"
#include "gpu/mali/cl/kernel_option/gemv_opt.h"
#include "gpu/mali/cl/kernel_option/gemm_tn_opt.h"
#include "gpu/mali/cl/kernel_option/conv_wino_opt.h"

inline void convolution_produce_algos_paras(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    DataFormat idf,
    DataFormat odf,
    GCLMemType inputMemType,
    GCLMemType outputMemType,
    std::vector<ConvolutionForwardAlgorithm> *convolutionAlgorithms,
    std::vector<U32> *algoNumIndex,
    std::vector<U32> *vecH,
    std::vector<U32> *vecC,
    std::vector<U32> *vecK)
{
    U32 in, ic, it, ih, iw, fn, ft, fh, fw, sh, sw, dh, dw;
    tensorSelectGet(inputDesc, NULL, NULL, &in, &ic, &ih, &iw, &it);
    tensorSelectGet(filterDesc, NULL, NULL, &fn, NULL, &fh, &fw, &ft);
    sh = convParamSpec.stride_h;
    sw = convParamSpec.stride_w;
    dh = convParamSpec.dilatedRate_h;
    dw = convParamSpec.dilatedRate_w;

    convolutionAlgorithms->push_back(CONVOLUTION_ALGORITHM_DIRECT);
    if (useGemvCalMode(inputDesc, convParamSpec, inputMemType, outputMemType)) {
        CHECK_STATUS(get_gemv_cal_scheme(vecH, vecC, vecK));
    } else if (useNchwCalMode(idf, fw, ic, dw, dh)) {
        CHECK_STATUS(get_conv_direct_nchw_to_nchwc4_cal_scheme(vecH, vecC, vecK, fw, sw));
    } else if (odf == DF_NCHW) {
        CHECK_STATUS(get_conv_direct_sh1_fn_spe_cal_scheme(vecH, vecC, vecK, fh, outputMemType));
    } else {
        if (dw * dh == 1) {
            CHECK_STATUS(get_conv_direct_cal_scheme(vecH, vecC, vecK, fh, sh, fn));
            if (fw * fh * ft * sw == 1 && inputMemType == GCL_MEM_BUF &&
                outputMemType == GCL_MEM_BUF) {
                CHECK_STATUS(
                    get_conv_direct_reuse_w_cal_scheme(vecH, vecC, vecK, iw, fn, inputMemType));
            }
            if (in > 1 && ft == 1 && inputMemType == GCL_MEM_BUF && outputMemType == GCL_MEM_BUF) {
                CHECK_STATUS(get_conv_direct_multi_batch_cal_scheme(vecH, vecC, vecK, fn));
            }
        } else {
            CHECK_STATUS(get_conv_direct_dila_cal_scheme(vecH, vecC, vecK, dh, fn));
        }
    }
    algoNumIndex->push_back(vecH->size());

    if (fw == 3 && fh == 3 && ft == 1 && sw == 1 && sh == 1 && dw == 1 && dh == 1 &&
        idf != DF_NCHW && odf != DF_NCHW) {
        convolutionAlgorithms->push_back(CONVOLUTION_ALGORITHM_WINOGRAD);
        GCLMemType mt = (check_qualcomm_device()) ? GCL_MEM_IMG_3D : GCL_MEM_BUF;
        get_gemm_tn_cal_scheme(vecH, vecC, vecK, mt, mt, GCL_MEM_BUF);
        algoNumIndex->push_back(vecH->size());
    }

    if (sw == 1 && sh == 1 && dw == 1 && dh == 1 && fw * fh > 1 && ft == 1 && idf != DF_NCHW &&
        odf != DF_NCHW && ic > iw * 4 && ic > ih * 4) {
        convolutionAlgorithms->push_back(CONVOLUTION_ALGORITHM_INVGEMM);
        CHECK_STATUS(get_conv_direct_cal_scheme(vecH, vecC, vecK, 1, 1, fn));
        algoNumIndex->push_back(vecH->size());
    }
}

inline void infer_align_val(ConvolutionForwardAlgorithm algo,
    U32 algoNum,
    bool useNchwMode,
    std::vector<U32> vecH,
    U32 ow,
    U32 oh,
    U32 in,
    U32 *w_align,
    U32 *h_align,
    U32 *n_align)
{
    U32 w_val = *w_align;
    U32 h_val = *h_align;
    U32 n_val = *n_align;
    if (useNchwMode) {
        if (algo == CONVOLUTION_ALGORITHM_WINOGRAD) {
            w_val = UNI_ALIGN(w_val, 4);
            h_val = UNI_ALIGN(h_val, 4);
        } else {
            for (U32 i = 0; i < algoNum; i++) {
                U32 item_w = vecH[i];
                w_val = std::max(UNI_ALIGN(ow, item_w), w_val);
            }
        }
    } else {
        for (U32 i = 0; i < algoNum; i++) {
            U32 item_h = vecH[i];
            if ((item_h >> 8) > 0) {
                item_h = 1;
            }
            if ((item_h >> 4) > 0) {
                U32 item_n = item_h >> 4;
                item_h = item_h & 15;
                n_val = std::max(UNI_ALIGN(in, item_n), n_val);
            }
            h_val = std::max(UNI_ALIGN(oh, item_h), h_val);
        }
    }
    *w_align = w_val;
    *h_align = h_val;
    *n_align = n_val;
}

EE convolution_padding_input_mali(TensorDesc inputDesc,
    TensorDesc filterDesc,
    ConvolutionParamSpec convParamSpec,
    TensorDesc *outputDesc,
    OclMemory *inputMem,
    OclMemory *outputMem)
{
    if (inputMem == nullptr || outputMem == nullptr || outputDesc == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    U32 sw = convParamSpec.stride_w;
    U32 sh = convParamSpec.stride_h;
    U32 dw = convParamSpec.dilatedRate_w;
    U32 dh = convParamSpec.dilatedRate_h;
    if (sw != 1 && sw != 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    if (sh != 1 && sh != 2) {
        CHECK_STATUS(NOT_SUPPORTED);
    }
    DataFormat idf, odf;
    U32 fw, fh, fc, fn, ft;
    idf = inputDesc.df;
    U32 in = inputDesc.dims[inputDesc.nDims - 1];
    U32 ic = inputDesc.dims[inputDesc.nDims - 2];
    tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw, &ft);
    GCLMemType imt = inputMem->gclMemType();
    GCLMemType omt = outputMem->gclMemType();

    odf = DF_NCHWC4;
    if (idf == DF_NCHWC4 && fn * sh * ft == 1 && omt == GCL_MEM_BUF) {  //spe case for fn = 1
        odf = DF_NCHW;
    }
    (*outputDesc).df = odf;

    bool useNchwMode = useNchwCalMode(idf, fw, ic, dw, dh);
    std::vector<ConvolutionForwardAlgorithm> convolutionAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    convolution_produce_algos_paras(inputDesc, filterDesc, convParamSpec, idf, odf, imt, omt,
        &convolutionAlgorithms, &algoNumIndex, &vecH, &vecC, &vecK);

    U32 ow = (*outputDesc).dims[0];
    U32 oh = (*outputDesc).dims[1];
    U32 w_align = ow;
    U32 h_align = oh;
    U32 n_align = in;
    for (U32 i = 0; i < convolutionAlgorithms.size(); i++) {
        infer_align_val(convolutionAlgorithms[i], algoNumIndex[i], useNchwMode, vecH, ow, oh, in,
            &w_align, &h_align, &n_align);
    }

    U32 pl = 0;
    U32 pr = 0;
    U32 pt = 0;
    U32 pb = 0;
    U32 pf = 0;
    U32 pa = 0;
    if (useNchwMode || idf == DF_NCHWC4) {
        calPaddingVal(inputDesc, filterDesc, convParamSpec, w_align, h_align, n_align, useNchwMode,
            &pl, &pr, &pt, &pb, &pa, &pf);
    }
    inputMem->padding(pl, pr, pt, pb, pf, pa);
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
    ActivationParamSpec activationMode,
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
    GCLMemType imt = inputMemDesc.memType;
    GCLMemType omt = outputMemDesc.memType;
    std::vector<TensorDesc> filterDescVec(1, filterDesc);
    std::vector<I32> flag = build_conv_forward_algorithm_flag(
        inputDesc, filterDescVec, OT_Conv, imt, omt, convParamSpec);
    if (gcl_get_runInfo_from_cache(handle, flag, forwardRunInfo)) {
        return SUCCESS;
    }
    DataType dt;
    U32 ic, ih, iw, fn, fh, fw, ft;
    tensorSelectGet(inputDesc, NULL, NULL, NULL, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, &dt, NULL, &fn, NULL, &fh, &fw, &ft);
    U32 sh = convParamSpec.stride_h;
    U32 dh = convParamSpec.dilatedRate_h;
    U32 dw = convParamSpec.dilatedRate_w;
    bool useNchwMode = useNchwCalMode(inputDesc.df, fw, ic, dw, dh);

    std::vector<ConvolutionForwardAlgorithm> convolutionAlgorithms;
    std::vector<U32> algoNumIndex;
    std::vector<U32> vecH;
    std::vector<U32> vecC;
    std::vector<U32> vecK;
    DataFormat idf = inputDesc.df;
    DataFormat odf = outputDesc.df;
    convolution_produce_algos_paras(inputDesc, filterDesc, convParamSpec, idf, odf, imt, omt,
        &convolutionAlgorithms, &algoNumIndex, &vecH, &vecC, &vecK);
    if (vecH.size() == 1) {
        forwardRunInfo->best_h[0] = vecH[0];
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
        GCLMem_t filterImg = gcl_create_gclmem();
        GCLMem_t output = gcl_create_gclmem();
        GCLMem_t bias = gcl_create_gclmem();
        GCLMem_t biasbuf = gcl_create_gclmem();
        GCLMem_t tmpbuf = gcl_create_gclmem();
        GCLMem_t tmpImgA = gcl_create_gclmem();
        GCLMem_t tmpImgB = gcl_create_gclmem();
        U32 maxFilterSize = 0;
        U32 maxBytes[7] = {0};
        U32 algosNum = 0;
        std::vector<ForwardRunInfoMali> runInfos;
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        TensorDesc ftmDesc;
        for (U32 i = 0; i < algoNumIndex.size(); i++) {
            U32 bytes[7] = {0};
            ForwardRunInfoMali runInfo;
            U32 be = (i == 0) ? 0 : algoNumIndex[i - 1];
            U32 end = algoNumIndex[i];
            runInfo.algorithm = convolutionAlgorithms[i];
            for (U32 j = be; j < end; j++) {
                TensorDesc desc;
                runInfo.best_h[0] = vecH[j];
                runInfo.best_c[0] = vecC[j];
                runInfo.best_k[0] = vecK[j];
                if (convolution_transform_filter_bytes_mali(filterDesc, &runInfo, &desc) != SUCCESS) {
                    continue;
                }
                if (convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, outputDesc,
                        convParamSpec, &runInfo, bytes) != SUCCESS) {
                    continue;
                }
                if (tensorNumBytes(desc) > maxFilterSize) {
                    ftmDesc = desc;
                    maxFilterSize = tensorNumBytes(desc);
                }
                for (U32 i = 0; i < 7; i++) {
                    maxBytes[i] = (maxBytes[i] < bytes[i]) ? bytes[i] : maxBytes[i];
                }
                runInfos.push_back(runInfo);
            }
        }

        stride[0] = (fn + 3) / 4;
        stride[1] = 1;
        stride[2] = 1;
        CHECK_STATUS(gclmem_set_desc_padding(
            &bias->desc, stride, offset, dt, DF_NHWC, GCL_MEM_IMG_1D, CL_MEM_READ_WRITE));
        stride[0] = fn;
        CHECK_STATUS(gclmem_set_desc_padding(
            &biasbuf->desc, stride, offset, dt, DF_NHWC, GCL_MEM_BUF, CL_MEM_READ_WRITE));
        stride[0] = ftmDesc.dims[0];
        stride[1] = ftmDesc.dims[1];
        stride[2] = ftmDesc.dims[2];
        CHECK_STATUS(gclmem_set_desc_padding(
            &filter->desc, stride, offset, dt, DF_NCHW, GCL_MEM_BUF, CL_MEM_READ_WRITE));
        bool useImg = check_qualcomm_device();
        bool useWinoFltImg = false;
        if (useImg) {
            if (CHECK_MEET_IMAGE_LIMITS(stride[0] / 4, stride[1], stride[2])) {
                stride[0] = stride[0] / 4;
                CHECK_STATUS(gclmem_set_desc_padding(&filterImg->desc, stride, offset, dt, DF_NCHW,
                    GCL_MEM_IMG_3D, CL_MEM_READ_WRITE));
                useWinoFltImg = true;
            }
        }

        algosNum = runInfos.size();
        if (algosNum == 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        TensorDesc scaleDesc = tensor1d(DT_F32, 0);
        TensorDesc biasDesc = tensor1d(dt, fn);
        outputMemDesc.need_pad = false;
        input->desc = inputMemDesc;
        output->desc = outputMemDesc;
        gcl_create_memory(handle, input);
        gcl_create_memory(handle, output);
        gcl_create_memory(handle, filter);
        gcl_create_memory(handle, bias);
        gcl_create_memory(handle, biasbuf);
        std::vector<GCLMem_t> tmpDir(1, NULL);
        std::vector<GCLMem_t> tmpInv(1, NULL);
        std::vector<GCLMem_t> tmpWino(3, NULL);
        std::vector<GCLMem_t> tmp(3, NULL);
        if (maxBytes[0]) {
            tmpbuf->desc.byteSize = maxBytes[0];
        } else {
            tmpbuf->desc.byteSize = 128;
        }
        gcl_create_memory(handle, tmpbuf);
        tmpDir[0] = tmpbuf;
        tmpWino[0] = tmpbuf;
        tmpInv[0] = tmpbuf;
        if (check_qualcomm_device() && maxBytes[1] > 0 && maxBytes[2] > 0 && maxBytes[3] > 0) {
            tmpImgA->desc.memType = GCL_MEM_IMG_3D;
            tmpImgA->desc.stride[0] = maxBytes[1];
            tmpImgA->desc.stride[1] = maxBytes[2];
            tmpImgA->desc.stride[2] = maxBytes[3];
            gcl_create_memory(handle, tmpImgA);
            tmpDir[0] = tmpImgA;
            tmpWino[1] = tmpImgA;
        }
        if (check_qualcomm_device() && maxBytes[4] > 0 && maxBytes[5] > 0 && maxBytes[6] > 0) {
            tmpImgB->desc.memType = GCL_MEM_IMG_3D;
            tmpImgB->desc.stride[0] = maxBytes[4];
            tmpImgB->desc.stride[1] = maxBytes[5];
            tmpImgB->desc.stride[2] = maxBytes[6];
            gcl_create_memory(handle, tmpImgB);
            tmpWino[2] = tmpImgB;
        }

        double minTime = DBL_MAX;
        double minTimeWinograd = DBL_MAX;
        double winogradPicTranTime = DBL_MAX;
        double winogradOutTranTime = DBL_MAX;
        double minTimeInvGemm = DBL_MAX;
        double invGemmCol2ImgTime = DBL_MAX;
        U32 runKernelBe = 0;
        U32 runKernelEnd = 0;
        ForwardRunInfoMali bestRunInfo, bestRunInfoWinograd, bestRunInfoInvGemm;
        UNI_MEMSET(&bestRunInfo, 0, sizeof(bestRunInfo));
        GCLMem_t fltMem = filter;
        tmp[0] = tmpDir[0];
        for (U32 i = 0; i < algosNum; i++) {
            GCLMem_t biasMem = (runInfos[i].best_k[0] == 0) ? biasbuf : bias;
            if (check_qualcomm_device()) {
                if (input->desc.memType == GCL_MEM_BUF && tmpbuf->desc.memType == GCL_MEM_BUF) {
                    if (ft > 1 && runInfos[i].best_h[0] >= 7 && !useNchwMode) {
                        break;
                    }
                    if (sh == 2 && dh > 2 && runInfos[i].best_h[0] >= 6) {
                        break;
                    }
                }
                if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_DIRECT) {
                    fltMem = filter;
                    tmp[0] = tmpDir[0];
                } else if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_INVGEMM) {
                    fltMem = filter;
                    tmp[0] = tmpInv[0];
                } else if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_WINOGRAD) {
                    if (useWinoFltImg) {
                        gcl_create_memory(handle, filterImg);
                        useWinoFltImg = false;
                    }
                    fltMem = filterImg;
                    for (U32 i = 0; i < 3; i++) {
                        tmp[i] = tmpWino[i];
                    }
                }
            }
            if (convolution_mali(handle, inputDesc, input, filterDesc, fltMem, convParamSpec,
                    &runInfos[i], scaleDesc, NULL, biasDesc, biasMem, maxBytes[0], tmp, outputDesc,
                    output, activationMode) == SUCCESS) {
                if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_DIRECT) {
                    runKernelEnd = handle->kernelVec->size();
                    gcl_run_kernelVec_timing(handle, runKernelEnd - 1, runKernelEnd);
                    if (minTime > handle->t_execute) {
                        minTime = handle->t_execute;
                        bestRunInfo = runInfos[i];
                    }
                    runKernelBe = runKernelEnd;
                }

                if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_WINOGRAD) {
                    if (winogradPicTranTime == DBL_MAX) {
                        runKernelEnd = (inputDesc.df == DF_NCHW) ? runKernelBe + 1 : runKernelBe + 2;
                        gcl_run_kernelVec_timing(handle, runKernelBe, runKernelEnd);
                        winogradPicTranTime = handle->t_execute;
                    }
                    runKernelEnd = handle->kernelVec->size();
                    if (winogradOutTranTime == DBL_MAX) {
                        gcl_run_kernelVec_timing(handle, runKernelEnd - 1, runKernelEnd);
                        winogradOutTranTime = handle->t_execute;
                    }
                    gcl_run_kernelVec_timing(handle, runKernelEnd - 2, runKernelEnd - 1);
                    if (minTimeWinograd > handle->t_execute) {
                        minTimeWinograd = handle->t_execute;
                        bestRunInfoWinograd = runInfos[i];
                    }
                    runKernelBe = runKernelEnd;
                }
                if (runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_INVGEMM) {
                    runKernelEnd = handle->kernelVec->size();
                    if (invGemmCol2ImgTime == DBL_MAX) {
                        gcl_run_kernelVec_timing(handle, runKernelEnd - 1, runKernelEnd);
                        invGemmCol2ImgTime = handle->t_execute;
                    }
                    gcl_run_kernelVec_timing(handle, runKernelEnd - 2, runKernelEnd - 1);
                    if (minTimeInvGemm > handle->t_execute) {
                        minTimeInvGemm = handle->t_execute;
                        bestRunInfoInvGemm = runInfos[i];
                    }
                    runKernelBe = runKernelEnd;
                }
            }
        }

        if (minTimeWinograd != DBL_MAX) {
            minTimeWinograd = minTimeWinograd + winogradPicTranTime + winogradOutTranTime;
        }
        if (minTimeWinograd < minTime) {
            minTime = minTimeWinograd;
            bestRunInfo = bestRunInfoWinograd;
        }
        if (minTimeInvGemm != DBL_MAX) {
            minTimeInvGemm = minTimeInvGemm + invGemmCol2ImgTime;
        }
        if (minTimeInvGemm < minTime) {
            minTime = minTimeInvGemm;
            bestRunInfo = bestRunInfoInvGemm;
        }
        if (minTime == DBL_MAX) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        *forwardRunInfo = bestRunInfo;
        gcl_set_runInfo_to_cache(handle, flag, bestRunInfo);
        CHECK_STATUS(gcl_finish(handle));
        gcl_destroy_gclmem(input);
        gcl_destroy_gclmem(filter);
        gcl_destroy_gclmem(filterImg);
        gcl_destroy_gclmem(output);
        gcl_destroy_gclmem(bias);
        gcl_destroy_gclmem(biasbuf);
        gcl_destroy_gclmem(tmpbuf);
        gcl_destroy_gclmem(tmpImgA);
        gcl_destroy_gclmem(tmpImgB);
        convolutionAlgorithms.clear();
        runInfos.clear();
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        CHECK_STATUS(gcl_clean_programMap(handle));
        CHECK_STATUS(gcl_off_queue_profiling(handle));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE convolution_transform_filter_bytes_mali(
    TensorDesc filterDesc, ForwardRunInfoMali_t forwardRunInfo, TensorDesc *ftmDesc)
{
    return convolution_transform_filter_bytes_mali_fp16(filterDesc, forwardRunInfo, ftmDesc);
}

EE convolution_transform_filter_mali(GCLHandle_t handle,
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
            ret = convolution_transform_filter_mali_fp16(
                handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem, tmp);
            break;
        }
        default:
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
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = convolution_infer_forward_tmp_bytes_mali_fp16(
                inputDesc, filterDesc, outputDesc, convParamSpec, forwardRunInfo, bytes);
            break;
        }
        default:
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
    std::vector<GCLMem_t> tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ActivationParamSpec activationMode)
{
    UNUSED(scaleDesc);
    UNUSED(scale);
    EE ret = NOT_SUPPORTED;
    switch (inputDesc.dt) {
        case DT_F16:
        case DT_F32: {
            ret = convolution_mali_fp16(handle, inputDesc, input, filterDesc, filter, convParamSpec,
                forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output,
                activationMode);
            break;
        }
        default:
            break;
    }
    return ret;
}
