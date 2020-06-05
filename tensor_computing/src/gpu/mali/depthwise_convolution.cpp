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
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/depthwise_convolution_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE depthwise_convolution_infer_output_size_mali(TensorDesc           inputDesc,
                                                TensorDesc           filterDesc,
                                                ConvolutionDesc      convDesc,
                                                TensorDesc*          outputDesc,
                                                GCLMemDesc_t         gclmemInputDesc,
                                                GCLMemDesc_t         gclmemOutputDesc,
                                                ForwardRunInfoMali_t forwardRunInfo) {
    UNUSED(forwardRunInfo);
    DataType   idt, fdt;
    DataFormat idf, fdf;
    U32 iw, ih, ic, in;
    U32 fw, fh, fc, fn;
    U32 ow, oh;
    U32 sw, sh, pw, ph, dw, dh, pr, pb;
    tensorSelectGet(inputDesc,  &idt, &idf, &in, &ic, &ih, &iw);
    tensorSelectGet(filterDesc, &fdt, &fdf, &fn, &fc, &fh, &fw);
    pw = convDesc.padding_left;
    pr = convDesc.padding_right;
    ph = convDesc.padding_top;
    pb = convDesc.padding_bottom;
    sw = convDesc.stride_w;
    sh = convDesc.stride_h;
    dw = convDesc.dilatedRate_w;
    dh = convDesc.dilatedRate_h;
    if (fw < 1  || fh < 1)        return NOT_SUPPORTED;
    if (dw != 1 || dh != 1)       return NOT_SUPPORTED;
    if (pw != ph || sw != sh)     return NOT_SUPPORTED;
    if (pb != ph || pr != pw)     return NOT_SUPPORTED;
    if ((fn & 3) != 0)     return NOT_SUPPORTED;
    ow = (iw + 2 * pw - fw) / sw + 1;
    oh = (ih + 2 * ph - fh) / sh + 1;
    if(outputDesc) *outputDesc = tensor4df(idt, idf, in, fn, oh, ow);
    
    U32 iw_align, item_w, ext_w;
    if(idf == DF_NCHW) {
        item_w = forwardRunInfo->best_w[0];
        ext_w = (fw / 2 < pw) ? pw : fw / 2;
        iw_align = (ow + item_w - 1) / item_w * item_w;
        iw_align = iw_align * sw;
        if(pw < ext_w) {
            iw_align = iw_align + 2 * (ext_w - pw);
            ext_w = pw;
        }
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih, ic, ext_w, ph, ow, oh, fn, idt, idt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE depthwise_convolution_infer_forward_algorithm_mali(GCLHandle_t          handle,
                                                      TensorDesc           inputDesc, 
                                                      TensorDesc           filterDesc, 
                                                      TensorDesc           outputDesc,
                                                      ConvolutionDesc      convDesc,
                                                      ConvolutionPolicy    policy, 
                                                      ActivationMode       depthwiseActivationMode,
                                                      ActivationMode       pointwiseActivationMode,
                                                      ForwardRunInfoMali_t forwardRunInfo) {
    
    if(forwardRunInfo == nullptr) CHECK_STATUS(NULL_POINTER);
    DepthwiseConvolutionForwardAlgorithm algorithm = (DepthwiseConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if(algorithm != DEPTHWISE_CONVOLUTION_ALGORITHM_NULL) return SUCCESS;
    if(policy == CONVOLUTION_LIBRARY_SEARCH) CHECK_STATUS(NOT_SUPPORTED);
    if(policy == CONVOLUTION_FASTEST)        CHECK_STATUS(NOT_SUPPORTED);

    if(policy == CONVOLUTION_TUNNING) {
        GCLHandle_t handle_tun;
        CHECK_STATUS(gcl_create_handle_profiling(&handle_tun));
        handle_tun->binMapPtr = handle->binMapPtr;
        GCLMem_t input  = gcl_create_gclmem();
        GCLMem_t filter = gcl_create_gclmem();
        GCLMem_t output = gcl_create_gclmem();
        GCLMem_t bias   = gcl_create_gclmem();
        GCLMem_t tmpbuf = gcl_create_gclmem();
        GCLMem_t filter_dp = gcl_create_gclmem();
        GCLMem_t bias_dp   = gcl_create_gclmem();
        GCLMem_t bias_buf  = gcl_create_gclmem();
        std::vector<DepthwiseConvolutionForwardAlgorithm> depthwiseConvolutionAlgorithms;
        DataFormat filterDF = filterDesc.df;
        if(filterDF == DF_NCHW) {
            depthwiseConvolutionAlgorithms.push_back(DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT);
        } else if (filterDF == DF_CHW_NC) {
            depthwiseConvolutionAlgorithms.push_back(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT);
            depthwiseConvolutionAlgorithms.push_back(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM);
        } else {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        U32 maxInputSize = 0;
        U32 maxOutputSize = 0;
        U32 maxFilterSize = 0;
        U32 maxBytes = 0;
        U32 algosNum = 0;
        U32 biasNum;
        U32 loop = 1;
        U32 ic, ih, iw, fn, fc, fh, fw;
        DataType dt;
        std::vector<ForwardRunInfoMali> runInfos;
        std::vector<GCLMemDesc> inputMemDescs;
        std::vector<GCLMemDesc> outputMemDescs;
        std::vector<GCLMemDesc> filterMemDescs;
        std::vector<GCLMemDesc> filterMemDescs_dp;
        std::vector<double> kernelTimeArray;
        U32 configInfo[3][8] = {{0}};
        U32 configInfo_dp[3][16] = {{0}};
        tensorSelectGet(inputDesc,  NULL, NULL, NULL, &ic, &ih, &iw);
        tensorSelectGet(filterDesc, &dt,  NULL, &fn,  &fc, &fh, &fw);

        for(auto p : depthwiseConvolutionAlgorithms) {
            ForwardRunInfoMali runInfo;
            U32 bytes;
            U32 stride[3] = {0, 0, 0};
            U32 offset[3] = {0, 0, 0};
            runInfo.algorithm = (I32)p;
            U32 configNum = 8;
            for(U32 i = 0; i < configNum; i++) {
                configInfo[0][i] = i + 1;//w
                configInfo[1][i] = 1;//c
                configInfo[2][i] = 4;//k
            }

            if(p == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) {
                for(U32 i = 0; i < configNum; i++) {
                    configInfo_dp[0][i] = i + 1;
                    configInfo_dp[1][i] = 4;
                    configInfo_dp[2][i] = 4;
                }
                if(fn % 8 == 0) {
                    for(U32 i = 0; i < configNum; i++) configInfo_dp[2][i + configNum] = 8;//k
                    loop = 2;
                }
            }

            if(p == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM) {
                for(U32 i = 0; i < configNum; i++) {
                    configInfo_dp[0][i] = i + 1;//w
                    configInfo_dp[1][i] = 1;//c
                    configInfo_dp[2][i] = 4;//k
                }
                if(fn % 8 == 0) {
                    for(U32 i = 0; i < configNum; i++) configInfo_dp[2][i + configNum] = 8;//k
                    loop = 2;
                }
            }
            
            for(U32 j = 0; j < loop; j++) {
                for(U32 i = 0; i < configNum; ++i) {
                    GCLMemDesc inputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                    GCLMemDesc outputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);;
                    GCLMemDesc filterMemDesc[2];
                    filterMemDesc[0] = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                    filterMemDesc[1] = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                    runInfo.best_w[0] = configInfo[0][i];
                    runInfo.best_c[0] = configInfo[1][i];
                    runInfo.best_k[0] = configInfo[2][i];
                    runInfo.best_w[1] = configInfo_dp[0][i];
                    runInfo.best_c[1] = configInfo_dp[1][i];
                    runInfo.best_k[1] = configInfo_dp[2][i + j * 8];
                    if(depthwise_convolution_infer_output_size_mali(inputDesc, filterDesc, convDesc, NULL, &inputMemDesc, &outputMemDesc, &runInfo) != SUCCESS) continue;
                    if(depthwise_convolution_transform_filter_bytes_mali(filterDesc, &runInfo, filterMemDesc, &bytes) != SUCCESS) continue;
                    if(maxBytes < bytes) maxBytes= bytes;
                    if(depthwise_convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc,  outputDesc, convDesc, &runInfo, &bytes) != SUCCESS) continue;
                    if(maxBytes < bytes) maxBytes= bytes;
                    if(maxInputSize  < inputMemDesc.byteSize)  maxInputSize  = inputMemDesc.byteSize;
                    if(maxOutputSize < outputMemDesc.byteSize) maxOutputSize = outputMemDesc.byteSize;
                    if(maxFilterSize < filterMemDesc[0].byteSize) maxFilterSize = filterMemDesc[0].byteSize;
                    if(maxFilterSize < filterMemDesc[1].byteSize) maxFilterSize = filterMemDesc[1].byteSize;
                    inputMemDescs.push_back(inputMemDesc);
                    outputMemDescs.push_back(outputMemDesc);
                    filterMemDescs.push_back(filterMemDesc[0]);
                    filterMemDescs_dp.push_back(filterMemDesc[1]);
                    runInfos.push_back(runInfo);
                }
            }
        }

        biasNum = (fn + 3) / 4;
        TensorDesc biasDesc = tensor1d(dt, fn);
        if (filterDF == DF_CHW_NC) {
            bias_dp->desc.memType    = GCL_MEM_IMG_1D;
            bias_dp->desc.byteSize   = biasNum * 4 * bytesOf(dt);
            bias_dp->desc.stride[0]  = biasNum;
            bias_dp->desc.stride[1]  = 1;
            bias_dp->desc.stride[2]  = 1;
            bias_dp->desc.offset[0]  = 0;
            bias_dp->desc.offset[1]  = 0;
            bias_dp->desc.offset[2]  = 0;
            bias_dp->desc.num        = biasNum;
            bias_dp->desc.memFormat  = DF_NHWC;
            gcl_create_memory(handle_tun, bias_dp);
        }

        biasNum = (fn + 7) / 8 * 8;
        if (filterDF == DF_CHW_NC) {
            bias_buf->desc.memType    = GCL_MEM_BUF;
            bias_buf->desc.byteSize   = biasNum * bytesOf(dt);
            bias_buf->desc.stride[0]  = biasNum;
            bias_buf->desc.stride[1]  = 1;
            bias_buf->desc.stride[2]  = 1;
            bias_buf->desc.offset[0]  = 0;
            bias_buf->desc.offset[1]  = 0;
            bias_buf->desc.offset[2]  = 0;
            bias_buf->desc.num        = biasNum;
            bias_buf->desc.memFormat  = DF_NHWC;
            gcl_create_memory(handle_tun, bias_buf);
            biasNum = (fc + 3) / 4;
            biasDesc = tensor1d(dt, fn + fc);
        }

        algosNum = runInfos.size();
        if(algosNum == 0) CHECK_STATUS(NOT_SUPPORTED);
        inputMemDescs[0].byteSize  = maxInputSize;
        outputMemDescs[0].byteSize = maxOutputSize;
        filterMemDescs[0].byteSize = maxFilterSize;
        input->desc  = inputMemDescs[0];
        output->desc = outputMemDescs[0];
        filter->desc = filterMemDescs[0];
        bias->desc.memType    = GCL_MEM_IMG_1D;
        bias->desc.byteSize   = biasNum * 4 * bytesOf(dt);
        bias->desc.stride[0]  = biasNum;
        bias->desc.stride[1]  = 1;
        bias->desc.stride[2]  = 1;
        bias->desc.offset[0]  = 0;
        bias->desc.offset[1]  = 0;
        bias->desc.offset[2]  = 0;
        bias->desc.num        = biasNum;
        bias->desc.memFormat  = DF_NHWC;
        tmpbuf->desc.byteSize = maxBytes;
        gcl_create_memory(handle_tun, input);
        gcl_create_memory(handle_tun, output);
        gcl_create_memory(handle_tun, filter);
        gcl_create_memory(handle_tun, bias);
        if (filterDF == DF_CHW_NC) {
            filterMemDescs_dp[0].byteSize = maxFilterSize;
            filter_dp->desc = filterMemDescs_dp[0];
            gcl_create_memory(handle_tun, filter_dp);
        }
        if(maxBytes) gcl_create_memory(handle_tun, tmpbuf);

        double minTime = DBL_MAX;
        double minTime_d_direct = DBL_MAX;
        double minTime_p_direct = DBL_MAX;
        double minTime_d_gemm   = DBL_MAX;
        double minTime_p_gemm   = DBL_MAX;
        U32 runKernelBe = 0;
        U32 runKernelEnd = 0;
        ForwardRunInfoMali bestRunInfo;
        ForwardRunInfoMali bestRunInfoDirect;
        ForwardRunInfoMali bestRunInfoGEMM;
        CHECK_STATUS(gcl_finish(handle_tun));
        for(U32 i = 0; i < algosNum; i++) {
            I32 algo = runInfos[i].algorithm;
            U32 best_wp = runInfos[i].best_w[1];
            U32 best_kp = runInfos[i].best_k[1];
            input->desc     = inputMemDescs[i];
            output->desc    = outputMemDescs[i];
            filter->desc    = filterMemDescs[i];
            filter_dp->desc = filterMemDescs_dp[i];
            GCLMem filterArray[2];
            GCLMem biasArray[2];
            filterArray[0] = *filter;
            filterArray[1] = *filter_dp;
            biasArray[0] = *bias;
            biasArray[1] = *bias_dp;
            if(algo == (I32)(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM)) biasArray[1] = *bias_buf;
            if(algo == (I32)(DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) && best_kp == 8 && best_wp > 4) continue;
            if(depthwise_convolution_mali(handle_tun, inputDesc, input, filterDesc, filterArray, convDesc, &runInfos[i], biasDesc, biasArray,
                maxBytes, tmpbuf, outputDesc, output, depthwiseActivationMode, pointwiseActivationMode) == SUCCESS) {
                if(algo == DEPTHWISE_CONVOLUTION_ALGORITHM_DIRECT) {
                    runKernelEnd = handle_tun->kernelVec.size();
                    gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
                    if(minTime > handle_tun->t_execute) {
                        minTime = handle_tun->t_execute;
                        bestRunInfo.algorithm = runInfos[i].algorithm;
                        bestRunInfo.best_w[0] = runInfos[i].best_w[0];
                        bestRunInfo.best_c[0] = runInfos[i].best_c[0];
                        bestRunInfo.best_k[0] = runInfos[i].best_k[0];
                    }
                    runKernelBe = runKernelEnd;
                }

                if(algo == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_DIRECT) {
                    if(best_kp == 4) {
                        runKernelEnd = handle_tun->kernelVec.size();
                        gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd, &kernelTimeArray);
                        if(minTime_d_direct > kernelTimeArray[0]) {
                            minTime_d_direct = kernelTimeArray[0];
                            bestRunInfoDirect.algorithm = runInfos[i].algorithm;
                            bestRunInfoDirect.best_w[0] = runInfos[i].best_w[0];
                            bestRunInfoDirect.best_c[0] = runInfos[i].best_c[0];
                            bestRunInfoDirect.best_k[0] = runInfos[i].best_k[0];
                        }
                        if(minTime_p_direct > kernelTimeArray[1]) {
                            minTime_p_direct = kernelTimeArray[1];
                            bestRunInfoDirect.best_w[1] = runInfos[i].best_w[1];
                            bestRunInfoDirect.best_c[1] = runInfos[i].best_c[1];
                            bestRunInfoDirect.best_k[1] = runInfos[i].best_k[1];
                        }
                        runKernelBe = runKernelEnd;
                        kernelTimeArray.clear();
                    }
                    if(best_kp == 8) {
                        runKernelEnd = handle_tun->kernelVec.size();
                        runKernelBe  = runKernelBe + 1;
                        gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
                        if(minTime_p_direct > handle_tun->t_execute) {
                            minTime_p_direct = handle_tun->t_execute;
                            bestRunInfoDirect.best_w[1] = runInfos[i].best_w[1];
                            bestRunInfoDirect.best_c[1] = runInfos[i].best_c[1];
                            bestRunInfoDirect.best_k[1] = runInfos[i].best_k[1];
                        }
                        runKernelBe = runKernelEnd;
                    }
                }

                if(algo == DEPTHWISE_POINTWISE_CONVOLUTION_ALGORITHM_GEMM) {
                    if(best_kp == 4) {
                        runKernelEnd = handle_tun->kernelVec.size();
                        gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd, &kernelTimeArray);
                        if(minTime_d_gemm > kernelTimeArray[0]) {
                           minTime_d_gemm = kernelTimeArray[0];
                           bestRunInfoGEMM.algorithm = runInfos[i].algorithm;
                           bestRunInfoGEMM.best_w[0] = runInfos[i].best_w[0];
                           bestRunInfoGEMM.best_c[0] = runInfos[i].best_c[0];
                           bestRunInfoGEMM.best_k[0] = runInfos[i].best_k[0];
                        }
                        if(minTime_p_gemm > kernelTimeArray[1]) {
                           minTime_p_gemm = kernelTimeArray[1];
                           bestRunInfoGEMM.best_w[1] = runInfos[i].best_w[1];
                           bestRunInfoGEMM.best_c[1] = runInfos[i].best_c[1];
                           bestRunInfoGEMM.best_k[1] = runInfos[i].best_k[1];
                           runKernelBe = runKernelEnd;
                        }
                        kernelTimeArray.clear();
                    }
                    if(best_kp == 8) {
                        runKernelEnd = handle_tun->kernelVec.size();
                        runKernelBe  = runKernelBe + 1;
                        gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
                        if(minTime_p_gemm > handle_tun->t_execute) {
                            minTime_p_gemm = handle_tun->t_execute;
                            bestRunInfoGEMM.best_w[1] = runInfos[i].best_w[1];
                            bestRunInfoGEMM.best_c[1] = runInfos[i].best_c[1];
                            bestRunInfoGEMM.best_k[1] = runInfos[i].best_k[1];
                        }
                        runKernelBe = runKernelEnd;
                    }
                }
            }
        }

        if (filterDF == DF_CHW_NC) {
            double minTime_direct = minTime_d_direct + minTime_p_direct;
            double minTime_gemm   = minTime_d_gemm + minTime_p_gemm;
            if(minTime_direct <= minTime_gemm) {
                minTime = minTime_direct;
                bestRunInfo = bestRunInfoDirect;
            } else {
                minTime = minTime_gemm;
                bestRunInfo = bestRunInfoGEMM;
            }

        }
        if(minTime    == DBL_MAX) CHECK_STATUS(NOT_SUPPORTED);
        *forwardRunInfo = bestRunInfo;
        CHECK_STATUS(gcl_finish(handle_tun));
        gcl_destroy_gclmem(input);
        gcl_destroy_gclmem(filter);
        gcl_destroy_gclmem(output);
        gcl_destroy_gclmem(bias);
        gcl_destroy_gclmem(tmpbuf);
        if (filterDF == DF_CHW_NC) {
            gcl_destroy_gclmem(filter_dp);
            gcl_destroy_gclmem(bias_dp);
            gcl_destroy_gclmem(bias_buf);
        }
        depthwiseConvolutionAlgorithms.clear();
        runInfos.clear();
        inputMemDescs.clear();
        outputMemDescs.clear();
        filterMemDescs.clear();
        filterMemDescs_dp.clear();
        kernelTimeArray.clear();
        gcl_destroy_handle(handle_tun);
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE depthwise_convolution_transform_filter_bytes_mali(TensorDesc            filterDesc, 
                                                     ForwardRunInfoMali_t  forwardRunInfo,
                                                     GCLMemDesc_t          gclmemFilterDesc,
                                                     U32*                  bytes) {
    EE ret = SUCCESS;
    switch(filterDesc.dt) {
        case DT_F16:{
           ret = depthwise_convolution_transform_filter_bytes_mali_fp16(filterDesc, forwardRunInfo, gclmemFilterDesc, bytes);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_transform_filter_mali(GCLHandle_t          handle,
                                               TensorDesc           filterDesc,
                                               GCLMem_t             filter,
                                               ForwardRunInfoMali_t forwardRunInfo,
                                               TensorDesc*          fltmemDesc,
                                               GCLMem_t             fltmem) {
    EE ret = SUCCESS;
    switch(filterDesc.dt) {
        case DT_F16:{
           ret = depthwise_convolution_transform_filter_mali_fp16(handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

EE depthwise_convolution_infer_forward_tmp_bytes_mali(TensorDesc            inputDesc, 
                                                      TensorDesc            filterDesc, 
                                                      TensorDesc            outputDesc,
                                                      ConvolutionDesc       convDesc, 
                                                      ForwardRunInfoMali_t  forwardRunInfo,
                                                      U32*                  bytes) {
    EE ret = SUCCESS;
    switch(inputDesc.dt) {
        case DT_F16:{
            ret = depthwise_convolution_infer_forward_tmp_bytes_mali_fp16(inputDesc, filterDesc, outputDesc, convDesc, forwardRunInfo, bytes);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}
EE depthwise_convolution_mali(GCLHandle_t          handle,
                              TensorDesc           inputDesc, 
                              const GCLMem_t       input,
                              TensorDesc           filterDesc, 
                              const GCLMem_t       filter,
                              ConvolutionDesc      convDesc,
                              ForwardRunInfoMali_t forwardRunInfo,
                              TensorDesc           biasDesc, 
                              const GCLMem_t       bias,
                              U32                  tmpBytes, 
                              GCLMem_t             tmpBuf,
                              TensorDesc           outputDesc, 
                              GCLMem_t             output,
                              ActivationMode       depthwiseActivationMode,
                              ActivationMode       pointwiseActivationMode) {
    EE ret = SUCCESS;
    switch(inputDesc.dt) {
        case DT_F16:{
            ret = depthwise_convolution_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, 
                    depthwiseActivationMode, pointwiseActivationMode);
            break;
        }
        case DT_I8:{
            ret = NOT_SUPPORTED;
            break;
        }
        default:
            ret = NOT_SUPPORTED;
            break;
    }
    return ret;
}

