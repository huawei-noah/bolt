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
#include "type.h"
#include "tensor_desc.h"
#include "error.h"
#include "gpu/mali/tensor_computing_mali.h"
#include "gpu/mali/fp16/convolution_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"

EE convolution_infer_output_size_mali(TensorDesc           inputDesc,
                                      TensorDesc           filterDesc,
                                      ConvolutionDesc      convDesc,
                                      TensorDesc*          outputDesc,
                                      GCLMemDesc_t         gclmemInputDesc,
                                      GCLMemDesc_t         gclmemOutputDesc,
                                      ForwardRunInfoMali_t forwardRunInfo) {
    DataType   idt, fdt;
    DataFormat idf, fdf;
    U32 iw, ih, ic, in;
    U32 fw, fh, fc, fn;
    U32 ow, oh;
    U32 sw, sh, pw, ph, dw, dh, fdw, fdh;
    U32 pr, pb;
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
    if (fw != 1 && fw != 3 && fw != 5) CHECK_STATUS(NOT_SUPPORTED);
    if (in != 1) CHECK_STATUS(NOT_SUPPORTED);
    if (fw < 1  || fh < 1)        CHECK_STATUS(NOT_SUPPORTED);
    if (dw != 1 || dh != 1)       CHECK_STATUS(NOT_SUPPORTED); 
    if (pw != ph || sw != sh)     CHECK_STATUS(NOT_SUPPORTED);
    if (pb != ph || pr != pw)     CHECK_STATUS(NOT_SUPPORTED);
    //if ((fn & 3) != 0 && fn != 1) CHECK_STATUS(NOT_SUPPORTED);
    fdw = (fw - 1) * dw + 1; 
    fdh = (fh - 1) * dh + 1; 
    ow = (iw + 2 * pw - fdw) / sw + 1;
    oh = (ih + 2 * ph - fdh) / sh + 1;
    U32 iw_align, ih_align, item_w, item_h, ext_w, ext_h;

    if(idf == DF_NCHW || (fw == 1 && fh == 1 && iw == 1 && ih == 1)) {
        if(outputDesc) *outputDesc = tensor4df(idt, DF_NCHW, in, fn, oh, ow);
        item_w = forwardRunInfo->best_w[0];
        item_h = 1;
        ih_align = ih;
        ext_h    = ph;
        if(forwardRunInfo->algorithm == CONVOLUTION_ALGORITHM_WINOGRAD) {
            item_w = 16;
            ext_h  = 0;//no need to pad h axis
        }    
        iw_align = (ow + item_w - 1) / item_w * item_w; 
        iw_align = iw_align * sw;
        ext_w = (fw / 2 < pw) ? pw : fw / 2;//if fw / 2 < pw, use pw as offset
        if(pw < ext_w) {//if fw / 2 > pw, use pw as offset, and pad (ext_w - pw) * 2 in the end
            iw_align = iw_align + 2 * (ext_w - pw);
            ext_w = pw;
        }
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih_align, ic, ext_w, ext_h, ow, oh, fn, idt, idt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    }

    if(idf == DF_NCHW_ORG_MALI) {
        if(outputDesc) *outputDesc = tensor4df(idt, DF_NCHW, in, fn, oh, ow);
        item_w = forwardRunInfo->best_w[0];
        item_h = 1;
        ih_align = ih;
        ext_h    = ph;
        iw_align = (ow + item_w - 1) / item_w * item_w; 
        iw_align = iw_align * sw;
        ext_w = (fw / 2 < pw) ? pw : fw / 2;//if fw / 2 < pw, use pw as offset
        if(pw < ext_w) {//if fw / 2 > pw, use pw as offset, and pad (ext_w - pw) * 2 in the end
            iw_align = iw_align + 2 * (ext_w - pw);
            ext_w = pw;
        }
        CHECK_STATUS(infer_gclmem_desc_nchw(iw_align, ih_align, ic, ext_w, ext_h, ow, oh, fn, idt, idt, gclmemInputDesc, NULL));
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw_align, ih_align, ic, ext_w, ext_h, ow, oh, fn, idt, idt, NULL, gclmemOutputDesc));
        return SUCCESS;
    }

    if(idf == DF_NCHWC3) {
        if(fn == 1 && fc == 3 && fw == 1){
            if(outputDesc) *outputDesc = tensor4df(idt, DF_NCHW, in, fn, oh, ow);
            item_w = 2;
            item_h = 1;
            iw_align = (iw + item_w - 1) / item_w * item_w;
            ih_align = (ih + item_h - 1) / item_h * item_h;
            CHECK_STATUS(infer_gclmem_desc_nchwc3_to_nchw(iw_align, ih_align, ic, pw, ph, ow, oh, fn, gclmemInputDesc, gclmemOutputDesc));
            return SUCCESS;
        }
    }
    return NOT_SUPPORTED;
}

EE convolution_infer_forward_algorithm_mali(GCLHandle_t          handle,
                                            TensorDesc           inputDesc, 
                                            TensorDesc           filterDesc, 
                                            ConvolutionDesc      convDesc,
                                            TensorDesc           outputDesc,
                                            ConvolutionPolicy    policy, 
                                            ActivationMode       activationMode,
                                            ForwardRunInfoMali_t forwardRunInfo){
    if(forwardRunInfo == nullptr) CHECK_STATUS(NULL_POINTER);
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if(algorithm != CONVOLUTION_ALGORITHM_NULL) return SUCCESS;
    if(policy == CONVOLUTION_LIBRARY_SEARCH) CHECK_STATUS(NOT_SUPPORTED);
    if(policy == CONVOLUTION_FASTEST)        CHECK_STATUS(NOT_SUPPORTED);
    DataType dt;
    DataFormat idf;
    U32 ic, ih, iw, fn, fh, fw, sh, sw;
    tensorSelectGet(inputDesc,  NULL, &idf, NULL,  &ic,  &ih, &iw);
    tensorSelectGet(filterDesc, &dt,  NULL, &fn,   NULL, &fh, &fw);
    sh = convDesc.stride_h;
    sw = convDesc.stride_w;
    if(idf == DF_NCHW_ORG_MALI && (ih != 1 || iw != 1 || fw != 1 || fh != 1)) {
        U32 item_w = (8 * sw - (fw - 1)) / sw;
        forwardRunInfo->best_w[0] = item_w;
        forwardRunInfo->best_k[0] = 4;
        forwardRunInfo->best_c[0] = 1;
        forwardRunInfo->algorithm = CONVOLUTION_ALGORITHM_DIRECT;
        return SUCCESS;
    }

    if(policy == CONVOLUTION_TUNNING) {
        GCLHandle_t handle_tun;
        CHECK_STATUS(gcl_create_handle_profiling(&handle_tun));
        handle_tun->binMapPtr = handle->binMapPtr;
        GCLMem_t input  = gcl_create_gclmem();
        GCLMem_t filter = gcl_create_gclmem();
        GCLMem_t output = gcl_create_gclmem();
        GCLMem_t bias   = gcl_create_gclmem();
        GCLMem_t tmpbuf = gcl_create_gclmem();
        std::vector<ConvolutionForwardAlgorithm> convolutionAlgorithms;
        U32 maxInputSize = 0;
        U32 maxOutputSize = 0;
        U32 maxFilterSize = 0;
        U32 maxBytes = 0;
        U32 algosNum = 0;
        U32 biasNum;
        std::vector<ForwardRunInfoMali> runInfos;
        std::vector<GCLMemDesc> inputMemDescs;
        std::vector<GCLMemDesc> outputMemDescs;
        std::vector<GCLMemDesc> filterMemDescs;
        U32 configInfo[3][64];
        convolutionAlgorithms.push_back(CONVOLUTION_ALGORITHM_DIRECT);
        if(fw == 3 && fh == 3 && sw == 1 && sh == 1)convolutionAlgorithms.push_back(CONVOLUTION_ALGORITHM_WINOGRAD);

        for(auto p : convolutionAlgorithms) {
            ForwardRunInfoMali runInfo;
            U32 configNum = 0;
            U32 bytes;
            U32 stride[3] = {0, 0, 0};
            U32 offset[3] = {0, 0, 0};
            runInfo.algorithm = (I32)(p);
            if(p == CONVOLUTION_ALGORITHM_DIRECT) {
                if(ih == 1 && iw == 1 && fh == 1 && fw == 1) {
                    configNum = 3;
                    if((ic & 15) != 0) configNum = 2;
                    if((ic & 7)  != 0) configNum = 1; 
                    for(U32 i = 0; i < configNum; i++) {
                        configInfo[0][i] = 1;
                        configInfo[1][i] = 1 << (2 + i);
                        configInfo[2][i] = 0;
                    }
                } else {
                    configNum = 8;
                    for(U32 i = 0; i < configNum; i++) {
                        configInfo[0][i] = i + 1;
                        configInfo[1][i] = 4;
                        configInfo[2][i] = 4;
                    }
                    if(fn % 8 == 0) {
                        for(U32 i = 0; i < 4; i++) {
                            configInfo[0][i + configNum] = i + 1;
                            configInfo[1][i + configNum] = 4;
                            configInfo[2][i + configNum] = 8;
                        }
                        configNum += 4;
                    }
                }
            }

            if(p == CONVOLUTION_ALGORITHM_WINOGRAD) {
                biasNum = fn;
                bias->desc.byteSize = biasNum * bytesOf(dt);
                bias->desc.memType    = GCL_MEM_BUF;
                configNum = 0;
                for(U32 i = 1; i <= 8; i++) {
                    for(U32 j = 1; j <= 8; j++) {
                        if(i * j <= 2) continue;
                        configInfo[0][configNum] = i;
                        configInfo[1][configNum] = 1;
                        configInfo[2][configNum] = j;
                        configNum++;
                    }
                }
            }

            for(U32 i = 0; i < configNum; ++i) {
                GCLMemDesc inputMemDesc  = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                GCLMemDesc outputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                GCLMemDesc filterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
                runInfo.best_w[0] = configInfo[0][i];
                runInfo.best_c[0] = configInfo[1][i];
                runInfo.best_k[0] = configInfo[2][i];
                if(convolution_infer_output_size_mali(inputDesc, filterDesc, convDesc, NULL, &inputMemDesc, &outputMemDesc, &runInfo) != SUCCESS) continue;
                if(convolution_transform_filter_bytes_mali(filterDesc, &runInfo, &filterMemDesc, &bytes) != SUCCESS) continue;
                if(maxBytes < bytes) maxBytes= bytes;
                if(convolution_infer_forward_tmp_bytes_mali(inputDesc, filterDesc,  outputDesc, convDesc, &runInfo, &bytes) != SUCCESS) continue;
                if(maxBytes < bytes) maxBytes= bytes;
                if(maxInputSize  < inputMemDesc.byteSize)  maxInputSize  = inputMemDesc.byteSize;
                if(maxOutputSize < outputMemDesc.byteSize) maxOutputSize = outputMemDesc.byteSize;
                if(maxFilterSize < filterMemDesc.byteSize) maxFilterSize = filterMemDesc.byteSize;
                inputMemDescs.push_back(inputMemDesc);
                outputMemDescs.push_back(outputMemDesc);
                filterMemDescs.push_back(filterMemDesc);
                runInfos.push_back(runInfo);
            }
        }

        if(ih == 1 && iw == 1 && fh == 1 && fw == 1) {
           biasNum = fn;
           bias->desc.byteSize = biasNum * bytesOf(dt);
           bias->desc.memType    = GCL_MEM_BUF;
        } else {
           biasNum = (fn + 3) / 4;
           bias->desc.byteSize = biasNum * 4 * bytesOf(dt);
           bias->desc.memType    = GCL_MEM_IMG_1D;
        }
        algosNum = runInfos.size();
        if(algosNum == 0) CHECK_STATUS(NOT_SUPPORTED);
        TensorDesc scaleDesc = tensor1d(DT_F32, 0);
        TensorDesc biasDesc  = tensor1d(dt, fn);
        inputMemDescs[0].byteSize  = maxInputSize;
        outputMemDescs[0].byteSize = maxOutputSize;
        filterMemDescs[0].byteSize = maxFilterSize;
        input->desc  = inputMemDescs[0];
        output->desc = outputMemDescs[0];
        filter->desc = filterMemDescs[0];
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
        if(maxBytes) gcl_create_memory(handle_tun, tmpbuf);

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
        for(U32 i = 0; i < algosNum; i++) {
            input->desc = inputMemDescs[i];
            output->desc = outputMemDescs[i];
            filter->desc = filterMemDescs[i];
            if(convolution_mali(handle_tun, inputDesc, input, filterDesc, filter, convDesc, &runInfos[i], scaleDesc, NULL, biasDesc, bias, 
                maxBytes, tmpbuf, outputDesc, output, activationMode) == SUCCESS) {
                if(runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_DIRECT) {
                    runKernelEnd = handle_tun->kernelVec.size();
                    gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
                    runKernelBe = runKernelEnd;
                    if(minTimeDirect > handle_tun->t_execute) {
                        minTimeDirect = handle_tun->t_execute;
                        bestRunInfoDirect = runInfos[i];
                    }
                }

                if(runInfos[i].algorithm == (I32)CONVOLUTION_ALGORITHM_WINOGRAD) {
                    if(winogradPicTranTime == DBL_MAX) {
                        runKernelEnd = runKernelBe + 2;
                        gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
                        winogradPicTranTime = handle_tun->t_execute;
                    }
                    runKernelBe += 2;
                    runKernelEnd = runKernelBe + 1;
                    gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
                    if(minTimeWinograd > handle_tun->t_execute) {
                        minTimeWinograd = handle_tun->t_execute;
                        bestRunInfoWinograd = runInfos[i];
                    }
                    runKernelBe += 36;
                    if(winogradOutTranTime == DBL_MAX) {
                        runKernelEnd = runKernelBe + 1;
                        gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
                        winogradOutTranTime = handle_tun->t_execute;
                    }
                    runKernelBe = handle_tun->kernelVec.size();
                }
            }
        }
        if(minTimeWinograd != DBL_MAX) minTimeWinograd = 36 * minTimeWinograd + winogradPicTranTime + winogradOutTranTime;
        minTime = minTimeDirect;
        bestRunInfo = bestRunInfoDirect;
        if(minTimeWinograd < minTime) {
            minTime = minTimeWinograd;
            bestRunInfo = bestRunInfoWinograd;
        }
        if(minTime == DBL_MAX) CHECK_STATUS(NOT_SUPPORTED);
        *forwardRunInfo = bestRunInfo;
        CHECK_STATUS(gcl_finish(handle_tun));
        gcl_destroy_gclmem(input);
        gcl_destroy_gclmem(filter);
        gcl_destroy_gclmem(output);
        gcl_destroy_gclmem(bias);
        gcl_destroy_gclmem(tmpbuf);
        convolutionAlgorithms.clear();
        runInfos.clear();
        inputMemDescs.clear();
        outputMemDescs.clear();
        filterMemDescs.clear();
        gcl_destroy_handle(handle_tun);
        return SUCCESS;
    
    }
    return NOT_SUPPORTED;
}

EE convolution_transform_filter_bytes_mali(TensorDesc            filterDesc, 
                                           ForwardRunInfoMali_t  forwardRunInfo,
                                           GCLMemDesc_t          gclmemFilterDesc,
                                           U32*                  bytes){
    EE ret = SUCCESS;
    switch(filterDesc.dt){
        case DT_F16:{
           ret = convolution_transform_filter_bytes_mali_fp16(filterDesc, forwardRunInfo, gclmemFilterDesc, bytes);
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

EE convolution_transform_filter_mali(GCLHandle_t          handle,
                                     TensorDesc           filterDesc,
                                     GCLMem_t             filter,
                                     ForwardRunInfoMali_t forwardRunInfo,
                                     TensorDesc*          fltmemDesc,
                                     GCLMem_t             fltmem,
                                     GCLMem_t             tmp){
    EE ret = SUCCESS;
    switch(filterDesc.dt){
        case DT_F16:{
           ret = convolution_transform_filter_mali_fp16(handle, filterDesc, filter, forwardRunInfo, fltmemDesc, fltmem, tmp);
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

EE convolution_infer_forward_tmp_bytes_mali(TensorDesc            inputDesc, 
                                            TensorDesc            filterDesc, 
                                            TensorDesc            outputDesc,
                                            ConvolutionDesc       convDesc, 
                                            ForwardRunInfoMali_t  forwardRunInfo,
                                            U32*                  bytes){
    EE ret = SUCCESS;
    switch(inputDesc.dt){
        case DT_F16:{
           ret = convolution_infer_forward_tmp_bytes_mali_fp16(inputDesc, filterDesc, outputDesc, convDesc, forwardRunInfo, bytes);
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
EE convolution_mali(GCLHandle_t          handle,
                    TensorDesc           inputDesc, 
                    const GCLMem_t       input,
                    TensorDesc           filterDesc, 
                    const GCLMem_t       filter,
                    ConvolutionDesc      convDesc,
                    ForwardRunInfoMali_t forwardRunInfo,
                    TensorDesc           scaleDesc, 
                    const GCLMem_t       scale,
                    TensorDesc           biasDesc, 
                    const GCLMem_t       bias,
                    U32                  tmpBytes, 
                    GCLMem_t             tmpBuf,
                    TensorDesc           outputDesc, 
                    GCLMem_t             output,
                    ActivationMode       activationMode){
    UNUSED(scaleDesc);
    UNUSED(scale);
    EE ret = SUCCESS;
    switch(inputDesc.dt){
        case DT_F16:{
            ret = convolution_mali_fp16(handle, inputDesc, input, filterDesc, filter, convDesc, forwardRunInfo, biasDesc, bias, tmpBytes, tmpBuf, outputDesc, output, activationMode);
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

