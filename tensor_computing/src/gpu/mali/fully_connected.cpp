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
#include "gpu/mali/fp16/fully_connected_mali_fp16.h"
#include "gpu/mali/infer_gclmem_desc_mali.h"
inline EE fully_connected_checkpara_mali(GCLHandle_t            handle,
                                         TensorDesc             inputDesc, 
                                         const GCLMem_t         input,
                                         TensorDesc             filterDesc, 
                                         std::vector<GCLMem_t>* filter,
                                         std::vector<GCLMem_t>* bias,
                                         TensorDesc             outputDesc, 
                                         std::vector<GCLMem_t>* output) {
    if(nullptr == handle || nullptr == input || nullptr == filter || nullptr == output || nullptr == bias) return NULL_POINTER;
    if(filter->size() != output->size() || filter->size() != bias->size() || bias->size() == 0) return NOT_MATCH;
    for(U32 i = 0; i < filter->size(); ++i) {
        if(nullptr == (*filter)[i] || nullptr == (*output)[i] || nullptr == (*bias)[i]) return NULL_POINTER;
    }
    if(inputDesc.df == DF_NCHW) {
        U32 in, ic, ih, iw;
        U32 fn, fc, fh, fw;
        U32 oc;
        CHECK_STATUS(tensorSelectGet(inputDesc,  NULL, NULL, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw));
        CHECK_STATUS(tensorSelectGet(outputDesc, NULL, NULL, NULL, &oc, NULL, NULL));
        if(filterDesc.df != DF_NCHW)                    return NOT_SUPPORTED;
        if(input->desc.memFormat != DF_NCWHC4)          return NOT_SUPPORTED;
        if((*filter)[0]->desc.memFormat != DF_NCWHN4C4) return NOT_SUPPORTED;
        if((*output)[0]->desc.memFormat != DF_NCWHC4)   return NOT_SUPPORTED;
        if(in > 1)   return NOT_SUPPORTED;
        if(filter->size() > 1) return NOT_SUPPORTED;
        if(fw != iw) return NOT_MATCH;
        if(fh != ih) return NOT_MATCH;
        if(fc != ic) return NOT_MATCH;
        if(fn != oc) return NOT_MATCH;
    }
    if(inputDesc.df == DF_MKT) {
        U32 k;
        U32 fw, fh, fc, fn;
        k = inputDesc.dims[1];
        CHECK_STATUS(tensorSelectGet(filterDesc, NULL, NULL, &fn, &fc, &fh, &fw));
        if(fh != 1 || fw != 1) return NOT_MATCH;
        if(k != fc) return NOT_MATCH;
    }
    return SUCCESS; 
}
EE fully_connected_infer_output_size_mali(TensorDesc           inputDesc,
                                          TensorDesc           filterDesc,
                                          TensorDesc*          outputDesc,
                                          GCLMemDesc_t         gclmemInputDesc,
                                          GCLMemDesc_t         gclmemOutputDesc,
                                          ForwardRunInfoMali_t forwardRunInfo) {
    U32 fn;
    tensorSelectGet(filterDesc, NULL, NULL, &fn,  NULL, NULL, NULL);
    if(inputDesc.df == DF_NCHW) {
        DataType   idt;
        DataFormat idf;
        U32 iw, ih, ic, in;
        tensorSelectGet(inputDesc, &idt, &idf, &in,  &ic,  &ih,  &iw);
        if(outputDesc) *outputDesc = tensor4df(idt, idf, in, fn, 1, 1);
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(iw, ih, ic, 0, 0, 1, 1, fn, idt, idt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    } else if(inputDesc.df == DF_MKT) {
        DataType dt;
        U32 m, k, t;
        get_nlp_mkt_val(inputDesc, &dt, &m, &k, &t);
        if(outputDesc) {
            *outputDesc = inputDesc;
            (*outputDesc).dims[1] = fn;
        }    
        U32 item_wh = forwardRunInfo->best_w[0];
        U32 igw, igh, igc;
        U32 ogw, ogh, ogc;
        map_nlp_mkt_to_ncwhc4(m, k, (t + item_wh - 1) / item_wh * item_wh, &igw, &igh, &igc);
        map_nlp_mkt_to_ncwhc4(m, fn, t, &ogw, &ogh, &ogc);
        igc = igc * 4;
        ogc = ogc * 4;
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(igw, igh, igc, 0, 0, ogw, ogh, ogc, dt, dt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    }
    CHECK_STATUS(NOT_SUPPORTED);
    return NOT_SUPPORTED;
}

EE fully_connected_infer_forward_algorithm_mali(GCLHandle_t              handle,
                                                TensorDesc               inputDesc,
                                                TensorDesc               filterDesc,
                                                std::vector<TensorDesc>  outputDescs,
                                                ForwardRunInfoMali_t     forwardRunInfo) {
    if(forwardRunInfo == nullptr) CHECK_STATUS(NULL_POINTER);
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if(algorithm != CONVOLUTION_ALGORITHM_NULL) return SUCCESS;
    DataType dt;
    U32 iw, ih, ic, fw, fh, fn;
    tensorSelectGet(filterDesc, &dt, NULL, &fn, NULL,  &fh, &fw);
    if(inputDesc.df == DF_NCHW) {
        tensorSelectGet(inputDesc,  NULL, NULL, NULL, &ic, &ih, &iw);
        if(ih != 1 || iw != 1 || fh != 1 || fw != 1) {
            U32 item_w = (64 + ih - 1) / ih;
            item_w = (item_w > iw) ? iw : item_w;
            forwardRunInfo->best_w[0] = item_w;
            forwardRunInfo->best_c[0] = 4;
            forwardRunInfo->best_k[0] = 4;
            forwardRunInfo->algorithm = CONVOLUTION_ALGORITHM_DIRECT;
            return SUCCESS;
        }
    }

    GCLHandle_t handle_tun;
    CHECK_STATUS(gcl_create_handle_profiling(&handle_tun));
    handle_tun->binMapPtr = handle->binMapPtr;
    U32 sliceNum = outputDescs.size();
    GCLMem_t input  = gcl_create_gclmem();
    GCLMem_t tmpbuf = gcl_create_gclmem();
    std::vector<GCLMem_t> filter;
    std::vector<GCLMem_t> bias;
    std::vector<GCLMem_t> output;
    for(U32 i = 0; i < sliceNum; ++i) {
        GCLMem_t filterTmp = gcl_create_gclmem();
        GCLMem_t biasTmp   = gcl_create_gclmem();
        GCLMem_t outTmp    = gcl_create_gclmem();
        filter.push_back(filterTmp);
        bias.push_back(biasTmp);
        output.push_back(outTmp);
    }

    std::vector<ForwardRunInfoMali> runInfos;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)CONVOLUTION_ALGORITHM_DIRECT;
    std::vector<GCLMemDesc> inputMemDescs;
    std::vector<GCLMemDesc> outputMemDescs;
    std::vector<GCLMemDesc> filterMemDescs;
    U32 configInfo[3][64];
    U32 configNum, bytes;
    U32 maxBytes = 0;
    U32 maxInputSize = 0;
    U32 maxOutputSize = 0;
    U32 maxFilterSize = 0;
    U32 biasNum;
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};

    if(inputDesc.df == DF_NCHW) {
        configNum = 3;
        if((ic & 15) != 0) configNum = 2;
        if((ic & 7)  != 0) configNum = 1;
        for(U32 i =0; i < configNum; ++i) {
            configInfo[0][i] = 1;
            configInfo[1][i] = 1 << (2 + i);
            configInfo[2][i] = 0;
        }
    } else if(inputDesc.df == DF_MKT) {
        configNum = 8;
        bool align8 = true;
        for(U32 i = 0; i < configNum; i++) {
            configInfo[0][i] = i + 1;
            configInfo[1][i] = 4;
            configInfo[2][i] = 4;
            if(outputDescs[i].dims[1] % 8 != 0) align8 = false;
        }
        if(align8) {
            for(U32 i = 0; i < 4; i++) {
                configInfo[0][i + configNum] = i + 1;
                configInfo[1][i + configNum] = 4;
                configInfo[2][i + configNum] = 8;
            }
            configNum += 4;
        }
    } else {return NOT_SUPPORTED;}

    for(U32 i = 0; i < configNum; ++i) {
        GCLMemDesc inputMemDesc  = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc outputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        GCLMemDesc filterMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        runInfo.best_w[0] = configInfo[0][i];
        runInfo.best_c[0] = configInfo[1][i];
        runInfo.best_k[0] = configInfo[2][i];
        if(fully_connected_infer_output_size_mali(inputDesc, filterDesc, NULL, &inputMemDesc, &outputMemDesc, &runInfo) != SUCCESS) continue;
        if(fully_connected_transform_filter_bytes_mali(filterDesc, &filterMemDesc, &bytes, &runInfo) != SUCCESS) continue;
        if(maxBytes < bytes) maxBytes= bytes;
        if(fully_connected_infer_forward_tmp_bytes_mali(inputDesc, filterDesc, &bytes, &runInfo) != SUCCESS) continue;
        if(maxBytes < bytes) maxBytes= bytes;
        if(maxInputSize  < inputMemDesc.byteSize)  maxInputSize  = inputMemDesc.byteSize;
        if(maxOutputSize < outputMemDesc.byteSize) maxOutputSize = outputMemDesc.byteSize;
        if(maxFilterSize < filterMemDesc.byteSize) maxFilterSize = filterMemDesc.byteSize;
        inputMemDescs.push_back(inputMemDesc);
        outputMemDescs.push_back(outputMemDesc);
        filterMemDescs.push_back(filterMemDesc);
        runInfos.push_back(runInfo);
    }

    if(inputDesc.df == DF_NCHW) {
        biasNum = fn;
        bias[0]->desc.byteSize = biasNum * bytesOf(dt);
        bias[0]->desc.memType  = GCL_MEM_BUF;
    }

    if(inputDesc.df == DF_MKT) {
        biasNum = (fn + 3) / 4;
        bias[0]->desc.byteSize = biasNum * 4 * bytesOf(dt);
        bias[0]->desc.memType  = GCL_MEM_IMG_1D;
    }
    U32 algosNum = runInfos.size();
    if(algosNum == 0) CHECK_STATUS(NOT_SUPPORTED);
    TensorDesc biasDesc  = tensor1d(dt, fn);
    inputMemDescs[0].byteSize  = maxInputSize;
    outputMemDescs[0].byteSize = maxOutputSize;
    filterMemDescs[0].byteSize = maxFilterSize;
    input->desc  = inputMemDescs[0];
    output[0]->desc = outputMemDescs[0];
    filter[0]->desc = filterMemDescs[0];
    bias[0]->desc.stride[0]  = biasNum;
    bias[0]->desc.stride[1]  = 1;
    bias[0]->desc.stride[2]  = 1;
    bias[0]->desc.offset[0]  = 0;
    bias[0]->desc.offset[1]  = 0;
    bias[0]->desc.offset[2]  = 0;
    bias[0]->desc.num        = biasNum;
    bias[0]->desc.memFormat  = DF_NHWC;
    tmpbuf->desc.byteSize = maxBytes;
    gcl_create_memory(handle_tun, input);
    for(U32 i = 0; i < sliceNum; ++i) {
        filter[i]->desc = filter[0]->desc;
        bias[i]->desc   = bias[0]->desc;
        output[i]->desc = output[0]->desc;
        filter[i]->desc.has_alloc = false;
        bias[i]->desc.has_alloc   = false;
        output[i]->desc.has_alloc = false;
        gcl_create_memory(handle_tun, filter[i]);
        gcl_create_memory(handle_tun, bias[i]);
        gcl_create_memory(handle_tun, output[i]);
    }
    if(maxBytes) gcl_create_memory(handle_tun, tmpbuf);

    U32 runKernelBe = 0;
    U32 runKernelEnd = 0;
    double minTime = DBL_MAX;
    ForwardRunInfoMali bestRunInfo;
    for(U32 i = 0; i < algosNum; i++) {
        input->desc = inputMemDescs[i];
        output[0]->desc = outputMemDescs[i];
        filter[0]->desc = filterMemDescs[i];
        if(sliceNum > 1) {
            U32 item_k = runInfos[i].best_k[0];
            for(U32 j = 0; j < sliceNum; j++) {
                U32 fn = outputDescs[j].dims[1];
                output[j]->desc.stride[2] = (fn + 3) / 4;
                filter[j]->desc.stride[2] = (fn + item_k - 1) / item_k;
                biasNum = (inputDesc.df == DF_NCHW) ? fn : (fn + 3) / 4;
                bias[j]->desc.stride[0] = biasNum;
            }
        }
        if(fully_connected_mali(handle_tun, inputDesc, input, filterDesc, &filter, biasDesc, &bias, 
            maxBytes, tmpbuf, outputDescs[0], &output, &runInfos[i]) == SUCCESS) {
            runKernelEnd = handle_tun->kernelVec.size();
            gcl_run_kernelVec_timing(handle_tun, runKernelBe, runKernelEnd);
            runKernelBe = runKernelEnd;
            if(minTime > handle_tun->t_execute) {
                minTime = handle_tun->t_execute;
                bestRunInfo = runInfos[i];
            }
        }
    }
    if(minTime == DBL_MAX) CHECK_STATUS(NOT_SUPPORTED);
    *forwardRunInfo = bestRunInfo;
    CHECK_STATUS(gcl_finish(handle_tun));
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(tmpbuf);
    for(auto p : filter) gcl_destroy_gclmem(p);
    for(auto p : output) gcl_destroy_gclmem(p);
    for(auto p : bias)   gcl_destroy_gclmem(p);
    runInfos.clear();
    inputMemDescs.clear();
    outputMemDescs.clear();
    filterMemDescs.clear();
    gcl_destroy_handle(handle_tun);
    return SUCCESS;
}
EE fully_connected_transform_filter_bytes_mali(TensorDesc           filterDesc, 
                                               GCLMemDesc_t         gclmemFilterDesc,
                                               U32*                 bytes,
                                               ForwardRunInfoMali_t forwardRunInfo) {
    EE ret = SUCCESS;
    switch(filterDesc.dt) {
        case DT_F16:{
           ret = fully_connected_transform_filter_bytes_mali_fp16(filterDesc, gclmemFilterDesc, bytes, forwardRunInfo);
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

EE fully_connected_transform_filter_mali(GCLHandle_t            handle,
                                         TensorDesc             filterDesc,
                                         GCLMem_t               filter,
                                         TensorDesc*            fltmemDesc,
                                         std::vector<GCLMem_t>* fltmem,
                                         ForwardRunInfoMali_t   forwardRunInfo) {
    EE ret = SUCCESS;
    switch(filterDesc.dt) {
        case DT_F16:{
           ret = fully_connected_transform_filter_mali_fp16(handle, filterDesc, filter, fltmemDesc, *fltmem, forwardRunInfo);
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

EE fully_connected_infer_forward_tmp_bytes_mali(TensorDesc           inputDesc, 
                                                TensorDesc           filterDesc, 
                                                U32*                 bytes,
                                                ForwardRunInfoMali_t forwardRunInfo) {
    EE ret = SUCCESS;
    switch(inputDesc.dt) {
        case DT_F16:{
            ret = fully_connected_infer_forward_tmp_bytes_mali_fp16(inputDesc, filterDesc, bytes, forwardRunInfo);
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

EE fully_connected_mali(GCLHandle_t            handle,
                        TensorDesc             inputDesc, 
                        const GCLMem_t         input,
                        TensorDesc             filterDesc, 
                        std::vector<GCLMem_t>* filter,
                        TensorDesc             biasDesc, 
                        std::vector<GCLMem_t>* bias,
                        U32                    tmpBytes, 
                        GCLMem_t               tmpBuf,
                        TensorDesc             outputDesc, 
                        std::vector<GCLMem_t>* output,
                        ForwardRunInfoMali_t   forwardRunInfo) {
    EE ret = SUCCESS;
    ret = fully_connected_checkpara_mali(handle, inputDesc, input, filterDesc, filter, bias, outputDesc, output);
    switch(inputDesc.dt) {
        case DT_F16:{
            ret = fully_connected_mali_fp16(handle, inputDesc, input, filterDesc, *filter, biasDesc, *bias, tmpBytes, tmpBuf, outputDesc, *output, forwardRunInfo);
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

