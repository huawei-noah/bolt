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
#include "gpu/mali/fp16/multihead_attention_mali_fp16.h"

inline bool find_vector(std::vector<U32> vec, U32 val)
{
    bool find = false;
    for (auto p : vec) {
        if (p == val) {
            find = true;
            break;
        }
    }
    return find;
}

EE multihead_attention_checkpara_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<void *> bias,
    std::vector<void *> layerNormAlpha,
    std::vector<void *> layerNormBeta,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output)
{
    if (nullptr == handle || nullptr == input || nullptr == output) {
        return NULL_POINTER;
    }
    U32 filterNum = filterDesc.size();
    U32 lnNum = layerNormAlpha.size();
    if (filterNum != filter.size() || filterNum != bias.size()) {
        return NOT_MATCH;
    }
    if (lnNum != layerNormBeta.size()) {
        return NOT_MATCH;
    }
    if (filterNum != 4 || lnNum != 2) {
        return NOT_SUPPORTED;
    }
    for (U32 i = 0; i < filterNum; ++i) {
        if (nullptr == filter[i] || nullptr == bias[i]) {
            return NULL_POINTER;
        }
    }
    for (U32 i = 0; i < lnNum; ++i) {
        if (nullptr == layerNormAlpha[i] || nullptr == layerNormBeta[i]) {
            return NULL_POINTER;
        }
    }

    if (inputDesc.df == DF_MKT || inputDesc.df == DF_MTK) {
        U32 m, k;
        U32 fw, fh, fc, fn;
        get_nlp_mkt_val(inputDesc, NULL, &m, &k, NULL);
        if (firstFCSliceNum[0] != firstFCSliceNum[1] || firstFCSliceNum[0] != firstFCSliceNum[2]) {
            return NOT_SUPPORTED;
        }
        if (firstFCSliceNum[0] % matmulSliceLen != 0) {
            return NOT_MATCH;
        }
        if (m != 1) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
        CHECK_STATUS(tensorSelectGet(filterDesc[0], NULL, NULL, &fn, &fc, &fh, &fw));
        if (fh != 1 || fw != 1) {
            return NOT_MATCH;
        }
        if (k != fc) {
            return NOT_MATCH;
        }
    }
    return SUCCESS;
}

EE multihead_attention_infer_output_size_mali(TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    TensorDesc *outputDesc,
    U32 *firstFCSliceNum,
    GCLMemDesc_t gclmemInputDesc,
    GCLMemDesc_t gclmemOutputDesc,
    ForwardRunInfoMali_t forwardRunInfo)
{
    if (inputDesc.df == DF_MTK || inputDesc.df == DF_MKT) {
        DataType dt;
        U32 m, k, t;
        get_nlp_mkt_val(inputDesc, &dt, &m, &k, &t);
        U32 filterNum = filterDesc.size();
        U32 fn;
        tensorSelectGet(filterDesc[filterNum - 1], NULL, NULL, &fn, NULL, NULL, NULL);
        if (filterNum == 1) {
            fn = firstFCSliceNum[2];
        }
        if (outputDesc) {
            *outputDesc = inputDesc;
            (*outputDesc).dims[1] = fn;
        }
        U32 igw, igh, igc;
        U32 ogw, ogh, ogc;
        map_nlp_mkt_to_ncwhc4(m, k, t, &igw, &igh, &igc);
        map_nlp_mkt_to_ncwhc4(m, fn, t, &ogw, &ogh, &ogc);
        CHECK_STATUS(infer_gclmem_desc_ncwhc4(
            igw, igh, igc * 4, 0, 0, ogw, ogh, ogc * 4, dt, dt, gclmemInputDesc, gclmemOutputDesc));
        return SUCCESS;
    }
    return NOT_SUPPORTED;
}

EE multihead_attention_infer_forward_algorithm_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    TensorDesc outputDesc,
    ForwardRunInfoMali_t forwardRunInfo)
{
    if (forwardRunInfo == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    ConvolutionForwardAlgorithm algorithm = (ConvolutionForwardAlgorithm)(forwardRunInfo->algorithm);
    if (algorithm != CONVOLUTION_ALGORITHM_NULL) {
        return SUCCESS;
    }

    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_enable_queue_profiling(handle));
    GCLMem_t input = gcl_create_gclmem();
    GCLMem_t tmpbuf = gcl_create_gclmem();
    GCLMem_t output = gcl_create_gclmem();
    std::vector<void *> filter;
    std::vector<void *> bias;
    std::vector<void *> layerNormAlpha;
    std::vector<void *> layerNormBeta;
    U32 fn[4];

    for (U32 i = 0; i < filterDesc.size(); ++i) {
        tensorSelectGet(filterDesc[i], NULL, NULL, &fn[i], NULL, NULL, NULL);
        GCLMem_t filterTmp = gcl_create_gclmem();
        GCLMem_t biasTmp = gcl_create_gclmem();
        filter.push_back((void *)filterTmp);
        bias.push_back((void *)biasTmp);
    }

    for (U32 i = 0; i < 2; ++i) {
        GCLMem_t alphaTmp = gcl_create_gclmem();
        GCLMem_t betaTmp = gcl_create_gclmem();
        layerNormAlpha.push_back((void *)alphaTmp);
        layerNormBeta.push_back((void *)betaTmp);
    }

    std::vector<ForwardRunInfoMali> runInfos;
    ForwardRunInfoMali runInfo;
    runInfo.algorithm = (I32)CONVOLUTION_ALGORITHM_GEMM;
    std::vector<GCLMemDesc> inputMemDescs;
    std::vector<GCLMemDesc> outputMemDescs;
    std::vector<GCLMemDesc> filterMemDescs0;
    std::vector<GCLMemDesc> filterMemDescs1;
    std::vector<GCLMemDesc> filterMemDescs2;
    std::vector<GCLMemDesc> filterMemDescs3;
    std::vector<std::vector<GCLMemDesc>> filterMemDescs;
    /*0: fc0
     * 1: tn
     * 2: nt
     * 3: fc1
     * 4: fc2
     * 5: fc3*/
    U32 configInfos[6][3][64];
    U32 configNum_fc0 = 0;
    U32 configNum_fc1 = 0;
    U32 configNum_fc2 = 0;
    U32 configNum_fc3 = 0;
    U32 configNum_tn = 0;
    U32 configNum_nt = 0;
    U32 bytes;
    U32 maxBytes = 0;
    U32 maxInputSize = 0;
    U32 maxOutputSize = 0;
    U32 maxFilterSize[4] = {0, 0, 0, 0};
    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};

    for (U32 i = 1; i <= 8; ++i) {
        for (U32 j = 1; j <= 8; ++j) {
            if (i * j <= 2) {
                continue;
            }
            configInfos[0][0][configNum_fc0] = j;
            configInfos[0][1][configNum_fc0] = 1;
            configInfos[0][2][configNum_fc0] = i;
            configInfos[1][0][configNum_tn] = j;
            configInfos[1][1][configNum_tn] = 1;
            configInfos[1][2][configNum_tn] = i;
            configNum_fc0++;
            configNum_tn++;
        }
    }

    for (U32 i = 4; i <= 8; i += 4) {
        for (U32 j = 1; j <= 8; ++j) {
            configInfos[3][0][configNum_fc1] = j;
            configInfos[3][1][configNum_fc1] = 1;
            configInfos[3][2][configNum_fc1] = i;
            configNum_fc1++;
        }
    }

    for (U32 j = 1; j <= 8; j++) {
        configInfos[4][0][configNum_fc2] = j;
        configInfos[4][1][configNum_fc2] = 4;
        configInfos[4][2][configNum_fc2] = 4;
        configInfos[5][0][configNum_fc3] = j;
        configInfos[5][1][configNum_fc3] = 4;
        configInfos[5][2][configNum_fc3] = 4;
        configNum_fc2++;
        configNum_fc3++;
    }

    if (fn[2] % 8 == 0) {
        for (U32 j = 1; j <= 4; j++) {
            configInfos[4][0][configNum_fc2] = j;
            configInfos[4][1][configNum_fc2] = 4;
            configInfos[4][2][configNum_fc2] = 8;
            configNum_fc2++;
        }
    }

    if (fn[3] % 8 == 0) {
        for (U32 j = 1; j <= 4; j++) {
            configInfos[5][0][configNum_fc3] = j;
            configInfos[5][1][configNum_fc3] = 4;
            configInfos[5][2][configNum_fc3] = 8;
            configNum_fc3++;
        }
    }

    for (U32 i = 1; i <= 8; ++i) {
        for (U32 j = 1; j <= 8; ++j) {
            if (i * j <= 2) {
                continue;
            }
            if (i == 6 && j > 7) {
                continue;
            }
            if (i == 7 && j > 6) {
                continue;
            }
            if (i == 8 && j > 5) {
                continue;
            }
            if (matmulSliceLen % i != 0) {
                continue;
            }
            configInfos[2][0][configNum_nt] = j;  // w
            configInfos[2][1][configNum_nt] = 2;  // c
            configInfos[2][2][configNum_nt] = i;  // k
            configNum_nt++;
        }
    }

    for (U32 i = 1; i <= 8; ++i) {
        for (U32 j = 1; j <= 8; ++j) {
            if (i * j <= 2) {
                continue;
            }
            if (i == 5 && j > 6) {
                continue;
            }
            if (i == 6 && j > 5) {
                continue;
            }
            if (i == 7 && j > 4) {
                continue;
            }
            if (i == 8 && j > 3) {
                continue;
            }
            if (matmulSliceLen % i != 0) {
                continue;
            }
            configInfos[2][0][configNum_nt] = j;  // w
            configInfos[2][1][configNum_nt] = 4;  // c
            configInfos[2][2][configNum_nt] = i;  // k
            configNum_nt++;
        }
    }
    std::vector<U32> configNums;
    configNums.push_back(configNum_fc0);
    configNums.push_back(configNum_tn);
    configNums.push_back(configNum_nt);
    configNums.push_back(configNum_fc1);
    configNums.push_back(configNum_fc2);
    configNums.push_back(configNum_fc3);

    DataType dt;
    U32 t, k;
    get_nlp_mkt_val(inputDesc, &dt, NULL, &k, &t);
    std::vector<TensorDesc> biasDesc;
    for (U32 i = 0; i < 2; ++i) {
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        U32 biasNum = fn[i] + 8;
        tmpDesc.stride[0] = biasNum;
        tmpDesc.stride[1] = 1;
        tmpDesc.stride[2] = 1;
        tmpDesc.offset[0] = 0;
        tmpDesc.offset[1] = 0;
        tmpDesc.offset[2] = 0;
        tmpDesc.num = biasNum;
        tmpDesc.byteSize = biasNum * bytesOf(dt);
        tmpDesc.flags = CL_MEM_READ_WRITE;
        tmpDesc.memFormat = DF_NHWC;
        tmpDesc.memType = GCL_MEM_BUF;
        TensorDesc biasDescTmp = tensor1d(dt, fn[i]);
        biasDesc.push_back(biasDescTmp);
        ((GCLMem_t)bias[i])->desc = tmpDesc;
        gcl_create_memory(handle, (GCLMem_t)bias[i]);
    }

    for (U32 i = 2; i < filterDesc.size(); ++i) {
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        U32 biasNum = (fn[i] + 3) / 4;
        tmpDesc.stride[0] = biasNum;
        tmpDesc.stride[1] = 1;
        tmpDesc.stride[2] = 1;
        tmpDesc.offset[0] = 0;
        tmpDesc.offset[1] = 0;
        tmpDesc.offset[2] = 0;
        tmpDesc.num = biasNum;
        tmpDesc.byteSize = biasNum * 4 * bytesOf(dt);
        tmpDesc.flags = CL_MEM_READ_WRITE;
        tmpDesc.memFormat = DF_NHWC;
        tmpDesc.memType = GCL_MEM_IMG_1D;
        TensorDesc biasDescTmp = tensor1d(dt, fn[i]);
        biasDesc.push_back(biasDescTmp);
        ((GCLMem_t)bias[i])->desc = tmpDesc;
        gcl_create_memory(handle, (GCLMem_t)bias[i]);
    }

    for (U32 i = 0; i < 2; ++i) {
        U32 layerNormNum = ALIGN(k, 4);
        if (i == 1) {
            tensorSelectGet(filterDesc[1], NULL, NULL, &layerNormNum, NULL, NULL, NULL);
            layerNormNum = ALIGN(layerNormNum, 4);
        }
        GCLMemDesc tmpDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        tmpDesc.stride[0] = layerNormNum;
        tmpDesc.stride[1] = 1;
        tmpDesc.stride[2] = 1;
        tmpDesc.offset[0] = 0;
        tmpDesc.offset[1] = 0;
        tmpDesc.offset[2] = 0;
        tmpDesc.num = layerNormNum;
        tmpDesc.byteSize = layerNormNum * bytesOf(dt);
        tmpDesc.flags = CL_MEM_READ_WRITE;
        tmpDesc.memFormat = DF_NHWC;
        tmpDesc.memType = GCL_MEM_BUF;
        ((GCLMem_t)layerNormAlpha[i])->desc = tmpDesc;
        ((GCLMem_t)layerNormBeta[i])->desc = tmpDesc;
        gcl_create_memory(handle, (GCLMem_t)layerNormAlpha[i]);
        gcl_create_memory(handle, (GCLMem_t)layerNormBeta[i]);
    }

    U32 runKernelBe = 0;
    U32 runKernelEnd = 0;
    ForwardRunInfoMali bestRunInfo;
    bestRunInfo.algorithm = (I32)CONVOLUTION_ALGORITHM_GEMM;
    for (U32 i = 0; i < configNums.size(); ++i) {
        bestRunInfo.best_w[i] = configInfos[i][0][0];
        bestRunInfo.best_c[i] = configInfos[i][1][0];
        bestRunInfo.best_k[i] = configInfos[i][2][0];
    }
    GCLMemDesc inputMemDesc;
    GCLMemDesc outputMemDesc;
    GCLMemDesc filterMemDesc[4];
    for (U32 i = 0; i < configNums.size(); ++i) {
        runInfo = bestRunInfo;
        for (U32 j = 0; j < configNums[i]; ++j) {
            runInfo.best_w[i] = configInfos[i][0][j];
            runInfo.best_c[i] = configInfos[i][1][j];
            runInfo.best_k[i] = configInfos[i][2][j];
            inputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
            outputMemDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
            for (U32 m = 0; m < filterDesc.size(); m++) {
                filterMemDesc[i] = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
            }
            if (multihead_attention_infer_output_size_mali(inputDesc, filterDesc, NULL,
                    firstFCSliceNum, &inputMemDesc, &outputMemDesc, &runInfo) != SUCCESS) {
                continue;
            }
            if (multihead_attention_transform_filter_bytes_mali(
                    filterDesc, filterMemDesc, &bytes, &runInfo) != SUCCESS) {
                continue;
            }
            if (maxBytes < bytes) {
                maxBytes = bytes;
            }
            if (multihead_attention_infer_forward_tmp_bytes_mali(inputDesc, filterDesc,
                    eltwiseWithLayerNormIn, firstFCSliceNum, matmulSliceLen, &bytes,
                    &runInfo) != SUCCESS) {
                continue;
            }
            if (maxBytes < bytes) {
                maxBytes = bytes;
            }
            if (maxInputSize < inputMemDesc.byteSize) {
                maxInputSize = inputMemDesc.byteSize;
            }
            if (maxOutputSize < outputMemDesc.byteSize) {
                maxOutputSize = outputMemDesc.byteSize;
            }
            if (maxFilterSize[0] < filterMemDesc[0].byteSize) {
                maxFilterSize[0] = filterMemDesc[0].byteSize;
            }
            if (maxFilterSize[1] < filterMemDesc[1].byteSize) {
                maxFilterSize[1] = filterMemDesc[1].byteSize;
            }
            if (maxFilterSize[2] < filterMemDesc[2].byteSize) {
                maxFilterSize[2] = filterMemDesc[2].byteSize;
            }
            if (maxFilterSize[3] < filterMemDesc[3].byteSize) {
                maxFilterSize[3] = filterMemDesc[3].byteSize;
            }
            inputMemDescs.push_back(inputMemDesc);
            outputMemDescs.push_back(outputMemDesc);
            filterMemDescs0.push_back(filterMemDesc[0]);
            filterMemDescs1.push_back(filterMemDesc[1]);
            filterMemDescs2.push_back(filterMemDesc[2]);
            filterMemDescs3.push_back(filterMemDesc[3]);
            runInfos.push_back(runInfo);
        }
        U32 algosNum = runInfos.size();
        if (algosNum == 0) {
            CHECK_STATUS(NOT_SUPPORTED);
        }

        if (maxInputSize > inputMemDescs[0].byteSize || i == 0) {
            inputMemDescs[0].byteSize = maxInputSize;
            if (i > 0) {
                CHECK_STATUS(gcl_release_memory(input))
            }
            input->desc = inputMemDescs[0];
            CHECK_STATUS(gcl_create_memory(handle, input));
        }
        if (maxOutputSize > outputMemDescs[0].byteSize || i == 0) {
            outputMemDescs[0].byteSize = maxOutputSize;
            if (i > 0) {
                CHECK_STATUS(gcl_release_memory(output))
            }
            output->desc = outputMemDescs[0];
            CHECK_STATUS(gcl_create_memory(handle, output));
        }
        filterMemDescs.push_back(filterMemDescs0);
        filterMemDescs.push_back(filterMemDescs1);
        filterMemDescs.push_back(filterMemDescs2);
        filterMemDescs.push_back(filterMemDescs3);
        for (U32 k = 0; k < filterDesc.size(); k++) {
            if (maxFilterSize[k] > filterMemDescs[k][0].byteSize || i == 0) {
                filterMemDescs[k][0].byteSize = maxFilterSize[k];
                if (i > 0) {
                    CHECK_STATUS(gcl_release_memory((GCLMem_t)filter[k]));
                }
                ((GCLMem_t)filter[k])->desc = filterMemDescs[k][0];
                CHECK_STATUS(gcl_create_memory(handle, (GCLMem_t)filter[k]));
            }
        }
        if (maxBytes > tmpbuf->desc.byteSize || i == 0) {
            tmpbuf->desc.byteSize = maxBytes;
            if (i > 0) {
                CHECK_STATUS(gcl_release_subMem(tmpbuf));
                CHECK_STATUS(gcl_release_memory(tmpbuf));
            }
            if (maxBytes) {
                gcl_create_memory(handle, tmpbuf);
            }
        }

        double minTime = DBL_MAX;
        for (U32 ii = 0; ii < algosNum; ii++) {
            input->desc = inputMemDescs[ii];
            output->desc = outputMemDescs[ii];
            ((GCLMem_t)filter[0])->desc = filterMemDescs0[ii];
            ((GCLMem_t)filter[1])->desc = filterMemDescs1[ii];
            ((GCLMem_t)filter[2])->desc = filterMemDescs2[ii];
            ((GCLMem_t)filter[3])->desc = filterMemDescs3[ii];
            U32 best_w = runInfos[ii].best_w[i];
            U32 best_c = runInfos[ii].best_c[i];
            U32 best_k = runInfos[ii].best_k[i];
            runKernelBe = handle->kernelVec->size();
            if (multihead_attention_mali(handle, inputDesc, input, filterDesc, filter, biasDesc,
                    bias, layerNormAlpha, layerNormBeta, multiplyAlpha, multiplyBeta,
                    firstFCSliceNum, matmulSliceLen, eltwiseWithLayerNormIn, activation, maxBytes,
                    tmpbuf, outputDesc, output, &runInfos[ii]) == SUCCESS) {
                runKernelEnd = handle->kernelVec->size();
                runKernelBe = runKernelBe + 1;
                auto kernelInfo = (*handle->kernelVec)[runKernelBe];
                if (kernelInfo.name == "unknow_fill_memory_zero_vec4_f16") {
                    runKernelBe = runKernelBe + 1;
                }
                if (i == 0) {
                    goto R00;
                }
                runKernelBe = runKernelBe + 1;
                kernelInfo = (*handle->kernelVec)[runKernelBe];
                if (kernelInfo.name == "unknow_fill_memory_zero_vec4_f16") {
                    runKernelBe = runKernelBe + 1;
                }
                if (i == 1) {
                    goto R00;
                }
                runKernelBe = runKernelBe + 2;
                if (i == 2) {
                    goto R00;
                }
                runKernelBe = runKernelBe + 1;
                if (i == 3) {
                    goto R00;
                }
                runKernelBe = runKernelBe + 2;
                if (i == 4) {
                    goto R00;
                }
                runKernelBe = runKernelBe + 1;
                if (runKernelBe >= runKernelEnd) {
                    CHECK_STATUS(NOT_MATCH);
                }
            R00:
                gcl_run_kernelVec_timing(handle, runKernelBe, runKernelBe + 1);
                if (minTime > handle->t_execute) {
                    minTime = handle->t_execute;
                    bestRunInfo.best_w[i] = best_w;
                    bestRunInfo.best_c[i] = best_c;
                    bestRunInfo.best_k[i] = best_k;
                }
            }
        }
        inputMemDescs.clear();
        outputMemDescs.clear();
        filterMemDescs.clear();
        filterMemDescs0.clear();
        filterMemDescs1.clear();
        filterMemDescs2.clear();
        filterMemDescs3.clear();
        runInfos.clear();
        CHECK_STATUS(gcl_finish(handle));
        CHECK_STATUS(gcl_clean_kernelVec(handle));
        if (minTime == DBL_MAX) {
            CHECK_STATUS(NOT_SUPPORTED);
        }
    }
    *forwardRunInfo = bestRunInfo;
    CHECK_STATUS(gcl_finish(handle));
    gcl_destroy_gclmem(input);
    gcl_destroy_gclmem(output);
    gcl_destroy_gclmem(tmpbuf);
    for (auto p : filter) {
        gcl_destroy_gclmem(GCLMem_t(p));
    }
    for (auto p : bias) {
        gcl_destroy_gclmem(GCLMem_t(p));
    }
    for (auto p : layerNormAlpha) {
        gcl_destroy_gclmem(GCLMem_t(p));
    }
    for (auto p : layerNormBeta) {
        gcl_destroy_gclmem(GCLMem_t(p));
    }
    runInfos.clear();
    inputMemDescs.clear();
    outputMemDescs.clear();
    filterMemDescs[0].clear();
    filterMemDescs[1].clear();
    filterMemDescs[2].clear();
    filterMemDescs[3].clear();
    CHECK_STATUS(gcl_clean_kernelVec(handle));
    CHECK_STATUS(gcl_clean_programMap(handle));
    CHECK_STATUS(gcl_off_queue_profiling(handle));
    return SUCCESS;
}
EE multihead_attention_transform_filter_bytes_mali(std::vector<TensorDesc> filterDesc,
    GCLMemDesc_t gclmemFilterDesc,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (filterDesc[0].dt) {
        case DT_F16: {
            ret = multihead_attention_transform_filter_bytes_mali_fp16(
                filterDesc, gclmemFilterDesc, bytes, forwardRunInfo);
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

EE multihead_attention_transform_filter_mali(GCLHandle_t handle,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> *fltmemDesc,
    std::vector<void *> fltmem,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (filterDesc[0].dt) {
        case DT_F16: {
            ret = multihead_attention_transform_filter_mali_fp16(
                handle, filterDesc, filter, fltmemDesc, fltmem, forwardRunInfo);
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

EE multihead_attention_infer_forward_tmp_bytes_mali(TensorDesc inputDesc,
    std::vector<TensorDesc> filterDesc,
    std::vector<bool> eltwiseWithLayerNormIn,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    U32 *bytes,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = multihead_attention_infer_forward_tmp_bytes_mali_fp16(inputDesc, filterDesc,
                eltwiseWithLayerNormIn, firstFCSliceNum, matmulSliceLen, bytes, forwardRunInfo);
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

EE multihead_attention_mali(GCLHandle_t handle,
    TensorDesc inputDesc,
    const GCLMem_t input,
    std::vector<TensorDesc> filterDesc,
    std::vector<void *> filter,
    std::vector<TensorDesc> biasDesc,
    std::vector<void *> bias,
    std::vector<void *> layerNormAlpha,
    std::vector<void *> layerNormBeta,
    void *multiplyAlpha,
    void *multiplyBeta,
    U32 *firstFCSliceNum,
    U32 matmulSliceLen,
    std::vector<bool> eltwiseWithLayerNormIn,
    ActivationMode activation,
    U32 tmpBytes,
    GCLMem_t tmpBuf,
    TensorDesc outputDesc,
    GCLMem_t output,
    ForwardRunInfoMali_t forwardRunInfo)
{
    EE ret = SUCCESS;
    ret = multihead_attention_checkpara_mali(handle, inputDesc, input, filterDesc, filter, bias,
        layerNormAlpha, layerNormBeta, multiplyAlpha, multiplyBeta, firstFCSliceNum, matmulSliceLen,
        eltwiseWithLayerNormIn, activation, tmpBuf, outputDesc, output);
    switch (inputDesc.dt) {
        case DT_F16: {
            ret = multihead_attention_mali_fp16(handle, inputDesc, input, filterDesc, filter,
                biasDesc, bias, layerNormAlpha, layerNormBeta, multiplyAlpha, multiplyBeta,
                firstFCSliceNum, matmulSliceLen, eltwiseWithLayerNormIn, activation, tmpBytes,
                tmpBuf, outputDesc, output, forwardRunInfo);
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
