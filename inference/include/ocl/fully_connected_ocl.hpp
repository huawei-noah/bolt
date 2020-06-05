// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


/**
 * Project deploy
 */
#ifndef _FCELTWISE_OCL_H
#define _FCELTWISE_OCL_H

#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "fully_connected.hpp"

class FullyConnectedOCL: public FullyConnected {
public:
    FullyConnectedOCL(DataType dt, U32 numInput, U32 numOutput, U32 numSlice,
        I32 *slicePoints):
    FullyConnected(dt, numInput, numOutput, numSlice, slicePoints) { }

    virtual EE init_weight_bias_from_model(U8** modelPtr) override
    {
        auto curOpWs = this->get_weightspec_ptr();
        if(modelPtr == nullptr){
            this->numInput = curOpWs.bytes_of_weight / this->numOutput / UNI_MAX(1, bytesOf(curOpWs.mdt));
        }
        TensorDesc weightDesc = tensor2df(this->dt, DF_NORMAL, this->numOutput, this->numInput);
        TensorDesc biasDesc   = tensor1d(this->dt, this->numOutput);

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor(this->handle));
        std::shared_ptr<Tensor> modelBiasTensor(new Tensor(this->handle));
        modelWeightTensor->set_desc(weightDesc);
        modelBiasTensor->set_desc(biasDesc);
        GCLMem_t weightMem = modelWeightTensor->get_val();
        U32 s0, s1, s2;
        U32 num, bytes;
        s0 = this->numInput;
        s1 = this->numOutput;
        s2 = 1;
        num = s0 * s1 * s2;
        bytes = num * bytesOf(this->dt);
        weightMem->desc.stride[0] = s0;
        weightMem->desc.stride[1] = s1;
        weightMem->desc.stride[2] = s2;
        weightMem->desc.offset[0] = 0;
        weightMem->desc.offset[1] = 0;
        weightMem->desc.offset[2] = 0;
        weightMem->desc.memType   = GCL_MEM_BUF;
        weightMem->desc.memFormat = DF_NORMAL;
        weightMem->desc.num       = num;
        weightMem->desc.byteSize  = bytes;
        weightMem->desc.flags     = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;

        GCLMem_t biasMem = modelBiasTensor->get_val();
        biasMem->desc.stride[0] = (this->numOutput + 3) / 4 * 4;
        biasMem->desc.stride[1] = 1;
        biasMem->desc.stride[2] = 1;
        biasMem->desc.offset[0] = 0;
        biasMem->desc.offset[1] = 0;
        biasMem->desc.memType   = GCL_MEM_BUF;
        biasMem->desc.memFormat = DF_NCHW;
        biasMem->desc.num       = (this->numOutput + 3) / 4 * 4;
        biasMem->desc.byteSize  = (this->numOutput + 3) / 4 * 4 * bytesOf(this->dt);

        if(modelPtr != nullptr){
            weightMem->desc.host_ptr = *modelPtr;
            *modelPtr += tensorNumBytes(weightDesc);
        } else {
            weightMem->desc.host_ptr = curOpWs.weight;
        }

        U8* biasVal = nullptr;
        U8* biasTmp = nullptr;
        if(modelPtr != nullptr){
            if(this->hasBias){
                biasVal = *modelPtr;
                *modelPtr += tensorNumBytes(biasDesc);
            }
        } else {
            if(this->hasBias) biasVal = curOpWs.vec; 
        }

        if(biasVal){
            biasMem->desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
            if((this->numOutput & 3) == 0){
                biasMem->desc.host_ptr = biasVal;
            } else {
                biasTmp = (U8*)operator new(biasMem->desc.byteSize);
                memset(biasTmp, 0, biasMem->desc.byteSize);
                memcpy(biasTmp, biasVal, this->numOutput * bytesOf(this->dt));
                biasMem->desc.host_ptr = biasTmp;
            }
        } else {
            biasMem->desc.host_ptr = nullptr;
            biasMem->desc.flags = CL_MEM_READ_WRITE;
        }
 
        modelWeightTensor->alloc();
        modelBiasTensor->alloc();
        this->weightTensors.push_back(*modelWeightTensor.get());
        this->biasTensors.push_back(*modelBiasTensor.get());
        if(biasTmp) delete biasTmp;
        if(curOpWs.weight) delete curOpWs.weight;
        if(curOpWs.vec)    delete curOpWs.vec;
        return SUCCESS;
    }

    virtual EE infer_forward_algorithm(HashMap<std::string, std::string> &algorithmMap) override
    {
        TensorDesc inputDesc = (this->inputTensors[0]).get_desc();
        std::vector<TensorDesc> outputDescs;
        this->oclExtInfo.maliInfo.forwardRunInfo->algorithm = CONVOLUTION_ALGORITHM_NULL;
        for(U32 i = 0; i < this->outputTensors.size(); ++i) outputDescs.push_back(this->outputTensors[i].get_desc());
        if (algorithmMap.find(this->name) != algorithmMap.end()) {
            I32 algo[4];
            Operator::getAlgorithmInfoFromMap(algorithmMap, this->name, algo, 4);
            this->runInfo.algorithm = (ConvolutionForwardAlgorithm)algo[0];
            this->runInfo.best_w[0] = algo[1];
            this->runInfo.best_c[0] = algo[2];
            this->runInfo.best_k[0] = algo[3];
        } else {
            CHECK_STATUS(fully_connected_infer_forward_algorithm(inputDesc, filterDesc4D, outputDescs,this->schedule, &this->oclExtInfo));
                I32 algo[4];
                algo[0] = this->runInfo.algorithm;
                algo[1] = this->runInfo.best_w[0];
                algo[2] = this->runInfo.best_c[0];
                algo[3] = this->runInfo.best_k[0];
                Operator::setAlgorithmInfoToMap(algorithmMap, this->name, algo, 4);
        }
        return SUCCESS;
    }

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        this->handle->curOpName = this->get_op_name();
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc  = inputTensor.get_desc();
        TensorDesc weightDesc = this->weightTensors[0].get_desc();
        TensorDesc biasDesc   = this->biasTensors[0].get_desc();
        TensorDesc outputDesc = this->outputTensors[0].get_desc();
        std::vector<GCLMem_t> outputGCLMemArray;
        for(U32 i = 0; i < numSlice; i++) outputGCLMemArray.push_back(this->outputTensors[i].get_val());

        Tensor outputTensor = this->outputTensors[0];

        CHECK_STATUS(fully_connected(inputDesc, inputTensor.get_val(),
                                     weightDesc, &wtmGCLMemArray,
                                     this->temp->get_val(), this->lenOfTemp,
                                     outputDesc, &outputGCLMemArray,
                                     biasDesc, &biasGCLMemArray, this->schedule, &this->oclExtInfo));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inputDesc  = inDims[0];
        U32 ic, ih, iw;
        if(inputDesc.df == DF_NCHW) tensorSelectGet(inputDesc, NULL, NULL, NULL, &ic, &ih, &iw);
        if(inputDesc.df == DF_MKT) {
            iw = 1;
            ih = 1;
            ic = inputDesc.dims[1];
        }
        filterDesc4D = tensor4df(this->dt, DF_NCHW, this->numOutput, ic, ih, iw);
        this->oclExtInfo.maliInfo.gclmemInputDesc  = NULL;
        this->oclExtInfo.maliInfo.gclmemOutputDesc = NULL;
        CHECK_STATUS(fully_connected_infer_output_size(inputDesc, filterDesc4D, &((*outDims)[0]), this->schedule, &this->oclExtInfo));
        if(this->numSlice > 1) {
                for(U32 i = 0; i < this->numSlice; ++i) {
                    (*outDims)[i] = (*outDims)[0];
                    (*outDims)[i].dims[1] = this->slicePoints[i];
                }
        }
        return SUCCESS;
    }

    virtual EE infer_gclmem_desc(Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        TensorDesc inputDesc  = this->inputTensors[0].get_desc();
        this->oclExtInfo.maliInfo.gclmemInputDesc  = &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        CHECK_STATUS(fully_connected_infer_output_size(inputDesc, filterDesc4D, NULL, this->schedule, &this->oclExtInfo));
        if(this->numSlice > 1) {
            U32 h_str = (*gclmemOutputDesc)[0].stride[0];
            U32 w_str = (*gclmemOutputDesc)[0].stride[1];
            U32 c_str = (this->slicePoints[0] + 3) / 4;
            U32 num   = h_str * w_str * c_str * 4;
            (*gclmemOutputDesc)[0].stride[2] = c_str;
            (*gclmemOutputDesc)[0].num       = num;
            (*gclmemOutputDesc)[0].byteSize  = num * bytesOf(this->dt);
            for(U32 i = 1; i < this->numSlice; ++i) {
                (*gclmemOutputDesc)[i] = (*gclmemOutputDesc)[0];
                c_str = (this->slicePoints[i] + 3) / 4;
                num   = h_str * w_str * c_str * 4;
                (*gclmemOutputDesc)[i].stride[2] = c_str;
                (*gclmemOutputDesc)[i].num       = num;
                (*gclmemOutputDesc)[i].byteSize  = num * bytesOf(this->dt);
            }
        }
        return SUCCESS;
    }

    virtual U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDesc = this->inputTensors[0].get_desc();
        U32 bytes = 0;
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(inputDesc, filterDesc4D, &bytes, this->schedule, &this->oclExtInfo));
        return bytes;
    }

    virtual GCLMemDesc infer_wtm_memory_size_mali() override
    {
        U32 stride[3] = {0, 0, 0};
        U32 offset[3] = {0, 0, 0};
        GCLMemDesc gclmemWtmDesc = gcl_mem_desc(stride, offset, DT_U8, DF_NCWHC4);
        U32 bytes = 0;
        this->oclExtInfo.maliInfo.gclmemFilterDesc = &gclmemWtmDesc;
        CHECK_STATUS(fully_connected_transform_filter_bytes(filterDesc4D, &bytes, this->schedule, &this->oclExtInfo));
        return gclmemWtmDesc;
    }

    virtual EE transform_filter() override
    {
        TensorDesc inputDesc = this->inputTensors[0].get_desc();
        auto wtmDesc = this->infer_wtm_memory_size_mali();
        if(this->numSlice == 1) {
            this->wtm = std::shared_ptr<Tensor>(new Tensor(this->handle));
            OclMemory* wtmMem = (OclMemory*)this->wtm->get_memory();
            wtmMem->set_mem_desc(wtmDesc);
            this->wtm->alloc();
            wtmGCLMemArray.push_back((GCLMem_t)(this->get_wtm()->get_val()));
        } else {
            U32 item_c = this->oclExtInfo.maliInfo.forwardRunInfo->best_c[0];
            U32 item_k = this->oclExtInfo.maliInfo.forwardRunInfo->best_k[0];
            for(U32 i = 0; i < this->numSlice; i++) {
                GCLMemDesc tmpDesc = wtmDesc;
                U32 s0 = wtmDesc.stride[0];
                U32 s1 = wtmDesc.stride[1];
                U32 s2 = (this->slicePoints[i] + item_k - 1) / item_k;
                U32 num = s0 * s1 * s2 * item_c * item_k / (item_k >> 2);
                tmpDesc.stride[2] = s2;
                tmpDesc.num       = num;
                tmpDesc.byteSize  = num * bytesOf(this->dt);
                auto tmpWtm = std::shared_ptr<Tensor>(new Tensor(this->handle));
                auto tmpMem = (OclMemory*)tmpWtm->get_memory();
                tmpMem->set_mem_desc(tmpDesc);
                tmpWtm->alloc();
                if(i == 0) {
                    this->wtm = tmpWtm;
                } else {
                    wtmArray.push_back(tmpWtm); 
                }
                wtmGCLMemArray.push_back((GCLMem_t)tmpWtm->get_val());
            }
        }
        TensorDesc wtmCpuDesc;
        GCLMem_t weightPtr = this->weightTensors[0].get_val();
        CHECK_STATUS(fully_connected_transform_filter(inputDesc, filterDesc4D, weightPtr, &wtmCpuDesc, &wtmGCLMemArray,
            this->schedule, &this->oclExtInfo));
        if(this->numSlice == 1) {
            this->get_wtm()->set_desc(wtmCpuDesc);
            this->weightTensors[0] = *this->get_wtm();
        }

        GCLMem_t biasBuf = biasTensors[0].get_val();
        U32 size[3] = {1, 1, 1};
        if(this->numSlice > 1) {
            U32 offset[4] = {0, 0, 0, 0};
            for(U32 i = 0; i < this->numSlice; ++i) {
                TensorDesc tmpCpuDesc = wtmCpuDesc;
                tmpCpuDesc.dims[3] = this->slicePoints[i];
                if(i == 0) {
                    this->get_wtm()->set_desc(tmpCpuDesc);
                    this->weightTensors[0] = *this->get_wtm();
                } else {
                    wtmArray[i - 1]->set_desc(tmpCpuDesc);
                    this->weightTensors.push_back(*(wtmArray[i - 1].get()));
                }

                GCLMemDesc tmpDesc = biasBuf->desc;
                U32 spNum = this->slicePoints[i];
                size[0] = (spNum + 3) / 4;
                tmpDesc.stride[0] = (spNum + 3) / 4;
                tmpDesc.num       = (spNum + 3) / 4;
                tmpDesc.byteSize  = spNum * bytesOf(this->dt);
                tmpDesc.memType   = GCL_MEM_IMG_1D;
                tmpDesc.has_alloc = false;
                auto tmpBias = std::shared_ptr<Tensor>(new Tensor(this->handle));
                auto tmpMem  = (OclMemory*)tmpBias->get_memory();
                tmpMem->set_mem_desc(tmpDesc);
                tmpBias->alloc();
                CHECK_STATUS(gcl_trans_memory(this->handle.get(), biasBuf, tmpBias->get_val(), size, DEVICE_BUF_TO_IMG, CL_TRUE, offset))
                offset[0] += spNum * bytesOf(this->dt);
                biasImgArray.push_back(tmpBias);
                biasGCLMemArray.push_back((GCLMem_t)tmpBias->get_val());
            }
        } else {
            if(inputDesc.df == DF_MKT) {
                GCLMemDesc tmpDesc = biasBuf->desc;
                U32 spNum = tmpDesc.stride[0];
                size[0] = (spNum + 3) / 4;
                tmpDesc.stride[0] = (spNum + 3) / 4;
                tmpDesc.num       = (spNum + 3) / 4;
                tmpDesc.byteSize  = spNum * bytesOf(this->dt);
                tmpDesc.memType   = GCL_MEM_IMG_1D;
                tmpDesc.has_alloc = false;
                auto tmpBias = std::shared_ptr<Tensor>(new Tensor(this->handle));
                auto tmpMem  = (OclMemory*)tmpBias->get_memory();
                tmpMem->set_mem_desc(tmpDesc);
                tmpBias->alloc();
                CHECK_STATUS(gcl_trans_memory(this->handle.get(), biasBuf, tmpBias->get_val(), size, DEVICE_BUF_TO_IMG, CL_TRUE))
                biasImgArray.push_back(tmpBias);
                biasGCLMemArray.push_back((GCLMem_t)tmpBias->get_val());
            } else {
                biasGCLMemArray.push_back(biasBuf);
            }
        }
        return SUCCESS;
    }
private:
    TensorDesc filterDesc4D;
    std::vector<GCLMem_t> wtmGCLMemArray;
    std::vector<std::shared_ptr<Tensor>> wtmArray;
    std::vector<GCLMem_t> biasGCLMemArray;
    std::vector<std::shared_ptr<Tensor>> biasImgArray;
};

#endif //_FCELTWISE_OCL_H
