// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SCALE_GPU_H
#define _SCALE_GPU_H

#include <iostream>
#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"
#include "scale.hpp"

class ScaleOCL: public Scale
{
public:
    ScaleOCL(DataType dt, int numChannels, int numSource) :
    Scale(dt, numChannels, numSource) {} 
    virtual EE init_weight_bias_from_model(U8** modelPtr) override
    {
        auto curOpWs = this->get_weightspec_ptr();
        if(modelPtr == nullptr){
            this->numChannels = curOpWs.bytes_of_weight / UNI_MAX(1, bytesOf(curOpWs.mdt));
        }
        if(this->numChannels == 0) return SUCCESS;

        TensorDesc weightDesc = tensor1d(this->dt, this->numChannels);
        TensorDesc biasDesc   = weightDesc;
        std::shared_ptr<Tensor> modelWeightTensor(new Tensor(this->handle));
        std::shared_ptr<Tensor> modelBiasTensor(new Tensor(this->handle));
        modelWeightTensor->set_desc(weightDesc);
        modelBiasTensor->set_desc(biasDesc);
        GCLMem_t weightMem = modelWeightTensor->get_val();
        U32 s0, s1, s2;
        U32 num, bytes;
        s0 = (this->numChannels + 3) / 4 * 4;
        s1 = 1;
        s2 = 1;
        num = s0 * s1 * s2;
        bytes = num * 4 * bytesOf(this->dt);
        weightMem->desc.stride[0] = s0;
        weightMem->desc.stride[1] = s1;
        weightMem->desc.stride[2] = s2;
        weightMem->desc.offset[0] = 0;
        weightMem->desc.offset[1] = 0;
        weightMem->desc.offset[2] = 0;
        weightMem->desc.memType   = GCL_MEM_BUF;
        weightMem->desc.memFormat = DF_NCHW;
        weightMem->desc.num       = num;
        weightMem->desc.byteSize  = bytes;
        weightMem->desc.flags     = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        GCLMem_t biasMem = modelBiasTensor->get_val();
        *biasMem = *weightMem;
        biasMem->desc.flags = CL_MEM_READ_WRITE;

        U32 weightBytes = tensorNumBytes(weightDesc);
        U8* weightTmp = nullptr;
        if(modelPtr != nullptr){
            weightMem->desc.host_ptr = *modelPtr;
            *modelPtr += weightBytes;
        } else {
            weightMem->desc.host_ptr = curOpWs.weight;
        }
        if((this->numChannels & 3) != 0){
            weightTmp = (U8*)operator new(weightMem->desc.byteSize);
            memset(weightTmp, 0, weightMem->desc.byteSize);
            memcpy(weightTmp, weightMem->desc.host_ptr, this->numChannels * bytesOf(this->dt));
            weightMem->desc.host_ptr = weightTmp;
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
            if((this->numChannels & 3) == 0){
                biasMem->desc.host_ptr = biasVal;
            } else {
                biasTmp = (U8*)operator new(biasMem->desc.byteSize);
                memset(biasTmp, 0, biasMem->desc.byteSize);
                memcpy(biasTmp, biasVal, this->numChannels * bytesOf(this->dt));
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
        if(weightTmp) delete weightTmp;
        if(biasTmp)   delete biasTmp;
        if(biasVal)   delete biasVal;
        return SUCCESS;
    }

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        int inputNum = this->inputTensors.size();
        Tensor inputTensor = this->inputTensors[this->dataID];
        Tensor outputTensor = this->outputTensors[0];
        GCLMem_t inPtr = inputTensor.get_val();
        GCLMem_t outPtr = outputTensor.get_val();
        TensorDesc inputDesc  = inputTensor.get_desc();
        TensorDesc outputDesc = outputTensor.get_desc();
        U32 ic;
        tensorSelectGet(inputDesc, NULL, NULL, NULL, &ic, NULL, NULL);

        if(inputNum == 1 && ic != this->numChannels) CHECK_STATUS(NOT_MATCH);
        if(inputNum == 1 && weightTensors.size() == 0) CHECK_STATUS(NOT_MATCH);
        if(inputNum > 1){
            U32 cNum = this->inputTensors[0].get_desc().dims[2];
            for(int i = 1; i < inputNum; i++){
                if(cNum != this->inputTensors[i].get_desc().dims[2]) CHECK_STATUS(NOT_MATCH);
            }
        }
        if (inputNum == 1) {
            CHECK_STATUS(scale(this->weightTensors[0].get_val(), this->biasTensors[0].get_val(),
                               inputDesc, inPtr, outputDesc, outPtr, this->schedule, &this->oclExtInfo));
        } else {
            CHECK_STATUS(scale(this->inputTensors[1 - this->dataID].get_val(), NULL,
                               inputDesc, inPtr, outputDesc, outPtr, this->schedule, &this->oclExtInfo));
        }
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override 
    {
        UNUSED(inDims);
        UNUSED(outDims);
        return NOT_SUPPORTED;
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims, Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        if(inDims.size() > 1 && tensorNumElements(inDims[1]) > tensorNumElements(inDims[0])) this->dataID = 1;
        TensorDesc inputDesc  = inDims[dataID];
        this->oclExtInfo.maliInfo.gclmemInputDesc  = &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        CHECK_STATUS(scale_infer_output_size(inputDesc, &((*outDims)[0]), this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }
};

#endif //_SCALE_CPU_H
