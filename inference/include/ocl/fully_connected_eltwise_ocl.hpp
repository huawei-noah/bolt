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
#include "fully_connected_eltwise.hpp"

class FullyConnectedEltwiseOCL: public FullyConnectedEltwise {
public:
    FullyConnectedEltwiseOCL(DataType dt, U32 numInput, U32 numOutput, std::optional<EltwiseType> eltwiseType):
    FullyConnectedEltwise(dt, numInput, numOutput, eltwiseType) { }

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
        biasMem->desc.stride[0] = (this->numOutput + 3) / 4;
        biasMem->desc.stride[1] = 1;
        biasMem->desc.stride[2] = 1;
        biasMem->desc.offset[0] = 0;
        biasMem->desc.offset[1] = 0;
        biasMem->desc.memType   = GCL_MEM_IMG_1D;
        biasMem->desc.memFormat = DF_NCHW;
        biasMem->desc.num       = (this->numOutput + 3) / 4;
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
        if(biasVal) delete biasVal;
        return SUCCESS;
    }

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor weightTensor = this->weightTensors[0];
        TensorDesc weightDesc = weightTensor.get_desc();

        Tensor biasTensor = this->biasTensors[0];
        TensorDesc biasDesc = biasTensor.get_desc();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        CHECK_STATUS(fully_connected(inputDesc, inputTensor.get_val(),
                                     weightDesc, weightTensor.get_val(),
                                     this->gclTempMem, this->lenOfTemp,
                                     outputDesc, outputTensor.get_val(),
                                     biasDesc, biasTensor.get_val(), this->schedule, &this->oclExtInfo));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        UNUSED(inDims);
        UNUSED(outDims);
        return NOT_SUPPORTED;
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims, Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        TensorDesc inputDesc  = inDims[0];
        U32 ic, ih, iw;
        tensorSelectGet(inputDesc, NULL, NULL, NULL, &ic, &ih, &iw);
        filterDesc4D = tensor4df(this->dt, DF_NCHW, this->numOutput, ic, ih, iw);
        this->oclExtInfo.maliInfo.gclmemInputDesc  = &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        CHECK_STATUS(fully_connected_infer_output_size(inputDesc, filterDesc4D, &((*outDims)[0]), this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }

    virtual U32 infer_tmp_memory_size() override
    {
        TensorDesc inputDesc = this->inputTensors[0].desc;
        U32 bytes = 0;
        CHECK_STATUS(fully_connected_infer_forward_tmp_bytes(inputDesc, filterDesc4D, &bytes, this->schedule));
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
        this->wtm = std::shared_ptr<Tensor>(new Tensor(this->handle));
        OclMemory* wtmMem = (OclMemory*)this->wtm->get_memory();
        wtmMem->set_mem_desc(wtmDesc);
        this->wtm->alloc();
        TensorDesc wtmCpuDesc;
        GCLMem_t weightPtr = this->weightTensors[0].get_val();
        CHECK_STATUS(fully_connected_transform_filter(inputDesc, filterDesc4D, weightPtr, &wtmCpuDesc, this->get_wtm()->get_val(),
            this->schedule, &this->oclExtInfo));

        this->get_wtm()->set_desc(wtmCpuDesc);
        this->weightTensors[0] = *this->get_wtm();
        return SUCCESS;
    }
private:
    TensorDesc filterDesc4D;
};

#endif //_FCELTWISE_OCL_H
