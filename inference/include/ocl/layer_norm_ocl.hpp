// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _LAYER_NORM_OCL_H
#define _LAYER_NORM_OCL_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "op_type.h"
#include "layer_norm.hpp"

class LayerNormOCL: public LayerNorm {
public:
    LayerNormOCL(DataType dt, U32 weightNum) : LayerNorm(dt, weightNum) {}

    EE init_weight_bias_from_model(U8** modelPtr) override
    {
        auto curOpWs = this->get_weightspec_ptr();
        if(modelPtr == nullptr) {
            this->weightNum = curOpWs.bytes_of_weight / bytesOf(curOpWs.mdt);
        }

        DataType dtNoQ = (DT_F16_8Q == this->dt) ? DT_F16 : this->dt;
        TensorDesc weightDesc = tensor1d(dtNoQ, this->weightNum);
        TensorDesc biasDesc = tensor1d(dtNoQ, this->weightNum);
        U32 weightBytes = tensorNumBytes(weightDesc);

        std::shared_ptr<Tensor> modelWeightTensor(new Tensor(this->handle));
        std::shared_ptr<Tensor> modelBiasTensor(new Tensor(this->handle));
        modelWeightTensor->set_desc(weightDesc);
        modelBiasTensor->set_desc(biasDesc);
        GCLMem_t weightMem = modelWeightTensor->get_val();

        U32 s0, s1, s2;
        U32 num, bytes;
        s0 = (this->weightNum + 3) / 4 * 4;
        s1 = 1;
        s2 = 1;
        num = s0 * s1 * s2;
        bytes = num * bytesOf(dtNoQ);
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
        biasMem->desc.stride[0] = s0;
        biasMem->desc.stride[1] = s1;
        biasMem->desc.stride[2] = s2;
        biasMem->desc.offset[0] = 0;
        biasMem->desc.offset[1] = 0;
        biasMem->desc.memType   = GCL_MEM_BUF;
        biasMem->desc.memFormat = DF_NCHW;
        biasMem->desc.num       = num;
        biasMem->desc.byteSize  = bytes;


        if(modelPtr != nullptr) {
            weightMem->desc.host_ptr = *modelPtr;
            *modelPtr += tensorNumBytes(weightDesc);
        } else {
            weightMem->desc.host_ptr = curOpWs.weight;
        }

        U8* weightTmp = nullptr;
        if((this->weightNum & 3) != 0) {
            weightTmp = (U8*)operator new(weightMem->desc.byteSize);
            memset(weightTmp, 0, weightMem->desc.byteSize);
            memcpy(weightTmp, weightMem->desc.host_ptr, weightBytes);
            weightMem->desc.host_ptr = weightTmp;
        }

        U8* biasVal = nullptr;
        U8* biasTmp = nullptr;
        if(modelPtr != nullptr) {
            if(this->hasBias) {
                biasVal = *modelPtr;
                *modelPtr += tensorNumBytes(biasDesc);
            }
        } else {
            if(this->hasBias) biasVal = curOpWs.vec; 
        }
        if(biasVal) {
            biasMem->desc.flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
            if((this->weightNum & 3) == 0) {
                biasMem->desc.host_ptr = biasVal;
            } else {
                biasTmp = (U8*)operator new(biasMem->desc.byteSize);
                memset(biasTmp, 0, biasMem->desc.byteSize);
                memcpy(biasTmp, biasVal, weightBytes);
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
        if(weightTmp)      delete weightTmp;
        if(biasTmp)        delete biasTmp;
        if(curOpWs.weight) delete curOpWs.weight;
        if(curOpWs.vec)    delete curOpWs.vec;
        return SUCCESS;
    }

    void run() override 
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        this->handle->curOpName = this->get_op_name();
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor weightTensor = this->weightTensors[0];
        Tensor biasTensor = this->biasTensors[0];

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        CHECK_STATUS(layer_normalization(weightTensor.get_val(), biasTensor.get_val(),
                                         inputDesc, inputTensor.get_val(),
                                         outputDesc, outputTensor.get_val(), this->schedule, &this->oclExtInfo));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc in_dim = inDims[0];
        this->oclExtInfo.maliInfo.gclmemInputDesc  = NULL;
        this->oclExtInfo.maliInfo.gclmemOutputDesc = NULL;
        CHECK_STATUS(normalization_infer_output_size(in_dim, &((*outDims)[0]), this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }

    virtual EE infer_gclmem_desc(Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        TensorDesc inputDesc  = this->inputTensors[0].get_desc();
        this->oclExtInfo.maliInfo.gclmemInputDesc  = &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        CHECK_STATUS(normalization_infer_output_size(inputDesc, NULL, this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }
};

#endif //_LAYER_NORM_OCL_H
