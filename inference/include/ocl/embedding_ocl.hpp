// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _EMBEDDING_OCL_H
#define _EMBEDDING_OCL_H
#include "weight_operator.hpp"
#include "tensor_computing.h"
#include "embedding.hpp"

class EmbeddingOCL: public Embedding {
public:
    EmbeddingOCL(DataType dt, U32 inputDim, U32 numOutput, bool transpose) :
    Embedding(dt, inputDim, numOutput, transpose) { }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        this->handle->curOpName = this->get_op_name();
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor weightTensor;
        if (this->weightTensors.size() > 0)
            weightTensor = this->weightTensors[0];
        else
            weightTensor = this->inputTensors[1];
        TensorDesc weightDesc = weightTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();
        CHECK_STATUS(embedding(inputDesc,  inputTensor.get_val(), 
                               weightDesc, weightTensor.get_val(),
                               outputDesc, outputTensor.get_val(),
                               this->inputDim, this->numOutput,
                               this->transpose, this->dt,
                               this->schedule, &this->oclExtInfo));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        TensorDesc inputDesc  = inDims[0];
        this->oclExtInfo.maliInfo.gclmemInputDesc  = NULL;
        this->oclExtInfo.maliInfo.gclmemOutputDesc = NULL;
        CHECK_STATUS(embedding_infer_output_size(inputDesc, &((*outDims)[0]), this->inputDim, this->numOutput, this->dt, this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }

    virtual EE infer_gclmem_desc(Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        TensorDesc inputDesc  = this->inputTensors[0].get_desc();
        this->oclExtInfo.maliInfo.gclmemInputDesc  = &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        CHECK_STATUS(embedding_infer_output_size(inputDesc, NULL, this->inputDim, this->numOutput, this->dt, this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }

    EE init_weight_bias_from_model(U8** modelPtr) override
    {
        TensorDesc weightDesc;
        if (transpose)
            weightDesc = tensor2df(this->dt, DF_TRANSPOSE, this->numOutput, this->inputDim);
        else
            weightDesc = tensor2df(this->dt, DF_NORMAL, this->inputDim, this->numOutput);
        std::shared_ptr<Tensor> modelWeightTensor(new Tensor(this->handle));
        modelWeightTensor->set_desc(weightDesc);

        GCLMem_t weightMem = modelWeightTensor->get_val();
        U32 s0, s1, s2;
        U32 num, bytes;
        s0 = weightDesc.dims[0];
        s1 = weightDesc.dims[1];
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


        bool set_ptr = false;
        auto curOpWs = this->get_weightspec_ptr();
        if(modelPtr != nullptr){
            weightMem->desc.host_ptr = *modelPtr;
            *modelPtr += tensorNumBytes(weightDesc);
            set_ptr = true;
        } else {
            if (curOpWs.weight != nullptr) {
                weightMem->desc.host_ptr = curOpWs.weight;
                set_ptr = true;
            }
        }
        if(set_ptr) {
            modelWeightTensor->alloc();
            this->weightTensors.push_back(*modelWeightTensor.get());
            if(curOpWs.weight) delete curOpWs.weight;
        }
        return SUCCESS;
    }
};

#endif //_EMBEDDING_OCL_H
