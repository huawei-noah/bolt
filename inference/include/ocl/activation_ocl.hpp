// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _ACTIVATION_OCL_H
#define _ACTIVATION_OCL_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "activation.hpp"

class ActivationOCL: public Activation
{
public:
    /**
    @param mode
    */
    ActivationOCL(ActivationMode activeMode): Activation(activeMode) {}

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        GCLMem_t inPtr = inputTensor.get_val();
        GCLMem_t outPtr = outputTensor.get_val();
        if(inPtr->mem != outPtr->mem) CHECK_STATUS(NOT_SUPPORTED);
        TensorDesc inputDesc = inputTensor.get_desc();
        CHECK_STATUS(activation(inputDesc, outPtr, this->activeMode, this->schedule, &this->oclExtInfo));

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        UNUSED(inDims);
        UNUSED(outDims);
        CHECK_STATUS(NOT_SUPPORTED);
        return NOT_SUPPORTED;
    }

    
    virtual EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims, Vec<GCLMemDesc>* gclmemInputDesc, Vec<GCLMemDesc>* gclmemOutputDesc) override
    {
        this->oclExtInfo.maliInfo.gclmemInputDesc =  &((*gclmemInputDesc)[0]);
        this->oclExtInfo.maliInfo.gclmemOutputDesc = &((*gclmemOutputDesc)[0]);
        CHECK_STATUS(activation_infer_output_size(inDims[0], &((*outDims)[0]), this->schedule, &this->oclExtInfo));
        return SUCCESS;
    }
};

#endif //_ACTIVATION_OCL_H
