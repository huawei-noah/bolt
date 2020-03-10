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


#ifndef _ELTWISE_CPU_H
#define _ELTWISE_CPU_H

#include "operator.hpp"
#include "eltwise.hpp"

class EltwiseCPU: public Eltwise {
public:
    EltwiseCPU(EltwiseMode eltMode, I32 coeffSize, F32* coeffValues):Eltwise(eltMode, coeffSize, coeffValues){}

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Vec<TensorDesc> inputDesc;
        Vec<void*> inputPtr;
        for (Tensor tensorIn: this->inputTensors) {
            inputDesc.push_back(tensorIn.get_desc());
            inputPtr.push_back((void*)tensorIn.get_val());
        }
        auto outputDesc = this->outputTensors[0].get_desc();
        auto outputPtr = this->outputTensors[0].get_val();

        if (inputDesc.size() == 2
            && inputDesc[1].nDims == 2
            && inputDesc[0].dims[inputDesc[0].nDims-1] == inputDesc[1].dims[0]) {
            CHECK_STATUS(scale(this->inputTensors[1].get_val(), nullptr, outputDesc, this->inputTensors[0].get_val(), outputDesc, NULL, this->schedule));
            memcpy(outputPtr, this->inputTensors[0].get_val(), tensorNumBytes(outputDesc));
        } else {
            CHECK_STATUS(eltwise(inputDesc, inputPtr, outputDesc, outputPtr, this->eltMode, this->schedule));
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        CHECK_STATUS(eltwise_infer_output_size(inDims, &((*outDims)[0]), this->schedule));
        return SUCCESS;
    }
};

#endif //_ELTWISE_CPU_H
