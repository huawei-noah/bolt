// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _CONCAT_CPU_H
#define _CONCAT_CPU_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "concat.hpp"

class ConcatCPU: public Concat {
public:
    ConcatCPU(U32 concatDim) : Concat(concatDim) {}

    virtual void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)

        Vec<TensorDesc> inputDesc;
        Vec<void*> inputPtr;
        Vec<F32> inputScales;

        for (Tensor tensorIn: this->inputTensors) {
            inputDesc.push_back(tensorIn.get_desc());
            inputPtr.push_back((void*)tensorIn.get_val());
            inputScales.push_back(tensorIn.get_scale());
        }
        auto outputDesc = this->outputTensors[0].get_desc();
        auto outputPtr = this->outputTensors[0].get_val();
        F32 outputScale = 1.0;

        CHECK_STATUS(concat(inputDesc, inputPtr, inputScales.data(), outputDesc, outputPtr, &outputScale, this->concatDim, this->schedule));

        if (DT_I8 == outputDesc.dt) {
            this->outputTensors[0].set_scale(outputScale);
        }

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override
    {
        CHECK_STATUS(concat_infer_output_size(inDims, &((*outDims)[0]), this->concatDim, this->schedule));
        return SUCCESS;
    }
};

#endif //_CONCAT_CPU_H
