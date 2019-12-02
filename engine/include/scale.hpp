// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _SCALE_H
#define _SCALE_H

#include "operator.hpp"
#include "tensor_computing.h"
#include "tensor_desc.h"
#include "model_tools.h"

template <Arch A>
class Scale: public Operator<A>
{
public:
    Scale(DataType dt)
    {
        this->dt = dt;
        this->set_op_type(OT_Scale);        
    }
 
    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        int inputTensorNumber = this->inputTensors.size();
        Tensor inputTensorFirst = this->inputTensors[0];
        Tensor inputTensorSecond;
        if (inputTensorNumber > 1) {
            inputTensorSecond = this->inputTensors[1];
        }
        TensorDesc inputDesc = inputTensorFirst.get_desc();
        auto dataPtr = inputTensorFirst.get_val();
        if (inputTensorNumber == 1) {
            CHECK_STATUS(scale(this->alpha, this->beta, inputDesc, dataPtr.get(), A));
        } else {
            CHECK_STATUS(scale(inputTensorSecond.get_val().get(), nullptr, inputDesc, dataPtr.get(), A));// alpha/beta/inputDesc/data
        }

        Tensor outputTensor = this->outputTensors[0];
        if(inputTensorFirst.get_val().get() != outputTensor.get_val().get()) {
            memcpy(outputTensor.get_val().get(), inputTensorFirst.get_val().get(), tensorNumBytes(inputDesc));
        }
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override 
    {
        CHECK_STATUS(scale_infer_output_size(inDims[0], &((*outDims)[0])));
        return SUCCESS;
    }

    bool can_input_output_the_same() override
    {
        return true;
    }

    void set_scale_alpha(F16* alpha) 
    {
        this->alpha = alpha;
    }

    F16* get_scale_alpha() 
    {
        return this->alpha;
    }

    void set_scale_beta(F16* beta)
    {
        this->beta = beta;
    }

    F16* get_scale_beta()
    {
        return this->beta;
    }

private:
    F16* alpha;
    F16* beta;
};

#endif //_SCALE_H
