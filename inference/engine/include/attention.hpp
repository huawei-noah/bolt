// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _ATTENTION_H
#define _ATTENTION_H

#include "operator.hpp"

class Attention : public Operator {
public:
    Attention(DataType dt, AttentionParamSpec p)
    {
        this->dt = dt;
        this->p = p;
    }

    std::shared_ptr<Operator> clone() override
    {
        std::shared_ptr<Attention> mem =
            std::shared_ptr<Attention>(new Attention(this->dt, this->p));
        *mem = *this;
        return mem;
    }

    OperatorType get_type() override
    {
        return OT_Attention;
    }

    void run() override
    {
        Tensor inputTensor = this->inputTensors[0];
        Tensor outputTensor = this->outputTensors[0];
        auto inDesc = inputTensor.get_desc();
        inDesc.dt = this->dt;
        inputTensor.resize(inDesc);
        CHECK_STATUS(attention(inputTensor, outputTensor, &this->archInfo));
    }

    EE infer_output_tensors_size(
        std::vector<Tensor *> inTensors, std::vector<Tensor *> outTensors) override
    {
        auto inTensor = *inTensors[0];
        auto inDesc = inTensor.get_desc();
        inDesc.dt = this->dt;
        inTensor.resize(inDesc);
        CHECK_STATUS(attention_infer_output_size(&inTensor, this->p, outTensors[0]));
        return SUCCESS;
    }

private:
    AttentionParamSpec p;
};

#endif  // _ATTENTION_H
