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
#include "tensor_computing.h"

class Attention: public Operator
{
public:
    Attention(DataType dt, U32 numHeads, U32 fromSequenceLength, U32 toSequenceLength)
    {
        this->dt = dt;
        this->numHeads = numHeads;
        this->fromSequenceLength = fromSequenceLength;
        this->toSequenceLength = toSequenceLength;
    }

    OperatorType get_op_type() override
    {
        return OT_Attention;
    }
 
    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor = this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();

        Tensor outputTensor = this->outputTensors[0];
        TensorDesc outputDesc = outputTensor.get_desc();

        inputDesc.dt = this->dt;
        CHECK_STATUS(attention(inputDesc, inputTensor.get_val(),
                               outputDesc, outputTensor.get_val(), this->schedule));
        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }

    EE infer_output_tensors_size(Vec<TensorDesc>inDims, Vec<TensorDesc>* outDims) override 
    {
        inDims[0].dt = this->dt;
        CHECK_STATUS(attention_infer_output_size(inDims[0],
                                 this->numHeads, this->fromSequenceLength, this->toSequenceLength,
                                 &((*outDims)[0])));
        return SUCCESS;
    }

private:
    U32 numHeads;
    U32 fromSequenceLength;
    U32 toSequenceLength;
};

#endif //_ATTENTION_H
