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
#ifndef _multiply_H
#define _multiply_H
#include "operator.hpp"
#include "tensor_computing.h"

class Multiply: public Operator {
public:
    Multiply(DataType dt, F32 scale, F32 bias)
    {
        this->dt = dt;
        this->alpha = scale;
        this->beta = bias;
    }

    OperatorType get_op_type() override
    {
        return OT_Multiply;
    }

    void run() override
    {
        UTIL_TIME_TIC(__CLASS_FUNCTION__)
        Tensor inputTensor =  this->inputTensors[0];
        TensorDesc inputDesc = inputTensor.get_desc();
        Tensor outputTensor = this->outputTensors[0];
        TensorDesc output_desc = outputTensor.get_desc();

        CHECK_STATUS(multiply(&(this->alpha), &(this->beta),
                              inputDesc, inputTensor.get_val(),
                              output_desc, outputTensor.get_val(), this->schedule));

        UTIL_TIME_TOC(__CLASS_FUNCTION__)
    }


    EE infer_output_tensors_size(Vec<TensorDesc> inDims, Vec<TensorDesc>* outDims) override
    {
        CHECK_STATUS(multiply_infer_output_size(inDims[0], &((*outDims)[0])));
        return SUCCESS;
    }


public:
    F32 alpha;
    F32 beta;
};

#endif //_MULTIPLY_H
