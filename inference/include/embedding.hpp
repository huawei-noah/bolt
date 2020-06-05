// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _EMBEDDING_H
#define _EMBEDDING_H
#include "weight_operator.hpp"
#include "tensor_computing.h"

class Embedding: public WeightOperator {
public:
    Embedding(DataType dt, U32 inputDim, U32 numOutput, bool transpose)
    {
        this->dt = dt;
        this->inputDim = inputDim;
        this->numOutput = numOutput;
        this->transpose = transpose;
    }

    OperatorType get_op_type() override
    {
        return OT_Embedding;
    }

    virtual EE init_weight_bias_from_model(U8** modelPtr) {
        UNUSED(modelPtr);
        return NOT_SUPPORTED;
    }
protected:
    U32 inputDim;
    U32 numOutput;
    bool transpose;
};

#endif //_EMBEDDING__H
