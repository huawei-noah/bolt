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
#ifndef _FCELTWISE_H
#define _FCELTWISE_H

#include "weight_operator.hpp"
#include "tensor_computing.h"

class FullyConnected: public WeightOperator {
public:
    FullyConnected(DataType dt, U32 numInput, U32 numOutput,
        U32 numSlice, I32* slicePoint)
    {
        this->dt = dt;
        this->numInput = numInput;
        this->numOutput = numOutput;
        this->numSlice = numSlice;
        slicePoints = Vec<I32>(numSlice);
        memcpy(slicePoints.data(), slicePoint, sizeof(I32)*numSlice);
        this->hasBias = false;
    }

    OperatorType get_op_type() override
    {
        return OT_FC;
    }

    virtual EE init_weight_bias_from_model(U8** modelPtr) = 0;
    virtual EE transform_filter() = 0;
    virtual EE infer_forward_algorithm(HashMap<std::string, std::string> &algorithmMap) {
        UNUSED(algorithmMap);
        return SUCCESS;
    }
public:
    U32 numInput;
    U32 numOutput;
    U32 numSlice;
    Vec<I32> slicePoints;
};

#endif //_FCELTWISE_H
